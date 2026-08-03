"""Protein datasets, collators, and packed-sidecar construction."""


# Source: data/packed.py


from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phaseflow.protein.contracts import assert_no_eval_only_training_path


LABEL_KEYS = (
    "residue_target",
    "residue_mask",
    "residue_weight",
    "core_target",
    "core_mask",
    "start_target",
    "end_target",
    "boundary_weight",
    "safe_background_mask",
    "ignore_mask",
)
PHASEFLOW_LLPS_HIDDEN_KEY = "phaseflow_llps_hidden"
LEGACY_LLPS_HIDDEN_KEY = "phase" + "gt_hidden"

RUNTIME_ARRAYS = (
    "plm",
    "biophys",
    "aa_ids",
    "modality_mask",
    "reliability",
    "neighbors",
    "edge_attr",
    "neighbor_mask",
    "neighbor_edge_type",
    PHASEFLOW_LLPS_HIDDEN_KEY,
    *LABEL_KEYS,
)


@dataclass(frozen=True)
class PackedPackedProteinRow:
    protein_id: str
    sequence_sha256: str
    sequence: str
    length: int
    shard_id: int
    residue_offset: int


class DPRV2HotpathSidecar:
    """Runtime reader for DPR v2 hot-path packed sidecar.

    This cache contains only immutable training inputs. PhaseFlow bridge outputs are
    intentionally absent and remain online trainable model computation.
    """

    def __init__(
        self,
        *,
        data_root: str | Path,
        sidecar_root: str | Path | None = None,
        mmap: bool = True,
        max_open_shards: int = 8,
        allow_eval_only_sidecar: bool = False,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.sidecar_root = Path(sidecar_root).resolve() if sidecar_root is not None else self.data_root / "packed/hotpath_v1"
        if not allow_eval_only_sidecar:
            assert_no_eval_only_training_path(self.sidecar_root)
        manifest_path = self.sidecar_root / "manifest.parquet"
        shards_path = self.sidecar_root / "shards.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing DPR v2 hotpath manifest: {manifest_path}")
        if not shards_path.exists():
            raise FileNotFoundError(f"Missing DPR v2 hotpath shards manifest: {shards_path}")
        self.manifest = pd.read_parquet(manifest_path)
        self.shards = pd.read_parquet(shards_path)
        self.mmap = bool(mmap)
        self.max_open_shards = max(1, int(max_open_shards))
        self._rows: dict[tuple[str, str], PackedPackedProteinRow] = {}
        for row in self.manifest.itertuples(index=False):
            key = (str(row.protein_id), str(row.sequence_sha256))
            self._rows[key] = PackedPackedProteinRow(
                protein_id=str(row.protein_id),
                sequence_sha256=str(row.sequence_sha256),
                sequence=str(row.sequence),
                length=int(row.length),
                shard_id=int(row.shard_id),
                residue_offset=int(row.residue_offset),
            )
        self._shard_paths = {
            int(row.shard_id): self._resolve_sidecar_path(str(row.path))
            for row in self.shards.itertuples(index=False)
        }
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def sample_from_schedule_row(self, row: Any) -> dict[str, Any]:
        protein_id = str(row.protein_id)
        sequence_sha256 = str(row.sequence_sha256)
        packed = self._rows.get((protein_id, sequence_sha256))
        if packed is None:
            raise KeyError(f"{protein_id}/{sequence_sha256} not present in DPR v2 hotpath sidecar")
        length = int(packed.length)
        offset = int(packed.residue_offset)
        shard = self._open_shard(int(packed.shard_id))
        pool = str(row.pool)
        tier = str(getattr(row, "training_tier", ""))
        sample_weight = float(getattr(row, "sample_weight", 1.0))
        sample: dict[str, Any] = {
            "sample_id": f"{protein_id}:{pool}:full",
            "protein_id": protein_id,
            "sequence_sha256": sequence_sha256,
            "sequence": packed.sequence,
            "kind": "protein",
            "pool": pool,
            "tier": tier,
            "length": length,
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
        }
        sample["bag_label"], sample["bag_weight"] = _bag_target(pool, sample_weight)
        sample["driver_label"], sample["driver_weight"] = _driver_target(pool, tier, sample_weight)
        sample["partner_label"], sample["partner_weight"] = _partner_target(pool, tier, sample_weight)
        sample["general_label"], sample["general_weight"] = _general_target(pool, sample_weight)
        sl = slice(offset, offset + length)
        for key in ("plm", "biophys", "modality_mask", "reliability", "edge_attr", PHASEFLOW_LLPS_HIDDEN_KEY):
            sample[key] = _tensor(shard[key][sl], np.float16)
        sample["aa_ids"] = _tensor(shard["aa_ids"][sl], np.int16)
        sample["neighbors"] = _tensor(shard["neighbors"][sl], np.int64)
        sample["neighbor_mask"] = _tensor(shard["neighbor_mask"][sl], np.bool_)
        sample["neighbor_edge_type"] = _tensor(shard["neighbor_edge_type"][sl], np.int64)
        for key in LABEL_KEYS:
            sample[key] = _tensor(shard[key][sl], np.float32)
        return sample

    def _open_shard(self, shard_id: int) -> dict[str, Any]:
        if shard_id in self._cache:
            arrays = self._cache.pop(shard_id)
            self._cache[shard_id] = arrays
            return arrays
        if shard_id not in self._shard_paths:
            raise KeyError(f"Unknown DPR v2 hotpath shard_id={shard_id}")
        shard_path = self._shard_paths[shard_id]
        mmap_mode = "r" if self.mmap else None
        arrays: dict[str, Any] = {"path": shard_path}
        for name in RUNTIME_ARRAYS:
            arrays[name] = _load_runtime_array(shard_path, name, root=self.sidecar_root, mmap_mode=mmap_mode)
        self._cache[shard_id] = arrays
        while len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return arrays

    def _resolve_sidecar_path(self, value: str) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.sidecar_root / path
        path = path.resolve()
        _assert_under(path, self.sidecar_root)
        return path


class ScheduledHotpathBatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        sidecar: DPRV2HotpathSidecar,
        mil_schedule: pd.DataFrame,
        region_schedule: pd.DataFrame,
        rank: int,
        updates: int,
        mode: str = "speed_smoke",
    ) -> None:
        self.sidecar = sidecar
        self.mil_by_update = _schedule_by_update_and_rank(mil_schedule, int(rank))
        self.region_by_update = _schedule_by_update_and_rank(region_schedule, int(rank))
        self.rank = int(rank)
        self.updates = int(updates)
        self.mode = str(mode)

    def __len__(self) -> int:
        return self.updates

    def __getitem__(self, index: int) -> dict[str, Any]:
        step = int(index) + 1
        update_type, update_no = self._step_to_update(step)
        table = self.region_by_update if update_type == "region" else self.mil_by_update
        rows = table.get(int(update_no))
        if rows is None or rows.empty:
            raise RuntimeError(f"No {update_type} schedule rows for rank={self.rank} update={update_no}")
        samples = [self.sidecar.sample_from_schedule_row(row) for row in rows.itertuples(index=False)]
        batch = dpr_v2_hotpath_collate(samples)
        batch["update_type"] = update_type
        batch["schedule_update"] = int(update_no)
        return batch

    def _step_to_update(self, step: int) -> tuple[str, int]:
        if self.mode == "pilot_3000":
            if step <= 1400:
                return "protein_mil", step
            if step <= 2100:
                return "region", step - 1400
            local = step - 2100
            if local % 3 == 0:
                return "region", 700 + local // 3
            return "protein_mil", 1400 + (local - local // 3)
        if step % 3 == 0:
            return "region", step // 3
        return "protein_mil", step - step // 3


def dpr_v2_hotpath_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate empty DPR v2 hotpath batch")
    bsz = len(samples)
    max_len = max(int(sample["length"]) for sample in samples)
    out: dict[str, Any] = {
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
        "protein_ids": [str(sample["protein_id"]) for sample in samples],
        "sequences": [str(sample["sequence"]) for sample in samples],
        "kinds": [str(sample.get("kind", "protein")) for sample in samples],
        "pools": [str(sample.get("pool", "")) for sample in samples],
        "tiers": [str(sample.get("tier", "")) for sample in samples],
        "lengths": torch.tensor([int(sample["length"]) for sample in samples], dtype=torch.long),
        "seq_mask": torch.zeros(bsz, max_len, dtype=torch.bool),
        "plm": torch.zeros(bsz, max_len, 1280, dtype=torch.float16),
        "biophys": torch.zeros(bsz, max_len, 112, dtype=torch.float16),
        "aa_ids": torch.zeros(bsz, max_len, dtype=torch.int16),
        "modality_mask": torch.zeros(bsz, max_len, 5, dtype=torch.float16),
        "reliability": torch.zeros(bsz, max_len, 5, dtype=torch.float16),
        "neighbors": torch.zeros(bsz, max_len, 96, dtype=torch.long),
        "edge_attr": torch.zeros(bsz, max_len, 96, 32, dtype=torch.float16),
        "neighbor_mask": torch.zeros(bsz, max_len, 96, dtype=torch.bool),
        "neighbor_edge_type": torch.zeros(bsz, max_len, 96, dtype=torch.long),
        PHASEFLOW_LLPS_HIDDEN_KEY: torch.zeros(bsz, max_len, 256, dtype=torch.float16),
        "sample_weight": torch.stack([sample["sample_weight"].float() for sample in samples]),
        "bag_label": torch.stack([sample["bag_label"].float() for sample in samples]),
        "bag_weight": torch.stack([sample["bag_weight"].float() for sample in samples]),
        "driver_label": torch.stack([sample["driver_label"].float() for sample in samples]),
        "driver_weight": torch.stack([sample["driver_weight"].float() for sample in samples]),
        "partner_label": torch.stack([sample["partner_label"].float() for sample in samples]),
        "partner_weight": torch.stack([sample["partner_weight"].float() for sample in samples]),
        "general_label": torch.stack([sample["general_label"].float() for sample in samples]),
        "general_weight": torch.stack([sample["general_weight"].float() for sample in samples]),
    }
    for key in LABEL_KEYS:
        out[key] = torch.zeros(bsz, max_len, dtype=torch.float32)
    for i, sample in enumerate(samples):
        length = int(sample["length"])
        out["seq_mask"][i, :length] = True
        for key in (
            "plm",
            "biophys",
            "aa_ids",
            "modality_mask",
            "reliability",
            "neighbors",
            "edge_attr",
            "neighbor_mask",
            "neighbor_edge_type",
            PHASEFLOW_LLPS_HIDDEN_KEY,
            *LABEL_KEYS,
        ):
            out[key][i, :length] = sample[key][:length]
    return out


def _schedule_by_update_and_rank(frame: pd.DataFrame, rank: int) -> dict[int, pd.DataFrame]:
    work = frame.loc[pd.to_numeric(frame["rank"], errors="coerce").astype(int).eq(int(rank))].copy()
    return {int(update): group.reset_index(drop=True) for update, group in work.groupby("update", sort=True)}


def _bag_target(pool: str, sample_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    if pool in {"negative_disordered", "negative_structured"}:
        return torch.tensor(0.0), torch.tensor(float(sample_weight))
    if pool == "unlabeled":
        return torch.tensor(0.0), torch.tensor(0.0)
    return torch.tensor(1.0), torch.tensor(float(sample_weight))


def _driver_target(pool: str, tier: str, sample_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    if pool in {"negative_disordered", "negative_structured"}:
        return torch.tensor(0.0), torch.tensor(float(sample_weight))
    if tier in {"driver_gold", "driver_high"}:
        return torch.tensor(1.0), torch.tensor(float(sample_weight))
    return torch.tensor(0.0), torch.tensor(0.0)


def _partner_target(pool: str, tier: str, sample_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    if pool in {"negative_disordered", "negative_structured"}:
        return torch.tensor(0.0), torch.tensor(float(sample_weight))
    if tier == "partner_dependent":
        return torch.tensor(1.0), torch.tensor(float(sample_weight))
    return torch.tensor(0.0), torch.tensor(0.0)


def _general_target(pool: str, sample_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    if pool in {"negative_disordered", "negative_structured"}:
        return torch.tensor(0.0), torch.tensor(float(sample_weight))
    if pool == "unlabeled":
        return torch.tensor(0.0), torch.tensor(0.0)
    return torch.tensor(1.0), torch.tensor(float(sample_weight))


def _tensor(array: Any, dtype: Any) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=dtype).copy())


def _load_runtime_array(shard_path: Path, name: str, *, root: Path, mmap_mode: str | None) -> np.ndarray:
    candidates = (name, LEGACY_LLPS_HIDDEN_KEY) if name == PHASEFLOW_LLPS_HIDDEN_KEY else (name,)
    for candidate in candidates:
        path = (shard_path / f"{candidate}.npy").resolve()
        _assert_under(path, root)
        if path.exists():
            return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    raise FileNotFoundError(shard_path / f"{name}.npy")


def _assert_under(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"DPR v2 hotpath sidecar path escapes root: {path}") from exc


# Source: data/graph_policy.py


import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phaseflow.protein.data import (
    LABEL_KEYS,
    PHASEFLOW_LLPS_HIDDEN_KEY,
    RUNTIME_ARRAYS,
    _load_runtime_array,
    dpr_v2_hotpath_collate,
)
try:
    from phaseflow.protein.contracts import assert_no_eval_only_training_path
except ImportError:  # 53 keeps the original training guard; keep v3 self-contained.
    def assert_no_eval_only_training_path(path: str | Path) -> None:
        normalized = str(path).replace("\\", "/").lower()
        if "data/processed/evaluation_only/" in normalized:
            raise RuntimeError(f"Eval-only sidecar path is forbidden for DPR v3 training: {path}")


NO_STARLING_EDGE_TYPE = 30
PROTENIX_EDGE_TYPE = 20


@dataclass(frozen=True)
class GraphPolicyPackedProteinRow:
    protein_id: str
    sequence_sha256: str
    sequence: str
    length: int
    shard_id: int
    residue_offset: int


def graph_policy_read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    return pd.read_csv(path)


class DPRV3NoStarlingSidecar:
    """Read immutable v2 packed tensors and enforce the v3 no-STARLING graph policy.

    The parent hotpath remains read-only. Before tensors are returned to the model,
    edge_type=30 entries are masked and zeroed. Protenix edge-source dropout is
    represented by additionally masking edge_type=20 for the base-only view.
    """

    def __init__(
        self,
        *,
        v2_data_root: str | Path,
        packed_root: str | Path,
        mmap: bool = True,
        max_open_shards: int = 8,
    ) -> None:
        self.v2_data_root = Path(v2_data_root).resolve()
        self.packed_root = Path(packed_root).resolve()
        assert_no_eval_only_training_path(self.packed_root)
        manifest_path = self.packed_root / "manifest.parquet"
        shards_path = self.packed_root / "shards.parquet"
        self.manifest = graph_policy_read_table(manifest_path)
        self.shards = graph_policy_read_table(shards_path)
        self.mmap = bool(mmap)
        self.max_open_shards = max(1, int(max_open_shards))
        self._rows: dict[tuple[str, str], GraphPolicyPackedProteinRow] = {}
        for row in self.manifest.itertuples(index=False):
            key = (str(row.protein_id), str(row.sequence_sha256))
            self._rows[key] = GraphPolicyPackedProteinRow(
                protein_id=str(row.protein_id),
                sequence_sha256=str(row.sequence_sha256),
                sequence=str(row.sequence),
                length=int(row.length),
                shard_id=int(row.shard_id),
                residue_offset=int(row.residue_offset),
            )
        self._shard_paths = {int(row.shard_id): self._resolve_path(str(row.path)) for row in self.shards.itertuples(index=False)}
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def sample_from_tier_row(self, row: Any, *, view: str) -> dict[str, Any]:
        protein_id = str(row.protein_id)
        sequence_sha256 = str(row.sequence_sha256)
        packed = self._rows.get((protein_id, sequence_sha256))
        if packed is None:
            raise KeyError(f"{protein_id}/{sequence_sha256} not present in parent v2 packed hotpath")
        length = int(packed.length)
        shard = self._open_shard(int(packed.shard_id))
        sl = slice(int(packed.residue_offset), int(packed.residue_offset) + length)
        tier = str(getattr(row, "v3_tier", getattr(row, "tier", "")))
        pool = str(getattr(row, "v3_pool", getattr(row, "pool", "")))
        confidence = str(getattr(row, "mil_confidence", ""))
        sample_weight = float(getattr(row, "sample_weight", 1.0))
        sample: dict[str, Any] = {
            "sample_id": f"{protein_id}:{tier}:{view}",
            "protein_id": protein_id,
            "sequence_sha256": sequence_sha256,
            "sequence": packed.sequence,
            "kind": "protein",
            "pool": pool,
            "tier": tier,
            "v3_tier": tier,
            "mil_confidence": confidence,
            "view_name": str(view),
            "length": length,
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
        }
        targets = _targets_for_v3(tier=tier, pool=pool, confidence=confidence, sample_weight=sample_weight)
        sample.update(targets)
        for key in ("plm", "biophys", "modality_mask", "reliability", "edge_attr", PHASEFLOW_LLPS_HIDDEN_KEY):
            sample[key] = torch.from_numpy(np.asarray(shard[key][sl], dtype=np.float16).copy())
        sample["aa_ids"] = torch.from_numpy(np.asarray(shard["aa_ids"][sl], dtype=np.int16).copy())
        sample["neighbors"] = torch.from_numpy(np.asarray(shard["neighbors"][sl], dtype=np.int64).copy())
        sample["neighbor_mask"] = torch.from_numpy(np.asarray(shard["neighbor_mask"][sl], dtype=np.bool_).copy())
        sample["neighbor_edge_type"] = torch.from_numpy(np.asarray(shard["neighbor_edge_type"][sl], dtype=np.int64).copy())
        _apply_v3_edge_policy(sample, view=view)
        for key in LABEL_KEYS:
            sample[key] = torch.from_numpy(np.asarray(shard[key][sl], dtype=np.float32).copy())
        return sample

    def _open_shard(self, shard_id: int) -> dict[str, Any]:
        if shard_id in self._cache:
            arrays = self._cache.pop(shard_id)
            self._cache[shard_id] = arrays
            return arrays
        shard_path = self._shard_paths[int(shard_id)]
        mmap_mode = "r" if self.mmap else None
        arrays: dict[str, Any] = {"path": shard_path}
        for name in RUNTIME_ARRAYS:
            arrays[name] = _load_runtime_array(shard_path, name, root=self.packed_root, mmap_mode=mmap_mode)
        self._cache[shard_id] = arrays
        while len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return arrays

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.packed_root / path
        path = path.resolve()
        path.relative_to(self.packed_root)
        return path


class DPRV3StageBatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        sidecar: DPRV3NoStarlingSidecar,
        tier_manifest: pd.DataFrame,
        rank: int,
        world_size: int,
        updates: int,
        seed: int,
        mode: str,
    ) -> None:
        self.sidecar = sidecar
        self.tier_manifest = tier_manifest.reset_index(drop=True).copy()
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.updates = int(updates)
        self.seed = int(seed)
        self.mode = str(mode)
        self.by_tier: dict[str, pd.DataFrame] = {}
        for tier in ("S", "W", "M", "ND", "NP"):
            sub = self.tier_manifest.loc[self.tier_manifest["v3_tier"].astype(str).eq(tier)].copy()
            if sub.empty:
                raise RuntimeError(f"DPR v3 tier {tier} is empty")
            self.by_tier[tier] = _stable_shuffle(sub, self.seed + self.rank + _tier_offset(tier)).reset_index(drop=True)
        self.m_by_conf: dict[str, pd.DataFrame] = {}
        m = self.by_tier["M"]
        for conf in ("gold", "pseudo_positive_high", "pseudo_positive_weak_preserved"):
            sub = m.loc[m["mil_confidence"].astype(str).eq(conf)].copy()
            if sub.empty:
                raise RuntimeError(f"DPR v3 MIL confidence group is empty: {conf}")
            self.m_by_conf[conf] = _stable_shuffle(sub, self.seed + self.rank + _tier_offset(conf)).reset_index(drop=True)
        self.plan = self._build_rank_plan()

    def __len__(self) -> int:
        return self.updates

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.plan[int(index)]
        row = item["row"]
        primary_view = "base_only" if bool(item["drop_protenix_primary"]) else "base_protenix"
        alt_view = "base_protenix" if primary_view == "base_only" else "base_only"
        primary = self.sidecar.sample_from_tier_row(row, view=primary_view)
        alternate = self.sidecar.sample_from_tier_row(row, view=alt_view)
        primary_batch = dpr_v3_collate([primary])
        alternate_batch = dpr_v3_collate([alternate])
        return {
            "primary": primary_batch,
            "alternate": alternate_batch,
            "stage": item["stage"],
            "v3_tier": item["tier"],
            "mil_confidence": item.get("mil_confidence", ""),
            "update": int(index) + 1,
            "rank": self.rank,
        }

    def _build_rank_plan(self) -> list[dict[str, Any]]:
        counters: Counter[str] = Counter()
        m_counters: Counter[str] = Counter()
        plan: list[dict[str, Any]] = []
        for update in range(1, self.updates + 1):
            stage = stage_for_update(update, mode=self.mode)
            global_slot = (update - 1) * self.world_size + self.rank
            tier = _cycle_choice(stage_ratios(stage), global_slot + self.seed)
            if tier == "M":
                conf = _cycle_choice({"gold": 30, "pseudo_positive_high": 50, "pseudo_positive_weak_preserved": 20}, counters["M"] + self.seed)
                frame = self.m_by_conf[conf]
                idx = (m_counters[conf] * self.world_size + self.rank) % len(frame)
                row = frame.iloc[int(idx)]
                m_counters[conf] += 1
            else:
                conf = ""
                frame = self.by_tier[tier]
                idx = (counters[tier] * self.world_size + self.rank) % len(frame)
                row = frame.iloc[int(idx)]
            counters[tier] += 1
            drop = ((global_slot + self.seed) % 2) == 0
            plan.append({"stage": stage, "tier": tier, "row": row, "mil_confidence": conf, "drop_protenix_primary": drop})
        return plan


def dpr_v3_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out = dpr_v2_hotpath_collate(samples)
    out["v3_tiers"] = [str(sample.get("v3_tier", sample.get("tier", ""))) for sample in samples]
    out["mil_confidences"] = [str(sample.get("mil_confidence", "")) for sample in samples]
    out["view_names"] = [str(sample.get("view_name", "")) for sample in samples]
    out["starling_edges_masked"] = torch.tensor([int(sample.get("starling_edges_masked", 0)) for sample in samples], dtype=torch.long)
    out["protenix_edges_masked"] = torch.tensor([int(sample.get("protenix_edges_masked", 0)) for sample in samples], dtype=torch.long)
    out["edges_after_policy"] = torch.tensor([int(sample.get("edges_after_policy", 0)) for sample in samples], dtype=torch.long)
    return out


def stage_for_update(update: int, *, mode: str = "train") -> str:
    if str(mode) == "smoke":
        # The 300-update smoke traverses all stage policies.
        if update <= 90:
            return "A"
        if update <= 150:
            return "B"
        if update <= 250:
            return "C"
        return "D"
    if update <= 6000:
        return "A"
    if update <= 9000:
        return "B"
    if update <= 14000:
        return "C"
    return "D"


def stage_ratios(stage: str) -> dict[str, int]:
    return {
        "A": {"M": 50, "ND": 20, "NP": 15, "S": 10, "W": 5},
        "B": {"M": 35, "ND": 20, "NP": 15, "S": 20, "W": 10},
        "C": {"M": 45, "ND": 20, "NP": 15, "S": 12, "W": 8},
        "D": {"M": 30, "ND": 30, "NP": 25, "S": 10, "W": 5},
    }[str(stage)]


def _apply_v3_edge_policy(sample: dict[str, Any], *, view: str) -> None:
    edge_type = sample["neighbor_edge_type"].long()
    mask = sample["neighbor_mask"].bool()
    starling = mask & edge_type.eq(NO_STARLING_EDGE_TYPE)
    protenix = mask & edge_type.eq(PROTENIX_EDGE_TYPE)
    drop_protenix = str(view) == "base_only"
    keep = mask & ~starling
    if drop_protenix:
        keep = keep & ~protenix
    sample["starling_edges_masked"] = int(starling.sum().item())
    sample["protenix_edges_masked"] = int(protenix.sum().item()) if drop_protenix else 0
    sample["edges_after_policy"] = int(keep.sum().item())
    sample["neighbor_mask"] = keep
    masked = ~keep
    sample["neighbors"] = sample["neighbors"].masked_fill(masked, 0)
    sample["edge_attr"] = sample["edge_attr"].masked_fill(masked.unsqueeze(-1), 0)
    sample["neighbor_edge_type"] = sample["neighbor_edge_type"].masked_fill(masked, 0)


def _targets_for_v3(*, tier: str, pool: str, confidence: str, sample_weight: float) -> dict[str, torch.Tensor]:
    positive = tier in {"S", "W", "M"}
    negative = tier in {"ND", "NP"}
    w = float(sample_weight)
    if negative:
        return {
            "bag_label": torch.tensor(0.0),
            "bag_weight": torch.tensor(w),
            "driver_label": torch.tensor(0.0),
            "driver_weight": torch.tensor(w),
            "partner_label": torch.tensor(0.0),
            "partner_weight": torch.tensor(w),
            "general_label": torch.tensor(0.0),
            "general_weight": torch.tensor(w),
        }
    if positive:
        driver = confidence in {"gold", "pseudo_positive_high"} or str(pool).startswith("driver")
        partner = confidence == "pseudo_positive_weak_preserved" or str(pool).startswith("partner")
        return {
            "bag_label": torch.tensor(1.0),
            "bag_weight": torch.tensor(w),
            "driver_label": torch.tensor(1.0 if driver else 0.0),
            "driver_weight": torch.tensor(w if driver else 0.0),
            "partner_label": torch.tensor(1.0 if partner else 0.0),
            "partner_weight": torch.tensor(w if partner else 0.0),
            "general_label": torch.tensor(1.0),
            "general_weight": torch.tensor(w),
        }
    return {
        "bag_label": torch.tensor(0.0),
        "bag_weight": torch.tensor(0.0),
        "driver_label": torch.tensor(0.0),
        "driver_weight": torch.tensor(0.0),
        "partner_label": torch.tensor(0.0),
        "partner_weight": torch.tensor(0.0),
        "general_label": torch.tensor(0.0),
        "general_weight": torch.tensor(0.0),
    }


def _cycle_choice(weights: dict[str, int], index: int) -> str:
    total = int(sum(weights.values()))
    if total <= 0:
        raise ValueError("empty cycle weights")
    pos = int(index) % total
    cursor = 0
    for key, weight in weights.items():
        cursor += int(weight)
        if pos < cursor:
            return key
    return next(reversed(weights))


def _stable_shuffle(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rng = np.random.default_rng(int(seed) % (2**32 - 1))
    order = np.arange(len(frame))
    rng.shuffle(order)
    return frame.iloc[order].reset_index(drop=True)


def _tier_offset(name: str) -> int:
    return sum(ord(ch) for ch in str(name)) * 997


def graph_policy_edge_policy_summary(batch: dict[str, Any]) -> dict[str, int]:
    mask = batch["neighbor_mask"].bool()
    et = batch["neighbor_edge_type"].long()
    return {
        "starling_edges_passed_to_model": int((mask & et.eq(NO_STARLING_EDGE_TYPE)).sum().item()),
        "protenix_edges_passed_to_model": int((mask & et.eq(PROTENIX_EDGE_TYPE)).sum().item()),
        "total_edges_passed_to_model": int(mask.sum().item()),
    }


# Source: data/dataset.py


import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phaseflow.protein.contracts import FeatureCacheReader
from phaseflow.protein.contracts import IGNORE_INDEX
from phaseflow.protein.features import make_bio_vec


class PhaseFlowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        feature_dir: str | Path | list[str | Path] | tuple[str | Path, ...],
        protein_ids: list[str] | None = None,
        phase_targets: str | Path | None = None,
        region_targets: str | Path | None = None,
        region_supervision: str = "feature",
        read_raw_edges: bool = True,
    ) -> None:
        if isinstance(feature_dir, (list, tuple)):
            self.feature_dirs = [Path(path) for path in feature_dir]
        else:
            self.feature_dirs = [Path(feature_dir)]
        if protein_ids is None:
            seen: set[str] = set()
            protein_ids = []
            for directory in self.feature_dirs:
                for path in sorted(directory.glob("*.h5")):
                    if path.stem not in seen:
                        protein_ids.append(path.stem)
                        seen.add(path.stem)
        self.protein_ids = [str(protein_id) for protein_id in protein_ids]
        self.phase_targets = _read_phase_targets(phase_targets)
        self.region_targets = _read_region_targets(region_targets)
        self.region_supervision = _normalize_region_supervision(region_supervision)
        self.read_raw_edges = bool(read_raw_edges)

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        protein_id = self.protein_ids[index]
        record = FeatureCacheReader.read_h5(self._feature_path(protein_id), read_raw_edges=self.read_raw_edges)
        phase = self.phase_targets.get(protein_id)
        if phase is None:
            phase_values = np.zeros(16, dtype=np.float32)
            phase_mask = np.zeros(16, dtype=np.float32)
            phase_aux_weight = 0.0
            phase_mean_pssi = float("nan")
            phase_low_pssi = float("nan")
        else:
            phase_values = phase["phase_values"]
            phase_mask = phase["phase_mask"]
            phase_aux_weight = float(phase["phase_aux_weight"])
            phase_mean_pssi = float(phase["phase_mean_pssi"])
            phase_low_pssi = float(phase["phase_low_pssi"])
        region_target = self.region_targets.get(protein_id)
        if region_target is None:
            region_target = _empty_region_target(record.length)
        else:
            region_target = _fit_region_target(region_target, record.length)
        y_dpr = record.y_dpr
        y_key = record.y_key
        y_weight = record.y_weight
        regions = record.regions
        if self.region_supervision == "none":
            y_dpr, y_key, y_weight = _empty_hard_region_labels(record.length)
            regions = []
        elif self.region_supervision == "region_targets":
            y_dpr, y_key, y_weight = _empty_hard_region_labels(record.length)
            regions = _regions_from_region_target(record.protein_id, region_target, record.length)
        sample = {
            "protein_id": record.protein_id,
            "sequence": record.sequence,
            "length": record.length,
            "plm": torch.from_numpy(record.plm).float(),
            "physchem": torch.from_numpy(record.physchem).float(),
            "disorder": torch.from_numpy(record.disorder).float(),
            "protenix_embed": torch.from_numpy(record.protenix_embed).float(),
            "starling_embed": torch.from_numpy(record.starling_embed).float(),
            "bio_vec": torch.from_numpy(
                make_bio_vec(
                    sequence=record.sequence,
                    physchem=record.physchem,
                    disorder=record.disorder,
                    plm=record.plm,
                    protenix=record.protenix_embed,
                    starling=record.starling_embed,
                    edge_src=record.edge_src,
                    edge_dst=record.edge_dst,
                    graph_num_nodes=record.length,
                    graph_num_edges=len(record.edge_src),
                )
            ).float(),
            "modality_mask": torch.from_numpy(record.modality_mask).float(),
            "reliability": torch.from_numpy(record.reliability).float(),
            "edge_src": torch.from_numpy(record.edge_src).long(),
            "edge_dst": torch.from_numpy(record.edge_dst).long(),
            "edge_type": torch.from_numpy(record.edge_type).long(),
            "edge_attr": torch.from_numpy(record.edge_attr).float(),
            "precomputed_neighbors": (
                torch.from_numpy(record.graph_neighbors).long() if record.graph_neighbors is not None else None
            ),
            "precomputed_edge_attr": (
                torch.from_numpy(record.graph_edge_attr).float() if record.graph_edge_attr is not None else None
            ),
            "precomputed_neighbor_mask": (
                torch.from_numpy(record.graph_neighbor_mask).bool() if record.graph_neighbor_mask is not None else None
            ),
            "y_llps": torch.tensor(record.y_llps, dtype=torch.float32),
            "sample_weight": torch.tensor(record.sample_weight, dtype=torch.float32),
            "teacher_llps": torch.tensor(record.teacher_llps, dtype=torch.float32),
            "teacher_llps_weight": torch.tensor(record.teacher_llps_weight, dtype=torch.float32),
            "self_llps": torch.tensor(record.self_llps, dtype=torch.float32),
            "self_llps_weight": torch.tensor(record.self_llps_weight, dtype=torch.float32),
            "region_bag_label": torch.tensor(record.region_bag_label, dtype=torch.float32),
            "region_bag_weight": torch.tensor(record.region_bag_weight, dtype=torch.float32),
            "negative_regularization_weight": torch.tensor(record.negative_regularization_weight, dtype=torch.float32),
            "y_dpr": torch.from_numpy(y_dpr).long(),
            "y_key": torch.from_numpy(y_key).long(),
            "y_weight": torch.from_numpy(y_weight).float(),
            "teacher_dpr": torch.from_numpy(record.teacher_dpr).float(),
            "teacher_dpr_weight": torch.from_numpy(record.teacher_dpr_weight).float(),
            "self_dpr": torch.from_numpy(record.self_dpr).float(),
            "self_dpr_weight": torch.from_numpy(record.self_dpr_weight).float(),
            "candidate_prior": torch.from_numpy(record.candidate_prior).float(),
            "candidate_prior_weight": torch.from_numpy(record.candidate_prior_weight).float(),
            "region_bag_type": record.region_bag_type,
            "label_quality": record.label_quality,
            "negative_type": record.negative_type,
            "source": record.source,
            "regions": regions,
            "structure_metadata": record.structure_metadata,
            "phase_values": torch.from_numpy(phase_values).float(),
            "phase_mask": torch.from_numpy(phase_mask).float(),
            "phase_aux_weight": torch.tensor(phase_aux_weight, dtype=torch.float32),
            "phase_mean_pssi": torch.tensor(phase_mean_pssi, dtype=torch.float32),
            "phase_low_pssi": torch.tensor(phase_low_pssi, dtype=torch.float32),
            "region_teacher_target": torch.from_numpy(region_target["region_teacher_target"]).float(),
            "region_teacher_weight": torch.from_numpy(region_target["region_teacher_weight"]).float(),
            "region_key_target": torch.from_numpy(region_target["region_key_target"]).float(),
            "region_key_weight": torch.from_numpy(region_target["region_key_weight"]).float(),
            "region_boundary_target": torch.from_numpy(region_target["region_boundary_target"]).float(),
            "region_boundary_weight": torch.from_numpy(region_target["region_boundary_weight"]).float(),
            "region_contrast_target": torch.from_numpy(region_target["region_contrast_target"]).float(),
            "region_contrast_weight": torch.from_numpy(region_target["region_contrast_weight"]).float(),
        }
        return sample

    def _feature_path(self, protein_id: str) -> Path:
        for directory in self.feature_dirs:
            path = directory / f"{protein_id}.h5"
            if path.exists():
                return path
        searched = ", ".join(str(path) for path in self.feature_dirs)
        raise FileNotFoundError(f"Missing feature cache for {protein_id}; searched: {searched}")


def _read_phase_targets(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    value_cols = [f"phase_value_{index:02d}" for index in range(16)]
    mask_cols = [f"phase_mask_{index:02d}" for index in range(16)]
    missing_values = [column for column in value_cols if column not in frame.columns]
    missing_masks = [column for column in mask_cols if column not in frame.columns]
    if "protein_id" not in frame.columns or missing_values or missing_masks:
        raise ValueError(
            f"Phase target file {path} must contain protein_id, {value_cols[0]}..{value_cols[-1]}, "
            f"and {mask_cols[0]}..{mask_cols[-1]}"
        )
    targets: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        protein_id = str(row["protein_id"])
        values = np.asarray([row[column] for column in value_cols], dtype=np.float32)
        mask = np.asarray([row[column] for column in mask_cols], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0).astype(np.float32, copy=False)
        mask = np.nan_to_num(mask, nan=0.0).astype(np.float32, copy=False)
        targets[protein_id] = {
            "phase_values": values,
            "phase_mask": mask,
            "phase_aux_weight": float(row.get("phase_aux_weight", 1.0)),
            "phase_mean_pssi": float(row.get("phase_mean_pssi", np.nan)),
            "phase_low_pssi": float(row.get("phase_low_pssi", np.nan)),
        }
    return targets


REGION_TARGET_KEYS = (
    "region_teacher_target",
    "region_teacher_weight",
    "region_key_target",
    "region_key_weight",
    "region_boundary_target",
    "region_boundary_weight",
    "region_contrast_target",
    "region_contrast_weight",
)


def _read_region_targets(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return _read_region_targets_cached(str(path.resolve()))


@lru_cache(maxsize=8)
def _read_region_targets_cached(path: str) -> dict[str, dict[str, Any]]:
    import h5py

    targets: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        policy = _decode_h5_attr(handle.attrs.get("policy", "region_targets"))
        for protein_id in handle:
            group = handle[protein_id]
            item: dict[str, Any] = {"target_policy": policy}
            for key in REGION_TARGET_KEYS:
                if key in group:
                    item[key] = np.asarray(group[key], dtype=np.float32)
            item["positive_spans"] = _read_json_attr(group.attrs.get("positive_spans_json", "[]"))
            item["negative_spans"] = _read_json_attr(group.attrs.get("negative_spans_json", "[]"))
            if item:
                targets[str(protein_id)] = item
    return targets


def _empty_region_target(length: int) -> dict[str, Any]:
    return {
        "region_teacher_target": np.full(length, np.nan, dtype=np.float32),
        "region_teacher_weight": np.zeros(length, dtype=np.float32),
        "region_key_target": np.full(length, np.nan, dtype=np.float32),
        "region_key_weight": np.zeros(length, dtype=np.float32),
        "region_boundary_target": np.full(length, np.nan, dtype=np.float32),
        "region_boundary_weight": np.zeros(length, dtype=np.float32),
        "region_contrast_target": np.full(length, np.nan, dtype=np.float32),
        "region_contrast_weight": np.zeros(length, dtype=np.float32),
        "positive_spans": [],
        "negative_spans": [],
        "target_policy": "",
    }


def _fit_region_target(target: dict[str, Any], length: int) -> dict[str, Any]:
    out = _empty_region_target(length)
    for key, default in out.items():
        if key in {"positive_spans", "negative_spans", "target_policy"}:
            continue
        if key not in target:
            continue
        value = np.asarray(target[key], dtype=np.float32)
        copy_len = min(length, int(value.shape[0]))
        out[key][:copy_len] = value[:copy_len]
        if copy_len < length and "weight" in key:
            out[key][copy_len:] = 0.0
    out["positive_spans"] = _clip_spans(target.get("positive_spans", []), length)
    out["negative_spans"] = _clip_spans(target.get("negative_spans", []), length)
    out["target_policy"] = str(target.get("target_policy", ""))
    return out


def _normalize_region_supervision(value: str) -> str:
    normalized = str(value or "feature").strip().lower()
    aliases = {
        "feature_cache": "feature",
        "features": "feature",
        "gold": "feature",
        "off": "none",
        "disabled": "none",
        "teacher_targets": "region_targets",
        "pseudo_targets": "region_targets",
        "final_region_targets": "region_targets",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"feature", "region_targets", "none"}:
        raise ValueError(
            "region_supervision must be one of 'feature', 'region_targets', or 'none', "
            f"got {value!r}"
        )
    return normalized


def _empty_hard_region_labels(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full(length, IGNORE_INDEX, dtype=np.int64),
        np.full(length, IGNORE_INDEX, dtype=np.int64),
        np.zeros(length, dtype=np.float32),
    )


def _regions_from_region_target(protein_id: str, target: dict[str, Any], length: int) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    policy = str(target.get("target_policy") or "region_targets")
    for index, span in enumerate(_clip_spans(target.get("positive_spans", []), length), start=1):
        start = int(span["start"])
        end = int(span["end"])
        if end < start:
            continue
        regions.append(
            {
                "protein_id": protein_id,
                "start": start,
                "end": end,
                "type": "DPR_teacher",
                "region_type": "DPR_teacher",
                "region_label": "positive",
                "confidence": float(span.get("confidence", 1.0)),
                "soft_weight": float(span.get("sample_weight", span.get("confidence", 1.0))),
                "evidence_level": "pseudo",
                "source": policy,
                "region_id": f"{protein_id}_pseudo_{index}",
            }
        )
    return regions


def _clip_spans(raw_spans: Any, length: int) -> list[dict[str, float | int]]:
    spans: list[dict[str, float | int]] = []
    if not isinstance(raw_spans, list):
        return spans
    for raw in raw_spans:
        if isinstance(raw, dict):
            start = int(raw.get("start", 0))
            end = int(raw.get("end", start))
            confidence = float(raw.get("confidence", raw.get("sample_weight", 1.0)))
            sample_weight = float(raw.get("sample_weight", confidence))
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            start = int(raw[0])
            end = int(raw[1])
            confidence = 1.0
            sample_weight = 1.0
        else:
            continue
        start = max(0, min(start, length - 1))
        end = max(start, min(end, length - 1))
        spans.append(
            {
                "start": start,
                "end": end,
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "sample_weight": float(np.clip(sample_weight, 0.0, 1.0)),
            }
        )
    return spans


def _read_json_attr(value: Any) -> Any:
    try:
        return json.loads(_decode_h5_attr(value))
    except (TypeError, json.JSONDecodeError):
        return []


def _decode_h5_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# Source: data/offline_dataset.py


import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from phaseflow.protein.data import (
    _empty_hard_region_labels,
    _empty_region_target,
    _fit_region_target,
    _read_phase_targets,
    _read_region_targets,
    _regions_from_region_target,
)
from phaseflow.protein.contracts import assert_offline_path_allowed
from phaseflow.protein.contracts import IGNORE_INDEX
from phaseflow.protein.contracts import Esm2ShardedStore, NpzMirrorStore
from phaseflow.protein.features import make_bio_vec


class PhaseFlowOfflineDataset(Dataset[dict[str, Any]]):
    """Read model-ready PhaseFlow tensors from the audited offline parquet/npz tree."""

    def __init__(
        self,
        dataset_root: str | Path = "data/processed/merged",
        sample_index: str | Path | None = None,
        input_contract: str | Path | None = None,
        protein_ids: list[str] | None = None,
        split: str | None = None,
        phase_targets: str | Path | None = None,
        region_targets: str | Path | None = None,
        region_labels_dir: str | Path | None = None,
        region_supervision: str = "none",
        esm2_store_metadata: str | Path | None = None,
        esm2_store_required: bool = False,
        npz_mirror_manifest: str | Path | None = None,
        allow_legacy_h5: bool = False,
        read_graph: bool = True,
    ) -> None:
        if allow_legacy_h5:
            raise RuntimeError("PhaseFlowOfflineDataset does not support legacy H5 reading.")
        self.dataset_root = Path(dataset_root)
        self.input_contract = Path(input_contract) if input_contract is not None else self.dataset_root / "configs/offline_input_contract.yaml"
        assert_offline_path_allowed(self.input_contract)
        if not self.input_contract.exists():
            raise FileNotFoundError(f"Missing offline input contract: {self.input_contract}")
        self.contract = yaml.safe_load(self.input_contract.read_text()) or {}
        contract_root = Path(str(self.contract.get("dataset_root", self.dataset_root)))
        if not contract_root.is_absolute():
            contract_root = Path.cwd() / contract_root
        self.dataset_root = contract_root.resolve()
        sample_index_path = (
            Path(sample_index)
            if sample_index is not None
            else self.dataset_root / str(self.contract.get("sample_index", "tables/training_sample_index.parquet"))
        )
        if not sample_index_path.is_absolute():
            sample_index_path = Path.cwd() / sample_index_path
        assert_offline_path_allowed(sample_index_path)
        self.sample_index_path = sample_index_path
        frame = pd.read_parquet(sample_index_path)
        if "dataset_index" not in frame.columns:
            frame = frame.assign(dataset_index=np.arange(len(frame), dtype=np.int64))
        self.esm2_store: Esm2ShardedStore | None = None
        self.npz_mirror_store: NpzMirrorStore | None = None
        self._npz_array_cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        cache_gb = float(os.environ.get("PHASEFLOW_NPZ_ARRAY_CACHE_GB", "0") or 0.0)
        self._npz_array_cache_max_bytes = int(max(0.0, cache_gb) * (1024**3))
        self._npz_array_cache_bytes = 0
        if esm2_store_metadata is not None:
            metadata_path = Path(esm2_store_metadata)
            if not metadata_path.is_absolute():
                metadata_path = Path.cwd() / metadata_path
            assert_offline_path_allowed(metadata_path)
            metadata = pd.read_parquet(metadata_path)
            keep = [
                "dataset_index",
                "esm2_store_shard_id",
                "esm2_store_offset",
                "esm2_store_path",
                "esm2_mask_store_path",
            ]
            existing = [col for col in keep if col in metadata.columns]
            frame = frame.merge(metadata[existing], on="dataset_index", how="left")
            self.esm2_store = Esm2ShardedStore(required=bool(esm2_store_required))
        if npz_mirror_manifest is not None:
            mirror_path = Path(npz_mirror_manifest)
            if not mirror_path.is_absolute():
                mirror_path = Path.cwd() / mirror_path
            assert_offline_path_allowed(mirror_path)
            self.npz_mirror_store = NpzMirrorStore(mirror_path)
        if split is not None and "split" in frame.columns:
            frame = frame.loc[frame["split"].astype(str) == str(split)]
        if protein_ids is not None:
            order = {str(protein_id): idx for idx, protein_id in enumerate(protein_ids)}
            frame = frame.loc[frame["protein_id"].astype(str).isin(order)]
            frame = frame.assign(_order=frame["protein_id"].astype(str).map(order)).sort_values("_order").drop(columns=["_order"])
        self.frame = frame.reset_index(drop=True)
        self.protein_ids = [str(value) for value in self.frame["protein_id"].tolist()]
        self.phase_targets = _read_phase_targets(phase_targets)
        self.region_targets = _read_region_targets(region_targets)
        self.region_labels_dir = Path(region_labels_dir) if region_labels_dir is not None else None
        if self.region_labels_dir is not None and not self.region_labels_dir.is_absolute():
            self.region_labels_dir = (self.dataset_root / self.region_labels_dir).resolve()
        self.region_supervision = str(region_supervision)
        self.read_graph = bool(read_graph)
        self.sample_count = len(self.frame)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        profile_timing = _profile_dataset_timing_enabled()
        getitem_start = time.perf_counter() if profile_timing else 0.0
        timing: dict[str, float] = {}
        paths: dict[str, str] = {}
        sizes: dict[str, int] = {}
        row = self.frame.iloc[int(index)]
        protein_id = str(row["protein_id"])
        sequence = str(row["sequence"])
        length = int(row["seq_len"])

        section_start = time.perf_counter() if profile_timing else 0.0
        esm2 = self._read_esm2(row, length)
        if profile_timing:
            timing["esm2_read_sec"] = time.perf_counter() - section_start
            self._profile_path(paths, sizes, "esm2", row.get("esm2_path"))
            section_start = time.perf_counter()
        biophys = self._read_biophys(row, length)
        if profile_timing:
            timing["biophys_read_sec"] = time.perf_counter() - section_start
            self._profile_path(paths, sizes, "biophys", row.get("biophys_shard"))
            section_start = time.perf_counter()
        protenix = self._read_protenix(row, length)
        if profile_timing:
            timing["protenix_read_sec"] = time.perf_counter() - section_start
            if _truthy(row.get("has_protenix")):
                self._profile_path(paths, sizes, "protenix", row.get("protenix_node_shard"))
            section_start = time.perf_counter()
        starling = self._read_starling(row, length)
        if profile_timing:
            timing["starling_read_sec"] = time.perf_counter() - section_start
            starling_paths = _parse_json_list(row.get("starling_node_shards")) if _truthy(row.get("has_starling")) else []
            if starling_paths:
                self._profile_path(paths, sizes, "starling", starling_paths[0])
            section_start = time.perf_counter()
        graph = self._read_merged_graph(row, length) if self.read_graph else _empty_graph(length)
        if profile_timing:
            timing["graph_read_sec"] = time.perf_counter() - section_start
            self._profile_path(paths, sizes, "graph", row.get("merged_graph_shard"))

        modality_mask, reliability = self._modality_matrices(row, length)
        phase = self.phase_targets.get(protein_id)
        if phase is None:
            phase_values = np.zeros(16, dtype=np.float32)
            phase_mask = np.zeros(16, dtype=np.float32)
            phase_aux_weight = 0.0
            phase_mean_pssi = float("nan")
            phase_low_pssi = float("nan")
        else:
            phase_values = phase["phase_values"]
            phase_mask = phase["phase_mask"]
            phase_aux_weight = float(phase["phase_aux_weight"])
            phase_mean_pssi = float(phase["phase_mean_pssi"])
            phase_low_pssi = float(phase["phase_low_pssi"])

        region_target = self.region_targets.get(protein_id)
        if region_target is None:
            region_target = self._read_region_label_npz(row, length)
        if region_target is None:
            region_target = _empty_region_target(length)
        else:
            region_target = _fit_region_target(region_target, length)

        y_dpr, y_key, y_weight = _empty_hard_region_labels(length)
        regions: list[dict[str, Any]] = []
        if self.region_supervision == "region_targets":
            regions = _regions_from_region_target(protein_id, region_target, length)

        sample = {
            "dataset_index": int(row.get("dataset_index", index)),
            "sample_id": str(row.get("sample_id", protein_id)),
            "protein_id": protein_id,
            "sequence": sequence,
            "length": length,
            "split": str(row.get("split", "")),
            "esm2_available_mask": torch.from_numpy(esm2["mask"]).float(),
            "plm": torch.from_numpy(esm2["node"]).float(),
            "physchem": torch.from_numpy(biophys["physchem"]).float(),
            "biophys_node": torch.from_numpy(biophys["node"]).float(),
            "disorder": torch.from_numpy(biophys["disorder"]).float(),
            "protenix_embed": torch.from_numpy(protenix).float(),
            "starling_embed": torch.from_numpy(starling).float(),
            "bio_vec": torch.from_numpy(
                make_bio_vec(
                    sequence=sequence,
                    physchem=biophys["physchem"],
                    disorder=biophys["disorder"],
                    plm=esm2["node"],
                    protenix=protenix,
                    starling=starling,
                    edge_src=graph["edge_src"],
                    edge_dst=graph["edge_dst"],
                    graph_num_nodes=row.get("graph_num_nodes", length),
                    graph_num_edges=row.get("graph_num_edges", len(graph["edge_src"])),
                )
            ).float(),
            "modality_mask": torch.from_numpy(modality_mask).float(),
            "reliability": torch.from_numpy(reliability).float(),
            "edge_src": torch.from_numpy(graph["edge_src"]).long(),
            "edge_dst": torch.from_numpy(graph["edge_dst"]).long(),
            "edge_type": torch.from_numpy(graph["edge_type"]).long(),
            "edge_attr": torch.from_numpy(graph["edge_attr"]).float(),
            "precomputed_neighbors": None,
            "precomputed_edge_attr": None,
            "precomputed_neighbor_mask": None,
            "y_llps": torch.tensor(_float_or_default(row.get("llps_label"), IGNORE_INDEX), dtype=torch.float32),
            "sample_weight": torch.tensor(_float_or_default(row.get("sample_weight"), 1.0), dtype=torch.float32),
            "teacher_llps": torch.tensor(_float_or_default(row.get("teacher_llps"), np.nan), dtype=torch.float32),
            "teacher_llps_weight": torch.tensor(_float_or_default(row.get("teacher_llps_weight"), 0.0), dtype=torch.float32),
            "self_llps": torch.tensor(_float_or_default(row.get("self_llps"), np.nan), dtype=torch.float32),
            "self_llps_weight": torch.tensor(_float_or_default(row.get("self_llps_weight"), 0.0), dtype=torch.float32),
            "region_bag_label": torch.tensor(_float_or_default(row.get("region_bag_label"), IGNORE_INDEX), dtype=torch.float32),
            "region_bag_weight": torch.tensor(_float_or_default(row.get("region_bag_weight"), 0.0), dtype=torch.float32),
            "negative_regularization_weight": torch.tensor(
                _float_or_default(row.get("negative_regularization_weight"), 0.0),
                dtype=torch.float32,
            ),
            "y_dpr": torch.from_numpy(y_dpr).long(),
            "y_key": torch.from_numpy(y_key).long(),
            "y_weight": torch.from_numpy(y_weight).float(),
            "teacher_dpr": torch.full((length,), float("nan"), dtype=torch.float32),
            "teacher_dpr_weight": torch.zeros(length, dtype=torch.float32),
            "self_dpr": torch.full((length,), float("nan"), dtype=torch.float32),
            "self_dpr_weight": torch.zeros(length, dtype=torch.float32),
            "candidate_prior": torch.zeros(length, dtype=torch.float32),
            "candidate_prior_weight": torch.zeros(length, dtype=torch.float32),
            "region_bag_type": str(row.get("region_bag_type", "mask")),
            "label_quality": str(row.get("llps_label_status", row.get("merged_label_tier", ""))),
            "negative_type": str(row.get("negative_type", "")),
            "llps_role": str(row.get("llps_role", row.get("role", ""))),
            "source": str(row.get("source_dataset", row.get("source", ""))),
            "regions": regions,
            "structure_metadata": {
                "offline_input_contract": str(self.input_contract),
                "training_graph_source": str(row.get("merged_graph_shard", "")),
                "graph_multiedge": str(self.contract.get("graph_contract", {}).get("multigraph", True)),
            },
            "phase_values": torch.from_numpy(phase_values).float(),
            "phase_mask": torch.from_numpy(phase_mask).float(),
            "phase_aux_weight": torch.tensor(phase_aux_weight, dtype=torch.float32),
            "phase_mean_pssi": torch.tensor(phase_mean_pssi, dtype=torch.float32),
            "phase_low_pssi": torch.tensor(phase_low_pssi, dtype=torch.float32),
            "region_teacher_target": torch.from_numpy(region_target["region_teacher_target"]).float(),
            "region_teacher_weight": torch.from_numpy(region_target["region_teacher_weight"]).float(),
            "region_key_target": torch.from_numpy(region_target["region_key_target"]).float(),
            "region_key_weight": torch.from_numpy(region_target["region_key_weight"]).float(),
            "region_boundary_target": torch.from_numpy(region_target["region_boundary_target"]).float(),
            "region_boundary_weight": torch.from_numpy(region_target["region_boundary_weight"]).float(),
            "region_contrast_target": torch.from_numpy(region_target["region_contrast_target"]).float(),
            "region_contrast_weight": torch.from_numpy(region_target["region_contrast_weight"]).float(),
        }
        if profile_timing:
            timing["getitem_sec"] = time.perf_counter() - getitem_start
            sample["__timing"] = timing
            sample["__io_paths"] = paths
            sample["__io_sizes"] = sizes
        return sample

    def _resolve_data_path(self, value: Any) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.dataset_root / path
        assert_offline_path_allowed(path)
        try:
            path.relative_to(self.dataset_root)
        except ValueError as exc:
            raise RuntimeError(f"Offline dataset path escapes dataset_root: {path}") from exc
        return path

    def _profile_path(self, paths: dict[str, str], sizes: dict[str, int], name: str, value: Any) -> None:
        try:
            path = self._resolve_data_path(value)
        except Exception:
            return
        paths[f"{name}_path"] = str(path)
        try:
            sizes[f"{name}_size_bytes"] = int(path.stat().st_size)
        except OSError:
            sizes[f"{name}_size_bytes"] = -1

    def _read_esm2(self, row: pd.Series, length: int) -> dict[str, np.ndarray]:
        if self.esm2_store is not None:
            sharded = self.esm2_store.read(row, length)
            if sharded is not None:
                return sharded
        path = self._resolve_data_path(row["esm2_path"])
        with _npz_open(str(path)) as data:
            node = np.asarray(data["esm2_node"], dtype=np.float32)
            if "esm2_available_mask" in data:
                mask = np.asarray(data["esm2_available_mask"], dtype=np.float32)
            else:
                mask = np.ones(length, dtype=np.float32)
        _check_node_shape("esm2_node", node, length, 1280)
        if mask.shape != (length,):
            raise ValueError(f"esm2_available_mask must have shape [{length}], got {mask.shape}")
        return {"node": node, "mask": mask}

    def _read_biophys(self, row: pd.Series, length: int) -> dict[str, np.ndarray]:
        offset = int(row["biophys_offset"])
        mirror = self._mirror_array(row["biophys_shard"], "biophys_node")
        if mirror is not None:
            node = np.asarray(mirror[offset : offset + length], dtype=np.float32)
        else:
            path = self._resolve_data_path(row["biophys_shard"])
            data_node = self._cached_npz_array(path, "biophys_node")
            node = np.asarray(data_node[offset : offset + length], dtype=np.float32)
        _check_node_shape("biophys_node", node, length, 112)
        physchem = node[:, :90].astype(np.float32, copy=False)
        disorder = np.stack(
            [
                node[:, 90],
                node[:, 91],
                node[:, 92],
                node[:, 93],
                node[:, 94],
                node[:, 95],
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        return {"node": node, "physchem": physchem, "disorder": disorder}

    def _read_protenix(self, row: pd.Series, length: int) -> np.ndarray:
        if _protenix_read_disabled():
            return np.zeros((length, 512), dtype=np.float32)
        if not _truthy(row.get("has_protenix")):
            return np.zeros((length, 512), dtype=np.float32)
        offset = int(row["protenix_node_offset"])
        mirror = self._mirror_array(row["protenix_node_shard"], "protenix_node_embed")
        if mirror is not None:
            node = np.asarray(mirror[offset : offset + length], dtype=np.float32)
        else:
            path = self._resolve_data_path(row["protenix_node_shard"])
            with _npz_open(str(path)) as data:
                node = np.asarray(data["protenix_node_embed"][offset : offset + length], dtype=np.float32)
        _check_node_shape("protenix_node_embed", node, length, 512)
        return node

    def _read_starling(self, row: pd.Series, length: int) -> np.ndarray:
        if _starling_read_disabled():
            return np.zeros((length, 512), dtype=np.float32)
        if not _truthy(row.get("has_starling")):
            return np.zeros((length, 512), dtype=np.float32)
        node = np.zeros((length, 512), dtype=np.float32)
        segment_ids = _parse_segment_ids(row.get("starling_segment_ids"))
        starts = _parse_int_list(row.get("starling_start_0based"))
        ends = _parse_int_list(row.get("starling_end_exclusive_0based"))
        paths = _parse_json_list(row.get("starling_node_shards"))
        offsets = _parse_int_list(row.get("starling_node_offsets"))
        lengths = _parse_int_list(row.get("starling_segment_lengths"))
        for sid, start, end, path_value, offset, seg_len in zip(segment_ids, starts, ends, paths, offsets, lengths):
            if end - start != seg_len:
                raise ValueError(f"STARLING segment length mismatch for {sid}: {start}-{end}, L={seg_len}")
            mirror = self._mirror_array(path_value, "starling_node_embed")
            if mirror is not None:
                segment = np.asarray(mirror[offset : offset + seg_len], dtype=np.float32)
            else:
                path = self._resolve_data_path(path_value)
                with _npz_open(str(path)) as data:
                    segment = np.asarray(data["starling_node_embed"][offset : offset + seg_len], dtype=np.float32)
            _check_node_shape("starling_node_embed", segment, seg_len, 512)
            node[start:end] = segment[: end - start]
        return node

    def _read_merged_graph(self, row: pd.Series, length: int) -> dict[str, np.ndarray]:
        path = self._resolve_data_path(row["merged_graph_shard"])
        if "graphs/merged_sparse" not in str(path).replace("\\", "/"):
            raise RuntimeError(f"Training graph must come from graphs/merged_sparse: {path}")
        offset = int(row["merged_graph_offset"])
        edge_count = int(row["graph_num_edges"])
        edge_index_store = self._mirror_array(row["merged_graph_shard"], "edge_index")
        edge_type_store = self._mirror_array(row["merged_graph_shard"], "edge_type")
        edge_attr_store = self._mirror_array(row["merged_graph_shard"], "edge_scalar_attr")
        if edge_index_store is not None and edge_type_store is not None and edge_attr_store is not None:
            edge_index = np.asarray(edge_index_store[:, offset : offset + edge_count], dtype=np.int64)
            edge_type = np.asarray(edge_type_store[offset : offset + edge_count], dtype=np.int64)
            edge_attr = np.asarray(edge_attr_store[offset : offset + edge_count], dtype=np.float32)
        else:
            edge_index_store = self._cached_npz_array(path, "edge_index")
            edge_type_store = self._cached_npz_array(path, "edge_type")
            edge_attr_store = self._cached_npz_array(path, "edge_scalar_attr")
            edge_index = np.asarray(edge_index_store[:, offset : offset + edge_count], dtype=np.int64)
            edge_type = np.asarray(edge_type_store[offset : offset + edge_count], dtype=np.int64)
            edge_attr = np.asarray(edge_attr_store[offset : offset + edge_count], dtype=np.float32)
        if edge_index.shape != (2, edge_count):
            raise ValueError(f"edge_index must have shape [2,{edge_count}], got {edge_index.shape}")
        if edge_attr.shape != (edge_count, 32):
            raise ValueError(f"edge_attr must have shape [{edge_count},32], got {edge_attr.shape}")
        if edge_count and (edge_index.min() < 0 or edge_index.max() >= length):
            raise ValueError(f"edge_index out of bounds for {row['protein_id']}")
        return {
            "edge_src": edge_index[0],
            "edge_dst": edge_index[1],
            "edge_type": edge_type,
            "edge_attr": edge_attr,
        }

    def _read_region_label_npz(self, row: pd.Series, length: int) -> dict[str, Any] | None:
        if self.region_labels_dir is None:
            return None
        protein_id = str(row["protein_id"])
        path = self.region_labels_dir / f"{protein_id}.npz"
        assert_offline_path_allowed(path)
        try:
            path.relative_to(self.dataset_root)
        except ValueError as exc:
            raise RuntimeError(f"Region label path escapes dataset_root: {path}") from exc
        if not path.exists():
            return None
        with _npz_open(str(path)) as data:
            residue_label = np.asarray(data["residue_label"], dtype=np.float32)
            residue_mask = np.asarray(data["residue_mask"], dtype=np.float32)
            residue_weight = np.asarray(data["residue_weight"], dtype=np.float32)
            boundary_label = (
                np.asarray(data["boundary_target"], dtype=np.float32)
                if "boundary_target" in data
                else None
            )
            boundary_weight = (
                np.asarray(data["boundary_mask"], dtype=np.float32)
                if "boundary_mask" in data
                else None
            )
            starts = np.asarray(data["span_start"], dtype=np.int64) if "span_start" in data else np.zeros(0, dtype=np.int64)
            ends = np.asarray(data["span_end"], dtype=np.int64) if "span_end" in data else np.zeros(0, dtype=np.int64)
            confidence = (
                np.asarray(data["span_confidence"], dtype=np.float32)
                if "span_confidence" in data
                else np.ones_like(starts, dtype=np.float32)
            )
            coordinate_system = str(np.asarray(data["coordinate_system"]).item()) if "coordinate_system" in data else "0-based inclusive"
        copy_len = min(length, int(residue_label.shape[0]), int(residue_mask.shape[0]), int(residue_weight.shape[0]))
        target = _empty_region_target(length)
        if copy_len <= 0:
            return target
        valid = (residue_mask[:copy_len] > 0.0) & np.isfinite(residue_label[:copy_len]) & (residue_weight[:copy_len] > 0.0)
        target["region_teacher_target"][:copy_len][valid] = np.clip(residue_label[:copy_len][valid], 0.0, 1.0)
        target["region_teacher_weight"][:copy_len][valid] = residue_weight[:copy_len][valid]
        target["region_contrast_target"][:copy_len][valid] = np.clip(residue_label[:copy_len][valid], 0.0, 1.0)
        target["region_contrast_weight"][:copy_len][valid] = residue_weight[:copy_len][valid]
        if boundary_label is not None and boundary_weight is not None:
            boundary_len = min(length, int(boundary_label.shape[0]), int(boundary_weight.shape[0]))
            boundary_valid = (
                np.isfinite(boundary_label[:boundary_len])
                & (boundary_weight[:boundary_len] > 0.0)
            )
            target["region_boundary_target"][:boundary_len][boundary_valid] = np.clip(
                boundary_label[:boundary_len][boundary_valid],
                0.0,
                1.0,
            )
            target["region_boundary_weight"][:boundary_len][boundary_valid] = boundary_weight[:boundary_len][boundary_valid]
        spans: list[dict[str, float | int]] = []
        for raw_start, raw_end, raw_confidence in zip(starts.tolist(), ends.tolist(), confidence.tolist(), strict=False):
            start = int(raw_start)
            end = int(raw_end)
            if "1-based" in coordinate_system.lower():
                start -= 1
                end_index = end - 1
            elif "half-open" in coordinate_system.lower() or "half_open" in coordinate_system.lower():
                end_index = end - 1
            else:
                end_index = end
            start = max(0, min(start, length - 1))
            end_index = max(start, min(end_index, length - 1))
            conf = float(np.clip(float(raw_confidence), 0.0, 1.0))
            spans.append({"start": start, "end": end_index, "confidence": conf, "sample_weight": conf})
            if boundary_label is None or boundary_weight is None:
                target["region_boundary_target"][start] = 1.0
                target["region_boundary_weight"][start] = max(float(target["region_boundary_weight"][start]), conf)
                target["region_boundary_target"][end_index] = 1.0
                target["region_boundary_weight"][end_index] = max(float(target["region_boundary_weight"][end_index]), conf)
        if not spans and (boundary_label is None or boundary_weight is None):
            labels = np.zeros(length, dtype=np.float32)
            labels[:copy_len] = np.clip(residue_label[:copy_len], 0.0, 1.0)
            weights = np.zeros(length, dtype=np.float32)
            weights[:copy_len] = np.where(valid, residue_weight[:copy_len], 0.0)
            transitions = np.zeros(length, dtype=np.float32)
            transitions[1:] = np.abs(labels[1:] - labels[:-1])
            transitions[:-1] = np.maximum(transitions[:-1], np.abs(labels[1:] - labels[:-1]))
            boundary_valid = (transitions > 0.0) & (weights > 0.0)
            target["region_boundary_target"][boundary_valid] = 1.0
            target["region_boundary_weight"][boundary_valid] = weights[boundary_valid]
        target["positive_spans"] = spans
        target["target_policy"] = "region_labels_npz"
        return target

    def _mirror_array(self, source_path: Any, array_name: str) -> np.ndarray | None:
        if self.npz_mirror_store is None:
            return None
        return self.npz_mirror_store.array(source_path, array_name, dataset_root=self.dataset_root)

    def _cached_npz_array(self, path: Path, array_name: str) -> np.ndarray:
        assert_offline_path_allowed(path)
        key = (str(path), str(array_name))
        if self._npz_array_cache_max_bytes > 0 and key in self._npz_array_cache:
            value = self._npz_array_cache.pop(key)
            self._npz_array_cache[key] = value
            return value
        with _npz_open(str(path)) as data:
            value = np.asarray(data[str(array_name)])
        if self._npz_array_cache_max_bytes <= 0:
            return value
        value_bytes = int(value.nbytes)
        if value_bytes > self._npz_array_cache_max_bytes:
            return value
        while self._npz_array_cache and self._npz_array_cache_bytes + value_bytes > self._npz_array_cache_max_bytes:
            _, evicted = self._npz_array_cache.popitem(last=False)
            self._npz_array_cache_bytes -= int(evicted.nbytes)
        self._npz_array_cache[key] = value
        self._npz_array_cache_bytes += value_bytes
        return value

    @staticmethod
    def _modality_matrices(row: pd.Series, length: int) -> tuple[np.ndarray, np.ndarray]:
        protenix_present = False if _protenix_read_disabled() else _truthy(row.get("modality_mask_protenix"))
        protenix_reliability = 0.0 if _protenix_read_disabled() else _float_or_default(row.get("reliability_protenix"), 0.0)
        starling_present = False if _starling_read_disabled() else _truthy(row.get("modality_mask_starling"))
        starling_reliability = 0.0 if _starling_read_disabled() else _float_or_default(row.get("reliability_starling"), 0.0)
        present = np.asarray(
            [
                _truthy(row.get("modality_mask_esm2")),
                _truthy(row.get("modality_mask_biophys")),
                _truthy(row.get("modality_mask_biophys")),
                protenix_present,
                starling_present,
            ],
            dtype=np.float32,
        )
        reliability = np.asarray(
            [
                _float_or_default(row.get("reliability_esm2"), 0.0),
                _float_or_default(row.get("reliability_biophys"), 0.0),
                _float_or_default(row.get("reliability_biophys"), 0.0),
                protenix_reliability,
                starling_reliability,
            ],
            dtype=np.float32,
        )
        missing_mask = 1.0 - present
        return (
            np.broadcast_to(missing_mask, (length, 5)).copy(),
            np.broadcast_to(reliability, (length, 5)).copy(),
        )


def _npz_open(path: str) -> np.lib.npyio.NpzFile:
    assert_offline_path_allowed(path)
    return np.load(path, allow_pickle=False)


def _check_node_shape(name: str, value: np.ndarray, length: int, dim: int) -> None:
    if value.shape != (length, dim):
        raise ValueError(f"{name} must have shape [{length},{dim}], got {value.shape}")


def _empty_graph(length: int) -> dict[str, np.ndarray]:
    return {
        "edge_src": np.zeros(0, dtype=np.int64),
        "edge_dst": np.zeros(0, dtype=np.int64),
        "edge_type": np.zeros(0, dtype=np.int64),
        "edge_attr": np.zeros((0, 32), dtype=np.float32),
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _profile_dataset_timing_enabled() -> bool:
    return str(os.environ.get("PHASEFLOW_PROFILE_DATASET_TIMING", "")).strip().lower() in {"1", "true", "yes", "on"}


def _starling_read_disabled() -> bool:
    return str(os.environ.get("PHASEFLOW_DISABLE_STARLING_READ", "")).strip().lower() in {"1", "true", "yes", "on"}


def _protenix_read_disabled() -> bool:
    return str(os.environ.get("PHASEFLOW_DISABLE_PROTENIX_READ", "")).strip().lower() in {"1", "true", "yes", "on"}


def _float_or_default(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if np.isnan(out):
            return float(default)
        return out
    except (TypeError, ValueError):
        return float(default)


def _parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    return [str(item) for item in parsed]


def _parse_int_list(value: Any) -> list[int]:
    return [int(item) for item in _parse_json_list(value)]


def _parse_segment_ids(value: Any) -> list[str]:
    return _parse_json_list(value)


# Source: data/collator.py


import os
import time
from dataclasses import dataclass
from typing import Any

import torch

from phaseflow.protein.contracts import IGNORE_INDEX


PHASEFLOW_LLPS_HIDDEN_LAYERS_KEY = "phaseflow_llps_hidden_layers"
LEGACY_LLPS_HIDDEN_LAYERS_KEY = "phase" + "gt_hidden_layers"


@dataclass(slots=True)
class PhaseFlowCollator:
    max_neighbors: int = 96
    edge_attr_dim: int | None = None
    require_precomputed_graph: bool = False

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        profile_timing = _profile_collate_timing_enabled()
        collate_start = time.perf_counter() if profile_timing else 0.0
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        batch_size = len(samples)
        max_len = max(int(sample["length"]) for sample in samples)
        edge_dim = self.edge_attr_dim or _graph_edge_dim(samples)

        batch: dict[str, Any] = {
            "dataset_indices": [int(sample.get("dataset_index", -1)) for sample in samples],
            "sample_ids": [str(sample.get("sample_id", sample["protein_id"])) for sample in samples],
            "protein_ids": [sample["protein_id"] for sample in samples],
            "sequences": [sample["sequence"] for sample in samples],
            "lengths": torch.tensor([sample["length"] for sample in samples], dtype=torch.long),
            "seq_mask": torch.zeros(batch_size, max_len, dtype=torch.bool),
            "esm2_available_mask": torch.zeros(batch_size, max_len, dtype=torch.float32),
            "regions": [sample["regions"] for sample in samples],
            "structure_metadata": [sample.get("structure_metadata", {}) for sample in samples],
            "label_quality": [str(sample.get("label_quality", "")) for sample in samples],
            "negative_type": [str(sample.get("negative_type", "")) for sample in samples],
            "llps_role": [str(sample.get("llps_role", "")) for sample in samples],
            "region_bag_type": [str(sample.get("region_bag_type", "")) for sample in samples],
            "source": [str(sample.get("source", "")) for sample in samples],
            "plan_pool_name": [str(sample.get("plan_pool_name", "")) for sample in samples],
            "plan_label_group": [str(sample.get("plan_label_group", "")) for sample in samples],
            "plan_tier": [str(sample.get("plan_tier", "")) for sample in samples],
            "plan_negative_type": [str(sample.get("plan_negative_type", "")) for sample in samples],
            "hard_mining_source": [str(sample.get("hard_mining_source", "")) for sample in samples],
            "mixed_positive_source": [str(sample.get("mixed_positive_source", "")) for sample in samples],
        }

        bio_dim = int(samples[0].get("bio_vec", torch.zeros(0, dtype=torch.float32)).numel())
        if bio_dim > 0:
            batch["bio_vec"] = torch.zeros(batch_size, bio_dim, dtype=torch.float32)

        node_names = (
            "plm",
            "physchem",
            "disorder",
            "protenix_embed",
            "starling_embed",
            "modality_mask",
            "reliability",
        )
        for name in node_names:
            dim = int(samples[0][name].shape[1])
            batch[name] = torch.zeros(batch_size, max_len, dim, dtype=torch.float32)
        hidden_cache_specs: dict[str, tuple[int, int]] = {}
        for name in (PHASEFLOW_LLPS_HIDDEN_LAYERS_KEY, "phaseflow_hidden_layers"):
            value = _get_hidden_cache(samples[0], name)
            if torch.is_tensor(value) and value.ndim == 3:
                hidden_cache_specs[name] = (int(value.shape[0]), int(value.shape[2]))
                batch[name] = torch.zeros(batch_size, int(value.shape[0]), max_len, int(value.shape[2]), dtype=torch.float32)

        batch["y_llps"] = torch.full((batch_size,), float(IGNORE_INDEX), dtype=torch.float32)
        batch["sample_weight"] = torch.ones(batch_size, dtype=torch.float32)
        batch["teacher_llps"] = torch.full((batch_size,), float("nan"), dtype=torch.float32)
        batch["teacher_llps_weight"] = torch.zeros(batch_size, dtype=torch.float32)
        batch["self_llps"] = torch.full((batch_size,), float("nan"), dtype=torch.float32)
        batch["self_llps_weight"] = torch.zeros(batch_size, dtype=torch.float32)
        batch["region_bag_label"] = torch.full((batch_size,), float(IGNORE_INDEX), dtype=torch.float32)
        batch["region_bag_weight"] = torch.zeros(batch_size, dtype=torch.float32)
        batch["negative_regularization_weight"] = torch.zeros(batch_size, dtype=torch.float32)
        batch["y_dpr"] = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        batch["y_key"] = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        batch["y_weight"] = torch.zeros(batch_size, max_len, dtype=torch.float32)
        batch["teacher_dpr"] = torch.full((batch_size, max_len), float("nan"), dtype=torch.float32)
        batch["teacher_dpr_weight"] = torch.zeros(batch_size, max_len, dtype=torch.float32)
        batch["self_dpr"] = torch.full((batch_size, max_len), float("nan"), dtype=torch.float32)
        batch["self_dpr_weight"] = torch.zeros(batch_size, max_len, dtype=torch.float32)
        batch["candidate_prior"] = torch.zeros(batch_size, max_len, dtype=torch.float32)
        batch["candidate_prior_weight"] = torch.zeros(batch_size, max_len, dtype=torch.float32)
        batch["phase_values"] = torch.zeros(batch_size, 16, dtype=torch.float32)
        batch["phase_mask"] = torch.zeros(batch_size, 16, dtype=torch.float32)
        batch["phase_aux_weight"] = torch.zeros(batch_size, dtype=torch.float32)
        batch["phase_mean_pssi"] = torch.full((batch_size,), float("nan"), dtype=torch.float32)
        batch["phase_low_pssi"] = torch.full((batch_size,), float("nan"), dtype=torch.float32)
        batch["region_teacher_target"] = torch.full((batch_size, max_len), float("nan"), dtype=torch.float32)
        batch["region_teacher_weight"] = torch.zeros(batch_size, max_len, dtype=torch.float32)
        batch["region_key_target"] = torch.full((batch_size, max_len), float("nan"), dtype=torch.float32)
        batch["region_key_weight"] = torch.zeros(batch_size, max_len, dtype=torch.float32)
        batch["region_boundary_target"] = torch.full((batch_size, max_len), float("nan"), dtype=torch.float32)
        batch["region_boundary_weight"] = torch.zeros(batch_size, max_len, dtype=torch.float32)
        batch["region_contrast_target"] = torch.full((batch_size, max_len), float("nan"), dtype=torch.float32)
        batch["region_contrast_weight"] = torch.zeros(batch_size, max_len, dtype=torch.float32)
        batch["neighbors"] = torch.zeros(batch_size, max_len, self.max_neighbors, dtype=torch.long)
        batch["edge_attr"] = torch.zeros(batch_size, max_len, self.max_neighbors, edge_dim, dtype=torch.float32)
        batch["neighbor_mask"] = torch.zeros(batch_size, max_len, self.max_neighbors, dtype=torch.bool)
        batch["neighbor_edge_type"] = torch.zeros(batch_size, max_len, self.max_neighbors, dtype=torch.long)

        for batch_index, sample in enumerate(samples):
            length = int(sample["length"])
            batch["seq_mask"][batch_index, :length] = True
            if "esm2_available_mask" in sample:
                batch["esm2_available_mask"][batch_index, :length] = sample["esm2_available_mask"][:length].float()
            else:
                batch["esm2_available_mask"][batch_index, :length] = 1.0
            for name in node_names:
                batch[name][batch_index, :length] = sample[name]
            for name, (num_layers, hidden_dim) in hidden_cache_specs.items():
                value = _get_hidden_cache(sample, name)
                if torch.is_tensor(value):
                    if value.shape != (num_layers, length, hidden_dim):
                        raise ValueError(f"{name} must have shape [{num_layers},{length},{hidden_dim}], got {tuple(value.shape)}")
                    batch[name][batch_index, :, :length] = value.float()
            batch["y_llps"][batch_index] = sample["y_llps"]
            batch["sample_weight"][batch_index] = float(sample.get("sample_weight", 1.0))
            if "bio_vec" in batch and "bio_vec" in sample:
                batch["bio_vec"][batch_index] = sample["bio_vec"].float()
            batch["teacher_llps"][batch_index] = sample["teacher_llps"]
            batch["teacher_llps_weight"][batch_index] = sample["teacher_llps_weight"]
            batch["self_llps"][batch_index] = sample["self_llps"]
            batch["self_llps_weight"][batch_index] = sample["self_llps_weight"]
            batch["region_bag_label"][batch_index] = sample["region_bag_label"]
            batch["region_bag_weight"][batch_index] = sample["region_bag_weight"]
            batch["negative_regularization_weight"][batch_index] = sample["negative_regularization_weight"]
            batch["y_dpr"][batch_index, :length] = sample["y_dpr"]
            batch["y_key"][batch_index, :length] = sample["y_key"]
            batch["y_weight"][batch_index, :length] = sample["y_weight"]
            batch["teacher_dpr"][batch_index, :length] = sample["teacher_dpr"]
            batch["teacher_dpr_weight"][batch_index, :length] = sample["teacher_dpr_weight"]
            batch["self_dpr"][batch_index, :length] = sample["self_dpr"]
            batch["self_dpr_weight"][batch_index, :length] = sample["self_dpr_weight"]
            batch["candidate_prior"][batch_index, :length] = sample["candidate_prior"]
            batch["candidate_prior_weight"][batch_index, :length] = sample["candidate_prior_weight"]
            batch["phase_values"][batch_index] = sample.get("phase_values", torch.zeros(16, dtype=torch.float32))
            batch["phase_mask"][batch_index] = sample.get("phase_mask", torch.zeros(16, dtype=torch.float32))
            batch["phase_aux_weight"][batch_index] = sample.get("phase_aux_weight", torch.tensor(0.0))
            batch["phase_mean_pssi"][batch_index] = sample.get("phase_mean_pssi", torch.tensor(float("nan")))
            batch["phase_low_pssi"][batch_index] = sample.get("phase_low_pssi", torch.tensor(float("nan")))
            batch["region_teacher_target"][batch_index, :length] = sample.get(
                "region_teacher_target",
                torch.full((length,), float("nan"), dtype=torch.float32),
            )
            batch["region_teacher_weight"][batch_index, :length] = sample.get(
                "region_teacher_weight",
                torch.zeros(length, dtype=torch.float32),
            )
            batch["region_key_target"][batch_index, :length] = sample.get(
                "region_key_target",
                torch.full((length,), float("nan"), dtype=torch.float32),
            )
            batch["region_key_weight"][batch_index, :length] = sample.get(
                "region_key_weight",
                torch.zeros(length, dtype=torch.float32),
            )
            batch["region_boundary_target"][batch_index, :length] = sample.get(
                "region_boundary_target",
                torch.full((length,), float("nan"), dtype=torch.float32),
            )
            batch["region_boundary_weight"][batch_index, :length] = sample.get(
                "region_boundary_weight",
                torch.zeros(length, dtype=torch.float32),
            )
            batch["region_contrast_target"][batch_index, :length] = sample.get(
                "region_contrast_target",
                torch.full((length,), float("nan"), dtype=torch.float32),
            )
            batch["region_contrast_weight"][batch_index, :length] = sample.get(
                "region_contrast_weight",
                torch.zeros(length, dtype=torch.float32),
            )
            if _has_usable_precomputed_graph(sample, self.max_neighbors, edge_dim):
                neighbors, edge_attr, neighbor_mask = _slice_precomputed_graph(sample, self.max_neighbors, edge_dim)
                neighbor_edge_type = torch.zeros_like(neighbors)
            else:
                if self.require_precomputed_graph:
                    protein_id = str(sample.get("protein_id", "<unknown>"))
                    raise ValueError(
                        f"{protein_id} is missing a usable precomputed graph cache for "
                        f"max_neighbors={self.max_neighbors}, edge_dim={edge_dim}"
                    )
                neighbors, edge_attr, neighbor_mask, neighbor_edge_type = _edge_list_to_neighbors(
                    length=length,
                    edge_src=sample["edge_src"],
                    edge_dst=sample["edge_dst"],
                    edge_type=sample["edge_type"],
                    edge_attr=sample["edge_attr"],
                    max_neighbors=self.max_neighbors,
                    edge_dim=edge_dim,
                )
            batch["neighbors"][batch_index, :length] = neighbors
            batch["edge_attr"][batch_index, :length] = edge_attr
            batch["neighbor_mask"][batch_index, :length] = neighbor_mask
            batch["neighbor_edge_type"][batch_index, :length] = neighbor_edge_type

        if profile_timing:
            batch["__sample_timing"] = [sample.get("__timing", {}) for sample in samples]
            batch["__sample_io_paths"] = [sample.get("__io_paths", {}) for sample in samples]
            batch["__sample_io_sizes"] = [sample.get("__io_sizes", {}) for sample in samples]
            batch["__collate_sec"] = time.perf_counter() - collate_start
        return batch


def _graph_edge_dim(samples: list[dict[str, Any]]) -> int:
    for sample in samples:
        precomputed_edge_attr = sample.get("precomputed_edge_attr")
        if torch.is_tensor(precomputed_edge_attr) and precomputed_edge_attr.numel():
            return int(precomputed_edge_attr.shape[2])
    for sample in samples:
        edge_attr = sample.get("edge_attr")
        if torch.is_tensor(edge_attr) and edge_attr.numel():
            return int(edge_attr.shape[1])
    return 8


def _profile_collate_timing_enabled() -> bool:
    return str(os.environ.get("PHASEFLOW_PROFILE_COLLATE_TIMING", "")).strip().lower() in {"1", "true", "yes", "on"}


def _get_hidden_cache(sample: dict[str, Any], name: str) -> Any:
    if name == PHASEFLOW_LLPS_HIDDEN_LAYERS_KEY:
        return sample.get(name, sample.get(LEGACY_LLPS_HIDDEN_LAYERS_KEY))
    return sample.get(name)


def _has_usable_precomputed_graph(sample: dict[str, Any], max_neighbors: int, edge_dim: int) -> bool:
    neighbors = sample.get("precomputed_neighbors")
    edge_attr = sample.get("precomputed_edge_attr")
    neighbor_mask = sample.get("precomputed_neighbor_mask")
    if not torch.is_tensor(neighbors) or not torch.is_tensor(edge_attr) or not torch.is_tensor(neighbor_mask):
        return False
    length = int(sample["length"])
    return (
        neighbors.ndim == 2
        and edge_attr.ndim == 3
        and neighbor_mask.ndim == 2
        and neighbors.shape[0] >= length
        and edge_attr.shape[0] >= length
        and neighbor_mask.shape[0] >= length
        and neighbors.shape[1] >= max_neighbors
        and edge_attr.shape[1] >= max_neighbors
        and neighbor_mask.shape[1] >= max_neighbors
        and edge_attr.shape[2] >= edge_dim
    )


def _slice_precomputed_graph(
    sample: dict[str, Any],
    max_neighbors: int,
    edge_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = int(sample["length"])
    neighbors = sample["precomputed_neighbors"][:length, :max_neighbors].long()
    edge_attr = sample["precomputed_edge_attr"][:length, :max_neighbors, :edge_dim].float()
    neighbor_mask = sample["precomputed_neighbor_mask"][:length, :max_neighbors].bool()
    return neighbors, edge_attr, neighbor_mask


def _edge_list_to_neighbors(
    length: int,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_type: torch.Tensor,
    edge_attr: torch.Tensor,
    max_neighbors: int,
    edge_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    neighbors = torch.zeros(length, max_neighbors, dtype=torch.long)
    neighbor_attr = torch.zeros(length, max_neighbors, edge_dim, dtype=torch.float32)
    neighbor_mask = torch.zeros(length, max_neighbors, dtype=torch.bool)
    neighbor_edge_type = torch.zeros(length, max_neighbors, dtype=torch.long)

    if length <= 0 or max_neighbors <= 0:
        return neighbors, neighbor_attr, neighbor_mask, neighbor_edge_type

    counts = torch.zeros(length, dtype=torch.long)
    if edge_src.numel():
        valid = (edge_src >= 0) & (edge_src < length) & (edge_dst >= 0) & (edge_dst < length)
        if valid.any():
            src = edge_src[valid].long()
            dst = edge_dst[valid].long()
            edge_type_valid = edge_type[valid].long().clamp_min(0)
            distance = (dst - src).abs()
            type_stride = length + 1
            src_stride = max(int(edge_type_valid.max().item()) + 1, 1) * type_stride
            sort_key = src * src_stride + edge_type_valid * type_stride + distance
            order = torch.argsort(sort_key, stable=True)

            src_sorted = src[order]
            dst_sorted = dst[order]
            counts = torch.bincount(src_sorted, minlength=length)
            starts = torch.cumsum(counts, dim=0) - counts
            positions = torch.arange(src_sorted.numel(), dtype=torch.long) - torch.repeat_interleave(starts, counts)
            keep = positions < max_neighbors
            if keep.any():
                kept_src = src_sorted[keep]
                kept_rank = positions[keep]
                neighbors[kept_src, kept_rank] = dst_sorted[keep]
                neighbor_mask[kept_src, kept_rank] = True
                if edge_attr.numel():
                    width = min(edge_dim, int(edge_attr.shape[1]))
                    attr_sorted = edge_attr[valid][order]
                    neighbor_attr[kept_src, kept_rank, :width] = attr_sorted[keep, :width].float()
                neighbor_edge_type[kept_src, kept_rank] = edge_type_valid[order][keep]

    missing = counts == 0
    if missing.any():
        missing_src = torch.arange(length, dtype=torch.long)[missing]
        neighbors[missing_src, 0] = missing_src
        neighbor_mask[missing_src, 0] = True
    return neighbors, neighbor_attr, neighbor_mask, neighbor_edge_type


# Source: data/batch_plan_dataset.py


import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from phaseflow.protein.data import PhaseFlowCollator
from phaseflow.protein.data import PhaseFlowOfflineDataset
from phaseflow.protein.contracts import IGNORE_INDEX


class BatchPlanEpochSampler(Sampler[tuple[int, int]]):
    """Sampler that makes persistent DataLoader workers epoch-aware.

    The epoch is part of the index tuple, so worker dataset copies do not need
    mutable `set_epoch()` state.
    """

    def __init__(self, steps_by_epoch: dict[int, list[int]]) -> None:
        self.steps_by_epoch = {int(epoch): [int(step) for step in steps] for epoch, steps in steps_by_epoch.items()}
        if not self.steps_by_epoch:
            raise ValueError("BatchPlanEpochSampler requires at least one epoch")
        self.epochs = sorted(self.steps_by_epoch)
        self._epoch = self.epochs[0]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = self.epochs[int(epoch) % len(self.epochs)]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for step in self.steps_by_epoch[self._epoch]:
            yield (int(self._epoch), int(step))

    def __len__(self) -> int:
        return len(self.steps_by_epoch[self._epoch])


class PhaseFlowBatchPlanDataset(Dataset[dict[str, Any]]):
    """Rank-local batches materialized on demand from an index-only plan."""

    def __init__(
        self,
        plan_dir: str | Path,
        sample_index: str | Path = "data/processed/merged/tables/training_sample_index.parquet",
        dataset_root: str | Path = "data/processed/merged",
        input_contract: str | Path | None = None,
        esm2_store_metadata: str | Path | None = None,
        npz_mirror_manifest: str | Path | None = None,
        region_supervision: str = "none",
        local_rank: int | None = None,
        rank: int | None = None,
        max_neighbors: int = 96,
        edge_attr_dim: int | None = None,
        require_precomputed_graph: bool = False,
        hot_cache_pools: list[str] | tuple[str, ...] | None = None,
        hot_cache_max_samples: int = 0,
    ) -> None:
        self.plan_dir = Path(plan_dir)
        self.local_rank = _batch_plan_dataset_resolve_rank(local_rank, "LOCAL_RANK")
        self.rank = _batch_plan_dataset_resolve_rank(rank, "RANK", default=self.local_rank)
        self.offline = PhaseFlowOfflineDataset(
            dataset_root=dataset_root,
            sample_index=sample_index,
            input_contract=input_contract,
            region_supervision=region_supervision,
            esm2_store_metadata=esm2_store_metadata,
            esm2_store_required=esm2_store_metadata is not None,
            npz_mirror_manifest=npz_mirror_manifest,
        )
        self.collator = PhaseFlowCollator(
            max_neighbors=int(max_neighbors),
            edge_attr_dim=edge_attr_dim,
            require_precomputed_graph=bool(require_precomputed_graph),
        )
        self.epoch_plans = self._load_epoch_plans()
        self.steps_by_epoch = {epoch: sorted(plan) for epoch, plan in self.epoch_plans.items()}
        self.sample_count = int(sum(len(indices) for indices in self.epoch_plans[min(self.epoch_plans)].values()))
        self.hot_cache_pools = {str(item) for item in (hot_cache_pools or [])}
        self.hot_cache_max_samples = int(hot_cache_max_samples)
        self._hot_cache_indices = self._collect_hot_cache_indices()
        self._hot_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def make_sampler(self) -> BatchPlanEpochSampler:
        return BatchPlanEpochSampler(self.steps_by_epoch)

    def __len__(self) -> int:
        first_epoch = min(self.epoch_plans)
        return len(self.epoch_plans[first_epoch])

    def __getitem__(self, index: tuple[int, int] | int) -> dict[str, Any]:
        if isinstance(index, tuple):
            epoch, step = int(index[0]), int(index[1])
        else:
            epoch = min(self.epoch_plans)
            step = int(index)
        epoch = epoch % len(self.epoch_plans)
        start = time.perf_counter()
        dataset_indices = self.epoch_plans[epoch][step]
        plan_rows = self._rank_plan_row_maps[epoch][step]
        samples = [
            self._apply_plan_metadata(dict(self._get_sample(int(dataset_index))), plan_rows[offset])
            for offset, dataset_index in enumerate(dataset_indices)
        ]
        batch = self.collator(samples)
        batch["__batch_plan_epoch"] = int(epoch)
        batch["__batch_plan_global_step"] = int(step)
        batch["__batch_plan_rank"] = int(self.local_rank)
        batch["__batch_plan_dataset_indices"] = [int(item) for item in dataset_indices]
        batch["__batch_plan_getitem_sec"] = float(time.perf_counter() - start)
        batch["__batch_plan_shard_count"] = int(self._rank_shard_counts[epoch].get(step, 0))
        return batch

    def epoch_stats(self, epoch: int | None = None) -> dict[str, Any]:
        chosen = min(self.epoch_plans) if epoch is None else int(epoch) % len(self.epoch_plans)
        rows = self._rank_plan_frames[chosen]
        real = int(rows["length"].sum())
        padded = int(rows.groupby("global_step")["length"].max().sum() * rows.groupby("global_step").size().max())
        return {
            "epoch": int(chosen),
            "rank": int(self.rank),
            "local_rank": int(self.local_rank),
            "sampler": "tiered_batch_plan_sharded_store",
            "rank_batches": int(rows["global_step"].nunique()),
            "samples": int(len(rows)),
            "real_residues": real,
            "padded_residues": padded,
            "padding_ratio": (padded - real) / max(padded, 1),
        }

    def _load_epoch_plans(self) -> dict[int, dict[int, np.ndarray]]:
        epoch_files = sorted(self.plan_dir.glob("batch_plan_epoch_*.parquet"))
        if not epoch_files:
            fallback = self.plan_dir / "batch_plan.parquet"
            if not fallback.exists():
                raise FileNotFoundError(f"No batch_plan parquet files found under {self.plan_dir}")
            epoch_files = [fallback]
        self._rank_plan_frames: dict[int, pd.DataFrame] = {}
        self._rank_plan_row_maps: dict[int, dict[int, list[dict[str, Any]]]] = {}
        self._rank_shard_counts: dict[int, dict[int, int]] = {}
        out: dict[int, dict[int, np.ndarray]] = {}
        for path in epoch_files:
            frame = pd.read_parquet(path)
            rank_col = "local_rank" if "local_rank" in frame.columns else "rank"
            frame = frame[frame[rank_col].astype(int).eq(int(self.local_rank))].copy()
            if frame.empty:
                raise ValueError(f"No plan rows for local_rank={self.local_rank} in {path}")
            epoch = int(frame["epoch"].iloc[0])
            frame = frame.sort_values(["global_step", "local_slot"]).reset_index(drop=True)
            self._rank_plan_frames[epoch] = frame
            self._rank_shard_counts[epoch] = (
                frame.groupby("global_step")["embedding_shard_id"].nunique().astype(int).to_dict()
            )
            self._rank_plan_row_maps[epoch] = {
                int(step): chunk.to_dict("records")
                for step, chunk in frame.groupby("global_step", sort=True)
            }
            index_col = "plan_dataset_index" if "plan_dataset_index" in frame.columns else "dataset_index"
            out[epoch] = {
                int(step): chunk[index_col].to_numpy(dtype=np.int64)
                for step, chunk in frame.groupby("global_step", sort=True)
            }
        return out

    def _apply_plan_metadata(self, sample: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        label_group = str(row.get("label_group", ""))
        if label_group == "positive":
            sample["y_llps"] = torch.tensor(1.0, dtype=torch.float32)
        elif label_group == "negative":
            sample["y_llps"] = torch.tensor(0.0, dtype=torch.float32)
        elif label_group == "pu":
            sample["y_llps"] = torch.tensor(float(IGNORE_INDEX), dtype=torch.float32)

        if label_group == "pu":
            sample["sample_weight"] = torch.tensor(0.0, dtype=torch.float32)
        elif "sample_weight" in row:
            sample["sample_weight"] = torch.tensor(float(row["sample_weight"]), dtype=torch.float32)

        sample["label_quality"] = str(row.get("tier", sample.get("label_quality", "")))
        sample["source"] = str(row.get("source", sample.get("source", "")))
        if "negative_type" in row:
            sample["negative_type"] = str(row.get("negative_type", sample.get("negative_type", "")))
        sample["llps_role"] = str(row.get("llps_role", sample.get("llps_role", "")))
        sample["plan_pool_name"] = str(row.get("pool_name", ""))
        sample["plan_label_group"] = label_group
        sample["plan_tier"] = str(row.get("tier", ""))
        sample["plan_negative_type"] = str(row.get("negative_type", ""))
        sample["hard_mining_source"] = str(row.get("hard_mining_source", ""))
        sample["mixed_positive_source"] = str(row.get("mixed_positive_source", ""))
        return sample

    def _collect_hot_cache_indices(self) -> set[int]:
        if not self.hot_cache_pools or self.hot_cache_max_samples <= 0:
            return set()
        values: set[int] = set()
        for frame in getattr(self, "_rank_plan_frames", {}).values():
            subset = frame[frame["pool_name"].astype(str).isin(self.hot_cache_pools)]
            index_col = "plan_dataset_index" if "plan_dataset_index" in subset.columns else "dataset_index"
            values.update(int(item) for item in subset[index_col].astype(int).tolist())
        return values

    def _get_sample(self, dataset_index: int) -> dict[str, Any]:
        if dataset_index not in self._hot_cache_indices or self.hot_cache_max_samples <= 0:
            return self.offline[int(dataset_index)]
        cached = self._hot_cache.get(dataset_index)
        if cached is not None:
            self._hot_cache.move_to_end(dataset_index)
            return cached
        sample = self.offline[int(dataset_index)]
        self._hot_cache[dataset_index] = sample
        self._hot_cache.move_to_end(dataset_index)
        while len(self._hot_cache) > self.hot_cache_max_samples:
            self._hot_cache.popitem(last=False)
        return sample


def _batch_plan_dataset_resolve_rank(value: int | None, env_name: str, default: int = 0) -> int:
    if value is not None:
        return int(value)
    try:
        return int(__import__("os").environ.get(env_name, default))
    except (TypeError, ValueError):
        return int(default)


# Source: data/dpr_offline_labels.py


from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HalfOpenSpan:
    start: int
    end: int
    confidence: float = 1.0
    sample_weight: float = 1.0


def build_dpr_label_arrays(
    *,
    length: int,
    spans: list[HalfOpenSpan],
    boundary_radius: int = 3,
) -> dict[str, np.ndarray]:
    """Build DPR offline labels from zero-based half-open spans.

    Positive residues are [start, end). The end boundary anchor is end - 1,
    never end. Residues outside provided spans stay masked out here; callers can
    overlay reliable negative masks separately.
    """

    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    residue_target = np.zeros(length, dtype=np.float32)
    residue_mask = np.zeros(length, dtype=np.float32)
    residue_weight = np.zeros(length, dtype=np.float32)
    start_target = np.full(length, np.nan, dtype=np.float32)
    end_target = np.full(length, np.nan, dtype=np.float32)
    boundary_weight = np.zeros(length, dtype=np.float32)

    radius = max(0, int(boundary_radius))
    for span in spans:
        start = max(0, min(int(span.start), length))
        end = max(start, min(int(span.end), length))
        if end <= start:
            continue
        weight = float(np.clip(float(span.confidence) * float(span.sample_weight), 0.0, 1.0))
        if weight <= 0.0:
            continue
        residue_target[start:end] = 1.0
        residue_mask[start:end] = 1.0
        residue_weight[start:end] = np.maximum(residue_weight[start:end], weight)
        _write_boundary(start_target, boundary_weight, center=start, radius=radius, weight=weight)
        _write_boundary(end_target, boundary_weight, center=end - 1, radius=radius, weight=weight)

    return {
        "residue_target": residue_target,
        "residue_mask": residue_mask,
        "residue_weight": residue_weight,
        "start_target": start_target,
        "end_target": end_target,
        "boundary_weight": boundary_weight,
    }


def _write_boundary(
    target: np.ndarray,
    weight_arr: np.ndarray,
    *,
    center: int,
    radius: int,
    weight: float,
) -> None:
    length = int(target.shape[0])
    center = max(0, min(int(center), length - 1))
    left = max(0, center - radius)
    right = min(length, center + radius + 1)
    for pos in range(left, right):
        if radius <= 0:
            value = 1.0 if pos == center else 0.0
        else:
            value = max(0.0, 1.0 - abs(pos - center) / float(radius + 1))
        if not np.isfinite(target[pos]) or value > float(target[pos]):
            target[pos] = np.float32(value)
        weight_arr[pos] = max(float(weight_arr[pos]), float(weight))


# Source: data/dpr_window_dataset.py


from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

from phaseflow.protein.data import PhaseFlowOfflineDataset


DEFAULT_STAGE2_SAMPLE_INDEXES = (
    "sample_indexes/residue_supervised_index.parquet",
    "sample_indexes/bag_positive_index.parquet",
    "sample_indexes/negative_index.parquet",
    "sample_indexes/unlabeled_index.parquet",
)


class DPRWindowDataset(Dataset[dict[str, Any]]):
    """Stage2 DPR crop dataset backed by the audited window index."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        window_index: str | Path | None = None,
        sample_indexes: Iterable[str | Path] = DEFAULT_STAGE2_SAMPLE_INDEXES,
        input_contract: str | Path | None = None,
        region_labels_dir: str | Path | None = None,
        split: str | None = None,
        window_types: Iterable[str] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.window_index_path = Path(window_index) if window_index is not None else self.dataset_root / "window_indexes/window_index.parquet"
        self.window_index = pd.read_parquet(self.window_index_path)
        if split is not None and "split" in self.window_index.columns:
            self.window_index = self.window_index.loc[self.window_index["split"].astype(str) == str(split)]
        if window_types is not None:
            allowed = {str(item) for item in window_types}
            self.window_index = self.window_index.loc[self.window_index["window_type"].astype(str).isin(allowed)]
        self.window_index = self.window_index.reset_index(drop=True)
        contract = Path(input_contract) if input_contract is not None else self.dataset_root / "configs/offline_input_contract.yaml"
        labels = Path(region_labels_dir) if region_labels_dir is not None else self.dataset_root / "region_labels"
        self.datasets: list[PhaseFlowOfflineDataset] = []
        self.pid_to_item: dict[str, tuple[int, int]] = {}
        for sample_index in sample_indexes:
            path = Path(sample_index)
            if not path.is_absolute():
                path = self.dataset_root / path
            if not path.exists():
                continue
            dataset = PhaseFlowOfflineDataset(
                dataset_root=self.dataset_root,
                sample_index=path,
                input_contract=contract,
                region_labels_dir=labels,
                region_supervision="region_targets",
            )
            dataset_id = len(self.datasets)
            self.datasets.append(dataset)
            for item_index, protein_id in enumerate(dataset.protein_ids):
                self.pid_to_item.setdefault(str(protein_id), (dataset_id, item_index))
        if not self.datasets:
            raise FileNotFoundError("No usable Stage2 sample indexes were found")
        self.window_index = self.window_index.loc[
            self.window_index["protein_id"].astype(str).isin(self.pid_to_item)
        ].reset_index(drop=True)

    def __len__(self) -> int:
        return int(len(self.window_index))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.window_index.iloc[int(index)]
        protein_id = str(row["protein_id"])
        dataset_id, item_index = self.pid_to_item[protein_id]
        sample = self.datasets[dataset_id][item_index]
        start = int(row["window_start"])
        end = int(row["window_end"])
        if end <= start:
            raise ValueError(f"Invalid window for {protein_id}: {start}-{end}")
        return _crop_sample(sample, start=start, end=end, row=row)


def _crop_sample(sample: dict[str, Any], *, start: int, end: int, row: pd.Series) -> dict[str, Any]:
    length = int(sample["length"])
    start = max(0, min(int(start), length))
    end = max(start + 1, min(int(end), length))
    crop_len = end - start
    out: dict[str, Any] = {}
    for key, value in sample.items():
        if key in {"edge_src", "edge_dst", "edge_type", "edge_attr"}:
            continue
        if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == length:
            out[key] = value[start:end].clone()
        else:
            out[key] = value
    out["sequence"] = str(sample["sequence"])[start:end]
    out["length"] = crop_len
    out["sample_id"] = f"{sample['protein_id']}:{start}-{end}:{row.get('window_id', '')}"
    out["window_id"] = str(row.get("window_id", ""))
    out["window_start"] = start
    out["window_end"] = end
    out["window_type"] = str(row.get("window_type", ""))
    out["boundary_type"] = str(row.get("boundary_type", ""))
    out["supervision_mode"] = str(row.get("supervision_mode", ""))
    out["tier"] = str(row.get("tier", ""))
    out["span_id"] = str(row.get("span_id", ""))
    out["sample_weight"] = torch.tensor(float(row.get("sample_weight", float(sample.get("sample_weight", 1.0)))), dtype=torch.float32)
    out["edge_src"], out["edge_dst"], out["edge_type"], out["edge_attr"] = _crop_edges(
        sample["edge_src"],
        sample["edge_dst"],
        sample["edge_type"],
        sample["edge_attr"],
        start=start,
        end=end,
    )
    out["regions"] = _crop_regions(sample.get("regions", []), start=start, end=end)
    return out


def _crop_edges(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_type: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = (edge_src >= start) & (edge_src < end) & (edge_dst >= start) & (edge_dst < end)
    if not bool(keep.any()):
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, int(edge_attr.shape[1]), dtype=torch.float32),
        )
    return (
        (edge_src[keep] - start).long(),
        (edge_dst[keep] - start).long(),
        edge_type[keep].long(),
        edge_attr[keep].float().clone(),
    )


def _crop_regions(regions: list[dict[str, Any]], *, start: int, end: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for region in regions:
        r_start = int(region.get("start", -1))
        r_end = int(region.get("end", -1))
        overlap_start = max(r_start, start)
        overlap_end = min(r_end, end - 1)
        if overlap_start <= overlap_end:
            copied = dict(region)
            copied["start"] = overlap_start - start
            copied["end"] = overlap_end - start
            out.append(copied)
    return out


# Source: data/packed_batches.py


import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


class PackedBatchDataset(Dataset[dict[str, Any]]):
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Packed batch directory does not exist: {self.directory}")
        self.paths = sorted(self.directory.glob("batch_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No packed batch files found in {self.directory}")
        self.sample_count = len(self.paths)
        manifest_path = self.directory.parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            split_summary = manifest.get("splits", {}).get(self.directory.name, {})
            self.sample_count = int(split_summary.get("samples", self.sample_count))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return torch.load(self.paths[index], map_location="cpu", weights_only=False)


class PhaseFlowPackedBatchDataset(Dataset[dict[str, Any]]):
    """Rank-local pre-collated training batches.

    Each item is a complete batch dict produced by PhaseFlowCollator and can be
    sent directly to the model.  The dataset intentionally uses LOCAL_RANK by
    default because the packed layout is single-node rank-local:
    ``epoch_seed_x/rank0/batch_000000.pt``.
    """

    def __init__(
        self,
        packed_dir: str | Path,
        batch_index: str | Path | None = None,
        epoch_dirs: list[str | Path] | None = None,
        epoch_index_files: list[str | Path] | None = None,
        rank: int | None = None,
        local_rank: int | None = None,
        epoch_seed: int | None = None,
    ) -> None:
        self.packed_dir = Path(packed_dir)
        self.batch_index_path = Path(batch_index) if batch_index else None
        self.epoch_dirs = [Path(item) for item in (epoch_dirs or [])]
        self.epoch_index_files = [Path(item) for item in (epoch_index_files or [])]
        self.local_rank = _resolve_rank(local_rank, "LOCAL_RANK")
        self.rank = _resolve_rank(rank, "RANK", default=self.local_rank)
        self.epoch_seed = epoch_seed
        self._epoch = 0
        self.rank_dir = self._resolve_rank_dir()
        self.paths, self.records = self._load_paths()
        if not self.paths:
            raise FileNotFoundError(f"No packed batch files found for rank {self.local_rank} under {self.rank_dir}")
        self.sample_count = int(sum(int(record.get("n_samples", 0) or 0) for record in self.records))
        if self.sample_count <= 0:
            self.sample_count = len(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        previous = list(self.paths)
        self.rank_dir = self._resolve_rank_dir()
        self.paths, self.records = self._load_paths()
        if not self.paths:
            raise FileNotFoundError(f"No packed batch files found for epoch {self._epoch}, rank {self.local_rank}")
        self.sample_count = int(sum(int(record.get("n_samples", 0) or 0) for record in self.records))
        if self.sample_count <= 0:
            self.sample_count = len(self.paths)
        if previous and previous == self.paths and self._has_multiple_epochs():
            raise RuntimeError(
                "Packed dataset is configured for multiple epochs, but set_epoch() did not change "
                f"rank {self.local_rank} paths for epoch {self._epoch}."
            )

    def epoch_stats(self, epoch: int | None = None) -> dict[str, Any]:
        real = sum(int(record.get("real_residues", 0) or 0) for record in self.records)
        padded = sum(int(record.get("padded_residues", 0) or 0) for record in self.records)
        return {
            "epoch": self._epoch if epoch is None else int(epoch),
            "rank": self.rank,
            "local_rank": self.local_rank,
            "sampler": "rank_local_packed_batch",
            "rank_batches": len(self.paths),
            "samples": self.sample_count,
            "real_residues": real,
            "padded_residues": padded,
            "padding_ratio": (padded - real) / max(padded, 1),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[int(index)]
        start = time.perf_counter()
        batch = torch.load(path, map_location="cpu", weights_only=False)
        load_sec = time.perf_counter() - start
        batch["__packed_load_sec"] = float(load_sec)
        batch["__packed_path"] = str(path)
        batch["__packed_rank"] = int(self.local_rank)
        batch["__packed_index"] = int(index)
        batch["__packed_epoch"] = int(self._epoch)
        return batch

    def _resolve_rank_dir(self) -> Path:
        if self.epoch_dirs:
            directory = self.epoch_dirs[self._epoch % len(self.epoch_dirs)]
            return directory if directory.name.startswith("rank") else directory / f"rank{self.local_rank}"
        if self.packed_dir.name.startswith("rank"):
            return self.packed_dir
        return self.packed_dir / f"rank{self.local_rank}"

    def _load_paths(self) -> tuple[list[Path], list[dict[str, Any]]]:
        index_path = self._current_index_path()
        if index_path is not None and index_path.exists():
            frame = pd.read_parquet(index_path)
            rank_column = "local_rank" if "local_rank" in frame.columns else "rank"
            frame = frame.loc[frame[rank_column].astype(int) == int(self.local_rank)].copy()
            if "epoch" in frame.columns:
                frame = frame.loc[frame["epoch"].astype(int) == int(self._epoch)].copy()
            if self.epoch_seed is not None and "epoch_seed" in frame.columns:
                frame = frame.loc[frame["epoch_seed"].astype(int) == int(self.epoch_seed)].copy()
            if "global_step" in frame.columns:
                frame = frame.sort_values("global_step")
            elif "rank_step" in frame.columns:
                frame = frame.sort_values("rank_step")
            paths: list[Path] = []
            records: list[dict[str, Any]] = []
            for _, row in frame.iterrows():
                raw = row.get("path", row.get("rel_path"))
                if raw is None:
                    continue
                path = Path(str(raw))
                if not path.is_absolute():
                    path = index_path.parent / path
                paths.append(path)
                records.append(row.to_dict())
            return paths, records
        paths = sorted(self.rank_dir.glob("batch_*.pt"))
        records = [{"path": str(path), "n_samples": 0} for path in paths]
        return paths, records

    def _current_index_path(self) -> Path | None:
        if self.epoch_index_files:
            return self.epoch_index_files[self._epoch % len(self.epoch_index_files)]
        return self.batch_index_path

    def _has_multiple_epochs(self) -> bool:
        if len(self.epoch_dirs) > 1 or len(self.epoch_index_files) > 1:
            return True
        if self.batch_index_path is None or not self.batch_index_path.exists():
            return False
        try:
            frame = pd.read_parquet(self.batch_index_path, columns=["epoch"])
        except Exception:
            return False
        return int(frame["epoch"].nunique()) > 1


def _resolve_rank(value: int | None, env_name: str, *, default: int = 0) -> int:
    if value is not None:
        return int(value)
    raw = os.environ.get(env_name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


# Source: data/sidecar.py


import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phaseflow.protein.data import (
    LABEL_KEYS,
    PHASEFLOW_LLPS_HIDDEN_KEY,
    RUNTIME_ARRAYS,
    _load_runtime_array,
    dpr_v2_hotpath_collate,
)
from phaseflow.protein.data import NO_STARLING_EDGE_TYPE, PROTENIX_EDGE_TYPE
try:
    from phaseflow.protein.contracts import assert_no_eval_only_training_path
except ImportError:
    def assert_no_eval_only_training_path(path: str | Path) -> None:
        text = str(path).replace("\\", "/").lower()
        if "data/processed/evaluation_only/" in text:
            raise RuntimeError(f"Eval-only sidecar path is forbidden for DPR v5 training: {path}")


V5_TIER_SLOTS: tuple[str, ...] = ("S", "W", "M", "M", "ND", "ND", "NP", "NP")


@dataclass(frozen=True)
class PackedProteinRow:
    protein_id: str
    sequence_sha256: str
    sequence: str
    length: int
    shard_id: int
    residue_offset: int


def sidecar_read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path, low_memory=False)
            raise
    return pd.read_csv(path, low_memory=False)


class DPRV5BaseOnlySidecar:
    """Read immutable v2 packed tensors and enforce v5 base-only graph policy."""

    def __init__(
        self,
        *,
        v2_data_root: str | Path,
        packed_root: str | Path,
        mmap: bool = True,
        max_open_shards: int = 8,
    ) -> None:
        self.v2_data_root = Path(v2_data_root).resolve()
        self.packed_root = Path(packed_root).resolve()
        assert_no_eval_only_training_path(self.packed_root)
        self.manifest = sidecar_read_table(self.packed_root / "manifest.parquet")
        self.shards = sidecar_read_table(self.packed_root / "shards.parquet")
        self.mmap = bool(mmap)
        self.max_open_shards = max(1, int(max_open_shards))
        self._rows: dict[tuple[str, str], PackedProteinRow] = {}
        for row in self.manifest.itertuples(index=False):
            key = (str(row.protein_id), str(row.sequence_sha256))
            self._rows[key] = PackedProteinRow(
                protein_id=str(row.protein_id),
                sequence_sha256=str(row.sequence_sha256),
                sequence=str(row.sequence),
                length=int(row.length),
                shard_id=int(row.shard_id),
                residue_offset=int(row.residue_offset),
            )
        self._shard_paths = {int(row.shard_id): self._resolve_path(str(row.path)) for row in self.shards.itertuples(index=False)}
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def sample_from_tier_row(self, row: Any) -> dict[str, Any]:
        protein_id = str(row.protein_id)
        sequence_sha256 = str(row.sequence_sha256)
        packed = self._rows.get((protein_id, sequence_sha256))
        if packed is None:
            raise KeyError(f"{protein_id}/{sequence_sha256} not present in parent v2 packed hotpath")
        length = int(packed.length)
        shard = self._open_shard(int(packed.shard_id))
        sl = slice(int(packed.residue_offset), int(packed.residue_offset) + length)
        tier = str(getattr(row, "v3_tier", getattr(row, "tier", "")))
        sample_weight = float(TIER_TO_WEIGHT.get(tier, float(getattr(row, "sample_weight", 1.0))))
        sample: dict[str, Any] = {
            "sample_id": f"{protein_id}:{tier}:base_only",
            "protein_id": protein_id,
            "sequence_sha256": sequence_sha256,
            "sequence": packed.sequence,
            "kind": "protein",
            "pool": str(getattr(row, "v3_pool", getattr(row, "pool", ""))),
            "tier": tier,
            "v3_tier": tier,
            "mil_confidence": str(getattr(row, "mil_confidence", "")),
            "view_name": "base_only",
            "length": length,
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "bag_label": torch.tensor(1.0 if tier in {"S", "W", "M"} else 0.0, dtype=torch.float32),
            "bag_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "driver_label": torch.tensor(0.0, dtype=torch.float32),
            "driver_weight": torch.tensor(0.0, dtype=torch.float32),
            "partner_label": torch.tensor(0.0, dtype=torch.float32),
            "partner_weight": torch.tensor(0.0, dtype=torch.float32),
            "general_label": torch.tensor(1.0 if tier in {"S", "W", "M"} else 0.0, dtype=torch.float32),
            "general_weight": torch.tensor(sample_weight, dtype=torch.float32),
        }
        for key in ("plm", "biophys", "modality_mask", "reliability", "edge_attr", PHASEFLOW_LLPS_HIDDEN_KEY):
            sample[key] = torch.from_numpy(np.asarray(shard[key][sl], dtype=np.float16).copy())
        sample["aa_ids"] = torch.from_numpy(np.asarray(shard["aa_ids"][sl], dtype=np.int16).copy())
        sample["neighbors"] = torch.from_numpy(np.asarray(shard["neighbors"][sl], dtype=np.int64).copy())
        sample["neighbor_mask"] = torch.from_numpy(np.asarray(shard["neighbor_mask"][sl], dtype=np.bool_).copy())
        sample["neighbor_edge_type"] = torch.from_numpy(np.asarray(shard["neighbor_edge_type"][sl], dtype=np.int64).copy())
        apply_v5_base_only_edge_policy(sample)
        for key in LABEL_KEYS:
            sample[key] = torch.from_numpy(np.asarray(shard[key][sl], dtype=np.float32).copy())
        return sample

    def _open_shard(self, shard_id: int) -> dict[str, Any]:
        if shard_id in self._cache:
            arrays = self._cache.pop(shard_id)
            self._cache[shard_id] = arrays
            return arrays
        shard_path = self._shard_paths[int(shard_id)]
        mmap_mode = "r" if self.mmap else None
        arrays: dict[str, Any] = {"path": shard_path}
        for name in RUNTIME_ARRAYS:
            arrays[name] = _load_runtime_array(shard_path, name, root=self.packed_root, mmap_mode=mmap_mode)
        self._cache[shard_id] = arrays
        while len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return arrays

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.packed_root / path
        path = path.resolve()
        path.relative_to(self.packed_root)
        return path


class DPRV5ExactUpdateDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        sidecar: DPRV5BaseOnlySidecar,
        tier_manifest: pd.DataFrame,
        rank: int,
        world_size: int,
        start_update: int,
        end_update: int,
        seed: int,
    ) -> None:
        if int(world_size) != len(V5_TIER_SLOTS):
            raise ValueError(f"DPR v5 exact sampler requires world_size=8, got {world_size}")
        self.sidecar = sidecar
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_update = int(start_update)
        self.end_update = int(end_update)
        self.seed = int(seed)
        self.tier_manifest = tier_manifest.reset_index(drop=True).copy()
        self.by_tier: dict[str, pd.DataFrame] = {}
        for tier in ("S", "W", "M", "ND", "NP"):
            sub = self.tier_manifest.loc[self.tier_manifest["v3_tier"].astype(str).eq(tier)].copy()
            if sub.empty:
                raise RuntimeError(f"DPR v5 tier {tier} is empty")
            self.by_tier[tier] = sidecar_stable_shuffle(sub, self.seed + sidecar_tier_offset(tier)).reset_index(drop=True)
        self.plan = [
            make_rank_plan_item(update, self.rank, self.world_size, self.by_tier)
            for update in range(self.start_update, self.end_update + 1)
        ]

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.plan[int(index)]
        row = item["row"]
        sample = self.sidecar.sample_from_tier_row(row)
        batch = dpr_v5_collate([sample])
        batch["update"] = int(item["update"])
        batch["rank"] = self.rank
        batch["global_slot"] = int(item["slot"])
        batch["tier_global_index"] = int(item["tier_global_index"])
        return batch

    def state_dict(self, consumed_updates: int) -> dict[str, Any]:
        next_update = self.start_update + int(consumed_updates)
        return {
            "format": "dpr_v5_exact_update_sampler_state",
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "start_update": self.start_update,
            "end_update": self.end_update,
            "next_update": next_update,
        }


def dpr_v5_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out = dpr_v2_hotpath_collate(samples)
    out["v3_tiers"] = [str(sample.get("v3_tier", sample.get("tier", ""))) for sample in samples]
    out["mil_confidences"] = [str(sample.get("mil_confidence", "")) for sample in samples]
    out["view_names"] = [str(sample.get("view_name", "")) for sample in samples]
    out["starling_edges_masked"] = torch.tensor([int(sample.get("starling_edges_masked", 0)) for sample in samples], dtype=torch.long)
    out["protenix_edges_masked"] = torch.tensor([int(sample.get("protenix_edges_masked", 0)) for sample in samples], dtype=torch.long)
    out["edges_after_policy"] = torch.tensor([int(sample.get("edges_after_policy", 0)) for sample in samples], dtype=torch.long)
    return out


def make_rank_plan_item(update: int, rank: int, world_size: int, by_tier: dict[str, pd.DataFrame]) -> dict[str, Any]:
    slot = (int(rank) + int(update) - 1) % int(world_size)
    tier = V5_TIER_SLOTS[slot]
    tier_index = tier_global_index(update=int(update), rank=int(rank), world_size=int(world_size), tier=tier)
    frame = by_tier[tier]
    row = frame.iloc[int(tier_index) % len(frame)]
    return {
        "update": int(update),
        "rank": int(rank),
        "slot": int(slot),
        "tier": tier,
        "tier_global_index": int(tier_index),
        "row": row,
    }


def tier_global_index(*, update: int, rank: int, world_size: int, tier: str) -> int:
    count = 0
    for prev_update in range(1, int(update)):
        for prev_rank in range(int(world_size)):
            prev_slot = (prev_rank + prev_update - 1) % int(world_size)
            if V5_TIER_SLOTS[prev_slot] == tier:
                count += 1
    for prev_rank in range(0, int(rank)):
        prev_slot = (prev_rank + int(update) - 1) % int(world_size)
        if V5_TIER_SLOTS[prev_slot] == tier:
            count += 1
    return count


def simulate_global_tiers(updates: int, *, world_size: int = 8) -> list[list[str]]:
    rows: list[list[str]] = []
    for update in range(1, int(updates) + 1):
        rows.append([V5_TIER_SLOTS[(rank + update - 1) % int(world_size)] for rank in range(int(world_size))])
    return rows


def apply_v5_base_only_edge_policy(sample: dict[str, Any]) -> None:
    edge_type = sample["neighbor_edge_type"].long()
    mask = sample["neighbor_mask"].bool()
    starling = mask & edge_type.eq(NO_STARLING_EDGE_TYPE)
    protenix = mask & edge_type.eq(PROTENIX_EDGE_TYPE)
    keep = mask & ~starling & ~protenix
    sample["starling_edges_masked"] = int(starling.sum().item())
    sample["protenix_edges_masked"] = int(protenix.sum().item())
    sample["edges_after_policy"] = int(keep.sum().item())
    masked = ~keep
    sample["neighbor_mask"] = keep
    sample["neighbors"] = sample["neighbors"].masked_fill(masked, 0)
    sample["edge_attr"] = sample["edge_attr"].masked_fill(masked.unsqueeze(-1), 0)
    sample["neighbor_edge_type"] = sample["neighbor_edge_type"].masked_fill(masked, 0)


def edge_policy_summary(batch: dict[str, Any]) -> dict[str, int]:
    mask = batch["neighbor_mask"].bool()
    edge_type = batch["neighbor_edge_type"].long()
    return {
        "starling_edges_passed_to_model": int((mask & edge_type.eq(NO_STARLING_EDGE_TYPE)).sum().item()),
        "protenix_edges_passed_to_model": int((mask & edge_type.eq(PROTENIX_EDGE_TYPE)).sum().item()),
        "total_edges_passed_to_model": int(mask.sum().item()),
    }


def sidecar_stable_shuffle(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed) % (2**32 - 1))
    order = np.arange(len(frame))
    rng.shuffle(order)
    return frame.iloc[order].reset_index(drop=True)


def sidecar_tier_offset(name: str) -> int:
    return sum(ord(ch) for ch in str(name)) * 1009


def tier_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame["v3_tier"].astype(str).value_counts().sort_index().items()}


def write_sampler_audit(path: Path, *, updates: int, world_size: int, tier_manifest: pd.DataFrame) -> None:
    rows = simulate_global_tiers(int(updates), world_size=int(world_size))
    violations = []
    expected = sorted(V5_TIER_SLOTS)
    exposures: Counter[str] = Counter()
    rank_exposures: dict[int, Counter[str]] = {rank: Counter() for rank in range(int(world_size))}
    for update, tiers in enumerate(rows, start=1):
        if sorted(tiers) != expected:
            violations.append({"update": update, "tiers": tiers})
        for rank, tier in enumerate(tiers):
            exposures[tier] += 1
            rank_exposures[rank][tier] += 1
    payload = {
        "format": "dpr_v5_sampler_audit",
        "updates": int(updates),
        "world_size": int(world_size),
        "slot_pattern": list(V5_TIER_SLOTS),
        "violations": violations[:10],
        "violation_count": len(violations),
        "global_exposure": dict(exposures),
        "rank_exposure": {str(rank): dict(counter) for rank, counter in rank_exposures.items()},
        "tier_manifest_counts": tier_counts(tier_manifest),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


TIER_TO_WEIGHT = {"S": 1.00, "W": 0.75, "M": 1.00, "ND": 0.50, "NP": 1.00}


# Source: data/schedule.py


import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phaseflow.protein.data import DPRV5BaseOnlySidecar, dpr_v5_collate


TIERS: tuple[str, ...] = ("S", "W", "M", "ND", "NP")
EVEN_SLOTS: tuple[str, ...] = ("S", "W", "M", "M", "ND", "NP", "NP", "NP")
ODD_SLOTS: tuple[str, ...] = ("M", "M", "M", "M", "ND", "NP", "NP", "NP")
TIER_TO_BAG_LABEL = {"S": 1.0, "W": 1.0, "M": 1.0, "ND": 0.0, "NP": 0.0}
TIER_TO_WEIGHT = {"S": 1.00, "W": 0.75, "M": 1.00, "ND": 0.50, "NP": 1.00}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path, low_memory=False)
            raise
    return pd.read_csv(path, low_memory=False)


def stable_shuffle(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed) % (2**32 - 1))
    order = np.arange(len(frame))
    rng.shuffle(order)
    return frame.iloc[order].reset_index(drop=True)


def tier_offset(tier: str) -> int:
    return sum(ord(ch) for ch in str(tier)) * 1009


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_official_benchmark_ids(package_root: str | Path) -> tuple[set[str], set[str]]:
    proteins = pd.read_parquet(Path(package_root) / "proteins.parquet")
    ids = set(proteins["protein_id"].astype(str))
    ids.update(proteins["accession"].astype(str))
    hashes = set(proteins["sequence_sha256"].astype(str))
    return ids, hashes


def load_v6_tier_manifest(
    path: str | Path,
    *,
    official_package_root: str | Path,
    require_no_benchmark_overlap: bool = True,
) -> pd.DataFrame:
    frame = read_table(path).copy()
    if "v3_tier" not in frame.columns:
        if "tier" in frame.columns:
            frame["v3_tier"] = frame["tier"].astype(str)
        elif "training_tier" in frame.columns:
            mapping = {
                "strong_core": "S",
                "weak_region": "W",
                "mil_positive": "M",
                "structured_negative": "ND",
                "proteome_background": "NP",
            }
            frame["v3_tier"] = frame["training_tier"].map(mapping).fillna(frame["training_tier"]).astype(str)
        else:
            raise KeyError("tier manifest must contain v3_tier, tier, or training_tier")
    if "benchmark_excluded" in frame.columns:
        frame = frame.loc[~frame["benchmark_excluded"].fillna(False).astype(bool)].copy()
    official_ids, official_hashes = load_official_benchmark_ids(official_package_root)
    overlap_id = frame["protein_id"].astype(str).isin(official_ids)
    overlap_hash = frame["sequence_sha256"].astype(str).isin(official_hashes)
    overlap_count = int((overlap_id | overlap_hash).sum())
    if overlap_count and require_no_benchmark_overlap:
        frame = frame.loc[~(overlap_id | overlap_hash)].copy()
    frame = frame.loc[frame["v3_tier"].astype(str).isin(TIERS)].copy()
    frame = frame.reset_index(drop=True)
    if require_no_benchmark_overlap:
        rem_id = int(frame["protein_id"].astype(str).isin(official_ids).sum())
        rem_hash = int(frame["sequence_sha256"].astype(str).isin(official_hashes).sum())
        if rem_id or rem_hash:
            raise RuntimeError(f"DPR v6 benchmark overlap after filtering: accession={rem_id} hash={rem_hash}")
    return frame


def slots_for_update(update: int) -> tuple[str, ...]:
    return EVEN_SLOTS if int(update) % 2 == 0 else ODD_SLOTS


def schedule_tier_for_rank(update: int, rank: int, world_size: int = 8) -> tuple[int, str]:
    if int(world_size) != 8:
        raise ValueError(f"DPR v6 fixed schedule requires world_size=8, got {world_size}")
    slot = (int(rank) + int(update) - 1) % int(world_size)
    return slot, slots_for_update(update)[slot]


def build_fixed_schedule(
    tier_manifest: pd.DataFrame,
    *,
    updates: int = 2000,
    world_size: int = 8,
    seed: int = 20260616,
) -> pd.DataFrame:
    if int(world_size) != 8:
        raise ValueError("DPR v6 schedules are defined for 8 ranks")
    by_tier: dict[str, pd.DataFrame] = {}
    for tier in TIERS:
        sub = tier_manifest.loc[tier_manifest["v3_tier"].astype(str).eq(tier)].copy()
        if sub.empty:
            raise RuntimeError(f"DPR v6 tier {tier} is empty")
        by_tier[tier] = stable_shuffle(sub, int(seed) + tier_offset(tier))
    tier_counters: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for update in range(1, int(updates) + 1):
        for rank in range(int(world_size)):
            slot, tier = schedule_tier_for_rank(update, rank, world_size)
            frame = by_tier[tier]
            idx = tier_counters[tier]
            rec = frame.iloc[int(idx) % len(frame)]
            tier_counters[tier] += 1
            rows.append(
                {
                    "update": int(update),
                    "rank": int(rank),
                    "slot": int(slot),
                    "slot_tier": tier,
                    "tier_global_index": int(idx),
                    "tier_cycle_index": int(idx) % len(frame),
                    "protein_id": str(rec.protein_id),
                    "sequence_sha256": str(rec.sequence_sha256),
                    "length": int(rec.length),
                    "v3_tier": tier,
                    "v3_pool": str(getattr(rec, "v3_pool", getattr(rec, "source_pool", ""))),
                    "sample_weight": float(TIER_TO_WEIGHT[tier]),
                    "bag_label": float(TIER_TO_BAG_LABEL[tier]),
                    "source": str(getattr(rec, "source", "")),
                    "cluster_id": str(getattr(rec, "cluster_id", "")),
                    "label_path": str(getattr(rec, "label_path", "")),
                }
            )
    return pd.DataFrame(rows)


def validate_schedule(schedule: pd.DataFrame, *, updates: int = 2000, world_size: int = 8) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    expected_counts_even = Counter(EVEN_SLOTS)
    expected_counts_odd = Counter(ODD_SLOTS)
    for update, sub in schedule.groupby("update"):
        tiers = list(sub.sort_values("rank")["v3_tier"].astype(str))
        expected = expected_counts_even if int(update) % 2 == 0 else expected_counts_odd
        if Counter(tiers) != expected:
            violations.append({"update": int(update), "tiers": tiers})
    exposure = Counter(schedule["v3_tier"].astype(str))
    unique = schedule.groupby("v3_tier")["protein_id"].nunique().to_dict()
    rank_exposure = {
        str(rank): dict(Counter(schedule.loc[schedule["rank"].eq(rank), "v3_tier"].astype(str)))
        for rank in range(int(world_size))
    }
    return {
        "format": "dpr_v6_schedule_audit",
        "updates": int(updates),
        "world_size": int(world_size),
        "even_slots": list(EVEN_SLOTS),
        "odd_slots": list(ODD_SLOTS),
        "violation_count": len(violations),
        "violations": violations[:20],
        "global_exposure": dict(exposure),
        "unique_coverage": {str(k): int(v) for k, v in unique.items()},
        "rank_exposure": rank_exposure,
    }


def write_schedule_artifacts(
    schedule: pd.DataFrame,
    *,
    schedule_path: str | Path,
    audit_path: str | Path,
) -> dict[str, Any]:
    schedule_path = Path(schedule_path)
    audit_path = Path(audit_path)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(schedule_path, index=False)
    audit = validate_schedule(schedule, updates=int(schedule["update"].max()), world_size=int(schedule["rank"].max()) + 1)
    audit["schedule_path"] = str(schedule_path)
    audit["schedule_sha256"] = sha256_file(schedule_path)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


@dataclass(frozen=True)
class ExtraFeatureSpec:
    name: str
    root: Path
    column: str
    dim: int


class ExtraFeatureStore:
    def __init__(self, specs: list[ExtraFeatureSpec] | None = None) -> None:
        self.specs = specs or []

    @classmethod
    def from_config(cls, entries: list[dict[str, Any]] | None) -> "ExtraFeatureStore":
        specs: list[ExtraFeatureSpec] = []
        for entry in entries or []:
            specs.append(
                ExtraFeatureSpec(
                    name=str(entry["name"]),
                    root=Path(entry["root"]),
                    column=str(entry.get("column", entry["name"])),
                    dim=int(entry["dim"]),
                )
            )
        return cls(specs)

    def add_to_sample(self, sample: dict[str, Any]) -> None:
        if not self.specs:
            return
        protein_id = str(sample["protein_id"])
        expected_len = int(sample["length"])
        for spec in self.specs:
            path = spec.root / f"{protein_id}.npy"
            if not path.exists():
                raise FileNotFoundError(f"missing DPR v6 extra feature {spec.name}: {path}")
            arr = np.load(path, allow_pickle=False)
            if arr.shape[0] != expected_len or arr.shape[1] != spec.dim:
                raise RuntimeError(f"{spec.name} shape mismatch for {protein_id}: got {arr.shape}, expected ({expected_len}, {spec.dim})")
            sample[spec.column] = torch.from_numpy(np.asarray(arr, dtype=np.float32))


class DPRV6ScheduleDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        sidecar: DPRV5BaseOnlySidecar,
        schedule: pd.DataFrame,
        rank: int,
        start_update: int,
        end_update: int,
        extra_features: ExtraFeatureStore | None = None,
    ) -> None:
        self.sidecar = sidecar
        self.rank = int(rank)
        self.start_update = int(start_update)
        self.end_update = int(end_update)
        self.extra_features = extra_features or ExtraFeatureStore()
        self.schedule = schedule.loc[
            schedule["rank"].astype(int).eq(self.rank)
            & schedule["update"].astype(int).between(self.start_update, self.end_update)
        ].sort_values("update").reset_index(drop=True)
        if len(self.schedule) != max(0, self.end_update - self.start_update + 1):
            raise RuntimeError(f"rank {rank} schedule length mismatch: {len(self.schedule)}")

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.schedule.iloc[int(index)]
        sample = self.sidecar.sample_from_tier_row(row)
        self.extra_features.add_to_sample(sample)
        batch = dpr_v5_collate([sample])
        for key in ("pstp_650d", "pstp_esm8", "pstp_alb"):
            if key in sample:
                batch[key] = sample[key].unsqueeze(0)
        batch["update"] = int(row["update"])
        batch["rank"] = int(row["rank"])
        batch["global_slot"] = int(row["slot"])
        batch["tier_global_index"] = int(row["tier_global_index"])
        return batch
