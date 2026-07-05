from __future__ import annotations

import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phaseflow.full_length.data.dpr_v2_hotpath import (
    LABEL_KEYS,
    PHASEFLOW_LLPS_HIDDEN_KEY,
    RUNTIME_ARRAYS,
    _load_runtime_array,
    dpr_v2_hotpath_collate,
)
try:
    from phaseflow.full_length.data.runtime_guard import assert_no_eval_only_training_path
except ImportError:  # 53 keeps the original training guard; keep v3 self-contained.
    def assert_no_eval_only_training_path(path: str | Path) -> None:
        normalized = str(path).replace("\\", "/").lower()
        if "data/processed/evaluation_only/" in normalized:
            raise RuntimeError(f"Eval-only sidecar path is forbidden for DPR v3 training: {path}")


NO_STARLING_EDGE_TYPE = 30
PROTENIX_EDGE_TYPE = 20


@dataclass(frozen=True)
class PackedProteinRow:
    protein_id: str
    sequence_sha256: str
    sequence: str
    length: int
    shard_id: int
    residue_offset: int


def read_table(path: Path) -> pd.DataFrame:
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
        self.manifest = read_table(manifest_path)
        self.shards = read_table(shards_path)
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


def edge_policy_summary(batch: dict[str, Any]) -> dict[str, int]:
    mask = batch["neighbor_mask"].bool()
    et = batch["neighbor_edge_type"].long()
    return {
        "starling_edges_passed_to_model": int((mask & et.eq(NO_STARLING_EDGE_TYPE)).sum().item()),
        "protenix_edges_passed_to_model": int((mask & et.eq(PROTENIX_EDGE_TYPE)).sum().item()),
        "total_edges_passed_to_model": int(mask.sum().item()),
    }
