from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phaseflow.full_length.data.runtime_guard import assert_no_eval_only_training_path


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
class PackedProteinRow:
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
