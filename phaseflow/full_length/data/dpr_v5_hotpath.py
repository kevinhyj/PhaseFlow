from __future__ import annotations

import json
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
from phaseflow.full_length.data.dpr_v3_hotpath import NO_STARLING_EDGE_TYPE, PROTENIX_EDGE_TYPE
try:
    from phaseflow.full_length.data.runtime_guard import assert_no_eval_only_training_path
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
        self.manifest = read_table(self.packed_root / "manifest.parquet")
        self.shards = read_table(self.packed_root / "shards.parquet")
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
            self.by_tier[tier] = stable_shuffle(sub, self.seed + tier_offset(tier)).reset_index(drop=True)
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


def stable_shuffle(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed) % (2**32 - 1))
    order = np.arange(len(frame))
    rng.shuffle(order)
    return frame.iloc[order].reset_index(drop=True)


def tier_offset(name: str) -> int:
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
