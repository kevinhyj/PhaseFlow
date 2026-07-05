from __future__ import annotations

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

from phaseflow.full_length.data.dpr_v5_hotpath import DPRV5BaseOnlySidecar, dpr_v5_collate


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
