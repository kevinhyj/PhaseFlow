from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from phaseflow.full_length.data.collator import PhaseFlowCollator
from phaseflow.full_length.data.offline_dataset import PhaseFlowOfflineDataset
from phaseflow.full_length.data.schemas import IGNORE_INDEX


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
        self.local_rank = _resolve_rank(local_rank, "LOCAL_RANK")
        self.rank = _resolve_rank(rank, "RANK", default=self.local_rank)
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


def _resolve_rank(value: int | None, env_name: str, default: int = 0) -> int:
    if value is not None:
        return int(value)
    try:
        return int(__import__("os").environ.get(env_name, default))
    except (TypeError, ValueError):
        return int(default)
