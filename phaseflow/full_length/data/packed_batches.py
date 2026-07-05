from __future__ import annotations

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
