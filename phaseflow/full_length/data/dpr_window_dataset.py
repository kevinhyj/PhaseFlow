from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

from phaseflow.full_length.data.offline_dataset import PhaseFlowOfflineDataset


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
