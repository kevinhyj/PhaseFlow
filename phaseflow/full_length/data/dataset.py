from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phaseflow.full_length.data.feature_cache import FeatureCacheReader
from phaseflow.full_length.data.schemas import IGNORE_INDEX
from phaseflow.full_length.features.bio_vec import make_bio_vec


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
