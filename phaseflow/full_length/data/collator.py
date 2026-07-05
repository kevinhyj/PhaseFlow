from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import torch

from phaseflow.full_length.data.schemas import IGNORE_INDEX


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
