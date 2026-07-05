from __future__ import annotations

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

from phaseflow.full_length.data.dataset import (
    _empty_hard_region_labels,
    _empty_region_target,
    _fit_region_target,
    _read_phase_targets,
    _read_region_targets,
    _regions_from_region_target,
)
from phaseflow.full_length.data.runtime_guard import assert_offline_path_allowed
from phaseflow.full_length.data.schemas import IGNORE_INDEX
from phaseflow.full_length.data.sharded_store import Esm2ShardedStore, NpzMirrorStore
from phaseflow.full_length.features.bio_vec import make_bio_vec


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
