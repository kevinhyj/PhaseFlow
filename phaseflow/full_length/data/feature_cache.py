from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from phaseflow.full_length.data.runtime_guard import assert_offline_path_allowed, strict_offline_enabled
from phaseflow.full_length.data.schemas import FeatureCacheRecord
from phaseflow.full_length.features.graph_cache import GRAPH_CACHE_VERSION


PROTENIX_EMBED_DIM = 512
STARLING_EMBED_DIM = 512
NODE_FEATURES = (
    "plm",
    "physchem",
    "disorder",
    "protenix_embed",
    "starling_embed",
    "modality_mask",
    "reliability",
)
EDGE_FEATURES = ("edge_src", "edge_dst", "edge_type", "edge_attr")
GRAPH_GROUP = "graph"
LABELS = ("y_dpr", "y_key", "y_weight", "candidate_prior", "candidate_prior_weight")
SOFT_LABELS = ("teacher_dpr", "teacher_dpr_weight", "self_dpr", "self_dpr_weight")


class FeatureCacheWriter:
    @staticmethod
    def write_h5(path: str | Path, record: FeatureCacheRecord) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record.ensure_labels()
        FeatureCacheReader.validate_shapes(record)

        with h5py.File(path, "w") as handle:
            handle.attrs["protein_id"] = record.protein_id
            handle.attrs["sequence"] = record.sequence
            handle.attrs["length"] = record.length
            handle.attrs["sample_weight"] = float(record.sample_weight)
            handle.attrs["teacher_llps"] = float(record.teacher_llps)
            handle.attrs["teacher_llps_weight"] = float(record.teacher_llps_weight)
            handle.attrs["self_llps"] = float(record.self_llps)
            handle.attrs["self_llps_weight"] = float(record.self_llps_weight)
            handle.attrs["region_bag_label"] = float(record.region_bag_label)
            handle.attrs["region_bag_weight"] = float(record.region_bag_weight)
            handle.attrs["region_bag_type"] = str(record.region_bag_type)
            handle.attrs["negative_regularization_weight"] = float(record.negative_regularization_weight)
            handle.attrs["label_quality"] = str(record.label_quality)
            handle.attrs["negative_type"] = str(record.negative_type)
            handle.attrs["source"] = str(record.source)
            handle.attrs["regions_json"] = json.dumps(record.regions)
            handle.attrs["structure_metadata_json"] = json.dumps(record.structure_metadata)

            for name in NODE_FEATURES:
                handle.create_dataset(name, data=getattr(record, name), compression="gzip")
            for name in EDGE_FEATURES:
                handle.create_dataset(name, data=getattr(record, name), compression="gzip")
            if record.graph_neighbors is not None:
                graph = handle.create_group(GRAPH_GROUP)
                graph.attrs["version"] = GRAPH_CACHE_VERSION
                graph.attrs["max_neighbors"] = int(record.graph_neighbors.shape[1])
                graph.attrs["edge_dim"] = int(record.graph_edge_attr.shape[2])
                graph.attrs["source_edge_count"] = int(record.edge_src.shape[0])
                graph.create_dataset("neighbors", data=record.graph_neighbors, compression="gzip")
                graph.create_dataset("edge_attr", data=record.graph_edge_attr, compression="gzip")
                graph.create_dataset("neighbor_mask", data=record.graph_neighbor_mask, compression="gzip")

            embeddings = handle.create_group("embeddings")
            for key, value in record.structure_metadata.items():
                embeddings.attrs[key] = str(value)

            handle.create_dataset("y_llps", data=np.asarray(record.y_llps, dtype=np.float32))
            for name in LABELS:
                handle.create_dataset(name, data=getattr(record, name), compression="gzip")
            soft = handle.create_group("soft_labels")
            for name in SOFT_LABELS:
                soft.create_dataset(name, data=getattr(record, name), compression="gzip")


class FeatureCacheReader:
    @staticmethod
    def read_h5(path: str | Path, *, read_raw_edges: bool = True) -> FeatureCacheRecord:
        path = Path(path)
        if strict_offline_enabled():
            assert_offline_path_allowed(path)
        with h5py.File(path, "r") as handle:
            protein_id = _decode_attr(handle.attrs["protein_id"])
            sequence = _decode_attr(handle.attrs["sequence"])
            regions_json = _decode_attr(handle.attrs.get("regions_json", "[]"))
            structure_metadata_json = _decode_attr(handle.attrs.get("structure_metadata_json", "{}"))
            structure_metadata = json.loads(structure_metadata_json)
            if "structure" in handle:
                structure_metadata.update({key: _decode_attr(value) for key, value in handle["structure"].attrs.items()})
            if "embeddings" in handle:
                structure_metadata.update({key: _decode_attr(value) for key, value in handle["embeddings"].attrs.items()})
            protenix_embed = _read_protenix_embedding(handle, len(sequence), structure_metadata)
            starling_embed = _read_starling_embedding(handle, len(sequence))
            modality_mask = _read_modality_matrix(handle, "modality_mask", len(sequence), structure_metadata, is_reliability=False)
            reliability = _read_modality_matrix(handle, "reliability", len(sequence), structure_metadata, is_reliability=True)
            if _starling_read_disabled():
                modality_mask = modality_mask.copy()
                reliability = reliability.copy()
                if modality_mask.shape[1] >= 5:
                    modality_mask[:, 4] = 1.0
                if reliability.shape[1] >= 5:
                    reliability[:, 4] = 0.0
                structure_metadata["starling_read_disabled"] = "true"
            graph_neighbors, graph_edge_attr, graph_neighbor_mask = _read_graph_cache(handle)
            edge_dim = _edge_attr_dim(handle, graph_edge_attr)
            if read_raw_edges:
                edge_src = np.asarray(handle["edge_src"], dtype=np.int64)
                edge_dst = np.asarray(handle["edge_dst"], dtype=np.int64)
                edge_type = np.asarray(handle["edge_type"], dtype=np.int64)
                edge_attr = np.asarray(handle["edge_attr"], dtype=np.float32)
            else:
                edge_src = np.zeros((0,), dtype=np.int64)
                edge_dst = np.zeros((0,), dtype=np.int64)
                edge_type = np.zeros((0,), dtype=np.int64)
                edge_attr = np.zeros((0, edge_dim), dtype=np.float32)
            record = FeatureCacheRecord(
                protein_id=protein_id,
                sequence=sequence,
                plm=np.asarray(handle["plm"], dtype=np.float32),
                physchem=np.asarray(handle["physchem"], dtype=np.float32),
                disorder=np.asarray(handle["disorder"], dtype=np.float32),
                protenix_embed=protenix_embed,
                starling_embed=starling_embed,
                modality_mask=modality_mask,
                reliability=reliability,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_type=edge_type,
                edge_attr=edge_attr,
                graph_neighbors=graph_neighbors,
                graph_edge_attr=graph_edge_attr,
                graph_neighbor_mask=graph_neighbor_mask,
                y_llps=float(np.asarray(handle["y_llps"], dtype=np.float32).item()),
                y_dpr=np.asarray(handle["y_dpr"], dtype=np.int64),
                y_key=np.asarray(handle["y_key"], dtype=np.int64),
                y_weight=np.asarray(handle["y_weight"], dtype=np.float32),
                teacher_llps=float(handle.attrs.get("teacher_llps", np.nan)),
                teacher_llps_weight=float(handle.attrs.get("teacher_llps_weight", 0.0)),
                self_llps=float(handle.attrs.get("self_llps", np.nan)),
                self_llps_weight=float(handle.attrs.get("self_llps_weight", 0.0)),
                region_bag_label=float(handle.attrs.get("region_bag_label", -100.0)),
                region_bag_weight=float(handle.attrs.get("region_bag_weight", 0.0)),
                region_bag_type=_decode_attr(handle.attrs.get("region_bag_type", "mask")),
                negative_regularization_weight=float(handle.attrs.get("negative_regularization_weight", 0.0)),
                teacher_dpr=_read_optional_vector(handle, "soft_labels/teacher_dpr", len(sequence), np.nan),
                teacher_dpr_weight=_read_optional_vector(handle, "soft_labels/teacher_dpr_weight", len(sequence), 0.0),
                self_dpr=_read_optional_vector(handle, "soft_labels/self_dpr", len(sequence), np.nan),
                self_dpr_weight=_read_optional_vector(handle, "soft_labels/self_dpr_weight", len(sequence), 0.0),
                candidate_prior=_read_optional_vector(handle, "candidate_prior", len(sequence), 0.0),
                candidate_prior_weight=_read_optional_vector(handle, "candidate_prior_weight", len(sequence), 0.0),
                sample_weight=float(handle.attrs.get("sample_weight", 1.0)),
                label_quality=_decode_attr(handle.attrs.get("label_quality", "")),
                negative_type=_decode_attr(handle.attrs.get("negative_type", "")),
                source=_decode_attr(handle.attrs.get("source", "")),
                regions=json.loads(regions_json),
                structure_metadata=structure_metadata,
            )
        FeatureCacheReader.validate_shapes(record)
        return record

    @staticmethod
    def validate_shapes(record: FeatureCacheRecord) -> None:
        record.ensure_labels()
        length = record.length
        for name in NODE_FEATURES:
            value = getattr(record, name)
            if value.ndim != 2 or value.shape[0] != length:
                raise ValueError(f"{name} must have shape [L, D], got {value.shape} for L={length}")
        for name in ("edge_src", "edge_dst", "edge_type"):
            value = getattr(record, name)
            if value.ndim != 1:
                raise ValueError(f"{name} must be 1D, got {value.shape}")
        edge_count = len(record.edge_src)
        if len(record.edge_dst) != edge_count or len(record.edge_type) != edge_count:
            raise ValueError("edge_src, edge_dst and edge_type must have the same length")
        if record.edge_attr.ndim != 2 or record.edge_attr.shape[0] != edge_count:
            raise ValueError("edge_attr must have shape [E, D]")
        has_graph = (
            record.graph_neighbors is not None
            or record.graph_edge_attr is not None
            or record.graph_neighbor_mask is not None
        )
        if has_graph:
            if record.graph_neighbors is None or record.graph_edge_attr is None or record.graph_neighbor_mask is None:
                raise ValueError("graph_neighbors, graph_edge_attr and graph_neighbor_mask must be provided together")
            if record.graph_neighbors.ndim != 2 or record.graph_neighbors.shape[0] != length:
                raise ValueError(f"graph_neighbors must have shape [L, K], got {record.graph_neighbors.shape}")
            if record.graph_edge_attr.ndim != 3 or record.graph_edge_attr.shape[:2] != record.graph_neighbors.shape:
                raise ValueError(
                    f"graph_edge_attr must have shape [L, K, D], got {record.graph_edge_attr.shape}"
                )
            if record.graph_neighbor_mask.ndim != 2 or record.graph_neighbor_mask.shape != record.graph_neighbors.shape:
                raise ValueError(
                    f"graph_neighbor_mask must have shape [L, K], got {record.graph_neighbor_mask.shape}"
                )
            if np.any(record.graph_neighbors < 0):
                raise ValueError("graph_neighbors indices must be non-negative")
            if record.graph_neighbors.size and np.any(record.graph_neighbors >= max(length, 1)):
                raise ValueError("graph_neighbors indices must be within sequence length")
        for name in LABELS:
            value = getattr(record, name)
            if value.ndim != 1 or value.shape[0] != length:
                raise ValueError(f"{name} must have shape [L], got {value.shape} for L={length}")
        for name in SOFT_LABELS:
            value = getattr(record, name)
            if value.ndim != 1 or value.shape[0] != length:
                raise ValueError(f"{name} must have shape [L], got {value.shape} for L={length}")
        if np.any(record.edge_src < 0) or np.any(record.edge_dst < 0):
            raise ValueError("edge indices must be non-negative")
        if edge_count and (np.any(record.edge_src >= length) or np.any(record.edge_dst >= length)):
            raise ValueError("edge indices must be within sequence length")


def _read_protenix_embedding(
    handle: h5py.File,
    length: int,
    structure_metadata: dict[str, Any],
) -> np.ndarray:
    if "protenix_embed" in handle:
        return np.asarray(handle["protenix_embed"], dtype=np.float32)
    if "embeddings" in handle and "protenix" in handle["embeddings"]:
        return np.asarray(handle["embeddings/protenix"], dtype=np.float32)
    if {"protenix_s", "protenix_z"}.issubset(handle.keys()):
        return np.concatenate(
            [
                np.asarray(handle["protenix_s"], dtype=np.float32),
                np.asarray(handle["protenix_z"], dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
    if "af_node" not in handle:
        return np.zeros((length, PROTENIX_EMBED_DIM), dtype=np.float32)

    legacy = np.asarray(handle["af_node"], dtype=np.float32)
    geom_dim = _metadata_int(structure_metadata, "structure_node_dim", 12)
    s_dim = _metadata_int(structure_metadata, "protenix_embedding_s_dim", 384)
    z_dim = _metadata_int(structure_metadata, "protenix_embedding_z_dim", 128)
    expected = geom_dim + s_dim + z_dim
    if legacy.shape[1] >= expected:
        s = legacy[:, geom_dim : geom_dim + s_dim]
        z = legacy[:, geom_dim + s_dim : geom_dim + s_dim + z_dim]
        return np.concatenate([s, z], axis=1).astype(np.float32, copy=False)

    return np.zeros((length, s_dim + z_dim), dtype=np.float32)


def _read_starling_embedding(handle: h5py.File, length: int) -> np.ndarray:
    if _starling_read_disabled():
        return np.zeros((length, STARLING_EMBED_DIM), dtype=np.float32)
    if "starling_embed" in handle:
        return np.asarray(handle["starling_embed"], dtype=np.float32)
    if "embeddings" in handle and "starling" in handle["embeddings"]:
        return np.asarray(handle["embeddings/starling"], dtype=np.float32)
    return np.zeros((length, STARLING_EMBED_DIM), dtype=np.float32)


def _starling_read_disabled() -> bool:
    value = os.environ.get("PHASEFLOW_DISABLE_STARLING_READ", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_graph_cache(handle: h5py.File) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if GRAPH_GROUP not in handle:
        return None, None, None
    graph = handle[GRAPH_GROUP]
    required = {"neighbors", "edge_attr", "neighbor_mask"}
    if not required.issubset(graph.keys()):
        return None, None, None
    return (
        np.asarray(graph["neighbors"], dtype=np.int64),
        np.asarray(graph["edge_attr"], dtype=np.float32),
        np.asarray(graph["neighbor_mask"], dtype=np.bool_),
    )


def _edge_attr_dim(handle: h5py.File, graph_edge_attr: np.ndarray | None) -> int:
    if "edge_attr" in handle and handle["edge_attr"].ndim == 2:
        return int(handle["edge_attr"].shape[1])
    if graph_edge_attr is not None and graph_edge_attr.ndim == 3:
        return int(graph_edge_attr.shape[2])
    return 8


def _read_modality_matrix(
    handle: h5py.File,
    name: str,
    length: int,
    structure_metadata: dict[str, Any],
    *,
    is_reliability: bool,
) -> np.ndarray:
    fill = 0.0 if is_reliability else 1.0
    if name not in handle:
        return np.full((length, 5), fill, dtype=np.float32)
    value = np.asarray(handle[name], dtype=np.float32)
    if value.shape[1] != 5:
        if value.shape[1] == 6:
            return value[:, [0, 1, 2, 3, 5]].astype(np.float32, copy=False)
        return value
    return value


def _metadata_int(metadata: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(str(metadata.get(key, default)))
    except (TypeError, ValueError):
        return default


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_optional_vector(handle: h5py.File, name: str, length: int, fill: float) -> np.ndarray:
    if name in handle:
        return np.asarray(handle[name], dtype=np.float32)
    return np.full(length, fill, dtype=np.float32)
