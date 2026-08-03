"""Protein data contracts, caches, validation, and portable utilities."""



# Source: data/schemas.py

from dataclasses import dataclass, field
from typing import Any

import numpy as np

IGNORE_INDEX = -100


@dataclass(slots=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    llps_label: int = IGNORE_INDEX
    uniprot_id: str = ""
    source: str = ""
    label_confidence: float = 1.0
    negative_type: str = "unknown"

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass(slots=True)
class RegionLabel:
    protein_id: str
    start: int
    end: int
    type: str = "DPR_candidate"
    confidence: float = 1.0
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "protein_id": self.protein_id,
            "start": int(self.start),
            "end": int(self.end),
            "type": self.type,
            "confidence": float(self.confidence),
            "source": self.source,
        }


@dataclass(slots=True)
class FeatureCacheRecord:
    protein_id: str
    sequence: str
    plm: np.ndarray
    physchem: np.ndarray
    disorder: np.ndarray
    protenix_embed: np.ndarray
    starling_embed: np.ndarray
    modality_mask: np.ndarray
    reliability: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_type: np.ndarray
    edge_attr: np.ndarray
    graph_neighbors: np.ndarray | None = None
    graph_edge_attr: np.ndarray | None = None
    graph_neighbor_mask: np.ndarray | None = None
    y_llps: float = float(IGNORE_INDEX)
    y_dpr: np.ndarray | None = None
    y_key: np.ndarray | None = None
    y_weight: np.ndarray | None = None
    teacher_llps: float = float("nan")
    teacher_llps_weight: float = 0.0
    self_llps: float = float("nan")
    self_llps_weight: float = 0.0
    region_bag_label: float = float(IGNORE_INDEX)
    region_bag_weight: float = 0.0
    region_bag_type: str = "mask"
    negative_regularization_weight: float = 0.0
    teacher_dpr: np.ndarray | None = None
    teacher_dpr_weight: np.ndarray | None = None
    self_dpr: np.ndarray | None = None
    self_dpr_weight: np.ndarray | None = None
    candidate_prior: np.ndarray | None = None
    candidate_prior_weight: np.ndarray | None = None
    sample_weight: float = 1.0
    label_quality: str = ""
    negative_type: str = ""
    source: str = ""
    regions: list[dict[str, Any]] = field(default_factory=list)
    structure_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def ensure_labels(self) -> None:
        length = self.length
        if self.y_dpr is None:
            self.y_dpr = np.full(length, IGNORE_INDEX, dtype=np.int64)
        if self.y_key is None:
            self.y_key = np.full(length, IGNORE_INDEX, dtype=np.int64)
        if self.y_weight is None:
            self.y_weight = np.zeros(length, dtype=np.float32)
        if self.teacher_dpr is None:
            self.teacher_dpr = np.full(length, np.nan, dtype=np.float32)
        if self.teacher_dpr_weight is None:
            self.teacher_dpr_weight = np.zeros(length, dtype=np.float32)
        if self.self_dpr is None:
            self.self_dpr = np.full(length, np.nan, dtype=np.float32)
        if self.self_dpr_weight is None:
            self.self_dpr_weight = np.zeros(length, dtype=np.float32)
        if self.candidate_prior is None:
            self.candidate_prior = np.zeros(length, dtype=np.float32)
        if self.candidate_prior_weight is None:
            self.candidate_prior_weight = np.zeros(length, dtype=np.float32)


def zero_record(
    protein_id: str,
    sequence: str,
    plm_dim: int = 32,
    phys_dim: int = 88,
    disorder_dim: int = 6,
    protenix_dim: int = 512,
    starling_dim: int = 512,
    edge_dim: int = 8,
) -> FeatureCacheRecord:
    length = len(sequence)
    return FeatureCacheRecord(
        protein_id=protein_id,
        sequence=sequence,
        plm=np.zeros((length, plm_dim), dtype=np.float32),
        physchem=np.zeros((length, phys_dim), dtype=np.float32),
        disorder=np.zeros((length, disorder_dim), dtype=np.float32),
        protenix_embed=np.zeros((length, protenix_dim), dtype=np.float32),
        starling_embed=np.zeros((length, starling_dim), dtype=np.float32),
        modality_mask=np.ones((length, 5), dtype=np.float32),
        reliability=np.ones((length, 5), dtype=np.float32),
        edge_src=np.zeros((0,), dtype=np.int64),
        edge_dst=np.zeros((0,), dtype=np.int64),
        edge_type=np.zeros((0,), dtype=np.int64),
        edge_attr=np.zeros((0, edge_dim), dtype=np.float32),
        graph_neighbors=None,
        graph_edge_attr=None,
        graph_neighbor_mask=None,
    )



# Source: data/runtime_guard.py

import builtins
import os
from pathlib import Path
from typing import Any, Callable


FORBIDDEN_PATH_TOKENS = (
    ".h5",
    ".hdf5",
    "retired_",
    "features_v2",
    "graphs_v2",
    "tables_v2",
    "audit_v2",
)

EVAL_ONLY_TRAINING_PATH_TOKENS = (
    "data/processed/evaluation_only/phasepro_pstp_v1",
)

_ORIGINAL_OPEN: Callable[..., Any] | None = None
_PATCHED = False


def strict_offline_enabled() -> bool:
    return str(os.environ.get("PHASEFLOW_STRICT_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}


def assert_offline_path_allowed(path: str | Path, *, allow_legacy_h5: bool = False) -> None:
    text = str(path)
    normalized = text.replace("\\", "/").lower()
    if allow_legacy_h5:
        tokens = tuple(token for token in FORBIDDEN_PATH_TOKENS if token not in {".h5", ".hdf5"})
    else:
        tokens = FORBIDDEN_PATH_TOKENS
    offenders = [token for token in tokens if token in normalized]
    if offenders:
        raise RuntimeError(f"Forbidden strict-offline data path: {text} (matched {offenders})")


def assert_no_eval_only_training_path(path: str | Path) -> None:
    text = str(path)
    normalized = text.replace("\\", "/").lower()
    offenders = [token for token in EVAL_ONLY_TRAINING_PATH_TOKENS if token in normalized]
    if offenders:
        raise RuntimeError(f"Eval-only sidecar path is forbidden for training data access: {text} (matched {offenders})")


def assert_no_runtime_build(enabled: bool, name: str) -> None:
    if strict_offline_enabled() and enabled:
        raise RuntimeError(f"Runtime {name} is forbidden when PHASEFLOW_STRICT_OFFLINE=1")


def assert_no_forbidden_dataset_write(path: str | Path, mode: str = "r") -> None:
    write_mode = any(flag in mode for flag in ("w", "a", "x", "+"))
    if not write_mode:
        return
    normalized = str(path).replace("\\", "/").lower()
    forbidden_roots = (
        "data/processed/merged/features/",
        "data/processed/merged/graphs/",
    )
    if any(root in normalized for root in forbidden_roots):
        raise RuntimeError(f"Writing training inputs is forbidden in strict offline mode: {path}")


def install_strict_offline_guard() -> None:
    global _ORIGINAL_OPEN, _PATCHED
    if _PATCHED:
        return
    if not strict_offline_enabled():
        return
    _ORIGINAL_OPEN = builtins.open

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, Path)):
            assert_offline_path_allowed(file)
            assert_no_forbidden_dataset_write(file, mode)
        return _ORIGINAL_OPEN(file, mode, *args, **kwargs)  # type: ignore[misc]

    builtins.open = guarded_open
    _PATCHED = True



# Source: data/config.py

from pathlib import Path
from typing import Any


DEFAULT_FORBIDDEN_DATA_TOKENS = ("phaseflow", "phase_diagram")


def resolve_feature_dirs(data_config: dict[str, Any]) -> list[str | Path]:
    if bool(data_config.get("allow_feature_dir_fallbacks", False)) and data_config.get("feature_dirs"):
        return list(data_config["feature_dirs"])
    if data_config.get("feature_dir"):
        return [data_config["feature_dir"]]
    if data_config.get("feature_dirs"):
        feature_dirs = list(data_config["feature_dirs"])
        if feature_dirs:
            return [feature_dirs[0]]
    raise ValueError("data.feature_dir is required; data.feature_dirs fallback is opt-in")


def phase_aux_data_enabled(data_config: dict[str, Any]) -> bool:
    return bool(data_config.get("allow_phase_aux_data", False))


def resolve_phase_targets(data_config: dict[str, Any]) -> str | Path | None:
    if not phase_aux_data_enabled(data_config):
        return None
    return data_config.get("phase_targets")


def validate_forbidden_data_paths(data_config: dict[str, Any], feature_dirs: list[str | Path]) -> None:
    if not bool(data_config.get("forbid_phaseflow_data", False)):
        return
    tokens = tuple(
        str(token).lower()
        for token in data_config.get("forbidden_data_path_tokens", DEFAULT_FORBIDDEN_DATA_TOKENS)
        if str(token).strip()
    )
    if not tokens:
        return
    candidate_paths: list[str] = [str(path) for path in feature_dirs]
    for key in ("phase_train_ids_file", "phase_targets"):
        value = data_config.get(key)
        if value:
            candidate_paths.append(str(value))
    offenders = sorted(
        path
        for path in candidate_paths
        if any(token in path.lower() for token in tokens)
    )
    if offenders:
        raise ValueError(
            "PhaseFlow/phase-diagram data is forbidden for this run, but these paths were configured: "
            + ", ".join(offenders)
        )



# Source: data/feature_cache.py

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from phaseflow.protein.contracts import assert_offline_path_allowed, strict_offline_enabled
from phaseflow.protein.contracts import FeatureCacheRecord
# This is a serialized-cache compatibility value. Keeping it here avoids a
# data-contract import cycle through the optional feature builders.
GRAPH_CACHE_VERSION = 1


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



# Source: data/sharded_store.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phaseflow.protein.contracts import assert_offline_path_allowed


@dataclass
class Esm2ShardedStore:
    """Indexable ESM2 store backed by large `.npy` shards.

    Metadata columns are carried on the sample-index rows:
    `esm2_store_path`, `esm2_mask_store_path`, `esm2_store_offset`, `length`.
    The shard arrays stay fp16 on disk and are converted to float32 only for
    the existing model/collator contract.
    """

    required: bool = True
    _node_cache: dict[str, np.memmap] = field(default_factory=dict, init=False)
    _mask_cache: dict[str, np.memmap] = field(default_factory=dict, init=False)

    def read(self, row: Any, length: int) -> dict[str, np.ndarray] | None:
        node_path_value = row.get("esm2_store_path")
        mask_path_value = row.get("esm2_mask_store_path")
        if node_path_value is None or str(node_path_value) in {"", "nan", "None"}:
            if self.required:
                protein_id = str(row.get("protein_id", "<unknown>"))
                raise FileNotFoundError(f"Missing ESM2 sharded-store metadata for {protein_id}")
            return None
        node_path = Path(str(node_path_value))
        mask_path = Path(str(mask_path_value)) if mask_path_value is not None else None
        offset = int(row.get("esm2_store_offset", -1))
        if offset < 0:
            protein_id = str(row.get("protein_id", "<unknown>"))
            raise ValueError(f"Invalid ESM2 sharded-store offset for {protein_id}: {offset}")
        node_shard = self._open_node(node_path)
        node = np.asarray(node_shard[offset : offset + int(length)], dtype=np.float32)
        if node.shape != (int(length), 1280):
            protein_id = str(row.get("protein_id", "<unknown>"))
            raise ValueError(f"ESM2 sharded node shape mismatch for {protein_id}: {node.shape}")
        if mask_path is not None and str(mask_path) not in {"", "nan", "None"}:
            mask_shard = self._open_mask(mask_path)
            mask = np.asarray(mask_shard[offset : offset + int(length)], dtype=np.float32)
        else:
            mask = np.ones(int(length), dtype=np.float32)
        if mask.shape != (int(length),):
            protein_id = str(row.get("protein_id", "<unknown>"))
            raise ValueError(f"ESM2 sharded mask shape mismatch for {protein_id}: {mask.shape}")
        return {"node": node, "mask": mask}

    def _open_node(self, path: Path) -> np.memmap:
        key = str(path)
        if key not in self._node_cache:
            assert_offline_path_allowed(path)
            self._node_cache[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._node_cache[key]

    def _open_mask(self, path: Path) -> np.memmap:
        key = str(path)
        if key not in self._mask_cache:
            assert_offline_path_allowed(path)
            self._mask_cache[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._mask_cache[key]


@dataclass
class NpzMirrorStore:
    """Mmap mirror for arrays originally packed inside `.npz` shard files."""

    manifest: str | Path
    _paths: dict[tuple[str, str], str] = field(default_factory=dict, init=False)
    _cache: dict[str, np.memmap] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        manifest_path = Path(self.manifest)
        assert_offline_path_allowed(manifest_path)
        frame = pd.read_parquet(manifest_path)
        for _, row in frame.iterrows():
            array_name = str(row["array_name"])
            mirror_path = str(row["mirror_path"])
            for col in ("source_path", "source_abs_path"):
                value = str(row.get(col, ""))
                if value and value.lower() != "nan":
                    self._paths[(value, array_name)] = mirror_path

    def array(self, source_path: Any, array_name: str, *, dataset_root: Path | None = None) -> np.memmap | None:
        candidates = [str(source_path)]
        path = Path(str(source_path))
        if dataset_root is not None and not path.is_absolute():
            candidates.append(str(dataset_root / path))
        for candidate in candidates:
            mirror_path = self._paths.get((candidate, str(array_name)))
            if mirror_path is not None:
                return self._open(mirror_path)
        return None

    def _open(self, path_value: str) -> np.memmap:
        if path_value not in self._cache:
            path = Path(path_value)
            assert_offline_path_allowed(path)
            self._cache[path_value] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._cache[path_value]



# Source: data/splits.py

from pathlib import Path


def resolve_split_ids(data_config: dict, split: str) -> list[str]:
    direct_key = f"{split}_ids"
    file_key = f"{split}_ids_file"
    if direct_key in data_config and data_config[direct_key]:
        return [str(protein_id) for protein_id in data_config[direct_key]]
    if file_key in data_config and data_config[file_key]:
        path = Path(data_config[file_key])
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    manifest = data_config.get("manifest")
    if manifest:
        import pandas as pd

        manifest_path = Path(manifest)
        frame = pd.read_parquet(manifest_path) if manifest_path.suffix.lower() == ".parquet" else pd.read_csv(manifest_path)
        if "split" not in frame.columns:
            raise ValueError(f"Manifest {manifest} has no split column; set {direct_key} or {file_key}")
        aliases = {split}
        if split == "valid":
            aliases.add("val")
        rows = frame.loc[frame["split"].astype(str).isin(aliases)]
        return [str(value) for value in rows["protein_id"].tolist()]
    raise ValueError(f"Could not resolve IDs for split '{split}'; set {direct_key}, {file_key}, or data.manifest")



# Source: data/full_benchmark_leakage.py

import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd


FULL_PPMC_BENCHMARK_SPLITS = {"train", "valid", "test_internal", "benchmark_holdout"}
LEGACY_PPMC_HELDOUT_SPLITS = {"valid", "test_internal", "benchmark_holdout"}
PPMC_RAW_REL = Path("data/benchmarks/protein_benchmark_ppmc/ppmc_ce_de_c_d_np_nd_raw.tsv")
PPMC_MANIFEST_REL = Path("data/benchmarks/protein_benchmark_ppmc/manifest.csv")
PPMC_SOURCE_RECORDS_REL = Path("data/benchmarks/protein_benchmark_ppmc/source_records.csv")
PHASEPRO_PROTEINS_REL = Path("data/benchmarks/dpr_benchmark_phasepro/proteins.csv")
PHASEPRO_SOURCE_RECORDS_REL = Path("data/benchmarks/dpr_benchmark_phasepro/source_records.csv")
MMSEQS40_CLUSTER_REL = Path("data/interim/server_final/mmseqs40_leakage_20260606_cluster.tsv")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def norm_accession(value: Any) -> str:
    text = norm_text(value).upper()
    if not text:
        return ""
    return text.split(".", 1)[0].split("-", 1)[0]


def clean_sequence(value: Any) -> str:
    return "".join(ch for ch in norm_text(value).upper() if ch.isalpha())


def sequence_md5(value: Any) -> str:
    sequence = clean_sequence(value)
    return hashlib.md5(sequence.encode("utf-8")).hexdigest() if sequence else ""


def sequence_sha256(value: Any) -> str:
    sequence = clean_sequence(value)
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest() if sequence else ""


def empty_key_sets() -> dict[str, set[str]]:
    return {"ids": set(), "sha256": set(), "md5": set()}


def merge_key_sets(*items: dict[str, set[str]]) -> dict[str, set[str]]:
    out = empty_key_sets()
    for item in items:
        for key in out:
            out[key].update(item.get(key, set()))
    return out


def full_ppmc_key_sets(root: Path | None = None) -> dict[str, set[str]]:
    root = (root or project_root()).resolve()
    keys = empty_key_sets()
    raw = root / PPMC_RAW_REL
    if raw.exists():
        frame = pd.read_csv(raw, sep="\t", dtype=str, keep_default_na=False)
        _collect_keys(frame, keys)
        if "mapped_protein_ids" in frame.columns:
            for value in frame["mapped_protein_ids"].astype(str):
                for part in value.split(";"):
                    acc = norm_accession(part)
                    if acc:
                        keys["ids"].add(acc)

    for rel in [PPMC_MANIFEST_REL, PPMC_SOURCE_RECORDS_REL]:
        path = root / rel
        if path.exists():
            _collect_keys(pd.read_csv(path, dtype=str, keep_default_na=False), keys)
    return _strip_empty(keys)


def ppmc_legacy_heldout_key_sets(root: Path | None = None) -> dict[str, set[str]]:
    root = (root or project_root()).resolve()
    path = root / PPMC_MANIFEST_REL
    keys = empty_key_sets()
    if path.exists():
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "split" in frame.columns:
            frame = frame[frame["split"].astype(str).isin(LEGACY_PPMC_HELDOUT_SPLITS)].copy()
        _collect_keys(frame, keys)
    return _strip_empty(keys)


def ppmc_final_eval_key_sets(root: Path | None = None) -> dict[str, set[str]]:
    return ppmc_legacy_heldout_key_sets(root)


def phasepro_key_sets(root: Path | None = None) -> dict[str, set[str]]:
    root = (root or project_root()).resolve()
    keys = empty_key_sets()
    for rel in [PHASEPRO_PROTEINS_REL, PHASEPRO_SOURCE_RECORDS_REL]:
        path = root / rel
        if path.exists():
            _collect_keys(pd.read_csv(path, dtype=str, keep_default_na=False), keys)
    return _strip_empty(keys)


def full_benchmark_key_sets(root: Path | None = None) -> dict[str, set[str]]:
    return merge_key_sets(full_ppmc_key_sets(root), phasepro_key_sets(root))


def overlap_flags(
    frame: pd.DataFrame,
    keys: dict[str, set[str]],
    *,
    prefix: str,
    cluster_path: Path | None = None,
) -> pd.DataFrame:
    row_ids = _row_id_sets(frame)
    row_sha256 = _row_hash_sets(frame, hash_kind="sha256")
    row_md5 = _row_hash_sets(frame, hash_kind="md5")
    key_ids = {norm_accession(value) for value in keys.get("ids", set()) if norm_accession(value)}
    key_sha256 = {norm_text(value).lower() for value in keys.get("sha256", set()) if norm_text(value)}
    key_md5 = {norm_text(value).lower() for value in keys.get("md5", set()) if norm_text(value)}

    direct = pd.Series([bool(ids & key_ids) for ids in row_ids], index=frame.index)
    hash_overlap = pd.Series(
        [bool(sha & key_sha256) or bool(md5 & key_md5) for sha, md5 in zip(row_sha256, row_md5, strict=False)],
        index=frame.index,
    )
    homolog = pd.Series(False, index=frame.index)
    if cluster_path and cluster_path.exists() and key_ids:
        cluster_map = read_cluster_map(cluster_path)
        benchmark_reps = {cluster_map.get(key, key) for key in key_ids}
        homolog = pd.Series(
            [any(cluster_map.get(row_id, row_id) in benchmark_reps for row_id in ids) for ids in row_ids],
            index=frame.index,
        )

    out = pd.DataFrame(index=frame.index)
    out[f"{prefix}_direct_overlap"] = direct.astype(bool)
    out[f"{prefix}_hash_overlap"] = hash_overlap.astype(bool)
    out[f"{prefix}_homolog_overlap"] = homolog.astype(bool)
    blocker = direct | hash_overlap
    out[f"{prefix}_benchmark_overlap"] = blocker.astype(bool)
    return out


def assert_no_full_benchmark_leakage(
    sample_index: str | Path,
    *,
    root: Path | None = None,
    report_dir: str | Path | None = None,
    context: str = "training",
) -> None:
    path = Path(sample_index)
    if not path.is_absolute():
        path = (root or project_root()) / path
    if not path.exists():
        return
    frame = _read_table(path)
    keys = full_benchmark_key_sets(root)
    cluster = (root or project_root()) / MMSEQS40_CLUSTER_REL
    flags = overlap_flags(frame, keys, prefix="full_benchmark", cluster_path=cluster)
    mask = flags["full_benchmark_benchmark_overlap"].astype(bool)
    if not bool(mask.any()):
        return

    sample_cols = [col for col in ["protein_id", "uniprot_id", "accession", "sequence_sha256", "sequence_hash", "sequence_md5", "source_dataset", "source"] if col in frame.columns]
    sample = pd.concat([frame.loc[mask, sample_cols].reset_index(drop=True), flags.loc[mask].reset_index(drop=True)], axis=1)
    report_path = None
    if report_dir is not None:
        out_dir = Path(report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"full_benchmark_leakage_blocker_{path.stem}.csv"
        sample.to_csv(report_path, index=False)
    detail = f"; report={report_path}" if report_path else ""
    raise RuntimeError(
        f"Full PPMC/PhasePro exact-duplicate leakage guard failed for {context} sample_index={path}: "
        f"overlap_rows={int(mask.sum())}, unique_proteins={int(frame.loc[mask, 'protein_id'].nunique()) if 'protein_id' in frame.columns else 'unknown'}"
        f"{detail}"
    )


def read_cluster_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep = norm_accession(parts[0])
            member = norm_accession(parts[1])
            if rep and member:
                mapping[member] = rep
    return mapping


def _collect_keys(frame: pd.DataFrame, keys: dict[str, set[str]]) -> None:
    for col in ["protein_id", "uniprot_id", "accession", "source_id", "source_record_id", "UniProt.Acc", "uniprot_accession_norm"]:
        if col in frame.columns:
            keys["ids"].update(acc for acc in frame[col].map(norm_accession).tolist() if acc)
    for col in ["sequence_hash", "sequence_sha256", "seq_hash"]:
        if col in frame.columns:
            keys["sha256"].update(norm_text(value).lower() for value in frame[col].tolist() if norm_text(value))
    for col in ["sequence_md5"]:
        if col in frame.columns:
            keys["md5"].update(norm_text(value).lower() for value in frame[col].tolist() if norm_text(value))
    for col in ["sequence", "Full.seq"]:
        if col in frame.columns:
            keys["sha256"].update(value for value in frame[col].map(sequence_sha256).tolist() if value)
            keys["md5"].update(value for value in frame[col].map(sequence_md5).tolist() if value)


def _row_id_sets(frame: pd.DataFrame) -> list[set[str]]:
    cols = [col for col in ["protein_id", "uniprot_id", "accession", "source_id"] if col in frame.columns]
    rows: list[set[str]] = []
    if cols:
        for _, row in frame[cols].iterrows():
            rows.append({acc for acc in (norm_accession(row[col]) for col in cols) if acc})
    else:
        rows = [set() for _ in range(len(frame))]
    return rows


def _row_hash_sets(frame: pd.DataFrame, *, hash_kind: str) -> list[set[str]]:
    rows = [set() for _ in range(len(frame))]
    if hash_kind == "sha256":
        cols = [col for col in ["sequence_sha256", "sequence_hash", "seq_hash"] if col in frame.columns]
        for col in cols:
            for idx, value in enumerate(frame[col].tolist()):
                text = norm_text(value).lower()
                if text:
                    rows[idx].add(text)
        if "sequence" in frame.columns:
            for idx, value in enumerate(frame["sequence"].map(sequence_sha256).tolist()):
                if value:
                    rows[idx].add(value)
    elif hash_kind == "md5":
        if "sequence_md5" in frame.columns:
            for idx, value in enumerate(frame["sequence_md5"].tolist()):
                text = norm_text(value).lower()
                if text:
                    rows[idx].add(text)
        if "sequence" in frame.columns:
            for idx, value in enumerate(frame["sequence"].map(sequence_md5).tolist()):
                if value:
                    rows[idx].add(value)
    else:
        raise ValueError(f"Unsupported hash kind: {hash_kind}")
    return rows


def _strip_empty(keys: dict[str, set[str]]) -> dict[str, set[str]]:
    return {key: {value for value in values if value} for key, values in keys.items()}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)



# Source: utils.py

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        return yaml.safe_load(handle)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(sanitize_json(data), handle, indent=2, sort_keys=True, allow_nan=False)


def dumps_json(data: Any, **kwargs: Any) -> str:
    return json.dumps(sanitize_json(data), allow_nan=False, **kwargs)


def sanitize_json(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: sanitize_json(value) for key, value in data.items()}
    if isinstance(data, list):
        return [sanitize_json(value) for value in data]
    if isinstance(data, tuple):
        return [sanitize_json(value) for value in data]
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved
