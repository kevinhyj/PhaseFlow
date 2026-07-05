from __future__ import annotations

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
