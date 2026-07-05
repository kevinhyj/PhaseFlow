import numpy as np
import pandas as pd

from phaseflow.full_length.data.feature_cache import FeatureCacheReader, FeatureCacheWriter
from phaseflow.full_length.data.schemas import zero_record
from phaseflow.full_length.features.build_features import build_feature_cache_from_manifest


def test_feature_cache_roundtrip(tmp_path) -> None:
    record = zero_record("toy", "ACDEFGHIKLMN")
    record.y_llps = 1.0
    record.y_dpr = np.zeros(record.length, dtype=np.int64)
    record.y_key = np.zeros(record.length, dtype=np.int64)
    record.y_weight = np.ones(record.length, dtype=np.float32)
    record.teacher_llps = 0.8
    record.teacher_llps_weight = 0.6
    record.teacher_dpr = np.full(record.length, 0.25, dtype=np.float32)
    record.teacher_dpr_weight = np.full(record.length, 0.5, dtype=np.float32)
    record.edge_src = np.array([0, 1], dtype=np.int64)
    record.edge_dst = np.array([1, 2], dtype=np.int64)
    record.edge_type = np.array([0, 0], dtype=np.int64)
    record.edge_attr = np.zeros((2, 8), dtype=np.float32)
    record.graph_neighbors = np.tile(np.arange(record.length, dtype=np.int64).reshape(-1, 1), (1, 3))
    record.graph_edge_attr = np.zeros((record.length, 3, 8), dtype=np.float32)
    record.graph_neighbor_mask = np.ones((record.length, 3), dtype=np.bool_)
    path = tmp_path / "toy.h5"
    FeatureCacheWriter.write_h5(path, record)
    loaded = FeatureCacheReader.read_h5(path)
    assert loaded.protein_id == "toy"
    assert loaded.sequence == record.sequence
    assert loaded.plm.shape == record.plm.shape
    assert loaded.y_llps == 1.0
    assert loaded.teacher_llps == 0.8
    assert loaded.teacher_dpr_weight.shape == (record.length,)
    assert loaded.graph_neighbors is not None
    assert loaded.graph_neighbors.shape == (record.length, 3)
    assert loaded.graph_edge_attr is not None
    assert loaded.graph_edge_attr.shape == (record.length, 3, 8)
    assert loaded.graph_neighbor_mask is not None
    assert loaded.graph_neighbor_mask.dtype == np.bool_


def test_feature_cache_manifest_preserves_sample_weight(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "protein_id": "toy",
                "sequence": "ACDEFGHIKLMN",
                "llps_label": 1,
                "sample_weight": 0.37,
                "label_confidence": 0.9,
                "teacher_consensus_score": 0.75,
                "teacher_confidence": 0.5,
            }
        ]
    ).to_csv(manifest, index=False)

    out_dir = tmp_path / "features"
    build_feature_cache_from_manifest(manifest, out_dir, mode="simple")
    loaded = FeatureCacheReader.read_h5(out_dir / "toy.h5")
    assert loaded.sample_weight == 0.37
    assert loaded.teacher_llps == 0.75
    assert loaded.teacher_llps_weight == 0.5
