"""Workflow-level regression checks for the protein release."""



# Source: test_af3_no_msa.py

import json

from phaseflow.protein import write_af3_input_json


def test_write_af3_input_json_no_msa_defaults(tmp_path) -> None:
    path = write_af3_input_json("p1", "ACDEFG", tmp_path)
    payload = json.loads(path.read_text())

    protein = payload["sequences"][0]["protein"]
    assert protein["sequence"] == "ACDEFG"
    assert protein["unpairedMsa"] == ""
    assert protein["pairedMsa"] == ""
    assert protein["templates"] == []


def test_write_af3_input_json_full_pipeline_omits_manual_msa(tmp_path) -> None:
    path = write_af3_input_json("p1", "ACDEFG", tmp_path, msa_mode="full_pipeline")
    protein = json.loads(path.read_text())["sequences"][0]["protein"]

    assert "unpairedMsa" not in protein
    assert "pairedMsa" not in protein
    assert "templates" not in protein



# Source: test_biophys_features.py


import numpy as np

from phaseflow.protein import compute_biophys_node
from phaseflow.protein import compute_disorder_features
from phaseflow.protein import compute_physchem_features


def test_biophys_node_preserves_base_features_and_has_dpr_width() -> None:
    sequence = "AKRDEGPY"

    node, names = compute_biophys_node(sequence)
    physchem, _ = compute_physchem_features(sequence)
    disorder, _, _, _ = compute_disorder_features(sequence, mode="simple")

    assert node.shape == (len(sequence), 112)
    assert len(names) == 112
    np.testing.assert_allclose(node[:, :90], physchem)
    np.testing.assert_allclose(node[:, 90:96], disorder)
    assert node.dtype == np.float32
    assert np.all(node[:, -1] == 1.0)



# Source: test_data_config.py

from pathlib import Path

import pytest

from phaseflow.protein import resolve_feature_dirs, resolve_phase_targets, validate_forbidden_data_paths
from scripts.protein.workflows.training import _phase_train_ids, _validate_train_region_supervision


def test_feature_dir_fallbacks_are_opt_in() -> None:
    data_config = {
        "feature_dir": "artifacts/data/features/merged_h5_teacher",
        "feature_dirs": [
            "artifacts/data/features/merged_h5_teacher_phaseflow",
            "external_artifacts/phase_diagram_original_scale_h5",
        ],
    }

    assert resolve_feature_dirs(data_config) == ["artifacts/data/features/merged_h5_teacher"]

    data_config["allow_feature_dir_fallbacks"] = True
    assert resolve_feature_dirs(data_config) == [
        "artifacts/data/features/merged_h5_teacher_phaseflow",
        "external_artifacts/phase_diagram_original_scale_h5",
    ]


def test_phase_aux_ids_and_targets_are_opt_in(tmp_path: Path) -> None:
    ids_file = tmp_path / "phase_ids.txt"
    ids_file.write_text("phase_a\nphase_b\n")
    data_config = {
        "phase_train_ids_file": str(ids_file),
        "phase_targets": "artifacts/data/processed/phaseflow/phase_targets.csv",
    }

    assert _phase_train_ids(data_config) == []
    assert resolve_phase_targets(data_config) is None

    data_config["allow_phase_aux_data"] = True
    assert _phase_train_ids(data_config) == ["phase_a", "phase_b"]
    assert resolve_phase_targets(data_config) == "artifacts/data/processed/phaseflow/phase_targets.csv"


def test_forbidden_phaseflow_paths_fail_fast() -> None:
    data_config = {
        "forbid_phaseflow_data": True,
        "phase_targets": "artifacts/data/processed/phaseflow/phase_targets.csv",
    }

    with pytest.raises(ValueError, match="forbidden"):
        validate_forbidden_data_paths(data_config, ["artifacts/data/features/merged_h5_teacher"])


def test_feature_region_supervision_with_dpr_losses_fails_fast() -> None:
    config = {"training": {"loss_weights": {"region_gold": 0.1, "region": 0.1}}}
    with pytest.raises(ValueError, match="gold labels is forbidden"):
        _validate_train_region_supervision(config, "feature")


def test_pstp_region_supervision_requires_target_file_for_final_losses() -> None:
    config = {"data": {}, "training": {"loss_weights": {"region_teacher": 1.0}}}
    with pytest.raises(ValueError, match="requires data.region_targets"):
        _validate_train_region_supervision(config, "region_targets")


def test_pstp_region_supervision_rejects_mixed_dpr_teacher_losses() -> None:
    config = {
        "data": {"region_targets": "artifacts/data/processed/pstp_scan_region_targets.h5"},
        "training": {"loss_weights": {"region_teacher": 1.0, "teacher_dpr": 0.1, "region": 0.1}},
    }
    with pytest.raises(ValueError, match="Pure PSTP-Scan DPR training"):
        _validate_train_region_supervision(config, "region_targets")



# Source: test_dataset_collator.py

from phaseflow.protein import PhaseFlowCollator
from phaseflow.protein.data import _edge_list_to_neighbors
from phaseflow.protein import FeatureCacheWriter
from phaseflow.protein import PhaseFlowDataset
from phaseflow.protein import PhaseFlowOfflineDataset
from phaseflow.protein import zero_record
from phaseflow.protein import build_edges
from phaseflow.protein import edge_list_to_precomputed_graph
import pytest
import torch


def test_dataset_collator_shapes(tmp_path) -> None:
    for protein_id, sequence in [("a", "ACDEFG"), ("b", "ACDEFGHIK")]:
        record = zero_record(protein_id, sequence)
        edges = build_edges(len(sequence), local_window=1)
        record.edge_src = edges.edge_src
        record.edge_dst = edges.edge_dst
        record.edge_type = edges.edge_type
        record.edge_attr = edges.edge_attr
        FeatureCacheWriter.write_h5(tmp_path / f"{protein_id}.h5", record)
    dataset = PhaseFlowDataset(tmp_path, ["a", "b"])
    batch = PhaseFlowCollator(max_neighbors=4)([dataset[0], dataset[1]])
    assert batch["seq_mask"].shape == (2, 9)
    assert batch["neighbors"].shape == (2, 9, 4)
    assert batch["neighbor_mask"].any()
    assert batch["y_dpr"][0, 6].item() == -100
    assert batch["region_bag_label"].shape == (2,)
    assert batch["candidate_prior"].shape == (2, 9)


def test_dataset_collator_loads_phase_targets(tmp_path) -> None:
    for protein_id, sequence in [("phase_a", "ACDEFG"), ("plain_b", "ACDEFGHIK")]:
        record = zero_record(protein_id, sequence)
        edges = build_edges(len(sequence), local_window=1)
        record.edge_src = edges.edge_src
        record.edge_dst = edges.edge_dst
        record.edge_type = edges.edge_type
        record.edge_attr = edges.edge_attr
        FeatureCacheWriter.write_h5(tmp_path / f"{protein_id}.h5", record)
    target_path = tmp_path / "phase_targets.csv"
    phase_columns = []
    for index in range(16):
        phase_columns.extend([f"phase_value_{index:02d}", f"phase_mask_{index:02d}"])
    values = ["0.0", "0.0"] * 16
    values[0] = "-1.0"
    values[1] = "1.0"
    values[2] = "0.5"
    values[3] = "1.0"
    target_path.write_text(
        ",".join(
            [
                "protein_id",
                "sequence",
                "phase_aux_weight",
                "phase_mean_pssi",
                "phase_low_pssi",
                *phase_columns,
            ]
        )
        + "\n"
        + ",".join(["phase_a", "ACDEFG", "0.75", "-0.25", "0.5", *values])
        + "\n"
    )

    dataset = PhaseFlowDataset(tmp_path, ["phase_a", "plain_b"], phase_targets=target_path)
    batch = PhaseFlowCollator(max_neighbors=4)([dataset[0], dataset[1]])

    assert batch["phase_values"].shape == (2, 16)
    assert batch["phase_mask"].shape == (2, 16)
    assert batch["phase_mask"][0].sum().item() == 2
    assert batch["phase_mask"][1].sum().item() == 0
    assert batch["phase_aux_weight"].tolist() == [0.75, 0.0]


def test_dataset_collator_loads_final_region_targets(tmp_path) -> None:
    import h5py
    import numpy as np

    for protein_id, sequence in [("a", "ACDEFG"), ("b", "ACDEFGHIK")]:
        record = zero_record(protein_id, sequence)
        edges = build_edges(len(sequence), local_window=1)
        record.edge_src = edges.edge_src
        record.edge_dst = edges.edge_dst
        record.edge_type = edges.edge_type
        record.edge_attr = edges.edge_attr
        FeatureCacheWriter.write_h5(tmp_path / f"{protein_id}.h5", record)
    target_path = tmp_path / "region_targets.h5"
    with h5py.File(target_path, "w") as handle:
        group = handle.create_group("a")
        group.create_dataset("region_teacher_target", data=np.asarray([1, 1, 0, 0, np.nan, np.nan], dtype=np.float32))
        group.create_dataset("region_teacher_weight", data=np.asarray([1, 1, 0.5, 0.5, 0, 0], dtype=np.float32))
        group.create_dataset("region_key_target", data=np.asarray([np.nan, 1, np.nan, 0, np.nan, np.nan], dtype=np.float32))
        group.create_dataset("region_key_weight", data=np.asarray([0, 1, 0, 0.25, 0, 0], dtype=np.float32))
        group.create_dataset("region_boundary_target", data=np.asarray([1, 0, 0, 1, np.nan, np.nan], dtype=np.float32))
        group.create_dataset("region_boundary_weight", data=np.asarray([1, 0.25, 0.25, 1, 0, 0], dtype=np.float32))
        group.create_dataset("region_contrast_target", data=np.asarray([1, 1, 0, 0, np.nan, np.nan], dtype=np.float32))
        group.create_dataset("region_contrast_weight", data=np.asarray([1, 1, 0.5, 0.5, 0, 0], dtype=np.float32))

    dataset = PhaseFlowDataset(tmp_path, ["a", "b"], region_targets=target_path)
    batch = PhaseFlowCollator(max_neighbors=4)([dataset[0], dataset[1]])

    assert batch["region_teacher_target"].shape == (2, 9)
    assert batch["region_teacher_weight"][0, :4].sum().item() == 3.0
    assert batch["region_teacher_weight"][1].sum().item() == 0.0
    assert batch["region_key_weight"][0].sum().item() == 1.25
    assert batch["region_boundary_weight"][0].sum().item() == 2.5


def test_offline_region_npz_uses_half_open_span_end_for_boundaries(tmp_path) -> None:
    import numpy as np
    import pandas as pd

    labels_dir = tmp_path / "region_labels"
    labels_dir.mkdir()
    length = 32
    residue_label = np.full(length, np.nan, dtype=np.float32)
    residue_mask = np.zeros(length, dtype=np.float32)
    residue_weight = np.zeros(length, dtype=np.float32)
    residue_label[10:20] = 1.0
    residue_mask[10:20] = 1.0
    residue_weight[10:20] = 1.0
    np.savez(
        labels_dir / "half_open_reconstructed.npz",
        residue_label=residue_label,
        residue_mask=residue_mask,
        residue_weight=residue_weight,
        span_start=np.asarray([10], dtype=np.int32),
        span_end=np.asarray([20], dtype=np.int32),
        span_confidence=np.asarray([1.0], dtype=np.float32),
        coordinate_system=np.asarray("0-based half-open"),
    )

    dataset = PhaseFlowOfflineDataset.__new__(PhaseFlowOfflineDataset)
    dataset.dataset_root = tmp_path
    dataset.region_labels_dir = labels_dir
    target = dataset._read_region_label_npz(pd.Series({"protein_id": "half_open_reconstructed"}), length)

    assert target["region_teacher_target"][9] != 1.0
    assert target["region_teacher_target"][10:20].tolist() == [1.0] * 10
    assert target["region_teacher_target"][20] != 1.0
    assert target["region_boundary_target"][10] == 1.0
    assert target["region_boundary_target"][19] == 1.0
    assert target["region_boundary_target"][20] != 1.0
    assert target["positive_spans"] == [{"start": 10, "end": 19, "confidence": 1.0, "sample_weight": 1.0}]

    boundary_target = np.full(length, np.nan, dtype=np.float32)
    boundary_weight = np.zeros(length, dtype=np.float32)
    boundary_target[[10, 19]] = 1.0
    boundary_weight[[10, 19]] = 0.75
    np.savez(
        labels_dir / "half_open_precomputed.npz",
        residue_label=residue_label,
        residue_mask=residue_mask,
        residue_weight=residue_weight,
        boundary_target=boundary_target,
        boundary_mask=boundary_weight,
        span_start=np.asarray([10], dtype=np.int32),
        span_end=np.asarray([20], dtype=np.int32),
        span_confidence=np.asarray([0.75], dtype=np.float32),
        coordinate_system=np.asarray("0-based half-open"),
    )
    target = dataset._read_region_label_npz(pd.Series({"protein_id": "half_open_precomputed"}), length)
    assert target["region_boundary_target"][10] == 1.0
    assert target["region_boundary_target"][19] == 1.0
    assert target["region_boundary_target"][20] != 1.0
    assert target["region_boundary_weight"][10] == 0.75
    assert target["region_boundary_weight"][19] == 0.75


def test_region_targets_supervision_masks_feature_gold_labels(tmp_path) -> None:
    import json
    import h5py
    import numpy as np

    record = zero_record("gold_region", "ACDEFG")
    record.ensure_labels()
    record.y_dpr[:] = 1
    record.y_weight[:] = 1.0
    record.regions = [
        {
            "start": 0,
            "end": 5,
            "type": "DPR_gold",
            "region_type": "DPR_gold",
            "region_label": "positive",
            "source": "PhaSePro",
        }
    ]
    edges = build_edges(record.length, local_window=1)
    record.edge_src = edges.edge_src
    record.edge_dst = edges.edge_dst
    record.edge_type = edges.edge_type
    record.edge_attr = edges.edge_attr
    FeatureCacheWriter.write_h5(tmp_path / "gold_region.h5", record)

    target_path = tmp_path / "region_targets.h5"
    with h5py.File(target_path, "w") as handle:
        handle.attrs["policy"] = "pstp_scan_only_multiscale_window_no_gold"
        group = handle.create_group("gold_region")
        group.create_dataset("region_teacher_target", data=np.full(record.length, 0.8, dtype=np.float32))
        group.create_dataset("region_teacher_weight", data=np.full(record.length, 0.7, dtype=np.float32))
        group.attrs["positive_spans_json"] = json.dumps(
            [{"start": 1, "end": 4, "confidence": 0.8, "sample_weight": 0.8}]
        )

    dataset = PhaseFlowDataset(
        tmp_path,
        ["gold_region"],
        region_targets=target_path,
        region_supervision="region_targets",
    )
    batch = PhaseFlowCollator(max_neighbors=4)([dataset[0]])

    assert batch["y_dpr"][0, : record.length].tolist() == [-100] * record.length
    assert batch["y_weight"][0, : record.length].sum().item() == 0.0
    assert batch["regions"][0][0]["source"] == "pstp_scan_only_multiscale_window_no_gold"
    assert batch["regions"][0][0]["type"] == "DPR_teacher"


def test_edge_list_to_neighbors_orders_truncates_and_self_fills() -> None:
    edge_src = torch.tensor([0, 0, 0, 1, 1, 4])
    edge_dst = torch.tensor([2, 1, 0, 2, 0, 0])
    edge_type = torch.tensor([1, 0, 0, 2, 0, 0])
    edge_attr = torch.arange(48, dtype=torch.float32).reshape(6, 8)

    neighbors, neighbor_attr, neighbor_mask, neighbor_edge_type = _edge_list_to_neighbors(
        length=3,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_type=edge_type,
        edge_attr=edge_attr,
        max_neighbors=2,
        edge_dim=8,
    )

    assert neighbors[0].tolist() == [0, 1]
    assert neighbors[1].tolist() == [0, 2]
    assert neighbors[2].tolist() == [2, 0]
    assert neighbor_mask[0].tolist() == [True, True]
    assert neighbor_mask[2].tolist() == [True, False]
    assert torch.equal(neighbor_attr[0, 0], edge_attr[2])
    assert neighbor_edge_type[0].tolist() == [0, 0]
    assert neighbor_edge_type[1].tolist() == [0, 2]


def test_collator_uses_precomputed_graph_cache(tmp_path) -> None:
    record = zero_record("cached", "ACDEFG")
    edges = build_edges(record.length, local_window=1)
    record.edge_src = edges.edge_src
    record.edge_dst = edges.edge_dst
    record.edge_type = edges.edge_type
    record.edge_attr = edges.edge_attr
    graph = edge_list_to_precomputed_graph(
        length=record.length,
        edge_src=edges.edge_src,
        edge_dst=edges.edge_dst,
        edge_type=edges.edge_type,
        edge_attr=edges.edge_attr,
        max_neighbors=3,
        edge_dim=8,
    )
    record.graph_neighbors = graph.neighbors
    record.graph_edge_attr = graph.edge_attr
    record.graph_neighbor_mask = graph.neighbor_mask
    FeatureCacheWriter.write_h5(tmp_path / "cached.h5", record)

    dataset = PhaseFlowDataset(tmp_path, ["cached"])
    batch = PhaseFlowCollator(max_neighbors=2, require_precomputed_graph=True)([dataset[0]])

    assert torch.equal(batch["neighbors"][0, : record.length], torch.from_numpy(graph.neighbors[:, :2]))
    assert torch.equal(batch["neighbor_mask"][0, : record.length], torch.from_numpy(graph.neighbor_mask[:, :2]))
    assert torch.equal(batch["edge_attr"][0, : record.length], torch.from_numpy(graph.edge_attr[:, :2]))


def test_collator_requires_precomputed_graph_when_configured(tmp_path) -> None:
    record = zero_record("legacy", "ACDEFG")
    edges = build_edges(record.length, local_window=1)
    record.edge_src = edges.edge_src
    record.edge_dst = edges.edge_dst
    record.edge_type = edges.edge_type
    record.edge_attr = edges.edge_attr
    FeatureCacheWriter.write_h5(tmp_path / "legacy.h5", record)

    dataset = PhaseFlowDataset(tmp_path, ["legacy"])
    with pytest.raises(ValueError, match="missing a usable precomputed graph cache"):
        PhaseFlowCollator(max_neighbors=2, require_precomputed_graph=True)([dataset[0]])



# Source: test_esm2_cache_interface.py

import numpy as np

from phaseflow.protein import build_feature_cache
from phaseflow.protein import FeatureCacheReader


def test_build_feature_cache_from_precomputed_esm2(tmp_path) -> None:
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">p1\nACDEFG\n")
    esm2_dir = tmp_path / "esm2"
    esm2_dir.mkdir()
    embedding = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    np.savez_compressed(
        esm2_dir / "p1.npz",
        protein_id=np.asarray("p1"),
        sequence=np.asarray("ACDEFG"),
        length=np.asarray(6, dtype=np.int64),
        embedding_last_hidden_state=embedding,
        model_name=np.asarray("test-esm2"),
    )

    build_feature_cache(fasta=fasta, out_dir=tmp_path / "cache", mode="esm2", esm2_dir=esm2_dir)

    record = FeatureCacheReader.read_h5(tmp_path / "cache" / "p1.h5")
    assert record.plm.shape == (6, 5)
    np.testing.assert_allclose(record.plm, embedding)
    assert record.modality_mask[:, 0].sum() == 0.0



# Source: test_external_feature_cache.py

import numpy as np
import h5py

from phaseflow.protein import FeatureCacheReader
from phaseflow.protein import build_feature_cache


def test_build_feature_cache_reads_starling512_and_distance_edges(tmp_path) -> None:
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">p1\nACDEFG\n")

    starling_embedding_dir = tmp_path / "starling_embedding"
    starling_embedding_dir.mkdir()
    np.savez_compressed(
        starling_embedding_dir / "p1.npz",
        protein_id=np.asarray("p1"),
        sequence=np.asarray("ACDEFG"),
        embedding=np.full((6, 512), 3.0, dtype=np.float32),
    )
    starling_distance_dir = tmp_path / "starling_distance"
    starling_distance_dir.mkdir()
    maps = np.full((4, 6, 6), 30.0, dtype=np.float32)
    maps[:, 1, 4] = 5.0
    maps[:, 4, 1] = 5.0
    with h5py.File(starling_distance_dir / "p1.h5", "w") as handle:
        handle.attrs["sequence"] = "ACDEFG"
        handle.create_dataset("distance_maps", data=maps)

    build_feature_cache(
        fasta=fasta,
        out_dir=tmp_path / "cache",
        starling_embedding_dir=starling_embedding_dir,
        starling_distance_dir=starling_distance_dir,
        graph_edge_dim=13,
        require_starling=True,
    )

    record = FeatureCacheReader.read_h5(tmp_path / "cache" / "p1.h5")
    assert record.protenix_embed.shape == (6, 512)
    assert record.starling_embed.shape == (6, 512)
    np.testing.assert_allclose(record.starling_embed, 3.0)
    assert record.modality_mask[:, 3].sum() == 6.0
    assert record.modality_mask[:, 4].sum() == 0.0
    assert 2 in set(record.edge_type.tolist())
    star_edges = record.edge_attr[record.edge_type == 2]
    assert star_edges.shape[1] == 13
    assert star_edges[:, 11].max() > 0.0


def test_build_feature_cache_masks_missing_embeddings(tmp_path) -> None:
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">p1\nACDEFG\n")

    build_feature_cache(fasta=fasta, out_dir=tmp_path / "cache")

    record = FeatureCacheReader.read_h5(tmp_path / "cache" / "p1.h5")
    assert record.protenix_embed.shape == (6, 512)
    assert record.starling_embed.shape == (6, 512)
    assert record.modality_mask[:, 3].sum() == 6.0
    assert record.modality_mask[:, 4].sum() == 6.0


def test_build_feature_cache_concats_protenix_embedding(tmp_path) -> None:
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">p1\nACDEFG\n")

    embedding_dir = tmp_path / "protenix_embedding"
    embedding_dir.mkdir()
    np.savez_compressed(
        embedding_dir / "p1.npz",
        s=np.ones((6, 3), dtype=np.float32),
        z=np.full((6, 2), 2.0, dtype=np.float32),
        single_mask=np.ones(6, dtype=np.float32),
        is_ligand=np.zeros(6, dtype=np.float32),
    )

    build_feature_cache(
        fasta=fasta,
        out_dir=tmp_path / "cache",
        protenix_embedding_dir=embedding_dir,
    )

    record = FeatureCacheReader.read_h5(tmp_path / "cache" / "p1.h5")
    assert record.protenix_embed.shape == (6, 5)
    np.testing.assert_allclose(record.protenix_embed[:, :3], 1.0)
    np.testing.assert_allclose(record.protenix_embed[:, 3:], 2.0)
    assert record.modality_mask[:, 3].sum() == 0.0
    assert record.reliability[:, 3].min() == 1.0
    assert record.structure_metadata["protenix_embedding_success"] == "1"
    assert record.structure_metadata["protenix_embedding_dim"] == "5"



# Source: test_feature_cache.py

import numpy as np
import pandas as pd

from phaseflow.protein import FeatureCacheReader, FeatureCacheWriter
from phaseflow.protein import zero_record
from phaseflow.protein import build_feature_cache_from_manifest


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


def test_feature_cache_accepts_the_release_parquet_manifest(tmp_path) -> None:
    manifest = tmp_path / "proteins.parquet"
    pd.DataFrame([{"protein_id": "toy", "sequence": "ACDEFG"}]).to_parquet(manifest, index=False)

    build_feature_cache_from_manifest(manifest, tmp_path / "features", mode="simple")

    assert (tmp_path / "features" / "toy.h5").is_file()



# Source: test_feature_script_interfaces.py


import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_protenix_cache_setup_supports_help() -> None:
    result = subprocess.run(
            ["bash", "scripts/protein/features/setup_protenix_cache.sh", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_af3_batch_runner_supports_help() -> None:
    result = subprocess.run(
            ["bash", "scripts/protein/features/run_af3_batch.sh", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout



# Source: test_physchem.py

import numpy as np

from phaseflow.protein import compute_physchem_features


def test_physchem_shape_and_no_nan() -> None:
    features, names = compute_physchem_features("ACDX")
    assert features.shape == (4, len(names))
    assert features.shape[1] == 90
    assert not np.isnan(features).any()


def test_physchem_short_sequence() -> None:
    features, _ = compute_physchem_features("X")
    assert features.shape[0] == 1
    assert not np.isnan(features).any()



# Source: test_prepare_weak_dataset.py

import json

import pandas as pd

from scripts.protein.workflows.data_tools import prepare_weak_dataset


def test_prepare_weak_dataset_writes_manifest_regions_and_splits(tmp_path) -> None:
    ppmc = tmp_path / "datasets.tsv"
    pd.DataFrame(
        [
            {
                "UniProt.Acc": "P_POS",
                "Gene.Name": "POS",
                "Datasets": "D-;DE",
                "GO": "C",
                "Frac.Order": 0.1,
                "Frac.Disorder": 0.9,
                "Full.seq": "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
            },
            {
                "UniProt.Acc": "P_NEG",
                "Gene.Name": "NEG",
                "Datasets": "ND",
                "GO": "C",
                "Frac.Order": 0.1,
                "Frac.Disorder": 0.9,
                "Full.seq": "YYYYYYYYYYACDEFGHIKLMNPQRSTV",
            },
        ]
    ).to_csv(ppmc, sep="\t", index=False)

    phasepro = tmp_path / "phasepro.json"
    phasepro.write_text(
        json.dumps(
            {
                "P_POS": {
                    "accession": "P_POS",
                    "sequence": "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
                    "boundaries": "2-5",
                    "segment": "test segment",
                    "gene": "POS",
                }
            }
        )
    )

    out_dir = tmp_path / "processed"
    report = prepare_weak_dataset(ppmc, phasepro, out_dir, max_records=None, seed=1)

    manifest = pd.read_csv(out_dir / "manifest.csv")
    assert set(manifest["protein_id"]) == {"P_POS", "P_NEG"}
    assert int(manifest.loc[manifest["protein_id"] == "P_POS", "llps_label"].iloc[0]) == 1
    assert int(manifest.loc[manifest["protein_id"] == "P_NEG", "llps_label"].iloc[0]) == 0
    assert (out_dir / "proteins.csv").exists()
    assert (out_dir / "protein_labels.csv").exists()
    assert (out_dir / "regions.csv").exists()
    assert (out_dir / "evidence.csv").exists()
    assert (out_dir / "source_map.csv").exists()

    region_rows = [json.loads(line) for line in (out_dir / "regions.jsonl").read_text().splitlines()]
    assert region_rows[0]["regions"][0]["start"] == 1
    assert region_rows[0]["regions"][0]["end"] == 4
    assert (out_dir / "splits" / "train_ids.txt").exists()
    assert report["total_records"] == 2
    assert report["phase1_tables"]["proteins"] == 2
    assert report["phase1_tables"]["protein_labels"] == 2


def test_prepare_weak_dataset_accepts_additional_sources_and_cd_code_csv_links(tmp_path) -> None:
    ppmc = tmp_path / "datasets.tsv"
    pd.DataFrame(
        [
            {
                "UniProt.Acc": "P_POS",
                "Gene.Name": "POS",
                "Datasets": "D-;DE",
                "GO": "C",
                "Frac.Order": 0.1,
                "Frac.Disorder": 0.9,
                "Full.seq": "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
            },
            {
                "UniProt.Acc": "P_NEG",
                "Gene.Name": "NEG",
                "Datasets": "ND",
                "GO": "C",
                "Frac.Order": 0.1,
                "Frac.Disorder": 0.9,
                "Full.seq": "YYYYYYYYYYACDEFGHIKLMNPQRSTV",
            },
        ]
    ).to_csv(ppmc, sep="\t", index=False)

    phasepro = tmp_path / "phasepro.json"
    phasepro.write_text(
        json.dumps(
            {
                "P_POS": {
                    "accession": "P_POS",
                    "sequence": "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
                    "boundaries": "2-5",
                    "segment": "test segment",
                    "gene": "POS",
                }
            }
        )
    )

    phasepdb = tmp_path / "phasepdb.csv"
    pd.DataFrame(
        [
            {
                "uniprot_id": "QPHA",
                "class_": "PS-self",
                "primary_name": "PHA",
            }
        ]
    ).to_csv(phasepdb, index=False)

    llpsdb = tmp_path / "llpsdb_positive.csv"
    pd.DataFrame(
        [
            {
                "uniprot_id": "QLLP",
                "sequence_clean": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN",
                "source_subset": "Phase_separation_Unambiguous",
                "protein_level_use": "silver_positive_candidate",
                "gene_name": "LLP",
            }
        ]
    ).to_csv(llpsdb, index=False)

    cd_code_proteins = tmp_path / "cd_code_proteins.csv"
    pd.DataFrame(
        [
            {
                "uniprot_id": "QCD01",
                "gene_name": "CD",
                "protein_level_use": "bronze_silver_condensate_member",
            }
        ]
    ).to_csv(cd_code_proteins, index=False)

    cd_code_links = tmp_path / "cd_code_links.csv"
    pd.DataFrame(
        [
            {
                "uniprotkb_ac": "QCD01",
                "condensate_id": "C1",
                "condensate_name": "Test condensate",
            }
        ]
    ).to_csv(cd_code_links, index=False)

    uniprot_fasta = tmp_path / "uniprot.fasta"
    uniprot_fasta.write_text(
        ">sp|P_POS|POS_HUMAN\n"
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIK\n"
        ">sp|P_NEG|NEG_HUMAN\n"
        "YYYYYYYYYYACDEFGHIKLMNPQRSTV\n"
        ">sp|QPHA|PHA_HUMAN\n"
        "MSTNPKPQRITAYYQQQGGGGGGGGGGGG\n"
        ">sp|QLLP|LLP_HUMAN\n"
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN\n"
        ">sp|QCD01|CD_HUMAN\n"
        "MSTNPKPQRITAYYQQQGGGGGGGGGGAA\n"
    )

    out_dir = tmp_path / "expanded"
    prepare_weak_dataset(
        ppmc,
        phasepro,
        out_dir,
        llpsdb_positive_csv=llpsdb,
        phasepdb_csv=phasepdb,
        cd_code_proteins_csv=cd_code_proteins,
        cd_code_links_csv=cd_code_links,
        uniprot_fasta=uniprot_fasta,
        max_records=None,
        seed=1,
    )

    manifest = pd.read_csv(out_dir / "manifest.csv")
    assert set(manifest["protein_id"]) == {"P_POS", "P_NEG", "QPHA", "QLLP", "QCD01"}
    assert manifest.loc[manifest["protein_id"] == "QPHA", "source"].iloc[0] == "PhaSepDB-3"
    assert manifest.loc[manifest["protein_id"] == "QLLP", "source"].iloc[0] == "LLPSDB-v2"
    assert manifest.loc[manifest["protein_id"] == "QCD01", "source"].iloc[0] == "CD-CODE"
    assert float(manifest.loc[manifest["protein_id"] == "QCD01", "sample_weight"].iloc[0]) == 0.22
    assert (out_dir / "proteins.csv").exists()
    assert (out_dir / "protein_labels.csv").exists()
    assert (out_dir / "regions.csv").exists()
    assert (out_dir / "evidence.csv").exists()
    assert (out_dir / "source_map.csv").exists()



# Source: test_protenix_pipeline.py

import json
from pathlib import Path

import numpy as np

from scripts.protein.workflows.structure_tools import write_protenix_input_json
from phaseflow.protein import parse_protenix_outputs


def test_write_protenix_input_json(tmp_path) -> None:
    path = write_protenix_input_json("p1", "ACDEFG", tmp_path, model_seeds=[101, 102])
    payload = json.loads(path.read_text())
    assert isinstance(payload, list)
    assert payload[0]["name"] == "p1"
    assert payload[0]["modelSeeds"] == [101, 102]
    assert payload[0]["sequences"][0]["proteinChain"]["sequence"] == "ACDEFG"


def test_parse_protenix_outputs_to_structure_npz(tmp_path) -> None:
    pred_dir = tmp_path / "output" / "p1" / "seed_101" / "predictions"
    pred_dir.mkdir(parents=True)
    _write_minimal_cif(pred_dir / "p1_sample_0.cif", "p1", "ACDEFG")
    (pred_dir / "p1_summary_confidence_sample_0.json").write_text(
        json.dumps({"plddt": 85.0, "gpde": 2.0, "ptm": 0.7, "ranking_score": 0.6, "has_clash": False})
    )

    written = parse_protenix_outputs(
        records=[("p1", "ACDEFG")],
        protenix_output=tmp_path / "output",
        out_dir=tmp_path / "features",
        contact_topk=2,
        contact_cutoff=8.0,
    )

    assert written == [tmp_path / "features" / "p1.npz"]
    with np.load(written[0], allow_pickle=False) as data:
        assert data["node"].shape == (6, 12)
        assert data["reliability"].shape == (6,)
        assert str(data["structure_provider"].item()) == "protenix"
        assert data["contacts"].shape[1] == 4


def _write_minimal_cif(path: Path, protein_id: str, sequence: str) -> None:
    three = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
    }
    lines = [
        f"data_{protein_id}",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_PDB_model_num",
    ]
    for index, aa in enumerate(sequence, start=1):
        lines.append(
            f"ATOM {index} C CA {three[aa]} A {index} {float(index * 3):.3f} 0.000 0.000 1.00 85.00 1"
        )
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")



# Source: test_region_targets.py

from argparse import Namespace

import h5py
import numpy as np
import pytest

from scripts.protein.workflows.region_targets import build_targets_for_group, parse_args


def test_region_target_builder_requires_explicit_teacher_scores() -> None:
    with pytest.raises(SystemExit):
        parse_args([])


def _args(use_phaseflow: bool) -> Namespace:
    return Namespace(
        policy="stratified",
        use_phaseflow=use_phaseflow,
        phaseflow_pos=0.62,
        phasemotif_pos=0.70,
        pstp_pos=0.75,
        catgranule_pos=0.70,
        psphunter_key_pos=0.65,
        consensus_pos=0.55,
        confidence_pos=0.55,
        consensus_neg=0.25,
        phaseflow_neg=0.20,
        phasemotif_neg=0.15,
        pstp_neg=0.20,
        catgranule_neg=0.58,
        psphunter_neg=0.05,
        disorder_hard_neg=0.55,
        min_pos_len=3,
        min_neg_len=3,
        merge_gap=0,
        boundary_radius=1,
    )


def test_region_targets_ignore_phaseflow_by_default(tmp_path) -> None:
    path = tmp_path / "teachers.h5"
    length = 8
    with h5py.File(path, "w") as handle:
        group = handle.create_group("p1")
        group.create_dataset("teacher_consensus", data=np.full(length, 0.9, dtype=np.float32))
        group.create_dataset("teacher_confidence", data=np.full(length, 0.9, dtype=np.float32))
        group.create_dataset("phaseflow_score", data=np.full(length, 0.95, dtype=np.float32))
        group.create_dataset("phasemotif_score", data=np.full(length, 0.05, dtype=np.float32))
        group.create_dataset("pstp_scan_score", data=np.full(length, 0.05, dtype=np.float32))
        group.create_dataset("catgranule_score", data=np.full(length, 0.05, dtype=np.float32))
        group.create_dataset("psphunter_key_score", data=np.full(length, 0.05, dtype=np.float32))

    with h5py.File(path, "r") as handle:
        no_phaseflow = build_targets_for_group("p1", handle["p1"], [], _args(False))
        with_phaseflow = build_targets_for_group("p1", handle["p1"], [], _args(True))

    assert int(np.nansum(no_phaseflow["region_teacher_target"] == 1.0)) == 0
    assert int(np.nansum(with_phaseflow["region_teacher_target"] == 1.0)) == length


def test_pstp_scan_policy_uses_only_pstp_profile(tmp_path) -> None:
    path = tmp_path / "teachers.h5"
    length = 10
    with h5py.File(path, "w") as handle:
        group = handle.create_group("p1")
        group.create_dataset("phaseflow_score", data=np.full(length, 0.95, dtype=np.float32))
        group.create_dataset("phasemotif_score", data=np.full(length, 0.95, dtype=np.float32))
        group.create_dataset("catgranule_score", data=np.full(length, 0.95, dtype=np.float32))
        group.create_dataset("psphunter_key_score", data=np.full(length, 0.95, dtype=np.float32))
        group.create_dataset("pstp_scan_score", data=np.full(length, 0.05, dtype=np.float32))

    args = _args(False)
    args.policy = "pstp_scan"
    with h5py.File(path, "r") as handle:
        low_pstp = build_targets_for_group("p1", handle["p1"], [], args)

    assert len(low_pstp["positive_spans"]) == 0
    assert int(np.sum(np.asarray(low_pstp["region_teacher_target"]) >= 0.5)) == 0

    with h5py.File(path, "a") as handle:
        del handle["p1"]["pstp_scan_score"]
        handle["p1"].create_dataset("pstp_scan_score", data=np.full(length, 0.9, dtype=np.float32))

    with h5py.File(path, "r") as handle:
        high_pstp = build_targets_for_group("p1", handle["p1"], [], args)

    assert len(high_pstp["positive_spans"]) == 1
    assert int(np.sum(np.asarray(high_pstp["region_teacher_target"]) >= 0.5)) == length



# Source: test_sparse_graph_transformer.py

import torch

from phaseflow.protein import SparseGraphTransformer


def test_sparse_graph_transformer_empty_neighbor_safe() -> None:
    model = SparseGraphTransformer(d_model=16, num_layers=1, num_heads=4, edge_dim=8, ffn_dim=32)
    x = torch.randn(2, 5, 16)
    neighbors = torch.zeros(2, 5, 3, dtype=torch.long)
    edge_attr = torch.zeros(2, 5, 3, 8)
    neighbor_mask = torch.zeros(2, 5, 3, dtype=torch.bool)
    neighbor_mask[:, :, 0] = True
    seq_mask = torch.ones(2, 5, dtype=torch.bool)
    out = model(x, neighbors, edge_attr, neighbor_mask, seq_mask)
    assert out.shape == x.shape
    assert not out.isnan().any()



# Source: test_starling_features.py

import numpy as np

from phaseflow.protein import (
    assemble_starling_segments,
    candidate_starling_segments,
    starling_features_from_distance_maps,
)


def test_starling_distance_maps_become_node_and_contacts() -> None:
    maps = np.stack(
        [
            np.asarray(
                [
                    [0.0, 4.0, 12.0],
                    [4.0, 0.0, 5.0],
                    [12.0, 5.0, 0.0],
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    [0.0, 6.0, 14.0],
                    [6.0, 0.0, 6.0],
                    [14.0, 6.0, 0.0],
                ],
                dtype=np.float32,
            ),
        ]
    )
    node, missing, reliability, contacts = starling_features_from_distance_maps(
        maps,
        "ACD",
        contact_threshold=11.0,
        contact_topk=2,
    )
    assert node.shape == (3, 8)
    assert missing.sum() == 0.0
    assert reliability.min() > 0.0
    assert contacts.shape[1] == 5


def test_long_sequence_starling_segments_map_to_whole_sequence() -> None:
    sequence = "A" * 400 + "GPGPGPGPGPGPGPGPGPGPGPGPGPGPGP" + "A" * 400
    segments = candidate_starling_segments("p1", sequence, max_segment_length=64, min_segment_length=16)
    assert segments
    segment = segments[0]
    segment_node = np.ones((len(segment.sequence), 8), dtype=np.float32)
    segment_missing = np.zeros(len(segment.sequence), dtype=np.float32)
    segment_reliability = np.ones(len(segment.sequence), dtype=np.float32)
    segment_contacts = np.asarray([[0, 1, 0.8, 5.0]], dtype=np.float32)
    node, missing, reliability, contacts = assemble_starling_segments(
        len(sequence),
        [(segment, segment_node, segment_missing, segment_reliability, segment_contacts)],
    )
    assert node.shape == (len(sequence), 8)
    assert missing[segment.start : segment.end].sum() == 0.0
    assert reliability[segment.start : segment.end].min() == 1.0
    assert contacts[0, 0] == segment.start



# Source: test_teacher_pseudo_labels.py

import json

import h5py
import pandas as pd

from scripts.protein.workflows.data_tools import build_teacher_pseudo_labels


def test_teacher_pseudo_labels_are_train_only_and_write_soft_profiles(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "protein_id": "P_TRAIN",
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "llps_label": -100,
                "sample_weight": 0.0,
            },
            {
                "protein_id": "P_VALID",
                "sequence": "YYYYYYYYYYACDEFGHIKLM",
                "llps_label": -100,
                "sample_weight": 0.0,
            },
        ]
    ).to_csv(manifest, index=False)

    regions = tmp_path / "regions.jsonl"
    regions.write_text(
        json.dumps(
            {
                "protein_id": "P_VALID",
                "regions": [
                    {
                        "start": 1,
                        "end": 4,
                        "region_type": "DPR_gold",
                        "region_label": "gold",
                        "confidence": 1.0,
                    }
                ],
            }
        )
        + "\n"
    )
    train_ids = tmp_path / "train_ids.txt"
    train_ids.write_text("P_TRAIN\n")

    out_dir = tmp_path / "teacher"
    raw_dir = out_dir / "raw"
    (raw_dir / "deephase").mkdir(parents=True)
    (raw_dir / "pscore").mkdir(parents=True)
    (raw_dir / "phasemotif").mkdir(parents=True)
    (raw_dir / "phaseflow").mkdir(parents=True)

    pd.DataFrame(
        [
            {"name": "P_TRAIN", "deephase_score": 0.95},
            {"name": "P_VALID", "deephase_score": 0.99},
        ]
    ).to_csv(raw_dir / "deephase" / "deephase.tsv", sep="\t", index=False)
    (raw_dir / "pscore" / "pscore.txt").write_text("PScore: 5.2 >P_TRAIN\nPScore: 5.5 >P_VALID\n")
    pd.DataFrame(
        [
            {"IDR Name": "P_TRAIN|start=2|end=12", "IDR": "CDEFGHIKLMN", "Predict Score": 0.9},
            {"IDR Name": "P_VALID|start=3|end=14", "IDR": "YYYYYYYYYYAC", "Predict Score": 0.9},
        ]
    ).to_csv(raw_dir / "phasemotif" / "phasemotif.csv", index=False)
    phaseflow_scores = [0.1] * 20
    phaseflow_scores[1:12] = [0.85] * 11
    (raw_dir / "phaseflow" / "phaseflow.jsonl").write_text(
        json.dumps({"record_id": "P_TRAIN", "length": 20, "score": phaseflow_scores}) + "\n"
        + json.dumps({"record_id": "P_VALID", "length": 20, "score": [0.9] * 20}) + "\n"
    )

    config = {
        "paths": {
            "manifest": str(manifest),
            "regions": str(regions),
            "train_ids_file": str(train_ids),
            "out_dir": str(out_dir),
        },
        "consensus": {
            "min_protein_teachers": 2,
            "min_protein_confidence": 0.6,
            "min_region_teachers": 1,
            "min_region_confidence": 0.6,
            "min_region_len": 8,
        },
        "predictors": {
            "deephase": {
                "enabled": True,
                "output": "{raw_dir}/deephase/deephase.tsv",
                "parser": "deephase_tsv",
                "threshold": 0.5,
                "direction": "high",
                "weight": 1.0,
            },
            "pscore": {
                "enabled": True,
                "output": "{raw_dir}/pscore/pscore.txt",
                "parser": "pscore_text",
                "threshold": 4.0,
                "direction": "high",
                "weight": 1.0,
            },
            "phasemotif": {
                "enabled": True,
                "output": "{raw_dir}/phasemotif/phasemotif.csv",
                "parser": "phasemotif_csv",
                "threshold": 0.5,
                "direction": "high",
                "weight": 1.0,
            },
            "phaseflow": {
                "enabled": True,
                "output": "{raw_dir}/phaseflow/phaseflow.jsonl",
                "parser": "phaseflow_window_jsonl",
                "threshold": 0.7,
                "direction": "high",
                "weight": 0.45,
                "min_region_len": 8,
                "merge_gap": 0,
            },
        },
    }

    report = build_teacher_pseudo_labels(config, run_predictors=False)

    assert report["protein_pseudo_labels"] == 1
    assert report["pseudo_regions"] == 1
    updated = pd.read_csv(out_dir / "manifest_with_teacher.csv")
    assert int(updated.loc[updated["protein_id"] == "P_TRAIN", "llps_label"].iloc[0]) == 1
    assert int(updated.loc[updated["protein_id"] == "P_VALID", "llps_label"].iloc[0]) == -100

    assert not (out_dir / "merged_regions.jsonl").exists()
    candidates = [json.loads(line) for line in (out_dir / "teacher_region_candidates.jsonl").read_text().splitlines()]
    assert {row["protein_id"] for row in candidates} == {"P_TRAIN"}
    with h5py.File(out_dir / "teacher_scores.h5", "r") as handle:
        assert "P_TRAIN" in handle
        assert "teacher_consensus" in handle["P_TRAIN"]
        assert "phaseflow_score" in handle["P_TRAIN"]
        assert handle["P_TRAIN"]["phaseflow_score"].shape == (20,)
        assert "P_VALID" not in handle



# Source: test_split_manifest.py


import pandas as pd

from phaseflow.protein import resolve_split_ids


def test_split_ids_accept_release_parquet_manifest(tmp_path) -> None:
    manifest = tmp_path / "training_units.parquet"
    pd.DataFrame(
        [
            {"protein_id": "train_protein", "split": "train"},
            {"protein_id": "validation_protein", "split": "val"},
        ]
    ).to_parquet(manifest, index=False)

    assert resolve_split_ids({"manifest": str(manifest)}, "train") == ["train_protein"]
    assert resolve_split_ids({"manifest": str(manifest)}, "valid") == ["validation_protein"]
