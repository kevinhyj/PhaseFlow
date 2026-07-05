from phaseflow.full_length.data.collator import PhaseFlowCollator
from phaseflow.full_length.data.collator import _edge_list_to_neighbors
from phaseflow.full_length.data.feature_cache import FeatureCacheWriter
from phaseflow.full_length.data.dataset import PhaseFlowDataset
from phaseflow.full_length.data.offline_dataset import PhaseFlowOfflineDataset
from phaseflow.full_length.data.schemas import zero_record
from phaseflow.full_length.features.edge_builder import build_edges
from phaseflow.full_length.features.graph_cache import edge_list_to_precomputed_graph
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
