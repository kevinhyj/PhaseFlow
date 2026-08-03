"""Workflow-level regression checks for the protein release."""



# Source: test_losses.py

import torch

from phaseflow.protein import dice_loss_with_logits
from phaseflow.protein import focal_loss_with_logits
from phaseflow.protein import (
    boundary_transition_loss,
    residue_contrastive_margin_loss,
    weighted_soft_bce_logits,
)
from phaseflow.protein import masked_bce_with_logits
from phaseflow.protein import phase_diagram_loss
from phaseflow.protein import region_query_loss
from phaseflow.protein import (
    calibration_loss_with_logits,
    negative_region_regularization,
    nnpu_loss_with_logits,
    region_bag_mil_loss,
    soft_bce_with_logits,
    teacher_distillation_mse,
    weak_region_mil_loss,
)


def test_losses_finite() -> None:
    logits = torch.randn(2, 4)
    targets = torch.tensor([[1, 0, -100, 1], [0, 0, 1, -100]])
    seq_mask = torch.ones(2, 4, dtype=torch.bool)
    assert torch.isfinite(masked_bce_with_logits(logits, targets, seq_mask))
    assert torch.isfinite(focal_loss_with_logits(logits, targets, seq_mask))
    assert torch.isfinite(dice_loss_with_logits(logits, targets, seq_mask))
    assert torch.isfinite(soft_bce_with_logits(logits, torch.rand(2, 4), torch.ones(2, 4)))
    assert torch.isfinite(weak_region_mil_loss(logits, torch.rand(2, 4), torch.ones(2, 4), seq_mask))
    assert torch.isfinite(teacher_distillation_mse(logits, torch.rand(2, 4), torch.ones(2, 4), seq_mask))
    assert torch.isfinite(region_bag_mil_loss(torch.randn(2), torch.tensor([1.0, 0.0]), torch.ones(2)))
    assert torch.isfinite(negative_region_regularization(logits, seq_mask, torch.tensor([0.2, 0.4])))


def test_teacher_losses_finite() -> None:
    logits = torch.randn(4)
    targets = torch.tensor([1.0, 0.0, -100.0, -100.0])
    weights = torch.ones(4)
    assert torch.isfinite(nnpu_loss_with_logits(logits, targets, weights))
    assert torch.isfinite(calibration_loss_with_logits(logits, targets))


def test_region_loss_finite() -> None:
    loss = region_query_loss(
        torch.randn(1, 3),
        torch.rand(1, 3),
        torch.rand(1, 3),
        [[{"start": 1, "end": 4, "type": "DPR_candidate"}]],
        torch.tensor([10]),
    )
    assert torch.isfinite(loss)


def test_phase_diagram_loss_finite() -> None:
    outputs = {"phase_values": torch.randn(2, 16)}
    batch = {
        "phase_values": torch.zeros(2, 16),
        "phase_mask": torch.tensor([[1.0, 1.0] + [0.0] * 14, [0.0] * 16]),
        "phase_aux_weight": torch.tensor([0.75, 0.0]),
        "phase_low_pssi": torch.tensor([0.5, 0.0]),
    }
    loss = phase_diagram_loss(outputs, batch)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_final_region_losses_finite() -> None:
    logits = torch.randn(2, 6)
    key_logits = torch.randn(2, 6)
    seq_mask = torch.ones(2, 6, dtype=torch.bool)
    target = torch.tensor([[1.0, 1.0, 0.0, 0.0, float("nan"), float("nan")], [float("nan")] * 6])
    weight = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.0, 0.0], [0.0] * 6])
    boundary_target = torch.tensor([[1.0, 0.0, 0.0, 1.0, float("nan"), float("nan")], [float("nan")] * 6])
    boundary_weight = torch.tensor([[1.0, 0.25, 0.25, 1.0, 0.0, 0.0], [0.0] * 6])
    assert torch.isfinite(weighted_soft_bce_logits(logits, target, weight, seq_mask))
    assert torch.isfinite(weighted_soft_bce_logits(key_logits, target, weight, seq_mask))
    assert torch.isfinite(boundary_transition_loss(logits, boundary_target, boundary_weight, seq_mask))
    assert torch.isfinite(residue_contrastive_margin_loss(logits, target, weight, seq_mask))



# Source: test_metrics.py

import numpy as np

from phaseflow.protein import key_topk_metrics
from phaseflow.protein import binary_classification_metrics
from phaseflow.protein import boundary_f1, region_metrics
from phaseflow.protein import residue_binary_metrics


def test_metrics_do_not_crash() -> None:
    assert "auc" in binary_classification_metrics(np.array([0, 1]), np.array([0.2, 0.8]))
    assert "residue_dice" in residue_binary_metrics(np.array([0, 1, -100]), np.array([0.1, 0.9, 0.5]))
    assert "key_top2_precision" in key_topk_metrics(np.array([[0, 1, -100]]), np.array([[0.2, 0.8, 0.1]]), k=2)
    assert "region_iou@0.5_precision" in region_metrics(
        [[{"start": 1, "end": 4, "score": 0.9}]],
        [[{"start": 1, "end": 4, "type": "DPR_candidate"}]],
    )
    assert "boundary_f1" in boundary_f1(
        [[{"start": 1, "end": 4, "score": 0.9}]],
        [[{"start": 1, "end": 4, "region_type": "DPR_gold"}]],
    )



# Source: test_model_forward.py

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phaseflow.protein import PhaseFlowCollator
from phaseflow.protein import PhaseFlowDataset
from phaseflow.protein import build_feature_cache
from phaseflow.protein import BIO_VEC_NAMES
from phaseflow.protein import compute_multitask_loss
from phaseflow.protein import PhaseFlowModel


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "protein"


def _test_config() -> dict:
    return {
        "model": {
            "model_type": "v2_region",
            "d_model": 64,
            "dropout": 0.1,
            "input_dims": {
                "plm": 32,
                "physchem": 90,
                "disorder": 6,
                "protenix_embed": 512,
                "starling_embed": 512,
            },
            "local_encoder": {"num_layers": 1, "kernels": [3, 5, 9], "dilations": [2, 4]},
            "graph_transformer": {
                "num_layers": 1,
                "num_heads": 4,
                "ffn_dim": 128,
                "edge_dim": 13,
                "num_edge_types": 8,
                "relative_position_bins": 32,
            },
            "region_decoder": {"num_queries": 8, "num_layers": 1},
        }
    }


def _decoupled_test_config() -> dict:
    config = _test_config()
    model_config = config["model"]
    model_config["model_type"] = "v3_decoupled"
    model_config["llps_head"] = {"use_dpr_pooling": False}
    model_config["dpr_summary"] = {"enabled": True, "detach": True, "hidden_dim": 32, "residual_scale": 0.4}
    model_config["decoupled"] = {
        "shared_local_layers": 1,
        "branch_local_layers": 1,
        "shared_graph_layers": 1,
        "branch_graph_layers": 1,
    }
    return config


def test_model_forward(tmp_path) -> None:
    build_feature_cache(
        fasta=FIXTURE_DIR / "toy_sequences.fasta",
        protein_labels=FIXTURE_DIR / "toy_labels.tsv",
        regions=FIXTURE_DIR / "toy_regions.jsonl",
        out_dir=tmp_path,
    )
    dataset = PhaseFlowDataset(tmp_path, ["toy_pos_1", "toy_neg_1"])
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=PhaseFlowCollator(max_neighbors=16))))
    model = PhaseFlowModel(_test_config())
    outputs = model(batch)
    assert outputs["llps_logits"].shape == (2,)
    assert outputs["raw_llps_logits"].shape == (2,)
    assert outputs["loss_llps_logits"].shape == (2,)
    assert outputs["dpr_logits"].shape[:2] == batch["seq_mask"].shape
    assert outputs["region_global_logits"].shape == (2,)
    assert outputs["key_logits"].shape[:2] == batch["seq_mask"].shape
    assert outputs["region_logits"].shape[0] == 2
    assert outputs["region_width"].shape == outputs["region_start"].shape
    assert not outputs["dpr_logits"].isnan().any()


def test_decoupled_model_forward(tmp_path) -> None:
    build_feature_cache(
        fasta=FIXTURE_DIR / "toy_sequences.fasta",
        protein_labels=FIXTURE_DIR / "toy_labels.tsv",
        regions=FIXTURE_DIR / "toy_regions.jsonl",
        out_dir=tmp_path,
    )
    dataset = PhaseFlowDataset(tmp_path, ["toy_pos_1", "toy_neg_1"])
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=PhaseFlowCollator(max_neighbors=16))))
    model = PhaseFlowModel(_decoupled_test_config())
    outputs = model(batch)
    assert outputs["llps_logits"].shape == (2,)
    assert outputs["raw_llps_logits"].shape == (2,)
    assert outputs["llps_logits"].shape == (2,)
    assert outputs["loss_llps_logits"].shape == (2,)
    assert outputs["dpr_summary_features"].shape == (2, 6)
    assert outputs["llps_residue_repr"].shape == outputs["dpr_residue_repr"].shape
    assert outputs["dpr_logits"].shape[:2] == batch["seq_mask"].shape
    assert not outputs["llps_logits"].isnan().any()
    assert not outputs["dpr_summary_features"].isnan().any()


def test_decoupled_llps_loss_does_not_update_dpr_branch(tmp_path) -> None:
    build_feature_cache(
        fasta=FIXTURE_DIR / "toy_sequences.fasta",
        protein_labels=FIXTURE_DIR / "toy_labels.tsv",
        regions=FIXTURE_DIR / "toy_regions.jsonl",
        out_dir=tmp_path,
    )
    dataset = PhaseFlowDataset(tmp_path, ["toy_pos_1", "toy_neg_1"])
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=PhaseFlowCollator(max_neighbors=16))))
    model = PhaseFlowModel(_decoupled_test_config())
    outputs = model(batch)
    loss, _ = compute_multitask_loss(
        outputs,
        batch,
        {
            "llps": 1.0,
            "teacher_llps": 0.0,
            "self_llps": 0.0,
            "nnpu": 0.0,
            "region_mil": 0.0,
            "region_gold": 0.0,
            "teacher_dpr": 0.0,
            "teacher_distill": 0.0,
            "self_dpr": 0.0,
            "negative_regularization": 0.0,
            "region": 0.0,
            "coverage": 0.0,
            "key": 0.0,
            "smoothness": 0.0,
            "phase_aux": 0.0,
            "region_teacher": 0.0,
            "region_key_teacher": 0.0,
            "region_boundary": 0.0,
            "region_contrastive": 0.0,
        },
    )
    loss.backward()
    dpr_grad = _max_grad(
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("dpr_branch")
        or name.startswith("dpr_local_encoder")
        or name.startswith("dpr_encoder")
        or name.startswith("dpr_head")
        or name.startswith("region_decoder")
        or name.startswith("key_head")
    )
    llps_grad = _max_grad(
        parameter for name, parameter in model.named_parameters() if name.startswith("llps_") or name.startswith("llps_head")
    )
    assert dpr_grad == 0.0
    assert llps_grad > 0.0


def test_model_forward_with_phase_head(tmp_path) -> None:
    build_feature_cache(
        fasta=FIXTURE_DIR / "toy_sequences.fasta",
        protein_labels=FIXTURE_DIR / "toy_labels.tsv",
        regions=FIXTURE_DIR / "toy_regions.jsonl",
        out_dir=tmp_path,
    )
    dataset = PhaseFlowDataset(tmp_path, ["toy_pos_1", "toy_neg_1"])
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=PhaseFlowCollator(max_neighbors=16))))
    config = _test_config()
    config["model"]["phase_aux"] = {"enabled": True, "phase_dim": 16}
    model = PhaseFlowModel(config)
    outputs = model(batch)
    assert outputs["phase_values"].shape == (2, 16)
    assert not outputs["phase_values"].isnan().any()


def test_no_protenix_starling_ablation_masks_modalities_and_edges(tmp_path) -> None:
    build_feature_cache(
        fasta=FIXTURE_DIR / "toy_sequences.fasta",
        protein_labels=FIXTURE_DIR / "toy_labels.tsv",
        regions=FIXTURE_DIR / "toy_regions.jsonl",
        out_dir=tmp_path,
    )
    dataset = PhaseFlowDataset(tmp_path, ["toy_pos_1", "toy_neg_1"])
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=PhaseFlowCollator(max_neighbors=16))))
    config = _test_config()
    config["model"]["ablation"] = {"name": "no_protenix_starling"}
    model = PhaseFlowModel(config)
    modality_mask, reliability = model._apply_ablation(batch["modality_mask"], batch["reliability"])
    assert modality_mask[..., 3].all()
    assert modality_mask[..., 4].all()
    assert modality_mask[..., 4].all()
    assert not reliability[..., 3].any()
    assert not reliability[..., 4].any()

    edge_attr = batch["edge_attr"].clone()
    neighbor_mask = batch["neighbor_mask"].clone()
    edge_attr[..., 3:] = 0.0
    edge_attr[:, :, 0, 5] = 1.0
    filtered = model._apply_edge_ablation(edge_attr, neighbor_mask)
    assert not filtered[:, :, 0].any()


def test_modality_ablation_zeros_matching_bio_vec_features() -> None:
    config = _test_config()
    config["model"]["bio_mlp"] = {"enabled": True, "input_dim": len(BIO_VEC_NAMES), "hidden": [16], "dropout": 0.0}
    config["model"]["ablation"] = {
        "name": "llps_embedding_ablation",
        "disabled_modalities": ["plm", "protenix_embed", "starling_embed"],
        "disabled_bio_vec_groups": ["physchem"],
        "disabled_bio_vec_features": ["idr_fraction"],
    }
    model = PhaseFlowModel(config)
    disabled = {BIO_VEC_NAMES[index] for index in model.disabled_bio_vec_indices}
    assert {"esm_mean", "esm_std"}.issubset(disabled)
    assert "protenix_available" in disabled
    assert {"starling_mean_norm", "starling_std_norm", "starling_compaction_proxy"}.issubset(disabled)
    assert "hydropathy_mean" in disabled
    assert "idr_fraction" in disabled


def test_named_starling_ablation_still_zeros_starling_bio_vec_features() -> None:
    config = _test_config()
    config["model"]["bio_mlp"] = {"enabled": True, "input_dim": len(BIO_VEC_NAMES), "hidden": [16], "dropout": 0.0}
    config["model"]["ablation"] = {"name": "no_starling"}
    model = PhaseFlowModel(config)
    disabled = {BIO_VEC_NAMES[index] for index in model.disabled_bio_vec_indices}
    assert disabled == {"starling_mean_norm", "starling_std_norm", "starling_compaction_proxy"}


def _max_grad(parameters) -> float:
    value = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = max(value, float(parameter.grad.detach().abs().max().cpu()))
    return value



# Source: test_phaseflow_fusion.py

import numpy as np

from scripts.protein.workflows.evaluation import (
    PSSI_MAX,
    PSSI_MIN,
    PhaseFlowFusionConfig,
    fuse_llps_probability,
    fuse_phaseflow_with_phaseflow,
    gated_lift,
    parse_window_sizes,
    pssi_to_score,
    rank_blend,
    usable_window_sizes,
)


def test_pssi_to_score_inverts_phaseflow_direction() -> None:
    scores = pssi_to_score(np.asarray([PSSI_MIN, PSSI_MAX], dtype=np.float32))
    assert scores[0] > scores[1]
    assert np.allclose(scores, [1.0, 0.0])


def test_gated_lift_only_changes_borderline_phaseflow_residues() -> None:
    phaseflow = np.asarray([0.40, 0.62, 0.66, 0.72], dtype=np.float32)
    phaseflow_rank = np.asarray([1.00, 0.69, 0.80, 0.95], dtype=np.float32)
    fused = gated_lift(
        phaseflow_scores=phaseflow,
        phaseflow_rank=phaseflow_rank,
        phaseflow_low=0.60,
        phaseflow_high=0.68,
        phaseflow_rank_gate=0.70,
        lift=0.70,
        lift_span=0.05,
    )
    assert np.isclose(fused[0], phaseflow[0])
    assert np.isclose(fused[1], phaseflow[1])
    assert fused[2] > phaseflow[2]
    assert np.isclose(fused[3], phaseflow[3])


def test_llps_fusion_is_non_decreasing_and_respects_gate() -> None:
    config = PhaseFlowFusionConfig(llps_gate=0.72, llps_boost_scale=0.70, llps_max_phaseflow=0.80)
    unchanged = fuse_llps_probability(phaseflow_probability=0.45, phaseflow_proxy=0.60, config=config)
    boosted = fuse_llps_probability(phaseflow_probability=0.45, phaseflow_proxy=0.90, config=config)
    high_confidence = fuse_llps_probability(phaseflow_probability=0.85, phaseflow_proxy=0.90, config=config)
    assert np.isclose(unchanged, 0.45)
    assert boosted > 0.45
    assert np.isclose(high_confidence, 0.85)


def test_rank_blend_uses_phaseflow_local_rank_bidirectionally() -> None:
    phaseflow = np.asarray([0.80, 0.20], dtype=np.float32)
    phaseflow_rank = np.asarray([0.00, 1.00], dtype=np.float32)
    fused = rank_blend(phaseflow_scores=phaseflow, phaseflow_rank=phaseflow_rank, alpha=0.15)
    assert fused[0] < phaseflow[0]
    assert fused[1] > phaseflow[1]


def test_full_fusion_exports_phaseflow_profiles_and_metadata() -> None:
    config = PhaseFlowFusionConfig()
    phaseflow = np.asarray([0.61, 0.64, 0.66, 0.50], dtype=np.float32)
    phaseflow = np.asarray([0.10, 0.90, 0.95, 0.20], dtype=np.float32)
    result = fuse_phaseflow_with_phaseflow(
        phaseflow_dpr=phaseflow,
        phaseflow_llps_probability=0.40,
        phaseflow_scores=phaseflow,
        config=config,
        window_sizes=(10, 20),
    )
    assert result.dpr_scores.shape == phaseflow.shape
    assert result.phaseflow_scores.shape == phaseflow.shape
    assert result.phaseflow_rank.shape == phaseflow.shape
    assert result.window_sizes == (10, 20)
    assert result.lifted_residues >= 1
    assert result.suppressed_residues >= 1
    assert result.llps_probability >= 0.40


def test_window_size_parsing_and_short_sequence_fallback() -> None:
    assert parse_window_sizes("20,10,20") == (10, 20)
    assert usable_window_sizes("ACDEFGHIK", (10, 20), min_sequence_len=5) == (9,)



# Source: test_infer.py

import numpy as np

from scripts.protein.workflows.evaluation import _evidence_for_sample, _public_regions


def test_public_regions_are_one_based_inclusive() -> None:
    regions = [{"start": 0, "end": 4, "score": 0.9, "source": "postprocess"}]
    assert _public_regions(regions) == [{"start": 1, "end": 5, "score": 0.9, "source": "postprocess"}]


def test_evidence_reports_available_modalities() -> None:
    weights = np.asarray(
        [[0.1, 0.2, 0.1, 0.35, 0.05, 0.2], [0.1, 0.2, 0.1, 0.35, 0.05, 0.2]],
        dtype=np.float32,
    )
    mask = np.zeros_like(weights)
    evidence = _evidence_for_sample(weights, mask, {"structure_provider": "protenix", "structure_success": "1"})
    assert evidence["important_modalities"][0] == "protenix_embed"
    assert evidence["structure_provider"] == "protenix"
