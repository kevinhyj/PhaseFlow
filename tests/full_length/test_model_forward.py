from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phaseflow.full_length.data.collator import PhaseFlowCollator
from phaseflow.full_length.data.dataset import PhaseFlowDataset
from phaseflow.full_length.features.build_features import build_feature_cache
from phaseflow.full_length.features.bio_vec import BIO_VEC_NAMES
from phaseflow.full_length.losses.multitask import compute_multitask_loss
from phaseflow.full_length.models.phaseflow import PhaseFlowModel


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "full_length"


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
