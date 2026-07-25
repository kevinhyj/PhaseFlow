import torch

from phaseflow.full_length.losses.dice import dice_loss_with_logits
from phaseflow.full_length.losses.focal import focal_loss_with_logits
from phaseflow.full_length.losses.region_supervision import (
    boundary_transition_loss,
    residue_contrastive_margin_loss,
    weighted_soft_bce_logits,
)
from phaseflow.full_length.losses.masked_bce import masked_bce_with_logits
from phaseflow.full_length.losses.phase_aux import phase_diagram_loss
from phaseflow.full_length.losses.region_loss import region_query_loss
from phaseflow.full_length.losses.teacher import (
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
