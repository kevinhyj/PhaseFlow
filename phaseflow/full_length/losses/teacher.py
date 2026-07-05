from __future__ import annotations

import torch
import torch.nn.functional as F

from phaseflow.full_length.data.schemas import IGNORE_INDEX


def soft_bce_with_logits(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    logits = logits.float()
    soft_targets = soft_targets.to(device=logits.device, dtype=torch.float32)
    weight = weight.to(device=logits.device, dtype=torch.float32)
    valid = torch.isfinite(soft_targets) & (weight > 0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    target_clean = torch.where(valid, soft_targets.clamp(0.0, 1.0), torch.zeros_like(soft_targets))
    loss = F.binary_cross_entropy_with_logits(logits, target_clean, reduction="none")
    item_weight = weight.clamp(min=0.0) * valid.float()
    return (loss * item_weight).sum() / item_weight.sum().clamp_min(1.0)


def nnpu_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    positive_prior: float = 0.1,
    beta: float = 0.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.to(device=logits.device)
    labels = targets.float()
    positive = labels == 1.0
    unlabeled = labels < 0.0
    if not torch.any(positive) or not torch.any(unlabeled):
        return logits.sum() * 0.0
    weights = torch.ones_like(labels) if sample_weight is None else sample_weight.to(device=logits.device).float().clamp(min=0.0)
    pos_risk = _weighted_bce(logits[positive], torch.ones_like(logits[positive]), weights[positive])
    pos_as_neg = _weighted_bce(logits[positive], torch.zeros_like(logits[positive]), weights[positive])
    unlabeled_risk = _weighted_bce(logits[unlabeled], torch.zeros_like(logits[unlabeled]), torch.ones_like(logits[unlabeled]))
    negative_risk = unlabeled_risk - positive_prior * pos_as_neg
    if negative_risk < -beta:
        return -gamma * negative_risk
    return positive_prior * pos_risk + negative_risk


def calibration_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> torch.Tensor:
    logits = logits.float()
    targets = targets.to(device=logits.device)
    valid = (targets == 0.0) | (targets == 1.0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    probs = torch.sigmoid(logits[valid])
    labels = targets[valid].float()
    loss = logits.sum() * 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        in_bin = (probs >= lower) & (probs < upper if index < bins - 1 else probs <= upper)
        if torch.any(in_bin):
            loss = loss + torch.abs(probs[in_bin].mean() - labels[in_bin].mean()) * in_bin.float().mean()
    return loss


def weak_region_mil_loss(
    residue_logits: torch.Tensor,
    teacher_dpr: torch.Tensor,
    teacher_weight: torch.Tensor,
    seq_mask: torch.Tensor,
) -> torch.Tensor:
    residue_logits = residue_logits.float()
    teacher_dpr = teacher_dpr.to(device=residue_logits.device, dtype=torch.float32)
    teacher_weight = teacher_weight.to(device=residue_logits.device, dtype=torch.float32)
    seq_mask = seq_mask.to(device=residue_logits.device).bool()
    valid = torch.isfinite(teacher_dpr) & (teacher_weight > 0) & seq_mask
    if not torch.any(valid):
        return residue_logits.sum() * 0.0
    target_clean = torch.where(valid, teacher_dpr.clamp(0.0, 1.0), torch.zeros_like(teacher_dpr))
    loss = F.binary_cross_entropy_with_logits(residue_logits, target_clean, reduction="none")
    item_weight = teacher_weight.clamp(min=0.0) * valid.float()
    return (loss * item_weight).sum() / item_weight.sum().clamp_min(1.0)


def region_bag_mil_loss(
    region_global_logits: torch.Tensor,
    bag_label: torch.Tensor,
    bag_weight: torch.Tensor,
) -> torch.Tensor:
    region_global_logits = region_global_logits.float()
    bag_label = bag_label.to(device=region_global_logits.device, dtype=torch.float32)
    bag_weight = bag_weight.to(device=region_global_logits.device, dtype=torch.float32)
    valid = (bag_label == 0.0) | (bag_label == 1.0)
    valid = valid & (bag_weight > 0)
    if not torch.any(valid):
        return region_global_logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(region_global_logits, bag_label.clamp(0.0, 1.0), reduction="none")
    item_weight = bag_weight.clamp(min=0.0) * valid.float()
    return (loss * item_weight).sum() / item_weight.sum().clamp_min(1.0)


def teacher_distillation_mse(
    residue_logits: torch.Tensor,
    teacher_dpr: torch.Tensor,
    teacher_weight: torch.Tensor,
    seq_mask: torch.Tensor,
) -> torch.Tensor:
    residue_logits = residue_logits.float()
    teacher_dpr = teacher_dpr.to(device=residue_logits.device, dtype=torch.float32)
    teacher_weight = teacher_weight.to(device=residue_logits.device, dtype=torch.float32)
    seq_mask = seq_mask.to(device=residue_logits.device).bool()
    valid = torch.isfinite(teacher_dpr) & (teacher_weight > 0) & seq_mask
    if not torch.any(valid):
        return residue_logits.sum() * 0.0
    probs = torch.sigmoid(residue_logits[valid])
    loss = torch.square(probs - teacher_dpr[valid].float())
    weight = teacher_weight[valid].float()
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def negative_region_regularization(
    residue_logits: torch.Tensor,
    seq_mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    residue_logits = residue_logits.float()
    seq_mask = seq_mask.to(device=residue_logits.device).bool()
    sample_weight = sample_weight.to(device=residue_logits.device, dtype=torch.float32)
    valid_samples = sample_weight > 0
    if not torch.any(valid_samples):
        return residue_logits.sum() * 0.0
    probs = torch.sigmoid(residue_logits)
    per_sample = []
    weights = []
    for index in torch.nonzero(valid_samples, as_tuple=False).flatten():
        mask = seq_mask[index].bool()
        if torch.any(mask):
            per_sample.append(probs[index][mask].mean())
            weights.append(sample_weight[index].float())
    if not per_sample:
        return residue_logits.sum() * 0.0
    values = torch.stack(per_sample)
    weight = torch.stack(weights)
    return (values * weight).sum() / weight.sum().clamp(min=1.0e-6)


def _weighted_bce(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    targets = targets.to(device=logits.device, dtype=torch.float32)
    weight = weight.to(device=logits.device, dtype=torch.float32).clamp(min=0.0)
    if logits.numel() == 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)
