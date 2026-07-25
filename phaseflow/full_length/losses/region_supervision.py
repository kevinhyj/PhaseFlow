from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_soft_bce_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    seq_mask: torch.Tensor,
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.to(device=logits.device, dtype=torch.float32)
    weight = weight.to(device=logits.device, dtype=torch.float32)
    seq_mask = seq_mask.to(device=logits.device).bool()
    valid = torch.isfinite(targets) & (weight > 0) & seq_mask
    if not torch.any(valid):
        return logits.sum() * 0.0
    target_clean = torch.where(valid, targets.clamp(0.0, 1.0), torch.zeros_like(targets))
    loss = F.binary_cross_entropy_with_logits(logits, target_clean, reduction="none")
    item_weight = weight.clamp(min=0.0) * valid.float()
    return (loss * item_weight).sum() / item_weight.sum().clamp_min(1.0)


def boundary_transition_loss(
    residue_logits: torch.Tensor,
    boundary_target: torch.Tensor,
    boundary_weight: torch.Tensor,
    seq_mask: torch.Tensor,
) -> torch.Tensor:
    residue_logits = residue_logits.float()
    boundary_target = boundary_target.to(device=residue_logits.device, dtype=torch.float32)
    boundary_weight = boundary_weight.to(device=residue_logits.device, dtype=torch.float32)
    seq_mask = seq_mask.to(device=residue_logits.device).bool()
    valid = torch.isfinite(boundary_target) & (boundary_weight > 0) & seq_mask
    if not torch.any(valid):
        return residue_logits.sum() * 0.0
    probs = torch.sigmoid(residue_logits)
    left = torch.zeros_like(probs)
    right = torch.zeros_like(probs)
    left[:, 1:] = torch.abs(probs[:, 1:] - probs[:, :-1])
    right[:, :-1] = torch.abs(probs[:, 1:] - probs[:, :-1])
    transition = torch.maximum(left, right).clamp(min=1.0e-6, max=1.0 - 1.0e-6)
    target = boundary_target[valid].float()
    prob = transition[valid].float()
    loss = -(target * torch.log(prob) + (1.0 - target) * torch.log1p(-prob))
    item_weight = boundary_weight[valid].float().clamp(min=0.0)
    return (loss * item_weight).sum() / item_weight.sum().clamp_min(1.0)


def residue_contrastive_margin_loss(
    residue_logits: torch.Tensor,
    contrast_target: torch.Tensor,
    contrast_weight: torch.Tensor,
    seq_mask: torch.Tensor,
    *,
    margin: float = 0.35,
) -> torch.Tensor:
    residue_logits = residue_logits.float()
    contrast_target = contrast_target.to(device=residue_logits.device, dtype=torch.float32)
    contrast_weight = contrast_weight.to(device=residue_logits.device, dtype=torch.float32)
    seq_mask = seq_mask.to(device=residue_logits.device).bool()
    valid = torch.isfinite(contrast_target) & (contrast_weight > 0) & seq_mask
    positive = valid & (contrast_target >= 0.5)
    negative = valid & (contrast_target < 0.5)
    if not torch.any(positive) or not torch.any(negative):
        return residue_logits.sum() * 0.0
    probs = torch.sigmoid(residue_logits)
    pos_weight = contrast_weight[positive].float()
    neg_weight = contrast_weight[negative].float()
    pos_score = (probs[positive] * pos_weight).sum() / pos_weight.sum().clamp_min(1.0)
    neg_score = (probs[negative] * neg_weight).sum() / neg_weight.sum().clamp_min(1.0)
    return F.softplus(torch.as_tensor(float(margin), device=residue_logits.device) - pos_score + neg_score)
