"""Protein training objectives and benchmark metrics."""


# Source: losses/consistency_loss.py


import torch


def smoothness_loss(logits: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] < 2:
        return logits.sum() * 0.0
    probs = torch.sigmoid(logits)
    valid = seq_mask[:, 1:] & seq_mask[:, :-1]
    diff = torch.abs(probs[:, 1:] - probs[:, :-1])
    if not torch.any(valid):
        return logits.sum() * 0.0
    return diff[valid].mean()


# Source: losses/dice.py


import torch

from phaseflow.protein.contracts import IGNORE_INDEX


def dice_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    seq_mask: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    valid = targets != ignore_index
    if seq_mask is not None:
        valid = valid & seq_mask.bool()
    if not torch.any(valid):
        return logits.sum() * 0.0
    probs = torch.sigmoid(logits)[valid]
    labels = targets.float().clamp(0, 1)[valid]
    intersection = torch.sum(probs * labels)
    denom = torch.sum(probs) + torch.sum(labels)
    return 1.0 - (2.0 * intersection + eps) / (denom + eps)


# Source: losses/focal.py


import torch
import torch.nn.functional as F

from phaseflow.protein.contracts import IGNORE_INDEX


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    seq_mask: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.to(device=logits.device)
    valid = targets != ignore_index
    if seq_mask is not None:
        valid = valid & seq_mask.to(device=logits.device).bool()
    if not torch.any(valid):
        return logits.sum() * 0.0
    target_float = targets.float().clamp(0, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target_float, reduction="none")
    p_t = torch.exp(-bce).clamp(1.0e-6, 1.0)
    alpha_t = alpha * target_float + (1.0 - alpha) * (1.0 - target_float)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    denom = valid.float()
    if weight is not None:
        item_weight = weight.to(device=logits.device, dtype=torch.float32).clamp(min=0.0)
        loss = loss * item_weight
        denom = denom * item_weight
    return (loss * valid.float()).sum() / denom.sum().clamp_min(1.0)


# Source: losses/masked_bce.py


import torch
import torch.nn.functional as F

from phaseflow.protein.contracts import IGNORE_INDEX


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    seq_mask: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.to(device=logits.device)
    valid = targets != ignore_index
    if seq_mask is not None:
        valid = valid & seq_mask.to(device=logits.device).bool()
    if not torch.any(valid):
        return logits.sum() * 0.0
    targets_float = targets.float().clamp(min=0.0, max=1.0)
    loss = F.binary_cross_entropy_with_logits(logits, targets_float, reduction="none")
    denom = valid.float()
    if weight is not None:
        item_weight = weight.to(device=logits.device, dtype=torch.float32).clamp(min=0.0)
        loss = loss * item_weight
        denom = denom * item_weight
    return (loss * valid.float()).sum() / denom.sum().clamp_min(1.0)


def protein_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    class_normalized: bool = False,
    alpha_pos: float = 0.5,
    alpha_neg: float = 0.5,
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.to(device=logits.device)
    valid = (targets == 0.0) | (targets == 1.0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits[valid], targets[valid].float(), reduction="none")
    if weight is not None:
        loss = loss * weight[valid].float()
    if class_normalized:
        valid_targets = targets[valid].float()
        pos = valid_targets == 1.0
        neg = valid_targets == 0.0
        if torch.any(pos) and torch.any(neg):
            return float(alpha_pos) * loss[pos].mean() + float(alpha_neg) * loss[neg].mean()
        if torch.any(pos):
            return loss[pos].mean()
        if torch.any(neg):
            return loss[neg].mean()
    return loss.mean()


def protein_bce_class_normalized_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    alpha_pos: float = 0.5,
    alpha_neg: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = logits.float()
    targets = targets.to(device=logits.device)
    valid = (targets == 0.0) | (targets == 1.0)
    if not torch.any(valid):
        zero = logits.sum() * 0.0
        return zero, {
            "protein_loss_pos": zero.detach(),
            "protein_loss_neg": zero.detach(),
            "protein_loss_pos_count": torch.zeros((), device=logits.device),
            "protein_loss_neg_count": torch.zeros((), device=logits.device),
            "protein_loss_missing_class": torch.ones((), device=logits.device),
        }
    valid_logits = logits[valid]
    valid_targets = targets[valid].float()
    valid_weight = torch.ones_like(valid_targets) if weight is None else weight[valid].float().clamp(min=0.0)
    raw = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, reduction="none")
    weighted = raw * valid_weight
    pos = valid_targets == 1.0
    neg = valid_targets == 0.0
    pos_count = pos.float().sum()
    neg_count = neg.float().sum()
    missing_class = ~(torch.any(pos) & torch.any(neg))
    if torch.any(pos):
        pos_loss = weighted[pos].sum() / valid_weight[pos].sum().clamp(min=1.0e-6)
    else:
        pos_loss = logits.sum() * 0.0
    if torch.any(neg):
        neg_loss = weighted[neg].sum() / valid_weight[neg].sum().clamp(min=1.0e-6)
    else:
        neg_loss = logits.sum() * 0.0
    if torch.any(pos) and torch.any(neg):
        total = float(alpha_pos) * pos_loss + float(alpha_neg) * neg_loss
    elif torch.any(pos):
        total = pos_loss
    else:
        total = neg_loss
    return total, {
        "protein_loss_pos": pos_loss.detach(),
        "protein_loss_neg": neg_loss.detach(),
        "protein_loss_pos_count": pos_count.detach(),
        "protein_loss_neg_count": neg_count.detach(),
        "protein_loss_missing_class": missing_class.float().detach(),
    }


# Source: losses/phase_aux.py


import torch
import torch.nn.functional as F


def phase_diagram_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    pssi_min: float = -2.17,
    pssi_max: float = 1.64,
    mean_weight: float = 0.25,
) -> torch.Tensor:
    pred = outputs.get("phase_values")
    if pred is None:
        return batch["phase_values"].sum() * 0.0
    target = batch["phase_values"].float()
    mask = batch["phase_mask"].float()
    sample_weight = batch["phase_aux_weight"].float().clamp(min=0.0)
    valid = (mask > 0) & torch.isfinite(target) & (sample_weight[:, None] > 0)
    if not torch.any(valid):
        return pred.sum() * 0.0

    value_loss = F.smooth_l1_loss(pred[valid], target[valid], reduction="none")
    value_weight = mask[valid] * sample_weight[:, None].expand_as(mask)[valid]
    value_term = (value_loss * value_weight).sum() / value_weight.sum().clamp(min=1.0e-6)

    low_target = batch.get("phase_low_pssi")
    if low_target is None:
        return value_term
    sample_valid = (mask.sum(dim=1) > 0) & torch.isfinite(low_target) & (sample_weight > 0)
    if not torch.any(sample_valid):
        return value_term
    masked_pred_mean = (pred * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    low_pred = ((float(pssi_max) - masked_pred_mean) / (float(pssi_max) - float(pssi_min))).clamp(0.0, 1.0)
    mean_loss = torch.square(low_pred[sample_valid] - low_target[sample_valid].float())
    mean_term = (mean_loss * sample_weight[sample_valid]).sum() / sample_weight[sample_valid].sum().clamp(min=1.0e-6)
    return value_term + float(mean_weight) * mean_term


# Source: losses/ranking_loss.py


import torch

from phaseflow.protein.contracts import IGNORE_INDEX


def pairwise_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    seq_mask: torch.Tensor | None = None,
    margin: float = 0.1,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    valid = targets != ignore_index
    if seq_mask is not None:
        valid = valid & seq_mask.bool()
    losses: list[torch.Tensor] = []
    for sample_logits, sample_targets, sample_valid in zip(logits, targets, valid, strict=False):
        pos = sample_logits[sample_valid & (sample_targets == 1)]
        neg = sample_logits[sample_valid & (sample_targets == 0)]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        losses.append(torch.relu(margin - pos.mean() + neg.mean()))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


# Source: losses/region_loss.py


import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def region_query_loss(
    region_logits: torch.Tensor,
    region_start: torch.Tensor,
    region_end: torch.Tensor,
    regions: list[list[dict[str, object]]],
    lengths: torch.Tensor,
    boundary_weight: float = 1.0,
    iou_weight: float = 1.0,
) -> torch.Tensor:
    region_logits = region_logits.float()
    region_start = region_start.to(device=region_logits.device, dtype=torch.float32)
    region_end = region_end.to(device=region_logits.device, dtype=torch.float32)
    lengths = lengths.to(device=region_logits.device)
    device = region_logits.device
    losses: list[torch.Tensor] = []
    for batch_index, sample_regions in enumerate(regions):
        length = max(int(lengths[batch_index].item()), 1)
        true_regions = _valid_regions(sample_regions, length)
        query_count = region_logits.shape[1]
        cls_target = torch.zeros(query_count, device=device)
        if not true_regions:
            losses.append(F.binary_cross_entropy_with_logits(region_logits[batch_index], cls_target))
            continue
        true_start = torch.tensor([float(region["start"]) / length for region in true_regions], device=device)
        true_end = torch.tensor([float(region["end"]) / length for region in true_regions], device=device)
        pred_start = region_start[batch_index]
        pred_end = region_end[batch_index]
        cost = _matching_cost(
            region_logits[batch_index].detach(),
            pred_start.detach(),
            pred_end.detach(),
            true_start.detach(),
            true_end.detach(),
        )
        pred_indices, true_indices = linear_sum_assignment(cost.cpu().numpy())
        cls_target[pred_indices] = 1.0
        cls_loss = F.binary_cross_entropy_with_logits(region_logits[batch_index], cls_target)
        matched_pred = torch.as_tensor(pred_indices, device=device, dtype=torch.long)
        matched_true = torch.as_tensor(true_indices, device=device, dtype=torch.long)
        boundary = (
            torch.abs(pred_start[matched_pred] - true_start[matched_true])
            + torch.abs(pred_end[matched_pred] - true_end[matched_true])
        ).mean()
        iou = region_loss_interval_iou(pred_start[matched_pred], pred_end[matched_pred], true_start[matched_true], true_end[matched_true])
        losses.append(cls_loss + boundary_weight * boundary + iou_weight * (1.0 - iou.mean()))
    if not losses:
        return region_logits.sum() * 0.0
    return torch.stack(losses).mean()


def region_coverage_loss(
    region_logits: torch.Tensor,
    region_start: torch.Tensor,
    region_end: torch.Tensor,
    regions: list[list[dict[str, object]]],
    lengths: torch.Tensor,
) -> torch.Tensor:
    region_logits = region_logits.float()
    region_start = region_start.to(device=region_logits.device, dtype=torch.float32)
    region_end = region_end.to(device=region_logits.device, dtype=torch.float32)
    lengths = lengths.to(device=region_logits.device)
    device = region_logits.device
    losses: list[torch.Tensor] = []
    probs = torch.sigmoid(region_logits)
    for batch_index, sample_regions in enumerate(regions):
        length = max(int(lengths[batch_index].item()), 1)
        true_regions = _valid_regions(sample_regions, length)
        if not true_regions:
            continue
        true_coverage = sum(max(0.0, float(region["end"]) - float(region["start"])) / length for region in true_regions)
        pred_width = torch.clamp(region_end[batch_index] - region_start[batch_index], min=0.0)
        pred_coverage = torch.sum(probs[batch_index] * pred_width)
        losses.append(torch.abs(pred_coverage - torch.as_tensor(true_coverage, device=device)))
    if not losses:
        return region_logits.sum() * 0.0
    return torch.stack(losses).mean()


def _matching_cost(
    logits: torch.Tensor,
    pred_start: torch.Tensor,
    pred_end: torch.Tensor,
    true_start: torch.Tensor,
    true_end: torch.Tensor,
) -> torch.Tensor:
    logits = logits.float()
    pred_start = pred_start.float()
    pred_end = pred_end.float()
    true_start = true_start.float()
    true_end = true_end.float()
    prob = torch.sigmoid(logits).unsqueeze(1)
    cls_cost = -prob.expand(-1, true_start.numel())
    boundary = torch.abs(pred_start.unsqueeze(1) - true_start.unsqueeze(0)) + torch.abs(pred_end.unsqueeze(1) - true_end.unsqueeze(0))
    iou = region_loss_interval_iou(pred_start.unsqueeze(1), pred_end.unsqueeze(1), true_start.unsqueeze(0), true_end.unsqueeze(0))
    return cls_cost + boundary + (1.0 - iou)


def region_loss_interval_iou(start_a: torch.Tensor, end_a: torch.Tensor, start_b: torch.Tensor, end_b: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    intersection = torch.clamp(torch.minimum(end_a, end_b) - torch.maximum(start_a, start_b), min=0.0)
    union = torch.clamp(torch.maximum(end_a, end_b) - torch.minimum(start_a, start_b), min=0.0)
    return intersection / (union + float(eps))


def _valid_regions(sample_regions: list[dict[str, object]], length: int) -> list[dict[str, object]]:
    valid: list[dict[str, object]] = []
    for region in sample_regions:
        if region.get("type") == "key_region":
            continue
        try:
            start = float(region["start"])
            end = float(region["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not torch.isfinite(torch.tensor([start, end])).all():
            continue
        start = max(0.0, min(start, float(length - 1)))
        end = max(start, min(end, float(length - 1)))
        valid.append({**region, "start": start, "end": end})
    return valid


# Source: losses/region_supervision.py


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


# Source: losses/teacher.py


import torch
import torch.nn.functional as F

from phaseflow.protein.contracts import IGNORE_INDEX


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


# Source: metrics/key_metrics.py


import numpy as np

from phaseflow.protein.contracts import IGNORE_INDEX


def key_topk_metrics(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> dict[str, float]:
    if isinstance(labels, np.ndarray) and labels.dtype != object:
        label_iter = list(labels)
    else:
        label_iter = list(labels)
    if isinstance(scores, np.ndarray) and scores.dtype != object:
        score_iter = list(scores)
    else:
        score_iter = list(scores)
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    for sample_labels, sample_scores in zip(label_iter, score_iter, strict=False):
        sample_labels = np.asarray(sample_labels)
        sample_scores = np.asarray(sample_scores)
        valid = sample_labels != IGNORE_INDEX
        if not np.any(valid):
            continue
        valid_labels = sample_labels[valid].astype(int)
        valid_scores = sample_scores[valid].astype(float)
        positives = int(np.sum(valid_labels == 1))
        if positives == 0:
            continue
        top_count = min(k, len(valid_scores))
        order = np.argsort(-valid_scores)[:top_count]
        hits = valid_labels[order]
        precisions.append(float(np.sum(hits == 1) / top_count))
        recalls.append(float(np.sum(hits == 1) / positives))
        gains = hits / np.log2(np.arange(top_count) + 2)
        ideal = np.ones(min(positives, top_count)) / np.log2(np.arange(min(positives, top_count)) + 2)
        ndcgs.append(float(np.sum(gains) / max(np.sum(ideal), 1.0e-8)))
    return {
        f"key_top{k}_precision": float(np.mean(precisions)) if precisions else np.nan,
        f"key_top{k}_recall": float(np.mean(recalls)) if recalls else np.nan,
        f"key_top{k}_ndcg": float(np.mean(ndcgs)) if ndcgs else np.nan,
    }


# Source: metrics/protein_metrics.py


import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


def binary_classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    valid = labels >= 0
    labels = labels[valid].astype(int)
    scores = scores[valid].astype(float)
    if labels.size == 0:
        return _nan_metrics()
    preds = (scores >= threshold).astype(int)
    tp = float(np.sum((preds == 1) & (labels == 1)))
    tn = float(np.sum((preds == 0) & (labels == 0)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))
    return {
        "auc": _protein_metrics_safe_metric(roc_auc_score, labels, scores),
        "prauc": _protein_metrics_safe_metric(average_precision_score, labels, scores),
        "f1": _protein_metrics_safe_metric(f1_score, labels, preds),
        "mcc": _protein_metrics_safe_metric(matthews_corrcoef, labels, preds),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "balanced_accuracy": 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp))) if (tp + fn) and (tn + fp) else np.nan,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "ece": expected_calibration_error(labels, scores),
    }


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if labels.size == 0:
        return float("nan")
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        in_bin = (scores >= lower) & (scores < upper if index < bins - 1 else scores <= upper)
        if np.any(in_bin):
            ece += float(np.mean(in_bin) * abs(np.mean(scores[in_bin]) - np.mean(labels[in_bin])))
    return ece


def _protein_metrics_safe_metric(func, labels: np.ndarray, values: np.ndarray) -> float:
    try:
        return float(func(labels, values))
    except ValueError:
        return float("nan")


def _nan_metrics() -> dict[str, float]:
    return {
        name: float("nan")
        for name in ("auc", "prauc", "f1", "mcc", "sensitivity", "specificity", "balanced_accuracy", "fpr", "ece")
    }


# Source: metrics/region_metrics.py


import numpy as np


def interval_iou(pred: tuple[int, int], truth: tuple[int, int]) -> float:
    start = max(pred[0], truth[0])
    end = min(pred[1], truth[1])
    intersection = max(0, end - start + 1)
    union = max(pred[1], truth[1]) - min(pred[0], truth[0]) + 1
    return float(intersection / union) if union > 0 else 0.0


def region_metrics(pred_regions: list[list[dict[str, float]]], true_regions: list[list[dict[str, object]]], iou_threshold: float = 0.5) -> dict[str, float]:
    tp = 0
    pred_total = 0
    true_total = 0
    boundary_errors: list[float] = []
    for preds, truths_raw in zip(pred_regions, true_regions, strict=False):
        truths = [region for region in truths_raw if _region_truth_kind(region) in {"positive", "negative"}]
        pred_total += len(preds)
        true_total += len(truths)
        used: set[int] = set()
        for pred in preds:
            best_index = -1
            best_iou = 0.0
            for index, truth in enumerate(truths):
                if index in used:
                    continue
                iou = interval_iou((int(pred["start"]), int(pred["end"])), (int(truth["start"]), int(truth["end"])))
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index >= 0 and best_iou >= iou_threshold:
                used.add(best_index)
                tp += 1
                truth = truths[best_index]
                boundary_errors.append(abs(int(pred["start"]) - int(truth["start"])) + abs(int(pred["end"]) - int(truth["end"])))
    return {
        f"region_iou@{iou_threshold:g}_precision": tp / pred_total if pred_total else np.nan,
        f"region_iou@{iou_threshold:g}_recall": tp / true_total if true_total else np.nan,
        f"region_iou@{iou_threshold:g}_f1": (2 * tp / (pred_total + true_total)) if (pred_total + true_total) else np.nan,
        "mean_boundary_error": float(np.mean(boundary_errors)) if boundary_errors else np.nan,
        "fragmentation_rate": pred_total / true_total if true_total else np.nan,
        "region_coverage": _region_coverage(pred_regions, true_regions),
    }


def boundary_f1(
    pred_regions: list[list[dict[str, float]]],
    true_regions: list[list[dict[str, object]]],
    tolerance: int = 5,
) -> dict[str, float]:
    tp = 0
    pred_total = 0
    true_total = 0
    for preds, truths_raw in zip(pred_regions, true_regions, strict=False):
        truths = [region for region in truths_raw if _region_truth_kind(region) in {"positive", "negative"}]
        pred_bounds = [(int(region["start"]), int(region["end"])) for region in preds]
        true_bounds = [(int(region["start"]), int(region["end"])) for region in truths]
        pred_total += 2 * len(pred_bounds)
        true_total += 2 * len(true_bounds)
        used: set[tuple[int, int]] = set()
        for pred_start, pred_end in pred_bounds:
            for pred_boundary in (pred_start, pred_end):
                for true_index, (true_start, true_end) in enumerate(true_bounds):
                    for side, true_boundary in enumerate((true_start, true_end)):
                        key = (true_index, side)
                        if key not in used and abs(pred_boundary - true_boundary) <= tolerance:
                            used.add(key)
                            tp += 1
                            break
                    else:
                        continue
                    break
    precision = tp / pred_total if pred_total else np.nan
    recall = tp / true_total if true_total else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and (precision + recall) else np.nan
    return {"boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1}


def _region_coverage(pred_regions: list[list[dict[str, float]]], true_regions: list[list[dict[str, object]]]) -> float:
    overlaps: list[float] = []
    for preds, truths_raw in zip(pred_regions, true_regions, strict=False):
        truths = [region for region in truths_raw if _region_truth_kind(region) in {"positive", "negative"}]
        for truth in truths:
            truth_interval = (int(truth["start"]), int(truth["end"]))
            truth_len = max(1, truth_interval[1] - truth_interval[0] + 1)
            covered = 0
            for pred in preds:
                start = max(truth_interval[0], int(pred["start"]))
                end = min(truth_interval[1], int(pred["end"]))
                covered += max(0, end - start + 1)
            overlaps.append(min(1.0, covered / truth_len))
    return float(np.mean(overlaps)) if overlaps else np.nan


def _region_truth_kind(region: dict[str, object]) -> str:
    evidence_level = str(region.get("evidence_level") or "").strip().lower()
    if evidence_level in {"candidate", "pseudo"}:
        return "ignore"
    label = region.get("region_label")
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized in {"1", "positive", "gold", "curated"}:
            return "positive"
        if normalized in {"0", "negative", "control"}:
            return "negative"
        if normalized in {"candidate", "unknown", "ignore", ""}:
            return "ignore"
    elif isinstance(label, (int, float)):
        if int(label) == 1:
            return "positive"
        if int(label) == 0:
            return "negative"
    region_type = str(region.get("region_type") or region.get("type") or "").strip()
    if region_type in {"DPR_gold", "DPR_curated"}:
        return "positive"
    if region_type in {"non_DPR_control"}:
        return "negative"
    return "ignore"


# Source: metrics/residue_metrics.py


import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from phaseflow.protein.contracts import IGNORE_INDEX


def residue_binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    valid = labels != IGNORE_INDEX
    labels = labels[valid].astype(int)
    scores = scores[valid].astype(float)
    if labels.size == 0:
        return {"residue_auc": np.nan, "residue_prauc": np.nan, "residue_f1": np.nan, "residue_dice": np.nan}
    preds = (scores >= threshold).astype(int)
    intersection = np.sum((preds == 1) & (labels == 1))
    denom = np.sum(preds == 1) + np.sum(labels == 1)
    return {
        "residue_auc": _safe_metric(roc_auc_score, labels, scores),
        "residue_prauc": _safe_metric(average_precision_score, labels, scores),
        "residue_f1": _safe_metric(f1_score, labels, preds),
        "residue_dice": float((2 * intersection) / denom) if denom else np.nan,
    }


def _safe_metric(func, labels: np.ndarray, values: np.ndarray) -> float:
    try:
        return float(func(labels, values))
    except ValueError:
        return float("nan")


# Source: metrics/phasepro.py

"""PhasePro residue-profile evaluation primitives used by the public DPR CLI."""


import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def build_phasepro_truths(proteins: pd.DataFrame, regions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build the frozen PhasePro labels from the released half-open intervals."""
    truths: dict[str, dict[str, Any]] = {}
    protein_meta = proteins.set_index("protein_id", drop=False)
    for row in proteins.itertuples(index=False):
        protein_id = str(row.protein_id)
        truths[protein_id] = {
            "label": np.zeros(int(row.sequence_length), dtype=np.int8),
            "regions": [],
        }
    for row in regions.itertuples(index=False):
        protein_id = str(row.protein_id)
        start = int(row.pstp_notebook_start_0based)
        end = int(row.pstp_notebook_end_exclusive)
        length = int(protein_meta.loc[protein_id, "sequence_length"])
        if start < 0 or end > length or end <= start:
            raise RuntimeError(f"Invalid PhasePro truth span {protein_id}:{start}-{end} length={length}")
        truths[protein_id]["label"][start:end] = 1
        truths[protein_id]["regions"].append({"start": start, "end": end})
    return truths


def build_truths(proteins: pd.DataFrame, regions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compatibility-preserving PhasePro truth construction for released reports."""
    truths: dict[str, dict[str, Any]] = {}
    protein_meta = proteins.set_index("protein_id", drop=False)
    for row in proteins.itertuples(index=False):
        protein_id = str(row.protein_id)
        truths[protein_id] = {
            "label": np.zeros(int(row.sequence_length), dtype=np.int8),
            "regions": [],
            "sequence": str(row.sequence),
            "gene_name": str(row.gene_name),
            "protein_name": str(row.protein_name),
        }
    for row in regions.itertuples(index=False):
        protein_id = str(row.protein_id)
        start = int(row.pstp_notebook_start_0based)
        end = int(row.pstp_notebook_end_exclusive)
        length = int(protein_meta.loc[protein_id, "sequence_length"])
        if start < 0 or end > length or end <= start:
            raise RuntimeError(f"Invalid PSTP-notebook truth span {protein_id}:{start}-{end} length={length}")
        truths[protein_id]["label"][start:end] = 1
        truths[protein_id]["regions"].append(
            {
                "region_id": str(row.region_id),
                "start": start,
                "end": end,
                "start_1based": int(row.start_raw),
                "end_1based": int(end),
            }
        )
    return truths


def per_protein_phasepro_metrics(profiles: dict[str, np.ndarray], truths: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protein_id in sorted(profiles):
        label = np.asarray(truths[protein_id]["label"], dtype=int)
        score = np.asarray(profiles[protein_id], dtype=float)
        valid_spearman = len(np.unique(label)) == 2 and int((label == 0).sum()) >= 20
        rows.append(
            {
                "protein_id": protein_id,
                "length": int(len(score)),
                "positive_count": int(label.sum()),
                "spearman": safe_spearman(label, score) if valid_spearman else math.nan,
                "auroc": safe_auc(label, score),
                "auprc": safe_ap(label, score),
            }
        )
    return pd.DataFrame(rows)


def per_protein_metrics(profiles: dict[str, np.ndarray], truths: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protein_id in sorted(profiles):
        label = np.asarray(truths[protein_id]["label"], dtype=int)
        score = np.asarray(profiles[protein_id], dtype=float)
        valid_spearman = len(np.unique(label)) == 2 and int((label == 0).sum()) >= 20
        rows.append(
            {
                "protein_id": protein_id,
                "length": int(len(score)),
                "positive_count": int(label.sum()),
                "positive_fraction": float(label.mean()),
                "region_count": int(len(truths[protein_id]["regions"])),
                "regions": format_regions(truths[protein_id]["regions"]),
                "spearman": safe_spearman(label, score) if valid_spearman else math.nan,
                "auroc": safe_auc(label, score),
                "auprc": safe_ap(label, score),
                "pred_fraction_0p5": float((score >= 0.5).mean()),
                "pos_mean": float(score[label == 1].mean()) if int(label.sum()) else math.nan,
                "neg_mean": float(score[label == 0].mean()) if int((label == 0).sum()) else math.nan,
                "pos_minus_neg_mean": float(score[label == 1].mean() - score[label == 0].mean())
                if int(label.sum()) and int((label == 0).sum())
                else math.nan,
                "max_score": float(score.max()) if len(score) else math.nan,
                "mean_score": float(score.mean()) if len(score) else math.nan,
                "std_score": float(score.std()) if len(score) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def phasepro_threshold_free_metrics(
    profiles: dict[str, np.ndarray], truths: dict[str, dict[str, Any]], per_protein: pd.DataFrame
) -> dict[str, Any]:
    labels, scores = concat_labels_scores(profiles, truths)
    return {
        "global_residue_AUROC": safe_auc(labels, scores),
        "global_residue_AUPRC": safe_ap(labels, scores),
        "global_residue_Spearman": safe_spearman(labels, scores),
        "per_protein_Spearman_mean": float(per_protein["spearman"].mean(skipna=True)),
        "per_protein_Spearman_median": float(per_protein["spearman"].median(skipna=True)),
        "per_protein_Spearman_valid_count": int(per_protein["spearman"].notna().sum()),
        "per_protein_Spearman_invalid_count": int(per_protein["spearman"].isna().sum()),
        "per_protein_AUROC_mean": float(per_protein["auroc"].mean(skipna=True)),
        "per_protein_AUROC_median": float(per_protein["auroc"].median(skipna=True)),
        "residue_n": int(len(labels)),
        "positive_residue_n": int(labels.sum()),
        "positive_residue_fraction": float(labels.mean()),
    }


def threshold_free_metrics(profiles: dict[str, np.ndarray], truths: dict[str, dict[str, Any]], per: pd.DataFrame) -> dict[str, Any]:
    labels, scores = concat_labels_scores(profiles, truths)
    return {
        "global_residue_AUROC": safe_auc(labels, scores),
        "global_residue_AUPRC": safe_ap(labels, scores),
        "global_residue_Spearman": safe_spearman(labels, scores),
        "per_protein_Spearman_mean": float(per["spearman"].mean(skipna=True)),
        "per_protein_Spearman_median": float(per["spearman"].median(skipna=True)),
        "per_protein_Spearman_valid_count": int(per["spearman"].notna().sum()),
        "per_protein_Spearman_invalid_count": int(per["spearman"].isna().sum()),
        "per_protein_AUROC_mean": float(per["auroc"].mean(skipna=True)),
        "per_protein_AUROC_median": float(per["auroc"].median(skipna=True)),
        "same_protein_pairwise": pairwise_accuracy_from_per(per),
        "residue_n": int(len(labels)),
        "positive_residue_n": int(labels.sum()),
        "positive_residue_fraction": float(labels.mean()),
    }


def concat_labels_scores(profiles: dict[str, np.ndarray], truths: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for protein_id in sorted(profiles):
        label = np.asarray(truths[protein_id]["label"], dtype=int).reshape(-1)
        score = np.asarray(profiles[protein_id], dtype=float).reshape(-1)
        if label.shape != score.shape:
            raise ValueError(f"profile/label length mismatch for {protein_id}: {len(score)} != {len(label)}")
        labels.append(label)
        scores.append(score)
    if not labels:
        raise ValueError("no profiles supplied for PhasePro evaluation")
    return np.concatenate(labels), np.concatenate(scores)


def safe_auc(label: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(label)) < 2:
        return math.nan
    return float(roc_auc_score(label, score))


def safe_ap(label: np.ndarray, score: np.ndarray) -> float:
    if int(np.sum(label)) == 0:
        return math.nan
    return float(average_precision_score(label, score))


def safe_spearman(label: np.ndarray, score: np.ndarray) -> float:
    label = np.asarray(label, dtype=float).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    if label.size != score.size or label.size == 0 or not np.isfinite(label).all() or not np.isfinite(score).all():
        return math.nan
    label_rank = average_ranks(label)
    score_rank = average_ranks(score)
    label_rank -= float(label_rank.mean())
    score_rank -= float(score_rank.mean())
    denominator = float(np.sqrt(np.sum(label_rank * label_rank) * np.sum(score_rank * score_rank)))
    return math.nan if denominator == 0.0 else float(np.sum(label_rank * score_rank) / denominator)


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    if not values.size:
        return ranks
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    ranks[order] = np.repeat(0.5 * (starts + 1 + ends), ends - starts)
    return ranks


def format_regions(regions: list[dict[str, Any]]) -> str:
    return ";".join(f"{int(region['start_1based'])}-{int(region['end_1based'])}" for region in regions)


def pairwise_accuracy_from_per(per: pd.DataFrame) -> float:
    valid = per["auroc"].dropna()
    return float(valid.mean()) if len(valid) else math.nan


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# Source: losses/multitask.py


from typing import Any

import torch
import torch.nn.functional as F

from phaseflow.protein.objectives import smoothness_loss
from phaseflow.protein.objectives import dice_loss_with_logits
from phaseflow.protein.objectives import focal_loss_with_logits
from phaseflow.protein.objectives import (
    boundary_transition_loss,
    residue_contrastive_margin_loss,
    weighted_soft_bce_logits,
)
from phaseflow.protein.objectives import masked_bce_with_logits, protein_bce_class_normalized_stats, protein_bce_with_logits
from phaseflow.protein.objectives import phase_diagram_loss
from phaseflow.protein.objectives import region_coverage_loss, region_query_loss
from phaseflow.protein.objectives import (
    calibration_loss_with_logits,
    negative_region_regularization,
    nnpu_loss_with_logits,
    region_bag_mil_loss,
    soft_bce_with_logits,
    teacher_distillation_mse,
    weak_region_mil_loss,
)


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    protein_logit_key = str(weights.get("protein_logit_key", "loss_llps_logits"))
    protein_logits = outputs.get(protein_logit_key)
    if protein_logits is None:
        protein_logits = outputs.get("loss_llps_logits", outputs["llps_logits"])
    protein_logits = protein_logits.float()
    if bool(weights.get("protein_bce_class_normalized", False)):
        llps, llps_stats = protein_bce_class_normalized_stats(
            protein_logits,
            batch["y_llps"],
            batch.get("sample_weight"),
            alpha_pos=float(weights.get("protein_bce_alpha_pos", 0.5)),
            alpha_neg=float(weights.get("protein_bce_alpha_neg", 0.5)),
        )
    else:
        llps = protein_bce_with_logits(protein_logits, batch["y_llps"], batch.get("sample_weight"))
        zero = llps.detach() * 0.0
        llps_stats = {
            "protein_loss_pos": zero,
            "protein_loss_neg": zero,
            "protein_loss_pos_count": zero,
            "protein_loss_neg_count": zero,
            "protein_loss_missing_class": zero,
        }
    weighted_focal_bce = (
        weighted_focal_bce_with_logits(
            protein_logits,
            batch,
            gamma=float(weights.get("focal_gamma", 1.5)),
            positive_weight=float(weights.get("positive_weight", 1.0)),
            negative_weight=float(weights.get("negative_weight", 1.0)),
            client_weight=float(weights.get("client_weight", 1.0)),
            nd_weight=float(weights.get("nd_weight", 1.0)),
        )
        if float(weights.get("weighted_focal_bce", 0.0)) > 0.0
        else protein_logits.sum() * 0.0
    )
    teacher_llps = soft_bce_with_logits(protein_logits, batch["teacher_llps"], batch["teacher_llps_weight"])
    self_llps = soft_bce_with_logits(protein_logits, batch["self_llps"], batch["self_llps_weight"])
    nnpu = nnpu_loss_with_logits(
        protein_logits,
        batch["y_llps"],
        batch.get("sample_weight"),
        positive_prior=float(weights.get("positive_prior", 0.1)),
    )
    calibration = calibration_loss_with_logits(protein_logits, batch["y_llps"])
    zero_dpr_loss = protein_logits.sum() * 0.0
    dpr_logits = outputs.get("dpr_logits")
    if torch.is_tensor(dpr_logits):
        zero_dpr_loss = zero_dpr_loss + dpr_logits.sum() * 0.0
    region_gold_weight = float(weights.get("region_gold", weights.get("dpr", 0.0)))
    region_gold = (
        0.5 * masked_bce_with_logits(outputs["dpr_logits"], batch["y_dpr"], batch["seq_mask"], batch["y_weight"])
        + 0.3 * dice_loss_with_logits(outputs["dpr_logits"], batch["y_dpr"], batch["seq_mask"])
        + 0.2 * focal_loss_with_logits(outputs["dpr_logits"], batch["y_dpr"], batch["seq_mask"], batch["y_weight"])
        if region_gold_weight > 0.0
        else zero_dpr_loss
    )
    region_mil_weight = float(weights.get("region_mil", weights.get("region_MIL", 0.0)))
    region_mil = (
        region_bag_mil_loss(
            outputs["region_global_logits"],
            batch["region_bag_label"],
            batch["region_bag_weight"],
        )
        if region_mil_weight > 0.0
        else zero_dpr_loss
    )
    teacher_dpr_bce = (
        weak_region_mil_loss(outputs["dpr_logits"], batch["teacher_dpr"], batch["teacher_dpr_weight"], batch["seq_mask"])
        if float(weights.get("teacher_dpr", 0.0)) > 0.0
        else zero_dpr_loss
    )
    teacher_distill = (
        teacher_distillation_mse(
            outputs["dpr_logits"],
            batch["teacher_dpr"],
            batch["teacher_dpr_weight"],
            batch["seq_mask"],
        )
        if float(weights.get("teacher_distill", 0.0)) > 0.0
        else zero_dpr_loss
    )
    self_dpr = (
        weak_region_mil_loss(outputs["dpr_logits"], batch["self_dpr"], batch["self_dpr_weight"], batch["seq_mask"])
        if float(weights.get("self_dpr", 0.0)) > 0.0
        else zero_dpr_loss
    )
    region = (
        region_query_loss(outputs["region_logits"], outputs["region_start"], outputs["region_end"], batch["regions"], batch["lengths"])
        if float(weights.get("region", 0.0)) > 0.0
        else zero_dpr_loss
    )
    coverage = (
        region_coverage_loss(outputs["region_logits"], outputs["region_start"], outputs["region_end"], batch["regions"], batch["lengths"])
        if float(weights.get("coverage", 0.0)) > 0.0
        else zero_dpr_loss
    )
    key = (
        masked_bce_with_logits(outputs["key_logits"], batch["y_key"], batch["seq_mask"], batch["y_weight"])
        if float(weights.get("key", 0.0)) > 0.0
        else zero_dpr_loss
    )
    smooth_weight = float(weights.get("smoothness", 0.05))
    smooth = smoothness_loss(outputs["dpr_logits"], batch["seq_mask"]) if smooth_weight > 0.0 else zero_dpr_loss
    negative_regularization_weight = float(weights.get("negative_regularization", 0.0))
    negative_regularization = (
        negative_region_regularization(
            outputs["dpr_logits"],
            batch["seq_mask"],
            batch["negative_regularization_weight"],
        )
        if negative_regularization_weight > 0.0
        else zero_dpr_loss
    )
    phase_aux_weight = float(weights.get("phase_aux", weights.get("phase", 0.0)))
    phase_aux = (
        phase_diagram_loss(
            outputs,
            batch,
            pssi_min=float(weights.get("phase_pssi_min", -2.17)),
            pssi_max=float(weights.get("phase_pssi_max", 1.64)),
            mean_weight=float(weights.get("phase_mean_weight", 0.25)),
        )
        if phase_aux_weight > 0.0
        else zero_dpr_loss
    )
    region_teacher = (
        weighted_soft_bce_logits(
            outputs["dpr_logits"],
            batch["region_teacher_target"],
            batch["region_teacher_weight"],
            batch["seq_mask"],
        )
        if float(weights.get("region_teacher", 0.0)) > 0.0
        else zero_dpr_loss
    )
    region_key_teacher = (
        weighted_soft_bce_logits(outputs["key_logits"], batch["region_key_target"], batch["region_key_weight"], batch["seq_mask"])
        if float(weights.get("region_key_teacher", 0.0)) > 0.0
        else zero_dpr_loss
    )
    region_boundary = (
        boundary_transition_loss(
            outputs["dpr_logits"],
            batch["region_boundary_target"],
            batch["region_boundary_weight"],
            batch["seq_mask"],
        )
        if float(weights.get("region_boundary", 0.0)) > 0.0
        else zero_dpr_loss
    )
    region_contrastive = (
        residue_contrastive_margin_loss(
            outputs["dpr_logits"],
            batch["region_contrast_target"],
            batch["region_contrast_weight"],
            batch["seq_mask"],
            margin=float(weights.get("region_contrastive_margin", 0.35)),
        )
        if float(weights.get("region_contrastive", 0.0)) > 0.0
        else zero_dpr_loss
    )
    ranking_weight = float(weights.get("ranking_loss_weight", 0.0))
    top_negative_ranking = (
        top_negative_ranking_loss(
            protein_logits,
            batch,
            margin=float(weights.get("ranking_loss_margin", 0.7)),
            topk_negatives=int(weights.get("ranking_loss_topk_negatives", 4)),
            positive_pool_names=weights.get("ranking_positive_pool_names"),
            negative_pool_names=weights.get("ranking_negative_pool_names"),
        )
        if bool(weights.get("ranking_loss_enabled", False)) and ranking_weight > 0.0
        else protein_logits.sum() * 0.0
    )
    hard_negative_focal_weight = float(weights.get("hard_negative_focal_weight", 0.0))
    hard_negative_focal = (
        hard_negative_focal_loss(
            protein_logits,
            batch,
            gamma=float(weights.get("hard_negative_focal_gamma", 2.0)),
            pool_names=weights.get("hard_negative_focal_pool_names"),
        )
        if hard_negative_focal_weight > 0.0
        else protein_logits.sum() * 0.0
    )
    pairwise_rank_weight = float(weights.get("pairwise_rank_loss_weight", weights.get("rank_loss_weight", 0.0)))
    pairwise_rank = (
        pairwise_logistic_ranking_loss(
            protein_logits,
            batch,
            topk_negatives=int(weights.get("rank_loss_topk_negatives", 16)),
            positive_pool_names=weights.get("ranking_positive_pool_names"),
            negative_pool_names=weights.get("ranking_negative_pool_names"),
            client_nd_weight=float(weights.get("client_nd_rank_weight", 1.5)),
            nd_pair_weight=float(weights.get("nd_rank_weight", 1.5)),
        )
        if pairwise_rank_weight > 0.0
        else protein_logits.sum() * 0.0
    )
    driver_aux_weight = float(weights.get("driver_head", weights.get("driver_loss_weight", 0.0)))
    client_aux_weight = float(weights.get("client_head", weights.get("client_loss_weight", 0.0)))
    negtype_aux_weight = float(weights.get("negtype_head", weights.get("negtype_loss_weight", 0.0)))
    driver_aux = (
        role_bce_aux_loss(outputs.get("driver_logits"), batch, role="driver")
        if driver_aux_weight > 0.0
        else protein_logits.sum() * 0.0
    )
    client_aux = (
        role_bce_aux_loss(outputs.get("client_logits"), batch, role="client")
        if client_aux_weight > 0.0
        else protein_logits.sum() * 0.0
    )
    negtype_aux = (
        negtype_ce_aux_loss(outputs.get("negtype_logits"), batch)
        if negtype_aux_weight > 0.0
        else protein_logits.sum() * 0.0
    )
    total = (
        float(weights.get("llps", 1.0)) * llps
        + float(weights.get("weighted_focal_bce", 0.0)) * weighted_focal_bce
        + float(weights.get("teacher_llps", 0.0)) * teacher_llps
        + float(weights.get("self_llps", 0.0)) * self_llps
        + float(weights.get("nnpu", 0.0)) * nnpu
        + float(weights.get("calibration", 0.0)) * calibration
        + region_gold_weight * region_gold
        + region_mil_weight * region_mil
        + float(weights.get("teacher_dpr", 0.0)) * teacher_dpr_bce
        + float(weights.get("teacher_distill", 0.0)) * teacher_distill
        + float(weights.get("self_dpr", 0.0)) * self_dpr
        + float(weights.get("region", 0.0)) * region
        + float(weights.get("coverage", 0.0)) * coverage
        + float(weights.get("key", 0.0)) * key
        + smooth_weight * smooth
        + negative_regularization_weight * negative_regularization
        + phase_aux_weight * phase_aux
        + float(weights.get("region_teacher", 0.0)) * region_teacher
        + float(weights.get("region_key_teacher", 0.0)) * region_key_teacher
        + float(weights.get("region_boundary", 0.0)) * region_boundary
        + float(weights.get("region_contrastive", 0.0)) * region_contrastive
        + ranking_weight * top_negative_ranking
        + hard_negative_focal_weight * hard_negative_focal
        + pairwise_rank_weight * pairwise_rank
        + driver_aux_weight * driver_aux
        + client_aux_weight * client_aux
        + negtype_aux_weight * negtype_aux
    )
    values = {
        "loss": float(total.detach().cpu()),
        "llps": float(llps.detach().cpu()),
        "protein_loss_pos": float(llps_stats["protein_loss_pos"].detach().cpu()),
        "protein_loss_neg": float(llps_stats["protein_loss_neg"].detach().cpu()),
        "protein_loss_pos_count": float(llps_stats["protein_loss_pos_count"].detach().cpu()),
        "protein_loss_neg_count": float(llps_stats["protein_loss_neg_count"].detach().cpu()),
        "protein_loss_missing_class": float(llps_stats["protein_loss_missing_class"].detach().cpu()),
        "teacher_llps": float(teacher_llps.detach().cpu()),
        "self_llps": float(self_llps.detach().cpu()),
        "nnpu": float(nnpu.detach().cpu()),
        "calibration": float(calibration.detach().cpu()),
        "dpr": float(region_gold.detach().cpu()),
        "region_gold": float(region_gold.detach().cpu()),
        "region_mil": float(region_mil.detach().cpu()),
        "teacher_dpr": float(teacher_dpr_bce.detach().cpu()),
        "teacher_distill": float(teacher_distill.detach().cpu()),
        "self_dpr": float(self_dpr.detach().cpu()),
        "region": float(region.detach().cpu()),
        "coverage": float(coverage.detach().cpu()),
        "key": float(key.detach().cpu()),
        "smoothness": float(smooth.detach().cpu()),
        "negative_regularization": float(negative_regularization.detach().cpu()),
        "phase_aux": float(phase_aux.detach().cpu()),
        "region_teacher": float(region_teacher.detach().cpu()),
        "region_key_teacher": float(region_key_teacher.detach().cpu()),
        "region_boundary": float(region_boundary.detach().cpu()),
        "region_contrastive": float(region_contrastive.detach().cpu()),
        "top_negative_ranking": float(top_negative_ranking.detach().cpu()),
        "hard_negative_focal": float(hard_negative_focal.detach().cpu()),
        "weighted_focal_bce": float(weighted_focal_bce.detach().cpu()),
        "pairwise_rank": float(pairwise_rank.detach().cpu()),
        "driver_aux": float(driver_aux.detach().cpu()),
        "client_aux": float(client_aux.detach().cpu()),
        "negtype_aux": float(negtype_aux.detach().cpu()),
    }
    return total, values


def weighted_focal_bce_with_logits(
    protein_logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    gamma: float,
    positive_weight: float,
    negative_weight: float,
    client_weight: float,
    nd_weight: float,
) -> torch.Tensor:
    logits = protein_logits.float().reshape(-1)
    labels = batch["y_llps"].to(device=logits.device, dtype=torch.float32).reshape(-1)
    valid = labels.ge(0.0) & labels.le(1.0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    logits = logits[valid]
    labels = labels[valid]
    weights = batch.get("sample_weight")
    if torch.is_tensor(weights):
        sample_weight = weights.to(device=logits.device, dtype=torch.float32).reshape(-1)[valid]
    else:
        sample_weight = torch.ones_like(labels)
    texts = _row_texts(batch)
    text_valid = [texts[index] for index, keep in enumerate(valid.detach().cpu().tolist()) if keep]
    role_weight = torch.ones_like(labels)
    role_weight = torch.where(labels.eq(1.0), role_weight * float(positive_weight), role_weight * float(negative_weight))
    client_mask = torch.tensor(["client" in text or "member" in text for text in text_valid], device=logits.device)
    nd_mask = torch.tensor(["disordered" in text or "nd" in text or "n_disordered" in text for text in text_valid], device=logits.device)
    role_weight = torch.where(client_mask & labels.eq(1.0), role_weight * float(client_weight), role_weight)
    role_weight = torch.where(nd_mask & labels.eq(0.0), role_weight * float(nd_weight), role_weight)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    pt = torch.exp(-bce).clamp(min=1.0e-6, max=1.0)
    loss = ((1.0 - pt).pow(float(gamma)) * bce) * sample_weight * role_weight
    denom = (sample_weight * role_weight).sum().clamp(min=1.0)
    return loss.sum() / denom


def pairwise_logistic_ranking_loss(
    protein_logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    topk_negatives: int,
    positive_pool_names: Any = None,
    negative_pool_names: Any = None,
    client_nd_weight: float = 1.5,
    nd_pair_weight: float = 1.5,
) -> torch.Tensor:
    scores = protein_logits.float().reshape(-1)
    labels = batch["y_llps"].to(device=scores.device, dtype=torch.float32).reshape(-1)
    pools = [str(item) for item in batch.get("plan_pool_name", [])]
    if len(pools) != scores.numel():
        pools = ["" for _ in range(int(scores.numel()))]
    positive_pools = _pool_name_set(positive_pool_names, set())
    negative_pools = _pool_name_set(negative_pool_names, set())
    pos_mask = labels.eq(1.0)
    if positive_pools:
        pos_mask &= torch.tensor([pool in positive_pools for pool in pools], device=scores.device, dtype=torch.bool)
    neg_mask = labels.eq(0.0)
    if negative_pools:
        neg_mask &= torch.tensor([pool in negative_pools for pool in pools], device=scores.device, dtype=torch.bool)
    positives = scores[pos_mask]
    negatives_all = scores[neg_mask]
    if positives.numel() == 0 or negatives_all.numel() == 0:
        return scores.sum() * 0.0
    k = max(1, min(int(topk_negatives), int(negatives_all.numel())))
    neg_values, neg_indices_local = torch.topk(negatives_all, k=k, largest=True)
    pos_texts = [text for text, keep in zip(_row_texts(batch), pos_mask.detach().cpu().tolist(), strict=False) if keep]
    neg_texts_all = [text for text, keep in zip(_row_texts(batch), neg_mask.detach().cpu().tolist(), strict=False) if keep]
    neg_texts = [neg_texts_all[int(index)] for index in neg_indices_local.detach().cpu().tolist()]
    pair_loss = F.softplus(-(positives.unsqueeze(1) - neg_values.unsqueeze(0)))
    pair_weight = torch.ones_like(pair_loss)
    pos_client = torch.tensor(
        ["client" in text or "member" in text for text in pos_texts],
        device=scores.device,
        dtype=torch.bool,
    ).unsqueeze(1)
    neg_nd = torch.tensor(
        ["disordered" in text or "nd" in text or "n_disordered" in text for text in neg_texts],
        device=scores.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    pair_weight = torch.where(neg_nd, pair_weight * float(nd_pair_weight), pair_weight)
    pair_weight = torch.where(pos_client & neg_nd, pair_weight * float(client_nd_weight), pair_weight)
    return (pair_loss * pair_weight).sum() / pair_weight.sum().clamp(min=1.0)


def role_bce_aux_loss(logits: torch.Tensor | None, batch: dict[str, Any], *, role: str) -> torch.Tensor:
    if logits is None:
        fallback = batch["y_llps"].float().sum() * 0.0
        return fallback
    out = logits.float().reshape(-1)
    labels = batch["y_llps"].to(device=out.device, dtype=torch.float32).reshape(-1)
    positive = labels.eq(1.0)
    if not torch.any(positive):
        return out.sum() * 0.0
    texts = _row_texts(batch)
    if role == "driver":
        targets = ["driver" in text or "scaffold" in text or "p_gold" in text for text in texts]
    elif role == "client":
        targets = ["client" in text or "member" in text for text in texts]
    else:
        raise ValueError(f"Unsupported role aux target: {role}")
    target = torch.tensor(targets, device=out.device, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(out[positive], target[positive])


def negtype_ce_aux_loss(logits: torch.Tensor | None, batch: dict[str, Any]) -> torch.Tensor:
    if logits is None:
        return batch["y_llps"].float().sum() * 0.0
    out = logits.float()
    labels = batch["y_llps"].to(device=out.device, dtype=torch.float32).reshape(-1)
    texts = _row_texts(batch)
    structured = ["structured" in text or "np" in text or "n_structured" in text for text in texts]
    disordered = ["disordered" in text or "nd" in text or "n_disordered" in text for text in texts]
    mask = labels.eq(0.0) & torch.tensor(
        [s or d for s, d in zip(structured, disordered, strict=False)],
        device=out.device,
        dtype=torch.bool,
    )
    if not torch.any(mask):
        return out.sum() * 0.0
    target = torch.tensor([1 if d else 0 for d in disordered], device=out.device, dtype=torch.long)
    return F.cross_entropy(out[mask], target[mask])


def _row_texts(batch: dict[str, Any]) -> list[str]:
    fields = [
        batch.get("plan_pool_name", []),
        batch.get("plan_tier", []),
        batch.get("plan_negative_type", []),
        batch.get("negative_type", []),
        batch.get("label_quality", []),
        batch.get("llps_role", []),
        batch.get("source", []),
    ]
    n = 0
    for field in fields:
        try:
            n = max(n, len(field))
        except TypeError:
            continue
    out: list[str] = []
    for index in range(n):
        parts = []
        for field in fields:
            try:
                value = field[index]
            except Exception:
                value = ""
            parts.append(str(value).lower())
        out.append(" ".join(parts))
    return out


def top_negative_ranking_loss(
    protein_logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    margin: float = 0.7,
    topk_negatives: int = 4,
    positive_pool_names: Any = None,
    negative_pool_names: Any = None,
) -> torch.Tensor:
    scores = torch.sigmoid(protein_logits.float()).reshape(-1)
    labels = batch["y_llps"].to(device=scores.device, dtype=torch.float32).reshape(-1)
    pools = [str(item) for item in batch.get("plan_pool_name", [])]
    if len(pools) != scores.numel():
        pools = ["" for _ in range(int(scores.numel()))]

    strong_positive_pools = _pool_name_set(
        positive_pool_names,
        {"P_gold", "P_curated", "P_pseudo_high", "P_mixed_curated_high"},
    )
    negative_pools = _pool_name_set(negative_pool_names, set())
    positive_mask = torch.tensor(
        [pool in strong_positive_pools for pool in pools],
        device=scores.device,
        dtype=torch.bool,
    ) & labels.eq(1.0)
    negative_mask = torch.tensor(
        [(pool in negative_pools) if negative_pools else pool.startswith("N_") for pool in pools],
        device=scores.device,
        dtype=torch.bool,
    ) & labels.eq(0.0)

    positives = scores[positive_mask]
    negatives = scores[negative_mask]
    if positives.numel() == 0 or negatives.numel() == 0:
        return scores.sum() * 0.0
    k = max(1, min(int(topk_negatives), int(negatives.numel())))
    hard_negatives = torch.topk(negatives, k=k, largest=True).values
    pairwise = float(margin) - positives.unsqueeze(1) + hard_negatives.unsqueeze(0)
    return F.relu(pairwise).mean()


def hard_negative_focal_loss(
    protein_logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    gamma: float = 2.0,
    pool_names: Any = None,
) -> torch.Tensor:
    logits = protein_logits.float().reshape(-1)
    labels = batch["y_llps"].to(device=logits.device, dtype=torch.float32).reshape(-1)
    pools = [str(item) for item in batch.get("plan_pool_name", [])]
    if len(pools) != logits.numel():
        pools = ["" for _ in range(int(logits.numel()))]
    hard_pools = _pool_name_set(pool_names, {"N_hard"})
    hard_mask = torch.tensor(
        [pool in hard_pools for pool in pools],
        device=logits.device,
        dtype=torch.bool,
    ) & labels.eq(0.0)
    if not torch.any(hard_mask):
        return logits.sum() * 0.0
    selected = logits[hard_mask]
    target = torch.zeros_like(selected)
    bce = F.binary_cross_entropy_with_logits(selected, target, reduction="none")
    prob = torch.sigmoid(selected)
    return (prob.pow(float(gamma)) * bce).mean()


def _pool_name_set(value: Any, default: set[str]) -> set[str]:
    if value is None:
        return set(default)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set(default)
        return {item.strip() for item in text.split(",") if item.strip()}
    try:
        items = {str(item).strip() for item in value if str(item).strip()}
    except TypeError:
        return set(default)
    return items or set(default)
