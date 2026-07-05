from __future__ import annotations

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
        iou = interval_iou(pred_start[matched_pred], pred_end[matched_pred], true_start[matched_true], true_end[matched_true])
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
    iou = interval_iou(pred_start.unsqueeze(1), pred_end.unsqueeze(1), true_start.unsqueeze(0), true_end.unsqueeze(0))
    return cls_cost + boundary + (1.0 - iou)


def interval_iou(start_a: torch.Tensor, end_a: torch.Tensor, start_b: torch.Tensor, end_b: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
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
