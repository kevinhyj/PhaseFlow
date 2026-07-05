from __future__ import annotations

import torch
import torch.nn.functional as F

from phaseflow.full_length.data.schemas import IGNORE_INDEX


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
