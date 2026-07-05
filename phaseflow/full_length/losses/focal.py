from __future__ import annotations

import torch
import torch.nn.functional as F

from phaseflow.full_length.data.schemas import IGNORE_INDEX


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
