from __future__ import annotations

import torch

from phaseflow.full_length.data.schemas import IGNORE_INDEX


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
