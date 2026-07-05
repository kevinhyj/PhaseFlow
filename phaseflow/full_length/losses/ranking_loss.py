from __future__ import annotations

import torch

from phaseflow.full_length.data.schemas import IGNORE_INDEX


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
