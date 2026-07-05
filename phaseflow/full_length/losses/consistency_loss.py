from __future__ import annotations

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
