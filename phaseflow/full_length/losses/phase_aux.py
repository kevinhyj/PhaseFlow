from __future__ import annotations

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
