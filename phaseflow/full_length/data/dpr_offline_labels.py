from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HalfOpenSpan:
    start: int
    end: int
    confidence: float = 1.0
    sample_weight: float = 1.0


def build_dpr_label_arrays(
    *,
    length: int,
    spans: list[HalfOpenSpan],
    boundary_radius: int = 3,
) -> dict[str, np.ndarray]:
    """Build DPR offline labels from zero-based half-open spans.

    Positive residues are [start, end). The end boundary anchor is end - 1,
    never end. Residues outside provided spans stay masked out here; callers can
    overlay reliable negative masks separately.
    """

    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    residue_target = np.zeros(length, dtype=np.float32)
    residue_mask = np.zeros(length, dtype=np.float32)
    residue_weight = np.zeros(length, dtype=np.float32)
    start_target = np.full(length, np.nan, dtype=np.float32)
    end_target = np.full(length, np.nan, dtype=np.float32)
    boundary_weight = np.zeros(length, dtype=np.float32)

    radius = max(0, int(boundary_radius))
    for span in spans:
        start = max(0, min(int(span.start), length))
        end = max(start, min(int(span.end), length))
        if end <= start:
            continue
        weight = float(np.clip(float(span.confidence) * float(span.sample_weight), 0.0, 1.0))
        if weight <= 0.0:
            continue
        residue_target[start:end] = 1.0
        residue_mask[start:end] = 1.0
        residue_weight[start:end] = np.maximum(residue_weight[start:end], weight)
        _write_boundary(start_target, boundary_weight, center=start, radius=radius, weight=weight)
        _write_boundary(end_target, boundary_weight, center=end - 1, radius=radius, weight=weight)

    return {
        "residue_target": residue_target,
        "residue_mask": residue_mask,
        "residue_weight": residue_weight,
        "start_target": start_target,
        "end_target": end_target,
        "boundary_weight": boundary_weight,
    }


def _write_boundary(
    target: np.ndarray,
    weight_arr: np.ndarray,
    *,
    center: int,
    radius: int,
    weight: float,
) -> None:
    length = int(target.shape[0])
    center = max(0, min(int(center), length - 1))
    left = max(0, center - radius)
    right = min(length, center + radius + 1)
    for pos in range(left, right):
        if radius <= 0:
            value = 1.0 if pos == center else 0.0
        else:
            value = max(0.0, 1.0 - abs(pos - center) / float(radius + 1))
        if not np.isfinite(target[pos]) or value > float(target[pos]):
            target[pos] = np.float32(value)
        weight_arr[pos] = max(float(weight_arr[pos]), float(weight))
