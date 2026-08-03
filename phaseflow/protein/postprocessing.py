"""Protein prediction post-processing utilities."""
from __future__ import annotations

# Source: postprocess.py


import numpy as np


def smooth_scores(scores: np.ndarray, window: int = 5) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if window <= 1 or scores.size == 0:
        return scores
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad = window // 2
    padded = np.pad(scores, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: scores.size]


def scores_to_regions(
    scores: np.ndarray,
    threshold: float = 0.5,
    smooth_window: int = 5,
    merge_gap: int = 5,
    min_region_len: int = 6,
) -> list[dict[str, float]]:
    smoothed = smooth_scores(scores, smooth_window)
    mask = smoothed >= threshold
    raw_segments = _segments(mask)
    merged = _merge_segments(raw_segments, merge_gap)
    regions: list[dict[str, float]] = []
    for start, end in merged:
        if end - start + 1 < min_region_len:
            continue
        region_scores = smoothed[start : end + 1]
        top_count = max(1, int(np.ceil(len(region_scores) * 0.3)))
        score = float(np.mean(np.sort(region_scores)[-top_count:]))
        regions.append({"start": int(start), "end": int(end), "score": score, "source": "postprocess"})
    return regions


def decoder_regions(
    logits: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    length: int,
    score_threshold: float = 0.5,
) -> list[dict[str, float]]:
    scores = 1.0 / (1.0 + np.exp(-logits))
    regions: list[dict[str, float]] = []
    for score, start_norm, end_norm in zip(scores, starts, ends, strict=False):
        if score < score_threshold:
            continue
        start = int(np.clip(round(start_norm * max(length - 1, 1)), 0, length - 1))
        end = int(np.clip(round(end_norm * max(length - 1, 1)), 0, length - 1))
        if end < start:
            start, end = end, start
        regions.append({"start": start, "end": end, "score": float(score), "source": "decoder"})
    return _nms_regions(regions)


def combine_regions(decoder: list[dict[str, float]], postprocessed: list[dict[str, float]]) -> list[dict[str, float]]:
    if decoder:
        return _nms_regions(decoder + postprocessed)
    return postprocessed


def top_key_residues(scores: np.ndarray, k: int = 10) -> list[dict[str, float]]:
    if scores.size == 0:
        return []
    order = np.argsort(-scores)[: min(k, scores.size)]
    return [{"index": int(index), "score": float(scores[index])} for index in order]


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _merge_segments(segments: list[tuple[int, int]], merge_gap: int) -> list[tuple[int, int]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= merge_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _nms_regions(regions: list[dict[str, float]], iou_threshold: float = 0.5) -> list[dict[str, float]]:
    ordered = sorted(regions, key=lambda region: float(region["score"]), reverse=True)
    kept: list[dict[str, float]] = []
    for region in ordered:
        if all(_iou(region, kept_region) < iou_threshold for kept_region in kept):
            kept.append(region)
    return sorted(kept, key=lambda region: int(region["start"]))


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    start = max(int(a["start"]), int(b["start"]))
    end = min(int(a["end"]), int(b["end"]))
    intersection = max(0, end - start + 1)
    union = max(int(a["end"]), int(b["end"])) - min(int(a["start"]), int(b["start"])) + 1
    return intersection / union if union else 0.0
