from __future__ import annotations

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
