from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from phaseflow.full_length.data.schemas import IGNORE_INDEX


def residue_binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    valid = labels != IGNORE_INDEX
    labels = labels[valid].astype(int)
    scores = scores[valid].astype(float)
    if labels.size == 0:
        return {"residue_auc": np.nan, "residue_prauc": np.nan, "residue_f1": np.nan, "residue_dice": np.nan}
    preds = (scores >= threshold).astype(int)
    intersection = np.sum((preds == 1) & (labels == 1))
    denom = np.sum(preds == 1) + np.sum(labels == 1)
    return {
        "residue_auc": _safe_metric(roc_auc_score, labels, scores),
        "residue_prauc": _safe_metric(average_precision_score, labels, scores),
        "residue_f1": _safe_metric(f1_score, labels, preds),
        "residue_dice": float((2 * intersection) / denom) if denom else np.nan,
    }


def _safe_metric(func, labels: np.ndarray, values: np.ndarray) -> float:
    try:
        return float(func(labels, values))
    except ValueError:
        return float("nan")
