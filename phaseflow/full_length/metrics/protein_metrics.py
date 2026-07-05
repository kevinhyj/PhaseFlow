from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


def binary_classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    valid = labels >= 0
    labels = labels[valid].astype(int)
    scores = scores[valid].astype(float)
    if labels.size == 0:
        return _nan_metrics()
    preds = (scores >= threshold).astype(int)
    tp = float(np.sum((preds == 1) & (labels == 1)))
    tn = float(np.sum((preds == 0) & (labels == 0)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))
    return {
        "auc": _safe_metric(roc_auc_score, labels, scores),
        "prauc": _safe_metric(average_precision_score, labels, scores),
        "f1": _safe_metric(f1_score, labels, preds),
        "mcc": _safe_metric(matthews_corrcoef, labels, preds),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "balanced_accuracy": 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp))) if (tp + fn) and (tn + fp) else np.nan,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "ece": expected_calibration_error(labels, scores),
    }


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if labels.size == 0:
        return float("nan")
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        in_bin = (scores >= lower) & (scores < upper if index < bins - 1 else scores <= upper)
        if np.any(in_bin):
            ece += float(np.mean(in_bin) * abs(np.mean(scores[in_bin]) - np.mean(labels[in_bin])))
    return ece


def _safe_metric(func, labels: np.ndarray, values: np.ndarray) -> float:
    try:
        return float(func(labels, values))
    except ValueError:
        return float("nan")


def _nan_metrics() -> dict[str, float]:
    return {
        name: float("nan")
        for name in ("auc", "prauc", "f1", "mcc", "sensitivity", "specificity", "balanced_accuracy", "fpr", "ece")
    }
