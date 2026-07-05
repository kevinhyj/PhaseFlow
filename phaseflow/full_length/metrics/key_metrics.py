from __future__ import annotations

import numpy as np

from phaseflow.full_length.data.schemas import IGNORE_INDEX


def key_topk_metrics(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> dict[str, float]:
    if isinstance(labels, np.ndarray) and labels.dtype != object:
        label_iter = list(labels)
    else:
        label_iter = list(labels)
    if isinstance(scores, np.ndarray) and scores.dtype != object:
        score_iter = list(scores)
    else:
        score_iter = list(scores)
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    for sample_labels, sample_scores in zip(label_iter, score_iter, strict=False):
        sample_labels = np.asarray(sample_labels)
        sample_scores = np.asarray(sample_scores)
        valid = sample_labels != IGNORE_INDEX
        if not np.any(valid):
            continue
        valid_labels = sample_labels[valid].astype(int)
        valid_scores = sample_scores[valid].astype(float)
        positives = int(np.sum(valid_labels == 1))
        if positives == 0:
            continue
        top_count = min(k, len(valid_scores))
        order = np.argsort(-valid_scores)[:top_count]
        hits = valid_labels[order]
        precisions.append(float(np.sum(hits == 1) / top_count))
        recalls.append(float(np.sum(hits == 1) / positives))
        gains = hits / np.log2(np.arange(top_count) + 2)
        ideal = np.ones(min(positives, top_count)) / np.log2(np.arange(min(positives, top_count)) + 2)
        ndcgs.append(float(np.sum(gains) / max(np.sum(ideal), 1.0e-8)))
    return {
        f"key_top{k}_precision": float(np.mean(precisions)) if precisions else np.nan,
        f"key_top{k}_recall": float(np.mean(recalls)) if recalls else np.nan,
        f"key_top{k}_ndcg": float(np.mean(ndcgs)) if ndcgs else np.nan,
    }
