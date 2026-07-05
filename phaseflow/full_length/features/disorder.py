from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DISORDER_FEATURE_NAMES = [
    "p_disorder",
    "p_lcr",
    "p_prld",
    "idr_segment_id_norm",
    "idr_segment_len_norm",
    "distance_to_idr_boundary_norm",
]

DISORDER_PROMOTING = frozenset("GPQNSRY")
ORDER_PROMOTING = frozenset("WCFILVM")
PRLD_AA = frozenset("PQNGSY")


def compute_disorder_features(
    sequence: str,
    mode: str = "simple",
    precomputed_path: str | Path | None = None,
    protein_id: str | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Return disorder features, missing mask and reliability for one sequence."""
    sequence = sequence.upper()
    if mode == "none":
        length = len(sequence)
        return (
            np.zeros((length, len(DISORDER_FEATURE_NAMES)), dtype=np.float32),
            DISORDER_FEATURE_NAMES.copy(),
            np.ones(length, dtype=np.float32),
            np.zeros(length, dtype=np.float32),
        )
    if mode == "precomputed":
        if precomputed_path is None:
            raise ValueError("precomputed_path is required when mode='precomputed'")
        features = _read_precomputed(sequence, precomputed_path, protein_id)
    elif mode == "simple":
        features = _simple_disorder(sequence)
    else:
        raise ValueError(f"Unsupported disorder mode: {mode}")

    missing = np.zeros(len(sequence), dtype=np.float32)
    reliability = np.full(len(sequence), 0.6 if mode == "simple" else 1.0, dtype=np.float32)
    return features.astype(np.float32), DISORDER_FEATURE_NAMES.copy(), missing, reliability


def _read_precomputed(sequence: str, path: str | Path, protein_id: str | None) -> np.ndarray:
    frame = pd.read_csv(path, sep=None, engine="python")
    if protein_id is not None and "protein_id" in frame.columns:
        frame = frame.loc[frame["protein_id"].astype(str) == str(protein_id)]
    required = {"pos", "p_disorder"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Precomputed disorder file is missing columns: {sorted(missing)}")
    frame = frame.sort_values("pos")
    if len(frame) != len(sequence):
        raise ValueError("Precomputed disorder length does not match sequence length")
    values = np.zeros((len(sequence), len(DISORDER_FEATURE_NAMES)), dtype=np.float32)
    values[:, 0] = frame["p_disorder"].to_numpy(dtype=np.float32)
    for index, optional_name in enumerate(["p_lcr", "p_prld"], start=1):
        if optional_name in frame.columns:
            values[:, index] = frame[optional_name].to_numpy(dtype=np.float32)
    _fill_segment_features(values)
    return values


def _simple_disorder(sequence: str) -> np.ndarray:
    length = len(sequence)
    features = np.zeros((length, len(DISORDER_FEATURE_NAMES)), dtype=np.float32)
    for index in range(length):
        start = max(0, index - 7)
        end = min(length, index + 8)
        window = sequence[start:end]
        disorder_fraction = sum(aa in DISORDER_PROMOTING for aa in window) / max(len(window), 1)
        order_fraction = sum(aa in ORDER_PROMOTING for aa in window) / max(len(window), 1)
        prld_fraction = sum(aa in PRLD_AA for aa in window) / max(len(window), 1)
        unique_fraction = len(set(window)) / max(len(window), 1)
        features[index, 0] = np.clip(0.25 + 0.9 * disorder_fraction - 0.45 * order_fraction, 0.0, 1.0)
        features[index, 1] = np.clip(1.0 - unique_fraction, 0.0, 1.0)
        features[index, 2] = np.clip(prld_fraction, 0.0, 1.0)
    _fill_segment_features(features)
    return features


def _fill_segment_features(features: np.ndarray, threshold: float = 0.5) -> None:
    mask = features[:, 0] >= threshold
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

    length = max(len(mask), 1)
    for segment_index, (start, end) in enumerate(segments, start=1):
        segment_len = end - start + 1
        for index in range(start, end + 1):
            features[index, 3] = segment_index / max(len(segments), 1)
            features[index, 4] = segment_len / length
            features[index, 5] = min(index - start, end - index) / max(segment_len, 1)
