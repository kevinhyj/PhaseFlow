from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from phaseflow.full_length.features.disorder import compute_disorder_features


STARLING_NODE_DIM = 8
STARLING_EMBED_DIM = 512


@dataclass(slots=True)
class StarlingSegment:
    start: int
    end: int
    sequence: str
    name: str


def zero_starling_features(length: int, dim: int = STARLING_NODE_DIM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node = np.zeros((length, dim), dtype=np.float32)
    missing = np.ones(length, dtype=np.float32)
    reliability = np.zeros(length, dtype=np.float32)
    return node, missing, reliability


def zero_starling_embedding(length: int, dim: int = STARLING_EMBED_DIM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embedding = np.zeros((length, dim), dtype=np.float32)
    missing = np.ones(length, dtype=np.float32)
    reliability = np.zeros(length, dtype=np.float32)
    return embedding, missing, reliability


def load_starling_embedding(path: str | Path, sequence: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        cached_sequence = str(data["sequence"].item()) if "sequence" in data else sequence
        if cached_sequence and cached_sequence != sequence:
            raise ValueError(f"STARLING embedding sequence mismatch for {path}")
        embedding = _optional_array(data, "embedding", "starling_embed")
        if embedding is None:
            raise ValueError(f"STARLING embedding file {path} has no embedding/starling_embed array")
        if embedding.ndim != 2 or embedding.shape[0] != len(sequence):
            raise ValueError(f"STARLING embedding in {path} must have shape [L, D], got {embedding.shape}")
    missing = np.zeros(len(sequence), dtype=np.float32)
    reliability = np.ones(len(sequence), dtype=np.float32)
    metadata = {
        "starling_embedding_success": "1",
        "starling_embedding_path": str(path),
        "starling_embedding_dim": str(int(embedding.shape[1])),
    }
    return embedding.astype(np.float32, copy=False), missing, reliability, metadata


def load_starling_distance_contacts(
    path: str | Path,
    sequence: str,
    *,
    contact_threshold: float = 11.0,
    contact_topk: int = 48,
    min_contact_probability: float = 0.05,
) -> tuple[np.ndarray, dict[str, object]]:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        cached_sequence = handle.attrs.get("sequence", sequence)
        if isinstance(cached_sequence, bytes):
            cached_sequence = cached_sequence.decode("utf-8")
        if cached_sequence and cached_sequence != sequence:
            raise ValueError(f"STARLING distance-map sequence mismatch for {path}")
        if "distance_maps" not in handle:
            raise ValueError(f"STARLING distance-map file {path} has no distance_maps dataset")
        distance_maps = np.asarray(handle["distance_maps"], dtype=np.float32)
    contacts = starling_contacts_from_distance_maps(
        distance_maps,
        sequence,
        contact_threshold=contact_threshold,
        contact_topk=contact_topk,
        min_contact_probability=min_contact_probability,
    )
    metadata = {
        "starling_distance_success": "1",
        "starling_distance_path": str(path),
        "starling_distance_conformations": str(int(distance_maps.shape[0])),
        "starling_distance_contact_topk": str(int(contact_topk)),
    }
    return contacts, metadata


def load_starling_features(path: str | Path, sequence: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load parsed STARLING full-length features from an intermediate npz file.

    Expected keys:
    - `node` or `node_features`: [L, d]
    - `missing_mask`: [L], 0 where STARLING features are available
    - optional `reliability`: [L]
    - optional `contacts`: [E, >=3] as src, dst, confidence
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        cached_sequence = str(data["sequence"].item()) if "sequence" in data else sequence
        if cached_sequence and cached_sequence != sequence:
            raise ValueError(f"STARLING sequence mismatch for {path}: cache sequence differs from feature cache sequence")
        node = _optional_array(data, "node", "node_features")
        if node is None:
            raise ValueError(f"STARLING feature file {path} has no node/node_features array")
        if node.ndim != 2 or node.shape[0] != len(sequence):
            raise ValueError(f"STARLING node features in {path} must have shape [L, D], got {node.shape}")
        missing = (
            np.asarray(data["missing_mask"], dtype=np.float32)
            if "missing_mask" in data
            else np.zeros(len(sequence), dtype=np.float32)
        )
        reliability = (
            np.asarray(data["reliability"], dtype=np.float32)
            if "reliability" in data
            else 1.0 - missing.astype(np.float32)
        )
        contacts = np.asarray(data["contacts"], dtype=np.float32) if "contacts" in data else None
    return node.astype(np.float32, copy=False), missing, reliability, contacts


def candidate_starling_segments(
    protein_id: str,
    sequence: str,
    *,
    max_segment_length: int = 384,
    min_segment_length: int = 16,
    merge_gap: int = 12,
    flank: int = 8,
    score_threshold: float = 0.5,
) -> list[StarlingSegment]:
    """Select full-length-aligned segments where STARLING evidence is meaningful.

    STARLING has a hard maximum sequence length. For proteins that fit, we run
    the full sequence so every residue can receive ensemble evidence. For longer
    proteins, only disorder/LCR/PrLD-like candidate windows are simulated and
    mapped back to the original residue indices.
    """
    sequence = sequence.upper()
    length = len(sequence)
    max_segment_length = max(int(max_segment_length), 1)
    min_segment_length = max(int(min_segment_length), 1)
    if length <= max_segment_length:
        return [StarlingSegment(0, length, sequence, protein_id)]

    disorder, _, _, _ = compute_disorder_features(sequence, mode="simple")
    score = np.max(disorder[:, :3], axis=1)
    mask = score >= float(score_threshold)
    spans = _mask_to_spans(mask, min_segment_length=min_segment_length, merge_gap=merge_gap, flank=flank, length=length)
    segments: list[StarlingSegment] = []
    for span_index, (start, end) in enumerate(spans, start=1):
        for chunk_index, (chunk_start, chunk_end) in enumerate(
            _split_span(start, end, max_segment_length=max_segment_length, overlap=min(32, max_segment_length // 4)),
            start=1,
        ):
            subseq = sequence[chunk_start:chunk_end]
            if len(subseq) < min_segment_length:
                continue
            name = f"{protein_id}_starling_{span_index:03d}_{chunk_index:02d}_{chunk_start + 1}_{chunk_end}"
            segments.append(StarlingSegment(chunk_start, chunk_end, subseq, name))
    return segments


def load_starling_ensemble_file(path: str | Path):
    """Load a STARLING `.starling` file without importing STARLING at module import time."""
    try:
        from starling.structure.ensemble import load_ensemble  # type: ignore
    except Exception as exc:  # pragma: no cover - only hit when optional dep is absent.
        raise RuntimeError("STARLING external mode requires the `starling` Python package") from exc
    return load_ensemble(str(path), ignore_structures=True)


def starling_features_from_distance_maps(
    distance_maps: np.ndarray,
    sequence: str,
    *,
    contact_threshold: float = 11.0,
    contact_topk: int = 16,
    min_contact_probability: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert STARLING ensemble distance maps into PhaseFlow node/contact features."""
    distance_maps = np.asarray(distance_maps, dtype=np.float32)
    if distance_maps.ndim != 3:
        raise ValueError(f"STARLING distance maps must have shape [N, L, L], got {distance_maps.shape}")
    length = len(sequence)
    if distance_maps.shape[1:] != (length, length):
        raise ValueError(
            f"STARLING distance maps shape {distance_maps.shape} does not match sequence length {length}"
        )
    if length == 0:
        return zero_starling_features(0)[0], np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    mean_distance = np.nanmean(distance_maps, axis=0)
    distance_variance = np.nanvar(distance_maps, axis=0)
    contact_probability = np.nanmean(distance_maps <= float(contact_threshold), axis=0).astype(np.float32)
    np.fill_diagonal(contact_probability, 0.0)

    denom = max(length - 1, 1)
    contact_degree = np.sum(contact_probability, axis=1) / denom
    contact_entropy = _contact_entropy(contact_probability)
    variance_norm = np.clip(np.nanmean(distance_variance, axis=1) / max(float(contact_threshold) ** 2, 1.0), 0.0, 1.0)
    compactness = _compactness_from_distance(mean_distance, contact_threshold)
    local_rg = _local_radius_from_mean_distance(mean_distance, window=15)
    rg_values = _radius_of_gyration(distance_maps)
    rg_norm = np.full(length, _normalize_length_scale(float(np.nanmean(rg_values)), length), dtype=np.float32)
    end_to_end = float(np.nanmean(distance_maps[:, 0, length - 1])) if length > 1 else 0.0
    re_norm = np.full(length, _normalize_length_scale(end_to_end, length), dtype=np.float32)
    availability = np.ones(length, dtype=np.float32)

    node = np.stack(
        [
            np.clip(contact_degree, 0.0, 1.0),
            np.clip(contact_entropy, 0.0, 1.0),
            variance_norm,
            np.clip(compactness, 0.0, 1.0),
            np.clip(local_rg, 0.0, 1.0),
            rg_norm,
            re_norm,
            availability,
        ],
        axis=1,
    ).astype(np.float32)
    missing = np.zeros(length, dtype=np.float32)
    reliability = np.full(length, min(1.0, math.sqrt(max(distance_maps.shape[0], 1) / 100.0)), dtype=np.float32)
    contacts = _contacts_from_contact_probability(
        contact_probability,
        mean_distance,
        topk=contact_topk,
        min_probability=min_contact_probability,
    )
    return node, missing, reliability, contacts


def starling_contacts_from_distance_maps(
    distance_maps: np.ndarray,
    sequence: str,
    *,
    contact_threshold: float = 11.0,
    contact_topk: int = 48,
    min_contact_probability: float = 0.05,
) -> np.ndarray:
    distance_maps = np.asarray(distance_maps, dtype=np.float32)
    if distance_maps.ndim != 3:
        raise ValueError(f"STARLING distance maps must have shape [N, L, L], got {distance_maps.shape}")
    length = len(sequence)
    if distance_maps.shape[1:] != (length, length):
        raise ValueError(
            f"STARLING distance maps shape {distance_maps.shape} does not match sequence length {length}"
        )
    if length == 0:
        return np.zeros((0, 5), dtype=np.float32)
    mean_distance = np.nanmean(distance_maps, axis=0)
    distance_variance = np.nanvar(distance_maps, axis=0)
    contact_probability = np.nanmean(distance_maps <= float(contact_threshold), axis=0).astype(np.float32)
    np.fill_diagonal(contact_probability, 0.0)
    return _contacts_from_contact_probability(
        contact_probability,
        mean_distance,
        distance_variance=distance_variance,
        topk=contact_topk,
        min_probability=min_contact_probability,
    )


def assemble_starling_segments(
    length: int,
    segment_results: list[tuple[StarlingSegment, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node, missing, reliability = zero_starling_features(length, dim=STARLING_NODE_DIM)
    node_sum = np.zeros_like(node)
    reliability_sum = np.zeros(length, dtype=np.float32)
    coverage = np.zeros(length, dtype=np.float32)
    contact_rows: list[np.ndarray] = []
    for segment, segment_node, _, segment_reliability, segment_contacts in segment_results:
        start, end = int(segment.start), int(segment.end)
        span = end - start
        if span <= 0:
            continue
        node_sum[start:end] += segment_node[:span]
        reliability_sum[start:end] += segment_reliability[:span]
        coverage[start:end] += 1.0
        if segment_contacts.size:
            shifted = segment_contacts.copy()
            shifted[:, 0] += start
            shifted[:, 1] += start
            contact_rows.append(shifted)
    covered = coverage > 0
    if np.any(covered):
        node[covered] = node_sum[covered] / coverage[covered, None]
        reliability[covered] = reliability_sum[covered] / coverage[covered]
        missing[covered] = 0.0
    contacts = np.concatenate(contact_rows, axis=0).astype(np.float32) if contact_rows else np.zeros((0, 4), dtype=np.float32)
    return node.astype(np.float32), missing.astype(np.float32), reliability.astype(np.float32), contacts


def heuristic_starling_features(sequence: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback IDR-like full-length STARLING proxy used only when explicitly requested.

    This is not a STARLING ensemble simulation. It creates an 8D node feature
    surface that lets downstream parsing and cache integration be tested without
    the external STARLING package.
    """
    length = len(sequence)
    node = np.zeros((length, 8), dtype=np.float32)
    disorder_like = set("GPSQNERK")
    aromatic = set("FYW")
    charged = set("DEKR")
    for index, aa in enumerate(sequence):
        start = max(0, index - 15)
        end = min(length, index + 16)
        window = sequence[start:end]
        denom = max(len(window), 1)
        node[index, 0] = sum(residue in disorder_like for residue in window) / denom
        node[index, 1] = sum(residue in aromatic for residue in window) / denom
        node[index, 2] = sum(residue in charged for residue in window) / denom
        node[index, 3] = window.count("G") / denom
        node[index, 4] = window.count("P") / denom
        node[index, 5] = window.count("Q") / denom
        node[index, 6] = window.count("N") / denom
        node[index, 7] = index / max(length - 1, 1)
    missing = np.ones(length, dtype=np.float32)
    reliability = np.zeros(length, dtype=np.float32)
    return node, missing, reliability


def _mask_to_spans(
    mask: np.ndarray,
    *,
    min_segment_length: int,
    merge_gap: int,
    flank: int,
    length: int,
) -> list[tuple[int, int]]:
    raw: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if bool(flag) and start is None:
            start = index
        elif not bool(flag) and start is not None:
            raw.append((start, index))
            start = None
    if start is not None:
        raw.append((start, len(mask)))
    if not raw:
        return []

    merged: list[tuple[int, int]] = []
    for start, end in raw:
        start = max(0, start - flank)
        end = min(length, end + flank)
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [(start, end) for start, end in merged if end - start >= min_segment_length]


def _split_span(start: int, end: int, *, max_segment_length: int, overlap: int) -> list[tuple[int, int]]:
    if end - start <= max_segment_length:
        return [(start, end)]
    chunks: list[tuple[int, int]] = []
    cursor = start
    step = max(max_segment_length - max(overlap, 0), 1)
    while cursor < end:
        chunk_end = min(cursor + max_segment_length, end)
        chunks.append((cursor, chunk_end))
        if chunk_end == end:
            break
        cursor += step
    return chunks


def _contact_entropy(contact_probability: np.ndarray) -> np.ndarray:
    p = np.clip(contact_probability.astype(np.float32), 1.0e-6, 1.0 - 1.0e-6)
    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    np.fill_diagonal(entropy, 0.0)
    return np.sum(entropy, axis=1) / max(contact_probability.shape[0] - 1, 1)


def _compactness_from_distance(mean_distance: np.ndarray, contact_threshold: float) -> np.ndarray:
    length = mean_distance.shape[0]
    denom = max(length - 1, 1)
    row_sum = np.sum(mean_distance, axis=1)
    mean_row_distance = row_sum / denom
    return 1.0 / (1.0 + mean_row_distance / max(contact_threshold, 1.0))


def _local_radius_from_mean_distance(mean_distance: np.ndarray, window: int) -> np.ndarray:
    length = mean_distance.shape[0]
    values = np.zeros(length, dtype=np.float32)
    half = max(window // 2, 1)
    for index in range(length):
        start = max(0, index - half)
        end = min(length, index + half + 1)
        sub = mean_distance[start:end, start:end]
        if sub.size == 0:
            continue
        rg = math.sqrt(float(np.sum(np.square(sub))) / max(2 * (end - start) ** 2, 1))
        values[index] = _normalize_length_scale(rg, end - start)
    return values


def _radius_of_gyration(distance_maps: np.ndarray) -> np.ndarray:
    length = distance_maps.shape[1]
    return np.sqrt(np.sum(np.square(distance_maps), axis=(1, 2)) / max(2 * length**2, 1)).astype(np.float32)


def _normalize_length_scale(value: float, length: int) -> float:
    scale = max(math.sqrt(max(length, 1)) * 3.8, 1.0)
    return float(np.clip(value / scale, 0.0, 1.0))


def _contacts_from_contact_probability(
    contact_probability: np.ndarray,
    mean_distance: np.ndarray,
    *,
    distance_variance: np.ndarray | None = None,
    topk: int,
    min_probability: float,
) -> np.ndarray:
    rows: list[tuple[int, int, float, float, float]] = []
    if distance_variance is None:
        distance_variance = np.zeros_like(mean_distance, dtype=np.float32)
    length = contact_probability.shape[0]
    for src in range(length):
        order = np.argsort(-contact_probability[src])
        added = 0
        for dst in order:
            if int(dst) == src:
                continue
            probability = float(contact_probability[src, dst])
            if probability < float(min_probability):
                break
            rows.append(
                (
                    src,
                    int(dst),
                    probability,
                    float(mean_distance[src, dst]),
                    float(distance_variance[src, dst]),
                )
            )
            added += 1
            if added >= topk:
                break
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def _optional_array(data: np.lib.npyio.NpzFile, *names: str) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name], dtype=np.float32)
    return None
