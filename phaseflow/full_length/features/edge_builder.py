from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EDGE_LOCAL_SEQUENCE = 0
EDGE_AF_CONTACT = 1
EDGE_STARLING_CONTACT = 2
EDGE_PHYSCHEM = 3
EDGE_CANDIDATE_SEGMENT = 4
EDGE_TYPE_OFFSET = 3
EDGE_TYPE_WIDTH = 8
EDGE_DISTANCE_MEAN_INDEX = EDGE_TYPE_OFFSET + EDGE_TYPE_WIDTH
EDGE_DISTANCE_VAR_INDEX = EDGE_DISTANCE_MEAN_INDEX + 1


@dataclass(slots=True)
class SparseEdges:
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_type: np.ndarray
    edge_attr: np.ndarray


def build_edges(
    length: int,
    local_window: int = 16,
    af_contacts: np.ndarray | None = None,
    star_contacts: np.ndarray | None = None,
    physchem: np.ndarray | None = None,
    segment_ids: np.ndarray | None = None,
    edge_dim: int = 8,
    af_topk: int = 16,
    star_topk: int = 16,
    physchem_topk: int = 8,
    segment_topk: int = 8,
) -> SparseEdges:
    rows: list[tuple[int, int, int, np.ndarray]] = []
    denom = max(length - 1, 1)
    for src in range(length):
        start = max(0, src - local_window)
        end = min(length, src + local_window + 1)
        for dst in range(start, end):
            rows.append((src, dst, EDGE_LOCAL_SEQUENCE, _edge_attr(src, dst, EDGE_LOCAL_SEQUENCE, 1.0, denom, edge_dim)))
    _add_contact_edges(rows, af_contacts, EDGE_AF_CONTACT, af_topk, denom, edge_dim)
    _add_contact_edges(rows, star_contacts, EDGE_STARLING_CONTACT, star_topk, denom, edge_dim)
    _add_physchem_edges(rows, physchem, EDGE_PHYSCHEM, physchem_topk, local_window, denom, edge_dim)
    _add_segment_edges(rows, segment_ids, EDGE_CANDIDATE_SEGMENT, segment_topk, local_window, denom, edge_dim)

    if not rows:
        return SparseEdges(
            edge_src=np.zeros((0,), dtype=np.int64),
            edge_dst=np.zeros((0,), dtype=np.int64),
            edge_type=np.zeros((0,), dtype=np.int64),
            edge_attr=np.zeros((0, edge_dim), dtype=np.float32),
        )
    edge_src = np.asarray([row[0] for row in rows], dtype=np.int64)
    edge_dst = np.asarray([row[1] for row in rows], dtype=np.int64)
    edge_type = np.asarray([row[2] for row in rows], dtype=np.int64)
    edge_attr = np.stack([row[3] for row in rows]).astype(np.float32)
    return SparseEdges(edge_src=edge_src, edge_dst=edge_dst, edge_type=edge_type, edge_attr=edge_attr)


def _add_contact_edges(
    rows: list[tuple[int, int, int, np.ndarray]],
    contacts: np.ndarray | None,
    edge_type: int,
    topk: int,
    denom: int,
    edge_dim: int,
) -> None:
    if contacts is None or contacts.size == 0:
        return
    if contacts.ndim != 2 or contacts.shape[1] < 3:
        raise ValueError("contacts must have shape [E, >=3] as src,dst,confidence")
    by_src: dict[int, list[tuple[int, float, float, float]]] = {}
    for row in contacts.tolist():
        src, dst, confidence = row[:3]
        mean_distance = float(row[3]) if len(row) > 3 else 0.0
        distance_variance = float(row[4]) if len(row) > 4 else 0.0
        by_src.setdefault(int(src), []).append((int(dst), float(confidence), mean_distance, distance_variance))
    for src, entries in by_src.items():
        entries.sort(key=lambda item: item[1], reverse=True)
        for dst, confidence, mean_distance, distance_variance in entries[:topk]:
            rows.append(
                (
                    src,
                    dst,
                    edge_type,
                    _edge_attr(
                        src,
                        dst,
                        edge_type,
                        confidence,
                        denom,
                        edge_dim,
                        mean_distance=mean_distance,
                        distance_variance=distance_variance,
                    ),
                )
            )


def _add_physchem_edges(
    rows: list[tuple[int, int, int, np.ndarray]],
    physchem: np.ndarray | None,
    edge_type: int,
    topk: int,
    local_window: int,
    denom: int,
    edge_dim: int,
) -> None:
    if physchem is None or physchem.size == 0 or topk <= 0:
        return
    if physchem.ndim != 2 or physchem.shape[1] < 29:
        return
    length = physchem.shape[0]
    positive = physchem[:, 20]
    negative = physchem[:, 21]
    hydrophobic = physchem[:, 26]
    sticker = physchem[:, 27]
    for src in range(length):
        scores: list[tuple[int, float]] = []
        for dst in range(length):
            if dst == src or abs(dst - src) <= local_window:
                continue
            charge_complement = positive[src] * negative[dst] + negative[src] * positive[dst]
            sticker_pair = sticker[src] * sticker[dst]
            hydrophobic_pair = hydrophobic[src] * hydrophobic[dst]
            score = float(charge_complement + 0.75 * sticker_pair + 0.35 * hydrophobic_pair)
            if score > 0.0:
                scores.append((dst, min(score, 1.0)))
        scores.sort(key=lambda item: item[1], reverse=True)
        for dst, confidence in scores[:topk]:
            rows.append((src, dst, edge_type, _edge_attr(src, dst, edge_type, confidence, denom, edge_dim)))


def _add_segment_edges(
    rows: list[tuple[int, int, int, np.ndarray]],
    segment_ids: np.ndarray | None,
    edge_type: int,
    topk: int,
    local_window: int,
    denom: int,
    edge_dim: int,
) -> None:
    if segment_ids is None or segment_ids.size == 0 or topk <= 0:
        return
    ids = np.asarray(segment_ids).reshape(-1)
    by_segment: dict[float, list[int]] = {}
    for index, value in enumerate(ids.tolist()):
        if float(value) <= 0.0:
            continue
        by_segment.setdefault(float(value), []).append(index)
    for members in by_segment.values():
        for src in members:
            candidates = [dst for dst in members if dst != src and abs(dst - src) > local_window]
            candidates.sort(key=lambda dst: abs(dst - src))
            for dst in candidates[:topk]:
                confidence = 1.0 / (1.0 + abs(dst - src) / max(denom, 1))
                rows.append((src, dst, edge_type, _edge_attr(src, dst, edge_type, confidence, denom, edge_dim)))


def _edge_attr(
    src: int,
    dst: int,
    edge_type: int,
    confidence: float,
    denom: int,
    edge_dim: int,
    *,
    mean_distance: float = 0.0,
    distance_variance: float = 0.0,
) -> np.ndarray:
    attr = np.zeros(edge_dim, dtype=np.float32)
    attr[0] = abs(dst - src) / max(denom, 1)
    if edge_dim > 1:
        attr[1] = np.sign(dst - src)
    if edge_dim > 2:
        attr[2] = confidence
    if edge_dim > EDGE_TYPE_OFFSET:
        type_index = EDGE_TYPE_OFFSET + min(edge_type, max(min(EDGE_TYPE_WIDTH, edge_dim - EDGE_TYPE_OFFSET) - 1, 0))
        if type_index < edge_dim:
            attr[type_index] = 1.0
    if edge_dim > EDGE_DISTANCE_MEAN_INDEX and mean_distance > 0.0:
        length_scale = max(np.sqrt(max(denom + 1, 1)) * 3.8, 1.0)
        attr[EDGE_DISTANCE_MEAN_INDEX] = float(np.clip(mean_distance / length_scale, 0.0, 1.0))
    if edge_dim > EDGE_DISTANCE_VAR_INDEX and distance_variance > 0.0:
        attr[EDGE_DISTANCE_VAR_INDEX] = float(np.clip(distance_variance / (11.0**2), 0.0, 1.0))
    return attr
