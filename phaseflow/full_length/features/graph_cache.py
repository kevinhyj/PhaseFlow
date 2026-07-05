from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GRAPH_CACHE_VERSION = 1


@dataclass(slots=True)
class PrecomputedGraph:
    neighbors: np.ndarray
    edge_attr: np.ndarray
    neighbor_mask: np.ndarray


def edge_list_to_precomputed_graph(
    *,
    length: int,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_type: np.ndarray,
    edge_attr: np.ndarray,
    max_neighbors: int,
    edge_dim: int,
) -> PrecomputedGraph:
    neighbors = np.zeros((length, max_neighbors), dtype=np.int64)
    neighbor_attr = np.zeros((length, max_neighbors, edge_dim), dtype=np.float32)
    neighbor_mask = np.zeros((length, max_neighbors), dtype=np.bool_)

    if length <= 0 or max_neighbors <= 0:
        return PrecomputedGraph(neighbors=neighbors, edge_attr=neighbor_attr, neighbor_mask=neighbor_mask)

    edge_src = np.asarray(edge_src, dtype=np.int64).reshape(-1)
    edge_dst = np.asarray(edge_dst, dtype=np.int64).reshape(-1)
    edge_type = np.asarray(edge_type, dtype=np.int64).reshape(-1)
    edge_attr = np.asarray(edge_attr, dtype=np.float32)
    counts = np.zeros(length, dtype=np.int64)

    if edge_src.size:
        valid = (edge_src >= 0) & (edge_src < length) & (edge_dst >= 0) & (edge_dst < length)
        if np.any(valid):
            src = edge_src[valid]
            dst = edge_dst[valid]
            edge_type_valid = np.maximum(edge_type[valid], 0)
            distance = np.abs(dst - src)
            type_stride = length + 1
            src_stride = max(int(edge_type_valid.max()) + 1, 1) * type_stride
            sort_key = src * src_stride + edge_type_valid * type_stride + distance
            order = np.argsort(sort_key, kind="stable")

            src_sorted = src[order]
            dst_sorted = dst[order]
            counts = np.bincount(src_sorted, minlength=length).astype(np.int64, copy=False)
            starts = np.cumsum(counts) - counts
            positions = np.arange(src_sorted.size, dtype=np.int64) - np.repeat(starts, counts)
            keep = positions < max_neighbors
            if np.any(keep):
                kept_src = src_sorted[keep]
                kept_rank = positions[keep]
                neighbors[kept_src, kept_rank] = dst_sorted[keep]
                neighbor_mask[kept_src, kept_rank] = True
                if edge_attr.size:
                    width = min(edge_dim, int(edge_attr.shape[1]))
                    attr_sorted = edge_attr[valid][order]
                    neighbor_attr[kept_src, kept_rank, :width] = attr_sorted[keep, :width].astype(
                        np.float32,
                        copy=False,
                    )

    missing = counts == 0
    if np.any(missing):
        missing_src = np.arange(length, dtype=np.int64)[missing]
        neighbors[missing_src, 0] = missing_src
        neighbor_mask[missing_src, 0] = True
    return PrecomputedGraph(neighbors=neighbors, edge_attr=neighbor_attr, neighbor_mask=neighbor_mask)
