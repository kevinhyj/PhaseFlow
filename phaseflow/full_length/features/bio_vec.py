from __future__ import annotations

import math
from typing import Any

import numpy as np


BIO_VEC_NAMES = [
    "log_length",
    "length_scaled",
    "idr_fraction",
    "ordered_fraction",
    "prld_fraction",
    "low_complexity_fraction",
    "ncpr",
    "charge_kappa_proxy",
    "sticker_spacer_kappa_proxy",
    "frac_g",
    "frac_p",
    "frac_r",
    "frac_y",
    "frac_f",
    "frac_w",
    "frac_fyw",
    "rgg_density",
    "aromatic_cluster_density",
    "charge_blockiness",
    "hydropathy_mean",
    "rna_binding_proxy",
    "dna_binding_proxy",
    "ptm_density_proxy",
    "contact_density",
    "protenix_available",
    "graph_node_log",
    "graph_edge_log",
    "esm_mean",
    "esm_std",
    "starling_mean_norm",
    "starling_std_norm",
    "starling_compaction_proxy",
    "long_range_contact_fraction",
]


BIO_VEC_DIM = len(BIO_VEC_NAMES)

HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}


def make_bio_vec(
    *,
    sequence: str,
    physchem: np.ndarray | None = None,
    disorder: np.ndarray | None = None,
    plm: np.ndarray | None = None,
    protenix: np.ndarray | None = None,
    starling: np.ndarray | None = None,
    edge_src: np.ndarray | None = None,
    edge_dst: np.ndarray | None = None,
    graph_num_nodes: Any = None,
    graph_num_edges: Any = None,
) -> np.ndarray:
    seq = "".join(ch for ch in str(sequence).upper() if ch.isalpha())
    length = max(len(seq), 1)
    counts = {aa: seq.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    frac = {aa: counts[aa] / length for aa in counts}
    pos = counts["K"] + counts["R"]
    neg = counts["D"] + counts["E"]
    charged = pos + neg
    aromatic = counts["F"] + counts["Y"] + counts["W"]
    fyw_positions = [idx for idx, aa in enumerate(seq) if aa in {"F", "Y", "W"}]
    charge_positions = [idx for idx, aa in enumerate(seq) if aa in {"K", "R", "D", "E"}]
    sticker_positions = [idx for idx, aa in enumerate(seq) if aa in {"R", "K", "F", "Y", "W"}]

    idr_fraction = _safe_mean(disorder[:, 0] if _matrix_has_rows(disorder) else None, default=_simple_idr_fraction(seq))
    prld_fraction = _simple_prld_fraction(seq)
    low_complexity = _low_complexity_fraction(seq)
    ncpr = (pos - neg) / length
    charge_kappa = abs(pos - neg) / max(charged, 1)
    sticker_kappa = _gap_cv(sticker_positions)
    charge_blockiness = _gap_cv(charge_positions)
    aromatic_cluster_density = _cluster_count(fyw_positions, max_gap=3) / length
    rgg_density = seq.count("RGG") / length
    hydropathy = sum(HYDROPATHY.get(aa, 0.0) for aa in seq) / length / 4.5
    rna_proxy = min(1.0, 0.5 * frac["R"] + 8.0 * rgg_density + 0.25 * frac["G"])
    dna_proxy = min(1.0, frac["K"] + frac["R"])
    ptm_proxy = frac["S"] + frac["T"] + frac["Y"]

    nodes = _float_or_default(graph_num_nodes, length)
    edges = _float_or_default(graph_num_edges, _edge_count(edge_src))
    contact_density = edges / max(length * max(length - 1, 1), 1)
    protenix_available = float(_matrix_has_signal(protenix))
    esm_mean = _safe_mean(plm, default=0.0)
    esm_std = _safe_std(plm, default=0.0)
    star_mean_norm, star_std_norm = _embedding_norm_summary(starling)
    star_compaction = star_mean_norm / max(star_std_norm, 1.0e-6) if star_mean_norm > 0.0 else 0.0
    long_range = _long_range_fraction(edge_src, edge_dst, min_separation=24)

    values = np.asarray(
        [
            math.log1p(length) / math.log1p(4096.0),
            min(length / 2048.0, 4.0),
            idr_fraction,
            1.0 - idr_fraction,
            prld_fraction,
            low_complexity,
            ncpr,
            charge_kappa,
            sticker_kappa,
            frac["G"],
            frac["P"],
            frac["R"],
            frac["Y"],
            frac["F"],
            frac["W"],
            aromatic / length,
            rgg_density,
            aromatic_cluster_density,
            charge_blockiness,
            hydropathy,
            rna_proxy,
            dna_proxy,
            ptm_proxy,
            min(contact_density, 1.0),
            protenix_available,
            math.log1p(max(nodes, 0.0)) / math.log1p(4096.0),
            math.log1p(max(edges, 0.0)) / math.log1p(200000.0),
            esm_mean,
            esm_std,
            star_mean_norm,
            star_std_norm,
            min(star_compaction, 10.0) / 10.0,
            long_range,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)


def _matrix_has_rows(value: np.ndarray | None) -> bool:
    return isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] > 0


def _matrix_has_signal(value: np.ndarray | None) -> bool:
    return _matrix_has_rows(value) and bool(np.isfinite(value).any()) and float(np.abs(value).sum()) > 0.0


def _safe_mean(value: np.ndarray | None, *, default: float) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return float(default)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float(default)


def _safe_std(value: np.ndarray | None, *, default: float) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return float(default)
    finite = arr[np.isfinite(arr)]
    return float(finite.std()) if finite.size else float(default)


def _float_or_default(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _simple_idr_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    disorder_promoting = set("ARGQSEKP")
    return sum(aa in disorder_promoting for aa in seq) / len(seq)


def _simple_prld_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    prld = set("QNGSY")
    return sum(aa in prld for aa in seq) / len(seq)


def _low_complexity_fraction(seq: str, window: int = 12, threshold: float = 0.55) -> float:
    if not seq:
        return 0.0
    if len(seq) < window:
        return float(max(seq.count(aa) for aa in set(seq)) / max(len(seq), 1) >= threshold)
    marked = np.zeros(len(seq), dtype=bool)
    for start in range(0, len(seq) - window + 1):
        part = seq[start : start + window]
        if max(part.count(aa) for aa in set(part)) / window >= threshold:
            marked[start : start + window] = True
    return float(marked.mean())


def _gap_cv(positions: list[int]) -> float:
    if len(positions) < 3:
        return 0.0
    gaps = np.diff(np.asarray(positions, dtype=np.float32))
    mean = float(gaps.mean())
    if mean <= 0.0:
        return 0.0
    return float(min(gaps.std() / mean, 5.0) / 5.0)


def _cluster_count(positions: list[int], *, max_gap: int) -> int:
    if not positions:
        return 0
    clusters = 1
    for prev, cur in zip(positions, positions[1:], strict=False):
        if cur - prev > max_gap:
            clusters += 1
    return clusters


def _edge_count(edge_src: np.ndarray | None) -> float:
    if edge_src is None:
        return 0.0
    arr = np.asarray(edge_src)
    return float(arr.size)


def _long_range_fraction(edge_src: np.ndarray | None, edge_dst: np.ndarray | None, *, min_separation: int) -> float:
    if edge_src is None or edge_dst is None:
        return 0.0
    src = np.asarray(edge_src, dtype=np.int64).reshape(-1)
    dst = np.asarray(edge_dst, dtype=np.int64).reshape(-1)
    if src.size == 0 or dst.size == 0:
        return 0.0
    n = min(src.size, dst.size)
    sep = np.abs(src[:n] - dst[:n])
    return float((sep >= int(min_separation)).mean())


def _embedding_norm_summary(value: np.ndarray | None) -> tuple[float, float]:
    if not _matrix_has_rows(value):
        return 0.0, 0.0
    arr = np.asarray(value, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1)
    finite = norms[np.isfinite(norms)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite.mean() / 100.0), float(finite.std() / 100.0)
