from __future__ import annotations

import math
from collections import Counter

import numpy as np

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AA20)}
KYTE_DOOLITTLE = {
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
POSITIVE = frozenset("KRH")
NEGATIVE = frozenset("DE")
POLAR = frozenset("STNQCY")
HYDROPHOBIC = frozenset("AILMFWV")
AROMATIC = frozenset("FWY")
STICKER = frozenset("YFWRLDM")
SPACER = frozenset("GPSQN")
SPECIAL = "GPRYC"
WINDOWS = (9, 15, 31, 63)
WINDOW_FEATURES = (
    "fraction_G",
    "fraction_P",
    "fraction_Y",
    "fraction_R",
    "fraction_aromatic",
    "fraction_charged",
    "fraction_polar",
    "fraction_hydrophobic",
    "NCPR",
    "FCR",
    "local_entropy",
    "sticker_density",
    "spacer_density",
    "sticker_spacer_ratio",
)


def compute_physchem_features(sequence: str, windows: tuple[int, ...] = WINDOWS) -> tuple[np.ndarray, list[str]]:
    sequence = sequence.upper()
    length = len(sequence)
    names: list[str] = [f"aa_{aa}" for aa in AA20]
    names.extend(["charge_positive", "charge_negative", "charge_neutral", "hydropathy"])
    names.extend(["aromatic", "polar", "hydrophobic", "sticker", "spacer"])
    names.extend([f"special_{aa}" for aa in SPECIAL])
    for window in windows:
        names.extend([f"w{window}_{name}" for name in WINDOW_FEATURES])

    features = np.zeros((length, len(names)), dtype=np.float32)
    for index, aa in enumerate(sequence):
        column = 0
        aa_index = AA_TO_INDEX.get(aa)
        if aa_index is not None:
            features[index, aa_index] = 1.0
        column += len(AA20)
        is_pos = aa in POSITIVE
        is_neg = aa in NEGATIVE
        features[index, column : column + 3] = [float(is_pos), float(is_neg), float(not is_pos and not is_neg)]
        column += 3
        features[index, column] = _normalize_hydropathy(KYTE_DOOLITTLE.get(aa, 0.0))
        column += 1
        features[index, column : column + 5] = [
            float(aa in AROMATIC),
            float(aa in POLAR),
            float(aa in HYDROPHOBIC),
            float(aa in STICKER),
            float(aa in SPACER),
        ]
        column += 5
        for special in SPECIAL:
            features[index, column] = float(aa == special)
            column += 1

        for window in windows:
            start = max(0, index - window // 2)
            end = min(length, index + window // 2 + 1)
            features[index, column : column + len(WINDOW_FEATURES)] = _window_features(sequence[start:end])
            column += len(WINDOW_FEATURES)

    return features, names


def _window_features(window_sequence: str) -> np.ndarray:
    if not window_sequence:
        return np.zeros(len(WINDOW_FEATURES), dtype=np.float32)
    aas = list(window_sequence)
    n = float(len(aas))
    pos = sum(aa in POSITIVE for aa in aas)
    neg = sum(aa in NEGATIVE for aa in aas)
    charged = pos + neg
    sticker = sum(aa in STICKER for aa in aas)
    spacer = sum(aa in SPACER for aa in aas)
    values = [
        aas.count("G") / n,
        aas.count("P") / n,
        aas.count("Y") / n,
        aas.count("R") / n,
        sum(aa in AROMATIC for aa in aas) / n,
        charged / n,
        sum(aa in POLAR for aa in aas) / n,
        sum(aa in HYDROPHOBIC for aa in aas) / n,
        (pos - neg) / n,
        charged / n,
        _entropy(aas),
        sticker / n,
        spacer / n,
        sticker / max(spacer, 1),
    ]
    return np.asarray(values, dtype=np.float32)


def _entropy(aas: list[str]) -> float:
    counts = Counter(aas)
    total = float(len(aas))
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    max_entropy = math.log2(min(20, len(aas))) if aas else 1.0
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def _normalize_hydropathy(value: float) -> float:
    return float((value + 4.5) / 9.0)
