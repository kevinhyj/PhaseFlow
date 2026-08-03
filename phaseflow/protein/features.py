"""Protein sequence, embedding, and feature-cache construction."""


# Source: features/plm_embedder.py


from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def simple_plm_embedding(sequence: str, dim: int = 32) -> np.ndarray:
    """Deterministic lightweight embedding for toy tests, not a biological PLM."""
    values = np.zeros((len(sequence), dim), dtype=np.float32)
    for index, aa in enumerate(sequence.upper()):
        digest = hashlib.sha256(f"{aa}:{index % 17}".encode("utf-8")).digest()
        byte_values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        tiled = np.resize(byte_values, dim)
        values[index] = (tiled / 127.5) - 1.0
        values[index, 0] = index / max(len(sequence) - 1, 1)
    return values


@dataclass(slots=True)
class ESM2Config:
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    model_dir: str | None = None
    device: str = "auto"
    dtype: str = "float32"
    storage_dtype: str = "float32"
    local_files_only: bool = False
    max_length_policy: str = "chunk"
    chunk_size: int | None = None
    overlap: int = 128

    @property
    def model_source(self) -> str:
        return self.model_dir or self.model_name


def download_esm2_model(model_name: str = "facebook/esm2_t33_650M_UR50D", model_dir: str | Path | None = None) -> Path:
    """Download an ESM-2 Hugging Face snapshot and return its local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Downloading ESM-2 requires the optional 'huggingface_hub' dependency. "
            "Install with `python -m pip install -e '.[plm]'`."
        ) from exc
    if model_dir is None:
        return Path(snapshot_download(repo_id=model_name))
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(repo_id=model_name, local_dir=str(model_dir)))


class ESM2Embedder:
    """Frozen residue-level ESM-2 embedding extractor.

    The returned array is aligned to the raw protein sequence: special tokens
    inserted by the tokenizer are removed before values leave this class.
    """

    def __init__(self, config: ESM2Config | None = None) -> None:
        self.config = config or ESM2Config()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ESM-2 embedding requires the optional 'transformers' dependency. "
                "Install with `python -m pip install -e '.[plm]'`."
            ) from exc

        self.device = _resolve_device(self.config.device)
        torch_dtype = _torch_dtype(self.config.dtype, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_source,
            local_files_only=self.config.local_files_only,
        )
        model_kwargs = {
            "local_files_only": self.config.local_files_only,
            "torch_dtype": torch_dtype,
        }
        try:
            self.model = AutoModel.from_pretrained(
                self.config.model_source,
                **model_kwargs,
            ).to(self.device)
        except TypeError:
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
            self.model = AutoModel.from_pretrained(
                self.config.model_source,
                **model_kwargs,
            ).to(self.device)
        self.model.eval()
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))

    @torch.inference_mode()
    def embed(self, sequence: str) -> np.ndarray:
        sequence = clean_protein_sequence(sequence)
        if not sequence:
            return np.zeros((0, self.hidden_size), dtype=_np_dtype(self.config.storage_dtype))
        max_residues = self._max_residues_per_forward()
        if len(sequence) <= max_residues:
            return self._embed_chunk(sequence).astype(_np_dtype(self.config.storage_dtype), copy=False)
        if self.config.max_length_policy != "chunk":
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds max residues {max_residues}; "
                "set max_length_policy='chunk' to enable overlapping chunks."
            )
        return self._embed_long_sequence(sequence, max_residues).astype(_np_dtype(self.config.storage_dtype), copy=False)

    @torch.inference_mode()
    def embed_many(self, records: Iterable[tuple[str, str]]) -> dict[str, np.ndarray]:
        return {protein_id: self.embed(sequence) for protein_id, sequence in records}

    def _embed_long_sequence(self, sequence: str, max_residues: int) -> np.ndarray:
        chunk_size = int(self.config.chunk_size or max_residues)
        chunk_size = max(1, min(chunk_size, max_residues))
        overlap = max(0, min(int(self.config.overlap), chunk_size - 1))
        step = max(1, chunk_size - overlap)
        total = np.zeros((len(sequence), self.hidden_size), dtype=np.float32)
        counts = np.zeros((len(sequence), 1), dtype=np.float32)
        for start in range(0, len(sequence), step):
            end = min(len(sequence), start + chunk_size)
            chunk = self._embed_chunk(sequence[start:end]).astype(np.float32, copy=False)
            total[start:end] += chunk
            counts[start:end] += 1.0
            if end == len(sequence):
                break
        return total / np.maximum(counts, 1.0)

    def _embed_chunk(self, sequence: str) -> np.ndarray:
        encoded = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state[0]
        token_count = int(encoded["attention_mask"][0].sum().item())
        residue = hidden[1 : token_count - 1]
        if residue.shape[0] != len(sequence):
            residue = _fallback_strip_special_tokens(hidden, len(sequence))
        if residue.shape[0] != len(sequence):
            raise RuntimeError(
                f"ESM-2 output length mismatch: got {residue.shape[0]} residues for sequence length {len(sequence)}"
            )
        return residue.detach().float().cpu().numpy()

    def _max_residues_per_forward(self) -> int:
        explicit = self.config.chunk_size
        if explicit is not None:
            return int(explicit)
        max_positions = getattr(self.tokenizer, "model_max_length", None)
        if max_positions is None or max_positions > 100_000:
            max_positions = getattr(self.model.config, "max_position_embeddings", 1024)
        return max(1, int(max_positions) - 2)


def esm2_embedding(sequence: str, **kwargs: object) -> np.ndarray:
    config = ESM2Config(**kwargs)
    return ESM2Embedder(config).embed(sequence)


def clean_protein_sequence(sequence: str) -> str:
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(aa for aa in sequence.upper() if aa in allowed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _torch_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype, torch.float32)


def _np_dtype(dtype: str) -> np.dtype:
    return np.dtype({"float16": np.float16, "fp16": np.float16, "float32": np.float32, "fp32": np.float32}.get(dtype, np.float32))


def _fallback_strip_special_tokens(hidden: torch.Tensor, length: int) -> torch.Tensor:
    if hidden.shape[0] >= length + 2:
        return hidden[1 : length + 1]
    if hidden.shape[0] >= length:
        return hidden[:length]
    return hidden


# Source: features/run_esm2.py


from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from phaseflow.protein.features import ESM2Config, ESM2Embedder, clean_protein_sequence, download_esm2_model


def run_esm2_embeddings(
    records: list[tuple[str, str]],
    out_dir: str | Path,
    config: ESM2Config,
    overwrite: bool = False,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = ESM2Embedder(config)
    written: list[Path] = []
    for protein_id, sequence in tqdm(records, desc="ESM-2 embeddings"):
        out_path = out_dir / f"{protein_id}.npz"
        if out_path.exists() and not overwrite:
            written.append(out_path)
            continue
        clean_sequence = clean_protein_sequence(sequence)
        embedding = embedder.embed(clean_sequence)
        np.savez_compressed(
            out_path,
            protein_id=np.asarray(protein_id),
            sequence=np.asarray(clean_sequence),
            length=np.asarray(len(clean_sequence), dtype=np.int64),
            embedding_last_hidden_state=embedding,
            model_name=np.asarray(config.model_name),
        )
        written.append(out_path)
    return written


def records_from_manifest(path: str | Path) -> list[tuple[str, str]]:
    frame = pd.read_csv(path)
    required = {"protein_id", "sequence"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return [(str(row["protein_id"]), str(row["sequence"])) for _, row in frame.iterrows()]


def records_from_fasta(path: str | Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    protein_id: str | None = None
    chunks: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if protein_id is not None:
                records.append((protein_id, "".join(chunks).upper()))
            protein_id = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if protein_id is not None:
        records.append((protein_id, "".join(chunks).upper()))
    return records



# Source: features/disorder.py


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


# Source: features/physchem.py


import math
from collections import Counter

import numpy as np

from phaseflow.protein.tokenizer import AA20

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


# Source: features/biophys.py

"""Deterministic 112-dimensional residue features required by DPR."""


from collections import Counter

import numpy as np

from phaseflow.protein.features import compute_disorder_features
from phaseflow.protein.features import compute_physchem_features


def compute_biophys_node(sequence: str) -> tuple[np.ndarray, list[str]]:
    """Return the public DPR biophysical feature matrix for one sequence."""

    physchem, physchem_names = compute_physchem_features(sequence)
    disorder, disorder_names, _, _ = compute_disorder_features(sequence, mode="simple")
    extra, extra_names = _extra_features(sequence, physchem, disorder)
    node = np.concatenate((physchem, disorder, extra), axis=1).astype(np.float32, copy=False)
    names = physchem_names + [f"disorder_{name}" for name in disorder_names] + extra_names
    if node.shape[1] != 112:
        raise RuntimeError(f"DPR biophys feature width must be 112, got {node.shape[1]}")
    return node, names


def _extra_features(sequence: str, physchem: np.ndarray, disorder: np.ndarray) -> tuple[np.ndarray, list[str]]:
    length = len(sequence)
    extra = np.zeros((length, 16), dtype=np.float32)
    if not length:
        return extra, _extra_names()
    positions = np.arange(length, dtype=np.float32)
    idr = disorder[:, 0]
    low_complexity = disorder[:, 1]
    prion_like = disorder[:, 2]
    extra[:, 0] = idr
    extra[:, 1] = idr >= 0.5
    extra[:, 2] = low_complexity
    extra[:, 3] = low_complexity >= 0.5
    extra[:, 4] = prion_like
    extra[:, 5] = prion_like >= 0.25
    extra[:, 6] = (low_complexity >= 0.5) | (idr >= 0.5)
    extra[:, 7] = positions / max(length - 1, 1)
    extra[:, 8] = positions / max(length, 1)
    extra[:, 9] = (length - 1 - positions) / max(length, 1)
    counts = Counter(sequence.upper())
    charge_bias = (counts["K"] + counts["R"] - counts["D"] - counts["E"]) / float(length)
    extra[:, 10] = charge_bias
    extra[:, 11] = physchem[:, 27] * 0.5 + physchem[:, 28] * 0.5
    extra[:, 12] = physchem[:, 24]
    extra[:, 13] = physchem[:, 34] + physchem[:, 35]
    extra[:, 14] = abs(charge_bias) * idr
    extra[:, 15] = 1.0
    return extra, _extra_names()


def _extra_names() -> list[str]:
    return [
        "idr_score",
        "idr_mask",
        "low_complexity_score",
        "low_complexity_mask",
        "prion_like_score",
        "prion_like_mask",
        "low_complexity_or_idr_mask",
        "normalized_position",
        "n_terminal_distance",
        "c_terminal_distance",
        "global_charge_bias",
        "sticker_spacer_proxy",
        "aromatic_density_proxy",
        "gly_pro_density_proxy",
        "kappa_like_proxy",
        "bias_feature",
    ]


# Source: features/bio_vec.py


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


# Source: features/starling_runner.py


import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from phaseflow.protein.features import compute_disorder_features


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
    """Load parsed STARLING protein features from an intermediate npz file.

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
    """Select protein-aligned segments where STARLING evidence is meaningful.

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
    """Fallback IDR-like protein STARLING proxy used only when explicitly requested.

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


# Source: features/run_starling.py


import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from phaseflow.protein.features import clean_protein_sequence
from phaseflow.protein.features import records_from_fasta, records_from_manifest
from phaseflow.protein.features import (
    assemble_starling_segments,
    candidate_starling_segments,
    heuristic_starling_features,
    load_starling_ensemble_file,
    starling_features_from_distance_maps,
)


def run_starling_features(
    records: list[tuple[str, str]],
    out_dir: str | Path,
    mode: str = "heuristic",
    starling_binary: str = "starling",
    overwrite: bool = False,
    conformations: int = 400,
    steps: int = 30,
    batch_size: int = 100,
    device: str | None = None,
    max_segment_length: int = 384,
    min_segment_length: int = 16,
    segment_score_threshold: float = 0.5,
    contact_threshold: float = 11.0,
    contact_topk: int = 16,
    min_contact_probability: float = 0.05,
    cleanup_raw: bool = False,
    ionic_strength: float = 150.0,
    require_complete: bool = True,
    api_sequence_batch_size: int = 8,
    external_segment_runner=None,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "python_distance_api":
        return _run_python_distance_api_starling(
            records,
            out_dir=out_dir,
            overwrite=overwrite,
            conformations=conformations,
            steps=steps,
            batch_size=batch_size,
            device=device,
            max_segment_length=max_segment_length,
            min_segment_length=min_segment_length,
            segment_score_threshold=segment_score_threshold,
            contact_threshold=contact_threshold,
            contact_topk=contact_topk,
            min_contact_probability=min_contact_probability,
            ionic_strength=ionic_strength,
            require_complete=require_complete,
            api_sequence_batch_size=api_sequence_batch_size,
        )
    written: list[Path] = []
    for protein_id, raw_sequence in records:
        sequence = clean_protein_sequence(raw_sequence)
        out_path = out_dir / f"{protein_id}.npz"
        if _complete_starling_npz(out_path, sequence, require_complete=require_complete, expected_node_dim=8) and not overwrite:
            written.append(out_path)
            continue
        if mode == "heuristic":
            node, missing, reliability = heuristic_starling_features(sequence)
            contacts = np.zeros((0, 3), dtype=np.float32)
        elif mode == "external":
            if external_segment_runner is None:
                raise ValueError("external STARLING mode must be launched through scripts.protein.workflows.features")
            node, missing, reliability, contacts, metadata = _run_external_starling(
                protein_id=protein_id,
                sequence=sequence,
                out_dir=out_dir,
                starling_binary=starling_binary,
                conformations=conformations,
                steps=steps,
                batch_size=batch_size,
                device=device,
                max_segment_length=max_segment_length,
                min_segment_length=min_segment_length,
                segment_score_threshold=segment_score_threshold,
                contact_threshold=contact_threshold,
                contact_topk=contact_topk,
                min_contact_probability=min_contact_probability,
                cleanup_raw=cleanup_raw,
                run_segment=external_segment_runner,
            )
        else:
            raise ValueError(f"Unsupported STARLING mode: {mode}")
        payload: dict[str, object] = {
            "protein_id": np.asarray(protein_id),
            "sequence": np.asarray(sequence),
            "node": node,
            "missing_mask": missing,
            "reliability": reliability,
            "contacts": contacts,
            "starling_mode": np.asarray(mode),
        }
        if mode == "external":
            payload.update({key: np.asarray(str(value)) for key, value in metadata.items()})
        np.savez_compressed(out_path, **payload)
        if mode == "external" and cleanup_raw:
            shutil.rmtree(out_dir / "_raw" / protein_id, ignore_errors=True)
        written.append(out_path)
    return written


def _run_python_distance_api_starling(
    records: list[tuple[str, str]],
    *,
    out_dir: Path,
    overwrite: bool,
    conformations: int,
    steps: int,
    batch_size: int,
    device: str | None,
    max_segment_length: int,
    min_segment_length: int,
    segment_score_threshold: float,
    contact_threshold: float,
    contact_topk: int,
    min_contact_probability: float,
    ionic_strength: float,
    require_complete: bool,
    api_sequence_batch_size: int,
) -> list[Path]:
    mpl_cache = out_dir / "_matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    try:
        from starling import configs as starling_configs  # type: ignore
        from starling.frontend.ensemble_generation import generate  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "STARLING python_distance_api mode requires `from starling.frontend.ensemble_generation import generate`"
        ) from exc

    device = device or "cuda:0"
    effective_max_segment_length = min(int(max_segment_length), int(starling_configs.MAX_SEQUENCE_LENGTH))
    pending_segments: dict[str, str] = {}
    segment_to_parent: dict[str, tuple[str, object]] = {}
    parent_segments: dict[str, list[object]] = {}
    record_sequences: dict[str, str] = {}
    segment_results_by_parent: dict[str, list[tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    written: list[Path] = []

    for protein_id, raw_sequence in records:
        sequence = clean_protein_sequence(raw_sequence)
        record_sequences[protein_id] = sequence
        out_path = out_dir / f"{protein_id}.npz"
        if _complete_starling_npz(out_path, sequence, require_complete=require_complete, expected_node_dim=8) and not overwrite:
            written.append(out_path)
            continue
        segments = candidate_starling_segments(
            protein_id,
            sequence,
            max_segment_length=effective_max_segment_length,
            min_segment_length=min_segment_length,
            score_threshold=segment_score_threshold,
        )
        parent_segments[protein_id] = segments
        segment_results_by_parent[protein_id] = []
        for segment in segments:
            if len(segment.sequence) > int(starling_configs.MAX_SEQUENCE_LENGTH):
                raise RuntimeError(
                    f"STARLING segment {segment.name} length {len(segment.sequence)} exceeds "
                    f"MAX_SEQUENCE_LENGTH={starling_configs.MAX_SEQUENCE_LENGTH}"
                )
            if segment.name in pending_segments:
                raise RuntimeError(f"Duplicate STARLING segment name in python_distance_api mode: {segment.name}")
            pending_segments[segment.name] = segment.sequence
            segment_to_parent[segment.name] = (protein_id, segment)

    segment_names = list(pending_segments)
    chunk_size = max(int(api_sequence_batch_size), 1)
    for start in tqdm(range(0, len(segment_names), chunk_size), desc="STARLING python_distance_api batches"):
        chunk_names = segment_names[start : start + chunk_size]
        sequence_dict = {name: pending_segments[name] for name in chunk_names}
        ensembles = generate(
            sequence_dict,
            conformations=int(conformations),
            ionic_strength=float(ionic_strength),
            device=device,
            steps=int(steps),
            return_structures=False,
            batch_size=int(batch_size),
            output_directory=None,
            return_data=True,
            verbose=False,
            show_progress_bar=False,
            show_per_step_progress_bar=False,
        )
        if ensembles is None:
            raise RuntimeError("STARLING generate returned None in python_distance_api mode")
        missing = sorted(set(sequence_dict).difference(ensembles))
        if missing:
            raise RuntimeError(f"STARLING generate missed {len(missing)} segments: {missing[:5]}")
        for segment_name in chunk_names:
            protein_id, segment = segment_to_parent[segment_name]
            ensemble = ensembles[segment_name]
            segment_node, segment_missing, segment_reliability, segment_contacts = starling_features_from_distance_maps(
                ensemble.distance_maps(return_mean=False),
                segment.sequence,
                contact_threshold=contact_threshold,
                contact_topk=contact_topk,
                min_contact_probability=min_contact_probability,
            )
            if segment_node.shape[1] != 8:
                raise RuntimeError(f"STARLING distance-map features for {segment.name} must be 8D, got {segment_node.shape}")
            segment_results_by_parent[protein_id].append(
                (segment, segment_node, segment_missing, segment_reliability, segment_contacts)
            )

    for protein_id, sequence in tqdm(record_sequences.items(), desc="STARLING python_distance_api features"):
        out_path = out_dir / f"{protein_id}.npz"
        if _complete_starling_npz(out_path, sequence, require_complete=require_complete, expected_node_dim=8) and not overwrite:
            if out_path not in written:
                written.append(out_path)
            continue
        node, missing, reliability, contacts = assemble_starling_segments(
            len(sequence),
            segment_results_by_parent.get(protein_id, []),
        )
        if node.shape != (len(sequence), 8):
            raise RuntimeError(
                f"STARLING python_distance_api output for {protein_id} must have shape {(len(sequence), 8)}, got {node.shape}"
            )
        payload: dict[str, object] = {
            "protein_id": np.asarray(protein_id),
            "sequence": np.asarray(sequence),
            "node": node,
            "missing_mask": missing,
            "reliability": reliability,
            "contacts": contacts,
            "starling_mode": np.asarray("python_distance_api"),
            "starling_api": np.asarray("starling.frontend.ensemble_generation.generate"),
            "starling_node_dim": np.asarray(node.shape[1], dtype=np.int64),
            "starling_segments": np.asarray(len(parent_segments.get(protein_id, [])), dtype=np.int64),
            "starling_conformations": np.asarray(int(conformations), dtype=np.int64),
            "starling_steps": np.asarray(int(steps), dtype=np.int64),
            "starling_batch_size": np.asarray(int(batch_size), dtype=np.int64),
            "starling_max_segment_length": np.asarray(int(effective_max_segment_length), dtype=np.int64),
            "starling_ionic_strength": np.asarray(float(ionic_strength), dtype=np.float32),
            "starling_contact_threshold": np.asarray(float(contact_threshold), dtype=np.float32),
            "starling_contact_topk": np.asarray(int(contact_topk), dtype=np.int64),
        }
        np.savez_compressed(out_path, **payload)
        written.append(out_path)
    return written


def _complete_starling_npz(
    path: Path,
    sequence: str,
    *,
    require_complete: bool = True,
    expected_node_dim: int | None = None,
) -> bool:
    if not path.exists():
        return False
    if not require_complete:
        return True
    try:
        with np.load(path, allow_pickle=False) as data:
            if "sequence" in data and str(data["sequence"].item()) != sequence:
                return False
            required = {"node", "missing_mask", "reliability", "contacts", "starling_mode"}
            if not required.issubset(data.files):
                return False
            node = np.asarray(data["node"])
            missing = np.asarray(data["missing_mask"])
            reliability = np.asarray(data["reliability"])
            contacts = np.asarray(data["contacts"])
            length = len(sequence)
            if node.ndim != 2 or node.shape[0] != length or node.shape[1] == 0:
                return False
            if expected_node_dim is not None and node.shape[1] != int(expected_node_dim):
                return False
            if missing.shape != (length,) or reliability.shape != (length,):
                return False
            if contacts.ndim != 2 or contacts.shape[1] < 3:
                return False
            if not np.isfinite(node).all() or not np.isfinite(missing).all() or not np.isfinite(reliability).all():
                return False
    except Exception:
        return False
    return True


def _run_external_starling(
    protein_id: str,
    sequence: str,
    out_dir: Path,
    starling_binary: str,
    *,
    conformations: int,
    steps: int,
    batch_size: int,
    device: str | None,
    max_segment_length: int,
    min_segment_length: int,
    segment_score_threshold: float,
    contact_threshold: float,
    contact_topk: int,
    min_contact_probability: float,
    cleanup_raw: bool,
    run_segment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    raw_dir = out_dir / "_raw" / protein_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache = out_dir / "_matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    segments = candidate_starling_segments(
        protein_id,
        sequence,
        max_segment_length=max_segment_length,
        min_segment_length=min_segment_length,
        score_threshold=segment_score_threshold,
    )
    segment_results = []
    for segment in segments:
        segment_dir = raw_dir / segment.name
        segment_dir.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(2):
            attempt_dir = segment_dir if attempt == 0 else raw_dir / f"{segment.name}_retry_{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            ensemble_path = run_segment(
                segment.sequence,
                segment.name,
                attempt_dir,
                starling_binary=starling_binary,
                env=env,
                conformations=conformations,
                steps=steps,
                batch_size=batch_size,
                device=device,
            )
            try:
                ensemble = load_starling_ensemble_file(ensemble_path)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Could not load STARLING ensemble for {segment.name}") from last_error
        segment_node, segment_missing, segment_reliability, segment_contacts = starling_features_from_distance_maps(
            ensemble.distance_maps(return_mean=False),
            segment.sequence,
            contact_threshold=contact_threshold,
            contact_topk=contact_topk,
            min_contact_probability=min_contact_probability,
        )
        segment_results.append((segment, segment_node, segment_missing, segment_reliability, segment_contacts))
        if cleanup_raw:
            shutil.rmtree(segment_dir, ignore_errors=True)

    node, missing, reliability, contacts = assemble_starling_segments(len(sequence), segment_results)
    metadata = {
        "starling_raw_dir": str(raw_dir),
        "starling_segments": len(segments),
        "starling_conformations": conformations,
        "starling_steps": steps,
        "starling_batch_size": batch_size,
        "starling_max_segment_length": max_segment_length,
        "starling_contact_threshold": contact_threshold,
        "starling_contact_topk": contact_topk,
    }
    return node, missing, reliability, contacts, metadata


# Source: features/build_features.py


import json
from pathlib import Path

import numpy as np
import pandas as pd

from phaseflow.protein.contracts import FeatureCacheWriter
from phaseflow.protein.contracts import FeatureCacheRecord, IGNORE_INDEX
from phaseflow.protein.features import compute_disorder_features
from phaseflow.protein.structure import build_edges
from phaseflow.protein.structure import edge_list_to_precomputed_graph
from phaseflow.protein.features import compute_physchem_features
from phaseflow.protein.features import ESM2Config, ESM2Embedder, clean_protein_sequence, simple_plm_embedding
from phaseflow.protein.features import (
    STARLING_EMBED_DIM,
    load_starling_distance_contacts,
    load_starling_embedding,
    zero_starling_embedding,
)

DEFAULT_PROTENIX_EMBEDDING_DIM = 512
DEFAULT_PROTENIX_S_DIM = 384
DEFAULT_PROTENIX_Z_DIM = 128
DEFAULT_GRAPH_EDGE_DIM = 13
DEFAULT_STARLING_DISTANCE_TOPK = 48


def build_feature_cache(
    fasta: str | Path,
    out_dir: str | Path,
    protein_labels: str | Path | None = None,
    regions: str | Path | None = None,
    mil_bags: str | Path | None = None,
    candidate_priors: str | Path | None = None,
    teacher_scores: str | Path | None = None,
    mode: str = "simple",
    esm2_dir: str | Path | None = None,
    esm2_config: ESM2Config | None = None,
    structure_dir: str | Path | None = None,
    protenix_embedding_dir: str | Path | None = None,
    protenix_embedding_dim: int = DEFAULT_PROTENIX_EMBEDDING_DIM,
    af3_dir: str | Path | None = None,
    starling_dir: str | Path | None = None,
    starling_embedding_dir: str | Path | None = None,
    starling_distance_dir: str | Path | None = None,
    local_window: int = 16,
    graph_max_neighbors: int | None = 96,
    graph_edge_dim: int = DEFAULT_GRAPH_EDGE_DIM,
    starling_distance_topk: int = DEFAULT_STARLING_DISTANCE_TOPK,
    require_structure: bool = False,
    require_starling: bool = False,
    overwrite: bool = True,
) -> list[Path]:
    out_dir = Path(out_dir)
    records = [(protein_id, clean_protein_sequence(sequence)) for protein_id, sequence in read_fasta(fasta)]
    label_frame = _read_labels(protein_labels)
    region_map = _read_regions(regions)
    mil_bag_map = _read_mil_bags(mil_bags)
    candidate_prior_map = _read_candidate_priors(candidate_priors)
    teacher_score_map = _read_teacher_scores(teacher_scores)
    embedder = ESM2Embedder(esm2_config) if mode == "esm2" and esm2_dir is None else None
    written: list[Path] = []
    for protein_id, sequence in records:
        out_path = out_dir / f"{protein_id}.h5"
        if out_path.exists() and not overwrite:
            written.append(out_path)
            continue
        plm, plm_missing, plm_reliability = _plm_features(
            protein_id=protein_id,
            sequence=sequence,
            mode=mode,
            esm2_dir=esm2_dir,
            embedder=embedder,
        )
        physchem, _ = compute_physchem_features(sequence)
        disorder, _, disorder_missing, disorder_reliability = compute_disorder_features(sequence, mode="simple")
        protenix_embed, protenix_missing, protenix_reliability, structure_metadata = _protenix_embedding_features(
            protein_id,
            sequence,
            protenix_embedding_dir=protenix_embedding_dir,
            protenix_embedding_dim=protenix_embedding_dim,
        )
        starling_embed, starling_missing, starling_reliability, starling_metadata = _starling_embedding_features(
            protein_id,
            sequence,
            starling_embedding_dir or starling_dir,
            require_starling=require_starling,
        )
        star_contacts, distance_metadata = _starling_distance_contacts(
            protein_id,
            sequence,
            starling_distance_dir=starling_distance_dir,
            contact_topk=starling_distance_topk,
        )
        structure_metadata.update(starling_metadata)
        structure_metadata.update(distance_metadata)
        modality_mask = np.stack(
            [
                plm_missing,
                np.zeros(len(sequence), dtype=np.float32),
                disorder_missing,
                protenix_missing,
                starling_missing,
            ],
            axis=1,
        )
        reliability = np.stack(
            [
                plm_reliability,
                np.ones(len(sequence), dtype=np.float32),
                disorder_reliability,
                protenix_reliability,
                starling_reliability,
            ],
            axis=1,
        )
        edges = build_edges(
            len(sequence),
            local_window=local_window,
            af_contacts=None,
            star_contacts=star_contacts,
            physchem=physchem,
            segment_ids=disorder[:, 3],
            edge_dim=graph_edge_dim,
            star_topk=starling_distance_topk,
        )
        graph = _precompute_graph(edges, len(sequence), graph_max_neighbors, edge_dim=graph_edge_dim)
        label_row = _label_row_for(label_frame, protein_id)
        llps_label = _label_for(label_frame, protein_id)
        sample_weight = _sample_weight_for(label_frame, protein_id, llps_label)
        dpr, key, weight, sample_regions, soft = _labels_from_regions(
            protein_id,
            len(sequence),
            llps_label,
            region_map,
            candidate_prior_map=candidate_prior_map,
            teacher_score_map=teacher_score_map,
        )
        bag = _mil_bag_for(mil_bag_map, label_row, protein_id, llps_label)
        cache_record = FeatureCacheRecord(
            protein_id=protein_id,
            sequence=sequence,
            plm=plm,
            physchem=physchem,
            disorder=disorder,
            protenix_embed=protenix_embed,
            starling_embed=starling_embed,
            modality_mask=modality_mask,
            reliability=reliability,
            edge_src=edges.edge_src,
            edge_dst=edges.edge_dst,
            edge_type=edges.edge_type,
            edge_attr=edges.edge_attr,
            graph_neighbors=graph.neighbors if graph is not None else None,
            graph_edge_attr=graph.edge_attr if graph is not None else None,
            graph_neighbor_mask=graph.neighbor_mask if graph is not None else None,
            y_llps=float(llps_label),
            sample_weight=sample_weight,
            y_dpr=dpr,
            y_key=key,
            y_weight=weight,
            teacher_llps=_float_or_nan(label_row.get("teacher_consensus_score", label_row.get("teacher_weighted", np.nan))),
            teacher_llps_weight=_teacher_weight_from_row(label_row),
            self_llps=_float_or_nan(label_row.get("self_training_score", label_row.get("self_llps", np.nan))),
            self_llps_weight=_float_or_zero(label_row.get("self_training_weight", label_row.get("self_llps_weight", 0.0))),
            region_bag_label=bag["region_bag_label"],
            region_bag_weight=bag["region_bag_weight"],
            region_bag_type=str(bag["region_bag_type"]),
            negative_regularization_weight=_negative_regularization_weight(label_row),
            teacher_dpr=soft["teacher_dpr"],
            teacher_dpr_weight=soft["teacher_dpr_weight"],
            self_dpr=soft["self_dpr"],
            self_dpr_weight=soft["self_dpr_weight"],
            candidate_prior=soft["candidate_prior"],
            candidate_prior_weight=soft["candidate_prior_weight"],
            label_quality=str(label_row.get("label_quality", "")),
            negative_type=str(label_row.get("negative_type", "")),
            source=str(label_row.get("source", "")),
            regions=sample_regions,
            structure_metadata=structure_metadata,
        )
        FeatureCacheWriter.write_h5(out_path, cache_record)
        written.append(out_path)
    return written


def build_feature_cache_from_manifest(
    manifest: str | Path,
    out_dir: str | Path,
    regions: str | Path | None = None,
    mil_bags: str | Path | None = None,
    candidate_priors: str | Path | None = None,
    teacher_scores: str | Path | None = None,
    mode: str = "simple",
    esm2_dir: str | Path | None = None,
    esm2_config: ESM2Config | None = None,
    structure_dir: str | Path | None = None,
    protenix_embedding_dir: str | Path | None = None,
    protenix_embedding_dim: int = DEFAULT_PROTENIX_EMBEDDING_DIM,
    af3_dir: str | Path | None = None,
    starling_dir: str | Path | None = None,
    starling_embedding_dir: str | Path | None = None,
    starling_distance_dir: str | Path | None = None,
    local_window: int = 16,
    graph_max_neighbors: int | None = 96,
    graph_edge_dim: int = DEFAULT_GRAPH_EDGE_DIM,
    starling_distance_topk: int = DEFAULT_STARLING_DISTANCE_TOPK,
    require_structure: bool = False,
    require_starling: bool = False,
    overwrite: bool = True,
) -> list[Path]:
    manifest = Path(manifest)
    frame = pd.read_parquet(manifest) if manifest.suffix.lower() == ".parquet" else pd.read_csv(manifest)
    required = {"protein_id", "sequence"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    out_dir = Path(out_dir)
    records = [
        (str(row["protein_id"]), clean_protein_sequence(str(row["sequence"])))
        for _, row in frame.iterrows()
    ]
    label_frame = _labels_from_manifest_frame(frame)
    region_map = _read_regions(regions)
    mil_bag_map = _read_mil_bags(mil_bags)
    candidate_prior_map = _read_candidate_priors(candidate_priors)
    teacher_score_map = _read_teacher_scores(teacher_scores)
    embedder = ESM2Embedder(esm2_config) if mode == "esm2" and esm2_dir is None else None
    written: list[Path] = []
    for protein_id, sequence in records:
        out_path = out_dir / f"{protein_id}.h5"
        if out_path.exists() and not overwrite:
            written.append(out_path)
            continue
        plm, plm_missing, plm_reliability = _plm_features(protein_id, sequence, mode, esm2_dir, embedder)
        physchem, _ = compute_physchem_features(sequence)
        disorder, _, disorder_missing, disorder_reliability = compute_disorder_features(sequence, mode="simple")
        protenix_embed, protenix_missing, protenix_reliability, structure_metadata = _protenix_embedding_features(
            protein_id,
            sequence,
            protenix_embedding_dir=protenix_embedding_dir,
            protenix_embedding_dim=protenix_embedding_dim,
        )
        starling_embed, starling_missing, starling_reliability, starling_metadata = _starling_embedding_features(
            protein_id,
            sequence,
            starling_embedding_dir or starling_dir,
            require_starling=require_starling,
        )
        star_contacts, distance_metadata = _starling_distance_contacts(
            protein_id,
            sequence,
            starling_distance_dir=starling_distance_dir,
            contact_topk=starling_distance_topk,
        )
        structure_metadata.update(starling_metadata)
        structure_metadata.update(distance_metadata)
        modality_mask = np.stack(
            [
                plm_missing,
                np.zeros(len(sequence), dtype=np.float32),
                disorder_missing,
                protenix_missing,
                starling_missing,
            ],
            axis=1,
        )
        reliability = np.stack(
            [
                plm_reliability,
                np.ones(len(sequence), dtype=np.float32),
                disorder_reliability,
                protenix_reliability,
                starling_reliability,
            ],
            axis=1,
        )
        edges = build_edges(
            len(sequence),
            local_window=local_window,
            af_contacts=None,
            star_contacts=star_contacts,
            physchem=physchem,
            segment_ids=disorder[:, 3],
            edge_dim=graph_edge_dim,
            star_topk=starling_distance_topk,
        )
        graph = _precompute_graph(edges, len(sequence), graph_max_neighbors, edge_dim=graph_edge_dim)
        label_row = _label_row_for(label_frame, protein_id)
        llps_label = _label_for(label_frame, protein_id)
        sample_weight = _sample_weight_for(label_frame, protein_id, llps_label)
        dpr, key, weight, sample_regions, soft = _labels_from_regions(
            protein_id,
            len(sequence),
            llps_label,
            region_map,
            candidate_prior_map=candidate_prior_map,
            teacher_score_map=teacher_score_map,
        )
        bag = _mil_bag_for(mil_bag_map, label_row, protein_id, llps_label)
        FeatureCacheWriter.write_h5(
            out_path,
            FeatureCacheRecord(
                protein_id=protein_id,
                sequence=sequence,
                plm=plm,
                physchem=physchem,
                disorder=disorder,
                protenix_embed=protenix_embed,
                starling_embed=starling_embed,
                modality_mask=modality_mask,
                reliability=reliability,
                edge_src=edges.edge_src,
                edge_dst=edges.edge_dst,
                edge_type=edges.edge_type,
                edge_attr=edges.edge_attr,
                graph_neighbors=graph.neighbors if graph is not None else None,
                graph_edge_attr=graph.edge_attr if graph is not None else None,
                graph_neighbor_mask=graph.neighbor_mask if graph is not None else None,
                y_llps=float(llps_label),
                sample_weight=sample_weight,
                y_dpr=dpr,
                y_key=key,
                y_weight=weight,
                teacher_llps=_float_or_nan(label_row.get("teacher_consensus_score", label_row.get("teacher_weighted", np.nan))),
                teacher_llps_weight=_teacher_weight_from_row(label_row),
                self_llps=_float_or_nan(label_row.get("self_training_score", label_row.get("self_llps", np.nan))),
                self_llps_weight=_float_or_zero(label_row.get("self_training_weight", label_row.get("self_llps_weight", 0.0))),
                region_bag_label=bag["region_bag_label"],
                region_bag_weight=bag["region_bag_weight"],
                region_bag_type=str(bag["region_bag_type"]),
                negative_regularization_weight=_negative_regularization_weight(label_row),
                teacher_dpr=soft["teacher_dpr"],
                teacher_dpr_weight=soft["teacher_dpr_weight"],
                self_dpr=soft["self_dpr"],
                self_dpr_weight=soft["self_dpr_weight"],
                candidate_prior=soft["candidate_prior"],
                candidate_prior_weight=soft["candidate_prior_weight"],
                label_quality=str(label_row.get("label_quality", "")),
                negative_type=str(label_row.get("negative_type", "")),
                source=str(label_row.get("source", "")),
                regions=sample_regions,
                structure_metadata=structure_metadata,
            ),
        )
        written.append(out_path)
    return written


def _precompute_graph(edges, length: int, graph_max_neighbors: int | None, edge_dim: int):
    if graph_max_neighbors is None or int(graph_max_neighbors) <= 0:
        return None
    return edge_list_to_precomputed_graph(
        length=length,
        edge_src=edges.edge_src,
        edge_dst=edges.edge_dst,
        edge_type=edges.edge_type,
        edge_attr=edges.edge_attr,
        max_neighbors=int(graph_max_neighbors),
        edge_dim=edge_dim,
    )


def _plm_features(
    protein_id: str,
    sequence: str,
    mode: str,
    esm2_dir: str | Path | None,
    embedder: ESM2Embedder | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "simple":
        plm = simple_plm_embedding(sequence, dim=32)
    elif mode == "esm2":
        if esm2_dir is not None:
            plm = _read_esm2_npz(esm2_dir, protein_id, sequence)
        elif embedder is not None:
            plm = embedder.embed(sequence)
        else:
            raise ValueError("mode='esm2' requires either esm2_dir or esm2_config")
    else:
        raise ValueError(f"Unsupported feature mode: {mode}")
    missing = np.zeros(len(sequence), dtype=np.float32)
    reliability = np.ones(len(sequence), dtype=np.float32)
    return plm.astype(np.float32, copy=False), missing, reliability


def _read_esm2_npz(esm2_dir: str | Path, protein_id: str, sequence: str) -> np.ndarray:
    path = Path(esm2_dir) / f"{protein_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing ESM-2 embedding file: {path}")
    with np.load(path, allow_pickle=False) as data:
        embedding = np.asarray(data["embedding_last_hidden_state"], dtype=np.float32)
        cached_sequence = str(data["sequence"].item()) if "sequence" in data else sequence
    if clean_protein_sequence(cached_sequence) != sequence:
        raise ValueError(f"Sequence mismatch for {protein_id}: ESM-2 npz does not match cache sequence")
    if embedding.ndim != 2 or embedding.shape[0] != len(sequence):
        raise ValueError(f"ESM-2 embedding for {protein_id} must have shape [L, D], got {embedding.shape}")
    return embedding


def _protenix_embedding_features(
    protein_id: str,
    sequence: str,
    protenix_embedding_dir: str | Path | None,
    protenix_embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    length = len(sequence)
    if protenix_embedding_dir is None:
        s_dim, z_dim = _default_protenix_split_dims(protenix_embedding_dim)
        return (
            np.zeros((length, s_dim + z_dim), dtype=np.float32),
            np.ones(length, dtype=np.float32),
            np.zeros(length, dtype=np.float32),
            {
                "protenix_embedding_success": "0",
                "protenix_embedding_path": "",
                "protenix_embedding_dim": str(int(s_dim + z_dim)),
            },
        )
    path = Path(protenix_embedding_dir) / f"{protein_id}.npz"
    if not path.exists():
        s_dim, z_dim = _default_protenix_split_dims(protenix_embedding_dim)
        return np.zeros((length, s_dim + z_dim), dtype=np.float32), np.ones(length, dtype=np.float32), np.zeros(length, dtype=np.float32), {
            "protenix_embedding_success": "0",
            "protenix_embedding_path": str(path),
            "protenix_embedding_dim": str(int(s_dim + z_dim)),
        }
    with np.load(path, allow_pickle=False) as data:
        if "s" not in data or "z" not in data:
            raise ValueError(f"Protenix embedding file {path} must contain s and z arrays")
        s = np.asarray(data["s"], dtype=np.float32)
        z = np.asarray(data["z"], dtype=np.float32)
        if s.ndim != 2 or z.ndim != 2 or s.shape[0] != length or z.shape[0] != length:
            raise ValueError(
                f"Protenix embedding for {protein_id} must have s/z shapes [L, D], "
                f"got s={s.shape}, z={z.shape}, L={length}"
            )
        if "single_mask" in data:
            single_mask = np.asarray(data["single_mask"], dtype=np.float32)
            if single_mask.shape != (length,):
                raise ValueError(
                    f"Protenix embedding single_mask for {protein_id} must have shape [{length}], got {single_mask.shape}"
                )
            available = np.clip(single_mask, 0.0, 1.0).astype(np.float32)
        else:
            available = np.ones(length, dtype=np.float32)
    missing = (available <= 0.0).astype(np.float32)
    reliability = available.astype(np.float32, copy=False)
    embedding = np.concatenate([s, z], axis=1).astype(np.float32, copy=False)
    return embedding, missing, reliability, {
        "protenix_embedding_success": "1",
        "protenix_embedding_path": str(path),
        "protenix_embedding_dim": str(embedding.shape[1]),
    }


def _default_protenix_split_dims(protenix_embedding_dim: int) -> tuple[int, int]:
    if int(protenix_embedding_dim) == DEFAULT_PROTENIX_EMBEDDING_DIM:
        return DEFAULT_PROTENIX_S_DIM, DEFAULT_PROTENIX_Z_DIM
    s_dim = int(round(float(protenix_embedding_dim) * 0.75))
    z_dim = int(protenix_embedding_dim) - s_dim
    return max(s_dim, 1), max(z_dim, 1)


def _starling_embedding_features(
    protein_id: str,
    sequence: str,
    starling_embedding_dir: str | Path | None,
    require_starling: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if starling_embedding_dir is None:
        if require_starling:
            raise ValueError("require_starling=True needs --starling-embedding-dir")
        embedding, missing, reliability = zero_starling_embedding(len(sequence), dim=STARLING_EMBED_DIM)
        return embedding, missing, reliability, {"starling_embedding_success": "0", "starling_embedding_path": ""}
    path = Path(starling_embedding_dir) / f"{protein_id}.npz"
    if not path.exists():
        if require_starling:
            raise FileNotFoundError(f"Missing required STARLING embedding file: {path}")
        embedding, missing, reliability = zero_starling_embedding(len(sequence), dim=STARLING_EMBED_DIM)
        return embedding, missing, reliability, {"starling_embedding_success": "0", "starling_embedding_path": str(path)}
    return load_starling_embedding(path, sequence)


def _starling_distance_contacts(
    protein_id: str,
    sequence: str,
    *,
    starling_distance_dir: str | Path | None,
    contact_topk: int,
) -> tuple[np.ndarray | None, dict[str, object]]:
    if starling_distance_dir is None:
        return None, {"starling_distance_success": "0", "starling_distance_path": ""}
    path = Path(starling_distance_dir) / f"{protein_id}.h5"
    if not path.exists():
        return None, {"starling_distance_success": "0", "starling_distance_path": str(path)}
    return load_starling_distance_contacts(path, sequence, contact_topk=contact_topk)


def _labels_from_manifest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"protein_id", "llps_label"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=["protein_id", "llps_label"])
    columns = ["protein_id", "llps_label"]
    for name in (
        "sample_weight",
        "label_confidence",
        "confidence",
        "negative_type",
        "role_label",
        "source",
        "label_quality",
        "evidence_level",
        "teacher_consensus_score",
        "teacher_weighted",
        "teacher_confidence",
        "teacher_agreement",
        "self_training_score",
        "self_training_weight",
        "self_llps",
        "self_llps_weight",
    ):
        if name in frame.columns and name not in columns:
            columns.append(name)
    return frame[columns].copy()


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    protein_id: str | None = None
    chunks: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if protein_id is not None:
                records.append((protein_id, "".join(chunks).upper()))
            protein_id = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if protein_id is not None:
        records.append((protein_id, "".join(chunks).upper()))
    return records


def _read_labels(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["protein_id", "llps_label"])
    path = Path(path)
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _read_regions(path: str | Path | None) -> dict[str, list[dict[str, object]]]:
    if path is None:
        return {}
    path = Path(path)
    region_map: dict[str, list[dict[str, object]]] = {}
    if path.suffix.lower() in {".csv", ".tsv", ".tab"}:
        frame = pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".tab"} else ",")
        if not frame.empty:
            for protein_id, group in frame.groupby("protein_id"):
                regions: list[dict[str, object]] = []
                for _, row in group.iterrows():
                    start_1 = int(row["start"])
                    end_1 = int(row["end"])
                    regions.append(
                        {
                            "protein_id": str(protein_id),
                            "start": max(0, start_1 - 1),
                            "end": max(0, end_1 - 1),
                            "type": str(row.get("region_type") or row.get("type") or "DPR_candidate"),
                            "region_type": str(row.get("region_type") or row.get("type") or "DPR_candidate"),
                            "region_label": row.get("region_label", "unknown"),
                            "confidence": float(row.get("confidence", 1.0)),
                            "soft_label": _float_or_nan(row.get("soft_label", row.get("score", np.nan))),
                            "soft_weight": _float_or_zero(row.get("soft_weight", row.get("sample_weight", row.get("confidence", 0.0)))),
                            "evidence_level": str(row.get("evidence_level") or "candidate"),
                            "source": str(row.get("source") or ""),
                            "assay": str(row.get("assay") or ""),
                            "notes": str(row.get("notes") or ""),
                        }
                    )
                region_map[str(protein_id)] = regions
        return region_map
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        protein_id = str(row["protein_id"])
        if "regions" in row:
            region_map[protein_id] = list(row.get("regions", []))
            continue
        dpr_spans = row.get("dpr_spans", [])
        regions: list[dict[str, object]] = []
        for span in dpr_spans:
            if isinstance(span, dict):
                start = int(span.get("start", 0))
                end = int(span.get("end", start))
                label_tier = str(span.get("label_tier", row.get("label_tier", "gold")))
                source = str(span.get("source", row.get("source", "")))
                confidence = float(span.get("confidence", row.get("sample_weight", 1.0)))
                sample_weight = float(span.get("sample_weight", row.get("sample_weight", confidence)))
            else:
                start, end = int(span[0]), int(span[1])
                label_tier = str(row.get("label_tier", "gold"))
                source = str(row.get("source", ""))
                confidence = float(row.get("sample_weight", 1.0))
                sample_weight = confidence
            regions.append(
                {
                    "protein_id": protein_id,
                    "start": start,
                    "end": end,
                    "type": "DPR_gold" if label_tier == "gold" else "DPR_curated",
                    "region_type": "DPR_gold" if label_tier == "gold" else "DPR_curated",
                    "region_label": "positive",
                    "confidence": confidence,
                    "soft_weight": sample_weight,
                    "evidence_level": label_tier,
                    "source": source,
                }
            )
        if bool(row.get("outside_is_negative", False)):
            for span in row.get("negative_spans", []):
                if isinstance(span, dict):
                    start = int(span.get("start", 0))
                    end = int(span.get("end", start))
                    confidence = float(span.get("sample_weight", row.get("outside_negative_weight", 0.1)))
                else:
                    start, end = int(span[0]), int(span[1])
                    confidence = float(row.get("outside_negative_weight", 0.1))
                regions.append(
                    {
                        "protein_id": protein_id,
                        "start": start,
                        "end": end,
                        "type": "non_DPR_control",
                        "region_type": "non_DPR_control",
                        "region_label": "negative",
                        "confidence": confidence,
                        "soft_weight": confidence,
                        "evidence_level": "negative_control",
                        "source": str(row.get("source", "")),
                    }
                )
        region_map[protein_id] = regions
    return region_map


def _read_mil_bags(path: str | Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    bags: dict[str, dict[str, object]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        bags[str(row["protein_id"])] = row
    return bags


def _read_candidate_priors(path: str | Path | None) -> dict[str, list[dict[str, object]]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    priors: dict[str, list[dict[str, object]]] = {}
    if path.suffix.lower() in {".jsonl", ".json"}:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            protein_id = str(row["protein_id"])
            priors[protein_id] = list(row.get("candidate_spans", row.get("regions", [])))
        return priors
    import h5py

    with h5py.File(path, "r") as handle:
        for protein_id in handle:
            group = handle[protein_id]
            spans = np.asarray(group.get("spans", np.zeros((0, 2))), dtype=np.int64)
            scores = np.asarray(group.get("scores", np.ones((len(spans),))), dtype=np.float32)
            types = group.attrs.get("types_json", "[]")
            if isinstance(types, bytes):
                types = types.decode("utf-8")
            type_values = json.loads(str(types))
            rows = []
            for index, span in enumerate(spans):
                rows.append(
                    {
                        "start": int(span[0]),
                        "end": int(span[1]),
                        "score": float(scores[index]) if index < len(scores) else 1.0,
                        "type": str(type_values[index]) if index < len(type_values) else "candidate_prior",
                    }
                )
            priors[str(protein_id)] = rows
    return priors


def _read_teacher_scores(path: str | Path | None) -> dict[str, dict[str, np.ndarray]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    import h5py

    scores: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as handle:
        for protein_id in handle:
            group = handle[protein_id]
            if "teacher_consensus" not in group:
                continue
            record = {"teacher_consensus": np.asarray(group["teacher_consensus"], dtype=np.float32)}
            if "teacher_uncertainty" in group:
                record["teacher_uncertainty"] = np.asarray(group["teacher_uncertainty"], dtype=np.float32)
            if "teacher_confidence" in group:
                record["teacher_confidence"] = np.asarray(group["teacher_confidence"], dtype=np.float32)
            scores[str(protein_id)] = record
    return scores


def _label_row_for(frame: pd.DataFrame, protein_id: str) -> dict[str, object]:
    if frame.empty:
        return {}
    rows = frame.loc[frame["protein_id"].astype(str) == str(protein_id)]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _label_for(frame: pd.DataFrame, protein_id: str) -> int:
    if frame.empty:
        return IGNORE_INDEX
    rows = frame.loc[frame["protein_id"].astype(str) == str(protein_id)]
    if rows.empty:
        return IGNORE_INDEX
    return int(rows.iloc[0]["llps_label"])


def _sample_weight_for(frame: pd.DataFrame, protein_id: str, llps_label: int) -> float:
    if frame.empty:
        return 1.0
    rows = frame.loc[frame["protein_id"].astype(str) == str(protein_id)]
    if rows.empty:
        return 1.0
    row = rows.iloc[0]
    if llps_label == IGNORE_INDEX:
        return 0.0
    if "sample_weight" in rows.columns and pd.notna(row.get("sample_weight")):
        return float(row["sample_weight"])
    if "label_confidence" in rows.columns and pd.notna(row.get("label_confidence")):
        return float(row["label_confidence"])
    if "confidence" in rows.columns and pd.notna(row.get("confidence")):
        return float(row["confidence"])
    return 1.0


def _labels_from_regions(
    protein_id: str,
    length: int,
    llps_label: int,
    region_map: dict[str, list[dict[str, object]]],
    candidate_prior_map: dict[str, list[dict[str, object]]] | None = None,
    teacher_score_map: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]], dict[str, np.ndarray]]:
    dpr = np.full(length, IGNORE_INDEX, dtype=np.int64)
    key = np.full(length, IGNORE_INDEX, dtype=np.int64)
    weight = np.zeros(length, dtype=np.float32)
    sample_regions = region_map.get(protein_id, [])
    teacher_dpr = np.full(length, np.nan, dtype=np.float32)
    teacher_dpr_weight = np.zeros(length, dtype=np.float32)
    self_dpr = np.full(length, np.nan, dtype=np.float32)
    self_dpr_weight = np.zeros(length, dtype=np.float32)
    candidate_prior = np.zeros(length, dtype=np.float32)
    candidate_prior_weight = np.zeros(length, dtype=np.float32)
    for region in sample_regions:
        start = max(0, int(region["start"]))
        end = min(length - 1, int(region["end"]))
        confidence = float(region.get("confidence", 1.0))
        label_kind = _region_label_kind(region)
        if label_kind == "positive":
            dpr[start : end + 1] = 1
            weight[start : end + 1] = confidence
        elif label_kind == "negative":
            dpr[start : end + 1] = 0
            weight[start : end + 1] = confidence
        elif label_kind == "key":
            key[start : end + 1] = 1
            weight[start : end + 1] = confidence
        soft_label = _region_soft_label(region, label_kind)
        soft_weight = _region_soft_weight(region, confidence, label_kind)
        if soft_label == soft_label and soft_weight > 0.0:
            source = str(region.get("source") or region.get("evidence_level") or "").lower()
            region_type = str(region.get("region_type") or region.get("type") or "").lower()
            if "self" in source or "self" in region_type:
                _write_soft_region(self_dpr, self_dpr_weight, start, end, soft_label, soft_weight)
            elif label_kind == "soft" or "pseudo" in source or "teacher" in source or "pseudo" in region_type:
                _write_soft_region(teacher_dpr, teacher_dpr_weight, start, end, soft_label, soft_weight)
    if candidate_prior_map:
        for prior in candidate_prior_map.get(protein_id, []):
            start = max(0, int(prior.get("start", 0)))
            end = min(length - 1, int(prior.get("end", start)))
            score = _float_or_nan(prior.get("score", prior.get("confidence", 1.0)))
            if score != score:
                score = 1.0
            prior_weight = _float_or_nan(prior.get("weight", prior.get("sample_weight", 0.2)))
            if prior_weight != prior_weight:
                prior_weight = 0.2
            _write_prior_region(candidate_prior, candidate_prior_weight, start, end, score, prior_weight)
    if teacher_score_map and protein_id in teacher_score_map:
        teacher = teacher_score_map[protein_id]
        consensus = _fit_vector_length(teacher["teacher_consensus"], length, fill=np.nan)
        if "teacher_confidence" in teacher:
            confidence = _fit_vector_length(teacher["teacher_confidence"], length, fill=0.0)
        else:
            uncertainty = _fit_vector_length(teacher.get("teacher_uncertainty", np.ones(length, dtype=np.float32)), length, fill=1.0)
            confidence = np.clip(1.0 - uncertainty, 0.0, 1.0)
        valid = np.isfinite(consensus) & (confidence > 0)
        teacher_dpr[valid] = np.clip(consensus[valid], 0.0, 1.0)
        teacher_dpr_weight[valid] = np.maximum(teacher_dpr_weight[valid], confidence[valid])
    soft = {
        "teacher_dpr": teacher_dpr,
        "teacher_dpr_weight": teacher_dpr_weight,
        "self_dpr": self_dpr,
        "self_dpr_weight": self_dpr_weight,
        "candidate_prior": candidate_prior,
        "candidate_prior_weight": candidate_prior_weight,
    }
    return dpr, key, weight, sample_regions, soft


def _region_label_kind(region: dict[str, object]) -> str:
    label = region.get("region_label")
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized in {"1", "positive", "gold", "curated"}:
            return "positive"
        if normalized in {"candidate", "prior"}:
            return "ignore"
        if normalized in {"0", "negative", "control"}:
            return "negative"
        if normalized in {"key", "key_region"}:
            return "key"
        if normalized in {"unknown", "ignore", ""}:
            return "ignore"
    elif isinstance(label, (int, float)):
        if int(label) == 1:
            return "positive"
        if int(label) == 0:
            return "negative"
    region_type = str(region.get("region_type") or region.get("type") or "").strip()
    if region_type in {"DPR_gold", "DPR_curated", "DPR_silver", "DPR_pseudo"}:
        return "positive"
    if region_type in {"DPR_candidate"}:
        return "ignore"
    if region_type in {"non_DPR_control"}:
        return "negative"
    if region_type in {"key_region"}:
        return "key"
    if region_type in {"DPR_soft", "DPR_teacher", "DPR_self_training"}:
        return "soft"
    return "ignore"


def _region_soft_label(region: dict[str, object], label_kind: str) -> float:
    for name in ("soft_label", "score", "mean_residue_score", "confidence"):
        value = _float_or_nan(region.get(name, np.nan))
        if value == value:
            return float(np.clip(value, 0.0, 1.0))
    if label_kind == "positive":
        return 1.0
    if label_kind == "negative":
        return 0.0
    return float("nan")


def _region_soft_weight(region: dict[str, object], confidence: float, label_kind: str) -> float:
    for name in ("soft_weight", "sample_weight", "weight"):
        value = _float_or_nan(region.get(name, np.nan))
        if value == value:
            return float(np.clip(value, 0.0, 1.0))
    if label_kind in {"positive", "negative", "soft"}:
        return float(np.clip(confidence, 0.0, 1.0))
    return 0.0


def _write_soft_region(target: np.ndarray, weight: np.ndarray, start: int, end: int, value: float, new_weight: float) -> None:
    old_weight = weight[start : end + 1]
    old_value = np.nan_to_num(target[start : end + 1], nan=0.0)
    denom = old_weight + new_weight
    merged = np.where(denom > 0.0, (old_value * old_weight + value * new_weight) / denom, value)
    target[start : end + 1] = merged.astype(np.float32)
    weight[start : end + 1] = np.maximum(old_weight, new_weight)


def _write_prior_region(target: np.ndarray, weight: np.ndarray, start: int, end: int, value: float, new_weight: float) -> None:
    target[start : end + 1] = np.maximum(target[start : end + 1], float(np.clip(value, 0.0, 1.0)))
    weight[start : end + 1] = np.maximum(weight[start : end + 1], float(np.clip(new_weight, 0.0, 1.0)))


def _fit_vector_length(values: np.ndarray, length: int, fill: float) -> np.ndarray:
    out = np.full(length, fill, dtype=np.float32)
    n = min(length, int(values.shape[0]))
    if n:
        out[:n] = values[:n].astype(np.float32, copy=False)
    return out


def _mil_bag_for(
    mil_bag_map: dict[str, dict[str, object]],
    label_row: dict[str, object],
    protein_id: str,
    llps_label: int,
) -> dict[str, object]:
    explicit = mil_bag_map.get(protein_id)
    if explicit is not None:
        label = explicit.get("bag_label", explicit.get("region_bag_label", IGNORE_INDEX))
        return {
            "region_bag_label": float(label if label is not None else IGNORE_INDEX),
            "region_bag_weight": float(explicit.get("bag_weight", explicit.get("region_bag_weight", 0.0))),
            "region_bag_type": str(explicit.get("bag_type", explicit.get("region_bag_type", "mask"))),
        }
    role = str(label_row.get("role_label", label_row.get("role_type", ""))).lower()
    tier = str(label_row.get("label_quality", label_row.get("label_tier", label_row.get("evidence_level", "")))).lower()
    negative_type = str(label_row.get("negative_type", "")).lower()
    sample_weight = _float_or_zero(label_row.get("sample_weight", label_row.get("label_confidence", 0.0)))
    if llps_label == 1 and any(token in role for token in ("driver", "scaffold", "self")):
        return {
            "region_bag_label": 1.0,
            "region_bag_weight": sample_weight if sample_weight > 0 else 0.75,
            "region_bag_type": "protein_positive_driver",
        }
    if llps_label == 0 and ("negative" in tier or "negative" in role or "negative" in negative_type):
        bag_type = "negative_disordered" if "disordered" in negative_type or "disordered" in role else "negative_structured"
        return {
            "region_bag_label": 0.0,
            "region_bag_weight": sample_weight if sample_weight > 0 else 0.75,
            "region_bag_type": bag_type,
        }
    return {"region_bag_label": float(IGNORE_INDEX), "region_bag_weight": 0.0, "region_bag_type": "mask"}


def _negative_regularization_weight(label_row: dict[str, object]) -> float:
    negative_type = str(label_row.get("negative_type", "")).lower()
    role = str(label_row.get("role_label", label_row.get("role_type", ""))).lower()
    llps = _float_or_nan(label_row.get("llps_label", np.nan))
    if llps != 0.0:
        return 0.0
    if "disordered" in negative_type or "disordered" in role:
        return 0.4
    if "structured" in negative_type or "structured" in role or "negative" in negative_type:
        return 0.2
    return 0.0


def _teacher_weight_from_row(row: dict[str, object]) -> float:
    explicit = _float_or_nan(row.get("teacher_confidence", row.get("teacher_llps_weight", np.nan)))
    if explicit == explicit:
        return float(np.clip(explicit, 0.0, 1.0))
    score = _float_or_nan(row.get("teacher_consensus_score", row.get("teacher_weighted", np.nan)))
    if score == score:
        agreement = _float_or_nan(row.get("teacher_agreement", np.nan))
        if agreement == agreement:
            return float(np.clip(agreement, 0.0, 1.0))
        return 0.5
    return 0.0


def _float_or_nan(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _float_or_zero(value: object) -> float:
    parsed = _float_or_nan(value)
    if parsed != parsed:
        return 0.0
    return float(np.clip(parsed, 0.0, 1.0))



# Source: features/build_features_sharded.py


import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from phaseflow.protein.features import build_feature_cache_from_manifest
from phaseflow.protein.features import ESM2Config


def build_sharded_feature_cache(args: Any) -> list[Path]:
    frame = pd.read_csv(args.manifest)
    workers = max(1, min(int(args.workers), len(frame) if len(frame) else 1))
    if workers == 1:
        return build_feature_cache_from_manifest(
            manifest=args.manifest,
            out_dir=args.out_dir,
            regions=args.regions,
            mil_bags=args.mil_bags,
            candidate_priors=args.candidate_priors,
            teacher_scores=args.teacher_scores,
            mode=args.mode,
            esm2_dir=args.esm2_dir,
            esm2_config=_esm2_config(args),
            structure_dir=args.structure_dir,
            protenix_embedding_dir=args.protenix_embedding_dir,
            protenix_embedding_dim=args.protenix_embedding_dim,
            af3_dir=args.af3_dir,
            starling_dir=args.starling_dir,
            starling_embedding_dir=args.starling_embedding_dir,
            starling_distance_dir=args.starling_distance_dir,
            local_window=args.local_window,
            graph_max_neighbors=args.graph_max_neighbors,
            graph_edge_dim=args.graph_edge_dim,
            starling_distance_topk=args.starling_distance_topk,
            require_structure=args.require_structure,
            require_starling=args.require_starling,
            overwrite=not args.no_overwrite,
        )

    shard_dir = Path(args.shard_dir or Path(args.out_dir) / f".feature_shards_{int(time.time())}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = _balanced_shards(frame, workers)
    shard_specs: list[dict[str, Any]] = []
    for index, row_indices in enumerate(shards):
        shard_path = shard_dir / f"manifest_shard_{index:03d}.csv"
        frame.iloc[row_indices].to_csv(shard_path, index=False)
        shard_specs.append(
            {
                "index": index,
                "manifest": str(shard_path),
                "records": len(row_indices),
                "args": vars(args),
            }
        )
    (shard_dir / "shards.json").write_text(json.dumps(shard_specs, indent=2, sort_keys=True) + "\n")

    written: list[Path] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_build_one_shard, spec) for spec in shard_specs if spec["records"]]
        for future in as_completed(futures):
            paths = future.result()
            written.extend(Path(path) for path in paths)
            print(f"finished_shard_paths={len(paths)} total_finished_paths={len(written)}", flush=True)
    return written


def _balanced_shards(frame: pd.DataFrame, workers: int) -> list[list[int]]:
    costs = [0 for _ in range(workers)]
    shards: list[list[int]] = [[] for _ in range(workers)]
    if "length" in frame.columns:
        items = [(index, int(row.get("length", 1) or 1)) for index, row in frame.iterrows()]
    else:
        items = [(index, len(str(row.get("sequence", "")))) for index, row in frame.iterrows()]
    for index, length in sorted(items, key=lambda item: item[1], reverse=True):
        target = min(range(workers), key=lambda worker: costs[worker])
        shards[target].append(index)
        costs[target] += max(length, 1)
    return shards


def _build_one_shard(spec: dict[str, Any]) -> list[str]:
    raw_args = spec["args"]
    paths = build_feature_cache_from_manifest(
        manifest=spec["manifest"],
        out_dir=raw_args["out_dir"],
        regions=raw_args.get("regions"),
        mil_bags=raw_args.get("mil_bags"),
        candidate_priors=raw_args.get("candidate_priors"),
        teacher_scores=raw_args.get("teacher_scores"),
        mode=raw_args.get("mode", "simple"),
        esm2_dir=raw_args.get("esm2_dir"),
        esm2_config=_esm2_config_from_mapping(raw_args),
        structure_dir=raw_args.get("structure_dir"),
        protenix_embedding_dir=raw_args.get("protenix_embedding_dir"),
        protenix_embedding_dim=int(raw_args.get("protenix_embedding_dim", 512)),
        af3_dir=raw_args.get("af3_dir"),
        starling_dir=raw_args.get("starling_dir"),
        starling_embedding_dir=raw_args.get("starling_embedding_dir"),
        starling_distance_dir=raw_args.get("starling_distance_dir"),
        local_window=int(raw_args.get("local_window", 16)),
        graph_max_neighbors=int(raw_args.get("graph_max_neighbors", 96)),
        graph_edge_dim=int(raw_args.get("graph_edge_dim", 13)),
        starling_distance_topk=int(raw_args.get("starling_distance_topk", 48)),
        require_structure=bool(raw_args.get("require_structure", False)),
        require_starling=bool(raw_args.get("require_starling", False)),
        overwrite=not bool(raw_args.get("no_overwrite", False)),
    )
    return [str(path) for path in paths]


def _esm2_config(args: Any) -> ESM2Config:
    return _esm2_config_from_mapping(vars(args))


def _esm2_config_from_mapping(values: dict[str, Any]) -> ESM2Config:
    return ESM2Config(
        model_name=values.get("esm2_model_name", "facebook/esm2_t33_650M_UR50D"),
        model_dir=values.get("esm2_model_dir"),
        device=values.get("esm2_device", "auto"),
        dtype=values.get("esm2_dtype", "float32"),
        storage_dtype=values.get("esm2_storage_dtype", "float32"),
        local_files_only=bool(values.get("esm2_local_files_only", False)),
        chunk_size=values.get("esm2_chunk_size"),
        overlap=int(values.get("esm2_overlap", 128)),
    )
