from __future__ import annotations

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
