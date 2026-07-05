from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def zero_af_features(length: int, dim: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node = np.zeros((length, dim), dtype=np.float32)
    missing = np.ones(length, dtype=np.float32)
    reliability = np.zeros(length, dtype=np.float32)
    return node, missing, reliability


def load_af3_features(path: str | Path, sequence: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load parsed AF3 features from an intermediate npz file.

    Expected keys are intentionally simple so different AF3 parser versions can
    feed the same training cache:
    - `single_embedding` or `single_embeddings`: [L, 384]
    - `node` or `node_features`: [L, d]
    - `contacts`: [E, >=3] as src, dst, confidence
    - optional `reliability`: [L]
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        cached_sequence = str(data["sequence"].item()) if "sequence" in data else sequence
        if cached_sequence and cached_sequence != sequence:
            raise ValueError(f"AF3 sequence mismatch for {path}: cache sequence differs from feature cache sequence")
        single = _optional_array(data, "single_embedding", "single_embeddings")
        node = _optional_array(data, "node", "node_features")
        if single is None and node is None:
            raise ValueError(f"AF3 feature file {path} has neither single_embedding nor node features")
        parts = [array for array in (single, node) if array is not None]
        for array in parts:
            if array.ndim != 2 or array.shape[0] != len(sequence):
                raise ValueError(f"AF3 feature array in {path} must have shape [L, D], got {array.shape}")
        features = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        reliability = np.asarray(data["reliability"], dtype=np.float32) if "reliability" in data else np.ones(len(sequence), dtype=np.float32)
        contacts = np.asarray(data["contacts"], dtype=np.float32) if "contacts" in data else None
    missing = np.zeros(len(sequence), dtype=np.float32)
    return features, missing, reliability, contacts


def write_af3_input_json(
    protein_id: str,
    sequence: str,
    out_dir: str | Path,
    model_seeds: list[int] | None = None,
    dialect: str = "alphafold3",
    version: int = 3,
    msa_mode: str = "no_msa",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_seeds = model_seeds or [1]
    protein = {"id": "A", "sequence": sequence}
    if msa_mode == "no_msa":
        protein.update({"unpairedMsa": "", "pairedMsa": "", "templates": []})
    elif msa_mode != "full_pipeline":
        raise ValueError(f"Unsupported AF3 msa_mode: {msa_mode}")
    payload = {
        "name": protein_id,
        "modelSeeds": [int(seed) for seed in model_seeds],
        "sequences": [{"protein": protein}],
        "dialect": dialect,
        "version": int(version),
    }
    path = out_dir / f"{protein_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _optional_array(data: np.lib.npyio.NpzFile, *names: str) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name], dtype=np.float32)
    return None
