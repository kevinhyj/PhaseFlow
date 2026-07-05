from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phaseflow.full_length.features.run_esm2 import records_from_fasta, records_from_manifest


def parse_af3_outputs(
    records: list[tuple[str, str]],
    af3_output: str | Path,
    out_dir: str | Path,
    contact_topk: int = 32,
) -> list[Path]:
    af3_output = Path(af3_output)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for protein_id, sequence in records:
        job_dir = _find_job_dir(af3_output, protein_id)
        if job_dir is None:
            continue
        single = _find_single_embedding(job_dir, len(sequence))
        node = _find_confidence_node(job_dir, len(sequence))
        if single is None and node is None:
            continue
        contacts = _contacts_from_pair_embedding(job_dir, len(sequence), contact_topk)
        reliability = _reliability_from_node(node, len(sequence))
        payload = {
            "protein_id": np.asarray(protein_id),
            "sequence": np.asarray(sequence),
            "reliability": reliability,
        }
        if single is not None:
            payload["single_embedding"] = single.astype(np.float32, copy=False)
        if node is not None:
            payload["node"] = node.astype(np.float32, copy=False)
        if contacts is not None:
            payload["contacts"] = contacts.astype(np.float32, copy=False)
        path = out_dir / f"{protein_id}.npz"
        np.savez_compressed(path, **payload)
        written.append(path)
    return written


def _find_job_dir(root: Path, protein_id: str) -> Path | None:
    candidates = [root / protein_id, root / protein_id.lower(), root / protein_id.upper()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(path for path in root.glob("**/*") if path.is_dir() and path.name.lower() == protein_id.lower())
    return matches[0] if matches else None


def _find_single_embedding(job_dir: Path, length: int) -> np.ndarray | None:
    for path in sorted(job_dir.glob("**/embeddings.npz")):
        with np.load(path, allow_pickle=False) as data:
            for key in ("single_embeddings", "single_embedding"):
                if key in data:
                    value = np.asarray(data[key], dtype=np.float32)
                    if value.ndim == 2 and value.shape[0] >= length:
                        return value[:length]
    return None


def _find_confidence_node(job_dir: Path, length: int) -> np.ndarray | None:
    json_paths = sorted(job_dir.glob("*confidence*.json")) + sorted(job_dir.glob("**/*confidence*.json"))
    for path in json_paths:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        plddt = _first_numeric_vector(data, ("atom_plddts", "plddt", "pae_plddt"))
        if plddt is None:
            continue
        plddt = plddt[:length]
        if plddt.shape[0] != length:
            continue
        return np.stack([plddt / 100.0, np.ones(length, dtype=np.float32)], axis=1)
    return None


def _contacts_from_pair_embedding(job_dir: Path, length: int, topk: int) -> np.ndarray | None:
    for path in sorted(job_dir.glob("**/embeddings.npz")):
        with np.load(path, allow_pickle=False) as data:
            for key in ("pair_embeddings", "pair_embedding"):
                if key not in data:
                    continue
                pair = np.asarray(data[key])
                if pair.ndim != 3 or pair.shape[0] < length or pair.shape[1] < length:
                    continue
                score = np.linalg.norm(pair[:length, :length], axis=-1).astype(np.float32)
                return _topk_contacts(score, topk)
    return None


def _topk_contacts(score: np.ndarray, topk: int) -> np.ndarray:
    rows: list[tuple[int, int, float]] = []
    length = score.shape[0]
    for src in range(length):
        values = score[src].copy()
        values[src] = -np.inf
        if topk < length:
            idx = np.argpartition(-values, topk)[:topk]
        else:
            idx = np.arange(length)
        idx = idx[np.isfinite(values[idx])]
        idx = idx[np.argsort(-values[idx])]
        for dst in idx[:topk]:
            rows.append((src, int(dst), float(values[dst])))
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)


def _reliability_from_node(node: np.ndarray | None, length: int) -> np.ndarray:
    if node is None or node.shape[0] != length:
        return np.ones(length, dtype=np.float32)
    return np.clip(node[:, 0], 0.0, 1.0).astype(np.float32)


def _first_numeric_vector(data: object, names: tuple[str, ...]) -> np.ndarray | None:
    if isinstance(data, dict):
        for name in names:
            if name in data:
                value = np.asarray(data[name], dtype=np.float32).reshape(-1)
                return value
        for value in data.values():
            found = _first_numeric_vector(value, names)
            if found is not None:
                return found
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse AF3 outputs into PhaseFlow intermediate npz files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--fasta")
    parser.add_argument("--af3-output", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--contact-topk", type=int, default=32)
    args = parser.parse_args()
    records = records_from_manifest(args.manifest) if args.manifest else records_from_fasta(args.fasta)
    written = parse_af3_outputs(records, args.af3_output, args.out_dir, args.contact_topk)
    print(f"Wrote {len(written)} parsed AF3 feature files to {args.out_dir}")


if __name__ == "__main__":
    main()
