from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[3]
PSTP_REPO = ROOT_DIR / "external" / "teachers" / "PSTP"
PSTP_PYDEPS = ROOT_DIR / "external" / "teachers" / "pstp_pydeps"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(chunks).upper()))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        records.append((current_id, "".join(chunks).upper()))
    return [(protein_id, clean_sequence(sequence)) for protein_id, sequence in records if clean_sequence(sequence)]


def clean_sequence(sequence: str) -> str:
    return "".join(residue if residue in VALID_AA else "L" for residue in sequence.upper())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PSTP-Scan residue-level profiles over FASTA records.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--compression", default="lzf", choices=("gzip", "lzf", "none"))
    args = parser.parse_args()

    os.environ.setdefault("TORCH_HOME", str(ROOT_DIR / "model_cache" / "pstp_torch"))
    os.environ.setdefault("HF_HOME", str(ROOT_DIR / "model_cache" / "pstp_hf"))
    if PSTP_PYDEPS.exists():
        sys.path.insert(0, str(PSTP_PYDEPS))
    sys.path.insert(0, str(PSTP_REPO))

    import torch

    print(
        "PSTP runtime:"
        f" CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
        f" cuda_available={torch.cuda.is_available()}"
        f" device_count={torch.cuda.device_count()}",
        flush=True,
    )

    from pstp.pstp_collections import (
        predict_by_mix_models,
        predict_by_pdps_models,
        predict_by_saps_models,
        pstp_embedding_by_batch,
    )

    records = read_fasta(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, int(args.batch_size))
    progress_every = max(1, int(args.progress_every))
    compression = None if args.compression == "none" else args.compression
    with h5py.File(args.output, "w") as handle:
        handle.attrs["teacher"] = "pstp"
        handle.attrs["dataset"] = "pstp_scan_score"
        for batch_index, start in enumerate(range(0, len(records), batch_size), start=1):
            batch = records[start : start + batch_size]
            ids = [item[0] for item in batch]
            seqs = [item[1] for item in batch]
            matrices = pstp_embedding_by_batch(seqs)
            for protein_id, sequence, matrix in zip(ids, seqs, matrices):
                saps_profile, saps_score = predict_by_saps_models(matrix)
                pdps_profile, pdps_score = predict_by_pdps_models(matrix)
                mix_profile, mix_score = predict_by_mix_models(matrix)
                stacked = np.vstack(
                    [
                        np.asarray(saps_profile, dtype=np.float32),
                        np.asarray(pdps_profile, dtype=np.float32),
                        np.asarray(mix_profile, dtype=np.float32),
                    ]
                )
                profile = np.nanmax(stacked, axis=0).astype(np.float32)
                group = handle.create_group(str(protein_id))
                group.create_dataset("pstp_scan_score", data=np.clip(profile, 0.0, 1.0), compression=compression)
                group.create_dataset("pstp_saps_score", data=np.clip(saps_profile, 0.0, 1.0), compression=compression)
                group.create_dataset("pstp_pdps_score", data=np.clip(pdps_profile, 0.0, 1.0), compression=compression)
                group.create_dataset("pstp_mix_score", data=np.clip(mix_profile, 0.0, 1.0), compression=compression)
                saps_protein_score = float(np.asarray(saps_score).reshape(-1)[0])
                pdps_protein_score = float(np.asarray(pdps_score).reshape(-1)[0])
                mix_protein_score = float(np.asarray(mix_score).reshape(-1)[0])
                group.attrs["protein_score"] = float(max(saps_protein_score, pdps_protein_score, mix_protein_score))
                group.attrs["pstp_saps_protein_score"] = saps_protein_score
                group.attrs["pstp_pdps_protein_score"] = pdps_protein_score
                group.attrs["pstp_mix_protein_score"] = mix_protein_score
                group.attrs["sequence_length"] = len(sequence)
            if batch_index % progress_every == 0 or start + batch_size >= len(records):
                handle.flush()
                print(f"PSTP-Scan processed {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    print(f"PSTP-Scan profiles written to {args.output}; records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
