from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[3]
PSTP_REPO = ROOT_DIR / "external" / "teachers" / "PSTP"
PSTP_PYDEPS = ROOT_DIR / "external" / "teachers" / "pstp_pydeps"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(sequence: str) -> str:
    return "".join(residue if residue in VALID_AA else "L" for residue in sequence.upper())


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
                    sequence = clean_sequence("".join(chunks))
                    if sequence:
                        records.append((current_id, sequence))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        sequence = clean_sequence("".join(chunks))
        if sequence:
            records.append((current_id, sequence))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PSTP as a protein-level teacher; writes no residue/profile output.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    os.environ.setdefault("TORCH_HOME", str(ROOT_DIR / "model_cache" / "pstp_torch"))
    os.environ.setdefault("HF_HOME", str(ROOT_DIR / "model_cache" / "pstp_hf"))
    if PSTP_PYDEPS.exists():
        sys.path.insert(0, str(PSTP_PYDEPS))
    sys.path.insert(0, str(PSTP_REPO))

    import torch

    print(
        "PSTP protein runtime:"
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
    fieldnames = [
        "protein_id",
        "pstp_score",
        "pstp_saps_score",
        "pstp_pdps_score",
        "pstp_mix_score",
        "sequence_length",
    ]
    processed = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for batch_index, start in enumerate(range(0, len(records), batch_size), start=1):
            batch = records[start : start + batch_size]
            ids = [item[0] for item in batch]
            seqs = [item[1] for item in batch]
            matrices = pstp_embedding_by_batch(seqs)
            for protein_id, sequence, matrix in zip(ids, seqs, matrices):
                _, saps_score = predict_by_saps_models(matrix)
                _, pdps_score = predict_by_pdps_models(matrix)
                _, mix_score = predict_by_mix_models(matrix)
                scores = [float(np.asarray(value).reshape(-1)[0]) for value in [saps_score, pdps_score, mix_score]]
                writer.writerow(
                    {
                        "protein_id": protein_id,
                        "pstp_score": max(scores),
                        "pstp_saps_score": scores[0],
                        "pstp_pdps_score": scores[1],
                        "pstp_mix_score": scores[2],
                        "sequence_length": len(sequence),
                    }
                )
                processed += 1
            if batch_index % progress_every == 0 or processed >= len(records):
                handle.flush()
                print(f"PSTP protein processed {processed}/{len(records)}", flush=True)
    print(f"PSTP protein scores written to {args.output}; records={processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
