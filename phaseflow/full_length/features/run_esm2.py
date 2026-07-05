from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from phaseflow.full_length.features.plm_embedder import ESM2Config, ESM2Embedder, clean_protein_sequence, download_esm2_model


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen residue-level ESM-2 embeddings.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="CSV with protein_id and sequence columns.")
    source.add_argument("--fasta", help="FASTA file used when a manifest is not available.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--model-dir")
    parser.add_argument("--download", action="store_true", help="Download the Hugging Face snapshot before extraction.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--storage-dtype", default="float32", choices=["float16", "fp16", "float32", "fp32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length-policy", default="chunk", choices=["chunk", "error"])
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir
    if args.download:
        model_dir = str(download_esm2_model(args.model_name, args.model_dir))
    records = records_from_manifest(args.manifest) if args.manifest else records_from_fasta(args.fasta)
    config = ESM2Config(
        model_name=args.model_name,
        model_dir=model_dir,
        device=args.device,
        dtype=args.dtype,
        storage_dtype=args.storage_dtype,
        local_files_only=args.local_files_only,
        max_length_policy=args.max_length_policy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    written = run_esm2_embeddings(records, args.out_dir, config, overwrite=args.overwrite)
    print(f"Wrote {len(written)} ESM-2 embedding files to {args.out_dir}")


if __name__ == "__main__":
    main()
