from __future__ import annotations

import argparse
from pathlib import Path

from phaseflow.full_length.features.af_parser import write_af3_input_json
from phaseflow.full_length.features.plm_embedder import clean_protein_sequence
from phaseflow.full_length.features.run_esm2 import records_from_fasta, records_from_manifest


def make_af3_jsons(
    records: list[tuple[str, str]],
    out_dir: str | Path,
    model_seeds: list[int],
    msa_mode: str = "no_msa",
) -> list[Path]:
    paths: list[Path] = []
    for protein_id, sequence in records:
        paths.append(
            write_af3_input_json(
                protein_id,
                clean_protein_sequence(sequence),
                out_dir,
                model_seeds=model_seeds,
                msa_mode=msa_mode,
            )
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AlphaFold 3 input JSON files from manifest or FASTA.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--fasta")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--msa-mode",
        choices=["no_msa", "full_pipeline"],
        default="no_msa",
        help="no_msa writes empty MSA/template fields so AF3 can run with --run_data_pipeline=false.",
    )
    args = parser.parse_args()
    records = records_from_manifest(args.manifest) if args.manifest else records_from_fasta(args.fasta)
    paths = make_af3_jsons(records, args.out_dir, args.model_seeds, msa_mode=args.msa_mode)
    print(f"Wrote {len(paths)} AF3 input JSON files to {args.out_dir}")


if __name__ == "__main__":
    main()
