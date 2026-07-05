from __future__ import annotations

import argparse
import json
from pathlib import Path

from phaseflow.full_length.features.plm_embedder import clean_protein_sequence
from phaseflow.full_length.features.run_esm2 import records_from_fasta, records_from_manifest


def write_protenix_input_json(
    protein_id: str,
    sequence: str,
    out_dir: str | Path,
    model_seeds: list[int] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": protein_id,
            "modelSeeds": [int(seed) for seed in (model_seeds or [101])],
            "covalent_bonds": [],
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": clean_protein_sequence(sequence),
                        "count": 1,
                        "modifications": [],
                    }
                }
            ],
        }
    ]
    path = out_dir / f"{protein_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def make_protenix_jsons(
    records: list[tuple[str, str]],
    out_dir: str | Path,
    model_seeds: list[int],
) -> list[Path]:
    return [write_protenix_input_json(protein_id, sequence, out_dir, model_seeds) for protein_id, sequence in records]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Protenix input JSON files from a PhaseFlow manifest or FASTA.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--fasta")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[101])
    args = parser.parse_args()
    records = records_from_manifest(args.manifest) if args.manifest else records_from_fasta(args.fasta)
    paths = make_protenix_jsons(records, args.out_dir, args.model_seeds)
    print(f"Wrote {len(paths)} Protenix input JSON files to {args.out_dir}")


if __name__ == "__main__":
    main()
