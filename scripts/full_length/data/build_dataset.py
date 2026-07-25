#!/usr/bin/env python3
"""Validate a full-length release package and write its training manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "llps": {
        "proteins": {"protein_id", "sequence_sha256", "sequence", "sequence_length"},
        "training_units": {"protein_id", "sequence_sha256", "llps_label", "sample_weight"},
    },
    "dpr": {
        "proteins": {"protein_id", "sequence_sha256", "sequence", "sequence_length"},
        "training_units": {"protein_id", "sequence_sha256", "training_stage"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a full-length dataset package and write a training manifest.")
    parser.add_argument("--task", choices=tuple(REQUIRED_COLUMNS), required=True)
    parser.add_argument("--package-root", type=Path, required=True, help="Directory containing data/ and metadata/.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON manifest.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_required_table(package_root: Path, name: str, columns: set[str]) -> tuple[Path, pd.DataFrame]:
    path = package_root / "data" / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"required dataset table is missing: {path}")
    frame = pd.read_parquet(path)
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no records")
    return path, frame


def build_manifest(task: str, package_root: Path) -> dict[str, object]:
    tables: dict[str, pd.DataFrame] = {}
    table_paths: dict[str, Path] = {}
    for name, columns in REQUIRED_COLUMNS[task].items():
        table_paths[name], tables[name] = read_required_table(package_root, name, columns)

    proteins = tables["proteins"]
    units = tables["training_units"]
    protein_ids = set(proteins["protein_id"].astype(str))
    unknown_ids = sorted(set(units["protein_id"].astype(str)).difference(protein_ids))
    if unknown_ids:
        raise ValueError(f"training units reference unknown proteins: {unknown_ids[:10]}")
    if proteins["protein_id"].astype(str).duplicated().any():
        raise ValueError("proteins.parquet contains duplicate protein_id values")

    return {
        "format": "phaseflow_full_length_dataset_manifest_v1",
        "task": task,
        "package_root": str(package_root),
        "tables": {
            name: {"path": str(path), "rows": int(len(tables[name])), "sha256": sha256_file(path)}
            for name, path in table_paths.items()
        },
        "proteins": int(len(proteins)),
        "training_units": int(len(units)),
    }


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args.task, args.package_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
