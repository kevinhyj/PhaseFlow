from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from phaseflow.full_length.features.run_esm2 import records_from_fasta, records_from_manifest
from phaseflow.full_length.features.structure_parser import parse_single_protenix_output


def parse_protenix_outputs_parallel(
    records: list[tuple[str, str]],
    protenix_output: str | Path,
    out_dir: str | Path,
    contact_topk: int = 32,
    contact_cutoff: float = 8.0,
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), len(records) if records else 1))
    stats = {"written": 0, "skipped": 0, "missing": 0, "failed": 0}
    jobs = [
        {
            "protein_id": protein_id,
            "sequence": sequence,
            "protenix_output": str(protenix_output),
            "out_dir": str(out_dir),
            "contact_topk": int(contact_topk),
            "contact_cutoff": float(contact_cutoff),
            "overwrite": bool(overwrite),
        }
        for protein_id, sequence in records
    ]
    if workers == 1:
        for job in jobs:
            status = _parse_one(job)
            stats[status] = stats.get(status, 0) + 1
        return stats
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_parse_one, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                status = future.result()
            except Exception:
                status = "failed"
            stats[status] = stats.get(status, 0) + 1
            if index == 1 or index % 500 == 0 or index == len(futures):
                print(
                    "parse_progress "
                    f"done={index}/{len(futures)} "
                    f"written={stats.get('written', 0)} "
                    f"skipped={stats.get('skipped', 0)} "
                    f"missing={stats.get('missing', 0)} "
                    f"failed={stats.get('failed', 0)}",
                    flush=True,
                )
    return stats


def _parse_one(job: dict[str, Any]) -> str:
    protein_id = str(job["protein_id"])
    sequence = str(job["sequence"])
    out_path = Path(str(job["out_dir"])) / f"{protein_id}.npz"
    if not bool(job["overwrite"]) and _existing_npz_is_valid(out_path, protein_id, sequence):
        return "skipped"
    parsed = parse_single_protenix_output(
        protein_id=protein_id,
        sequence=sequence,
        protenix_output=Path(str(job["protenix_output"])),
        contact_topk=int(job["contact_topk"]),
        contact_cutoff=float(job["contact_cutoff"]),
    )
    if parsed is None:
        return "missing"
    payload: dict[str, Any] = {
        "protein_id": np.asarray(protein_id),
        "sequence": np.asarray(sequence),
        "node": parsed.node.astype(np.float32, copy=False),
        "missing_mask": parsed.missing_mask.astype(np.float32, copy=False),
        "reliability": parsed.reliability.astype(np.float32, copy=False),
    }
    if parsed.contacts is not None:
        payload["contacts"] = parsed.contacts.astype(np.float32, copy=False)
    for key, value in parsed.metadata.items():
        if value is not None:
            payload[key] = np.asarray(str(value))
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp_path.replace(out_path)
    return "written"


def _existing_npz_is_valid(path: Path, protein_id: str, sequence: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            cached_id = str(data["protein_id"].item()) if "protein_id" in data else protein_id
            cached_sequence = str(data["sequence"].item()) if "sequence" in data else sequence
            node = np.asarray(data["node"], dtype=np.float32) if "node" in data else None
            missing = np.asarray(data["missing_mask"], dtype=np.float32) if "missing_mask" in data else None
            reliability = np.asarray(data["reliability"], dtype=np.float32) if "reliability" in data else None
        return (
            cached_id == protein_id
            and cached_sequence == sequence
            and node is not None
            and missing is not None
            and reliability is not None
            and node.ndim == 2
            and node.shape[0] == len(sequence)
            and missing.shape == (len(sequence),)
            and reliability.shape == (len(sequence),)
        )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Protenix outputs into PhaseFlow structure intermediate npz files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--fasta")
    parser.add_argument("--protenix-output", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--contact-topk", type=int, default=32)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    records = records_from_manifest(args.manifest) if args.manifest else records_from_fasta(args.fasta)
    stats = parse_protenix_outputs_parallel(
        records=records,
        protenix_output=args.protenix_output,
        out_dir=args.out_dir,
        contact_topk=args.contact_topk,
        contact_cutoff=args.contact_cutoff,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(
        "Parsed Protenix structure features "
        f"to {args.out_dir}: "
        f"written={stats.get('written', 0)} "
        f"skipped={stats.get('skipped', 0)} "
        f"missing={stats.get('missing', 0)} "
        f"failed={stats.get('failed', 0)}"
    )


if __name__ == "__main__":
    main()
