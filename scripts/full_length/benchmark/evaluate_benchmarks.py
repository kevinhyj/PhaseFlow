#!/usr/bin/env python3
"""Combine explicit LLPS and DPR evaluation exports into a benchmark summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine full-length LLPS and DPR benchmark metric tables.")
    parser.add_argument("--llps-metrics", type=Path, required=True)
    parser.add_argument("--dpr-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_metrics(path: Path, task: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{task} metrics file does not exist: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{task} metrics file contains no rows: {path}")
    frame = frame.copy()
    frame.insert(0, "task", task)
    return frame


def main() -> int:
    args = parse_args()
    llps = read_metrics(args.llps_metrics, "llps")
    dpr = read_metrics(args.dpr_metrics, "dpr")
    summary = pd.concat([llps, dpr], ignore_index=True, sort=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "benchmark_metrics.csv", index=False)
    payload = {
        "format": "phaseflow_full_length_benchmark_summary_v1",
        "inputs": {"llps_metrics": str(args.llps_metrics), "dpr_metrics": str(args.dpr_metrics)},
        "rows": int(len(summary)),
        "tasks": {"llps": int(len(llps)), "dpr": int(len(dpr))},
    }
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
