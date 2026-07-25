#!/usr/bin/env python3
"""Collect explicitly selected figure assets for manuscript assembly."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy selected full-length figure assets into one directory.")
    parser.add_argument("--asset", type=Path, action="append", required=True, help="Figure file to include; repeat for each asset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied: set[str] = set()
    for source in args.asset:
        if not source.is_file():
            raise FileNotFoundError(f"figure asset does not exist: {source}")
        destination = args.output_dir / source.name
        if destination.name in copied:
            raise ValueError(f"duplicate figure filename: {destination.name}")
        shutil.copy2(source, destination)
        copied.add(destination.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
