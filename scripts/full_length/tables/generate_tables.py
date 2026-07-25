#!/usr/bin/env python3
"""Render a CSV result table as Markdown for manuscript or documentation use."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a CSV table as Markdown.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", nargs="*", default=None, help="Optional ordered subset of CSV columns.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"table input does not exist: {args.input}")
    frame = pd.read_csv(args.input)
    if frame.empty:
        raise ValueError(f"table input contains no rows: {args.input}")
    if args.columns:
        missing = [column for column in args.columns if column not in frame]
        if missing:
            raise ValueError(f"requested columns are missing: {missing}")
        frame = frame.loc[:, args.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
