#!/usr/bin/env python3
"""Plot residue-score examples from a DPR profile archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.full_length.figures.common import configure_style, plt, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot residue-score examples from a DPR NPZ archive.")
    parser.add_argument("--profiles", type=Path, required=True, help="NPZ archive containing one-dimensional residue profiles.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-key", action="append", default=[], help="Archive key to include; defaults to the first twelve profiles.")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--font", type=Path)
    return parser.parse_args()


def profile_items(path: Path, requested: list[str]) -> list[tuple[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"profile archive does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        keys = requested or list(archive.files)
        items: list[tuple[str, np.ndarray]] = []
        for key in keys:
            if key not in archive:
                raise ValueError(f"profile key is missing: {key}")
            values = np.asarray(archive[key]).squeeze()
            if values.ndim != 1:
                continue
            items.append((key, values.astype(float, copy=False)))
    if not items:
        raise ValueError("the archive does not contain one-dimensional residue profiles")
    return items[:12]


def main() -> int:
    args = parse_args()
    items = profile_items(args.profiles, args.profile_key)
    columns = max(1, int(args.columns))
    rows = int(np.ceil(len(items) / columns))
    configure_style(args.font)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 2.6 * rows), squeeze=False)
    for axis, (protein_id, profile) in zip(axes.flat, items):
        axis.plot(np.arange(1, len(profile) + 1), profile, color="#56669e", linewidth=1.3)
        axis.set_title(protein_id)
        axis.set_xlabel("Residue")
        axis.set_ylabel("DPR score")
        axis.set_ylim(0.0, 1.0)
        axis.grid(color="#d9dde5", linewidth=0.7)
    for axis in axes.flat[len(items) :]:
        axis.set_visible(False)
    save_figure(fig, args.output_dir, "dpr_examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
