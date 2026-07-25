#!/usr/bin/env python3
"""Plot LLPS benchmark metrics from a tabular evaluation export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.full_length.figures.common import choose_column, configure_style, plt, read_metrics, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LLPS benchmark metrics from a CSV file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-column")
    parser.add_argument("--auprc-column", default="auprc")
    parser.add_argument("--auroc-column", default="auroc")
    parser.add_argument("--font", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = read_metrics(args.input)
    label = choose_column(frame, args.label_column, ("model", "method", "name", "arm_id"), "label")
    metrics = ((args.auprc_column, "AUPRC"), (args.auroc_column, "AUROC"))
    missing = [column for column, _ in metrics if column not in frame]
    if missing:
        raise ValueError(f"metrics input is missing columns: {missing}")
    configure_style(args.font)
    fig, axes = plt.subplots(1, 2, figsize=(9, max(3.4, 0.42 * len(frame))))
    for axis, (column, title) in zip(axes, metrics):
        rows = frame.sort_values(column, ascending=True)
        axis.barh(rows[label].astype(str), rows[column], color="#56669e")
        axis.set_title(title)
        axis.set_xlabel(title)
        axis.grid(axis="x", color="#d9dde5", linewidth=0.8)
        axis.set_axisbelow(True)
    save_figure(fig, args.output_dir, "llps_benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
