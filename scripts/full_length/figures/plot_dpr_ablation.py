#!/usr/bin/env python3
"""Plot DPR ablation metrics from a tabular evaluation export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.full_length.figures.common import choose_column, configure_style, plt, read_metrics, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DPR ablation metrics from a CSV file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-column")
    parser.add_argument("--metric", action="append", default=["auprc", "segment_f1", "spearman"])
    parser.add_argument("--font", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = read_metrics(args.input)
    label = choose_column(frame, args.label_column, ("arm_id", "label", "method", "name"), "label")
    metrics = list(dict.fromkeys(args.metric))
    missing = [column for column in metrics if column not in frame]
    if missing:
        raise ValueError(f"metrics input is missing columns: {missing}")
    configure_style(args.font)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.1 * len(metrics), max(3.4, 0.42 * len(frame))))
    for axis, column in zip(axes, metrics):
        rows = frame.sort_values(column, ascending=True)
        axis.plot(rows[column], rows[label].astype(str), "o", color="#728ab9")
        axis.set_title(column.replace("_", " "))
        axis.grid(axis="x", color="#d9dde5", linewidth=0.8)
        axis.set_axisbelow(True)
    save_figure(fig, args.output_dir, "dpr_ablation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
