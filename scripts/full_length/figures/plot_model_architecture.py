#!/usr/bin/env python3
"""Render the full-length PhaseFlow architecture diagram."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.full_length.figures.common import configure_style, plt, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the full-length PhaseFlow architecture diagram.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    return parser.parse_args()


def add_box(axis: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    box = FancyBboxPatch((x, y), 2.15, 0.72, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#1f2937", linewidth=1.1)
    axis.add_patch(box)
    axis.text(x + 1.075, y + 0.36, text, ha="center", va="center", fontsize=9)


def connect(axis: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    axis.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, linewidth=1.0, color="#4b5563"))


def main() -> int:
    args = parse_args()
    configure_style(args.font)
    fig, axis = plt.subplots(figsize=(11, 5.4))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5)
    axis.axis("off")
    inputs = ((0.5, 3.8, "Sequence", "#dce7f7"), (0.5, 2.7, "Biophysical features", "#e5f0e7"), (0.5, 1.6, "Structure features", "#f8e5e9"))
    for x, y, text, color in inputs:
        add_box(axis, x, y, text, color)
        connect(axis, x + 2.15, y + 0.36, 3.25, 2.85)
    add_box(axis, 3.3, 2.49, "Feature adapters\n+reliability fusion", "#f7ecd4")
    add_box(axis, 6.15, 2.49, "Local and graph\n+encoders", "#e9e1f5")
    add_box(axis, 8.95, 3.35, "LLPS readout", "#f8d9e5")
    add_box(axis, 8.95, 1.65, "DPR multi-scale\n+readout", "#d9edf0")
    connect(axis, 5.45, 2.85, 6.15, 2.85)
    connect(axis, 8.3, 2.85, 8.95, 3.7)
    connect(axis, 8.3, 2.85, 8.95, 2.0)
    axis.text(6, 4.65, "PhaseFlow full-length architecture", ha="center", va="center", fontsize=14, fontweight="bold")
    save_figure(fig, args.output_dir, "full_length_architecture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
