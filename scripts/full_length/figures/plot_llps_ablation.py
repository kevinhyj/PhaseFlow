#!/usr/bin/env python3
"""Render the LLPS embedding-ablation figure from an exported metric table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.full_length.figures.common import configure_style, plt, read_metrics, save_figure


GROUP_COLORS = {"phaseflow": "#F585BF", "wo": "#728AB9", "starling": "#9A9A9A", "pseudo": "#C9857B"}
DEFAULT_METRICS = (("auprc", "AUPRC"), ("auroc", "AUROC"), ("mcc_at_0.5", "MCC@0.5"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with one row per ablation arm and metric values or bootstrap intervals.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", action="append", help="Metric name to include; repeat to change the default paper panel set.")
    parser.add_argument("--font", type=Path)
    return parser.parse_args()


def value_column(frame: pd.DataFrame, metric: str) -> str:
    for column in (f"{metric}_value", metric):
        if column in frame:
            return column
    raise ValueError(f"metrics input is missing a value column for {metric}")


def prepare_frame(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    required = {"arm_id", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"metrics input is missing columns: {missing}")
    out = frame.copy()
    out["group"] = out.get("group", "wo").fillna("wo").astype(str)
    for metric in metrics:
        column = value_column(out, metric)
        out[f"{metric}_value"] = pd.to_numeric(out[column], errors="raise")
        for bound in ("ci_low", "ci_high"):
            column = f"{metric}_{bound}"
            out[column] = pd.to_numeric(out.get(column, out[f"{metric}_value"]), errors="raise")
    is_reference = out["arm_id"].eq("no_starling_reference")
    is_pseudo = out["group"].eq("pseudo") | out["arm_id"].str.contains("pseudo", case=False, na=False)
    out["_section"] = np.select([is_reference, is_pseudo], [0, 2], default=1)
    out = out.sort_values(["_section", "auprc_value"], ascending=[True, False], kind="mergesort").drop(columns="_section")
    return out.reset_index(drop=True)


def draw_metric(axis: plt.Axes, frame: pd.DataFrame, metric: str, label: str, show_labels: bool) -> None:
    values = frame[f"{metric}_value"].to_numpy(float)
    lows = frame[f"{metric}_ci_low"].to_numpy(float)
    highs = frame[f"{metric}_ci_high"].to_numpy(float)
    y = np.arange(len(frame))
    reference = float(values[frame["arm_id"].eq("no_starling_reference").argmax()])
    axis.axvline(reference, color="#8A8A8A", linewidth=1.1, linestyle=(0, (3, 2)), alpha=0.7, zorder=0)
    for index, (value, low, high, group) in enumerate(zip(values, lows, highs, frame["group"], strict=True)):
        color = GROUP_COLORS.get(str(group), "#728AB9")
        axis.errorbar(value, index, xerr=[[value - low], [high - value]], fmt="o", color=color, ecolor=color, markersize=7, markeredgecolor="white", markeredgewidth=0.9, elinewidth=1.4, capsize=3, zorder=3)
    pseudo_positions = np.flatnonzero(frame["group"].eq("pseudo") | frame["arm_id"].str.contains("pseudo", case=False, na=False))
    if len(pseudo_positions) and pseudo_positions[0] > 0:
        axis.axhline(pseudo_positions[0] - 0.5, color="#666666", linewidth=0.9, linestyle=(0, (3, 2)))
    span = max(float(highs.max() - lows.min()), 0.02)
    axis.set_xlim(float(lows.min() - span * 0.1), float(highs.max() + span * 0.1))
    axis.set_ylim(-0.5, len(frame) - 0.5)
    axis.set_yticks(y)
    axis.invert_yaxis()
    axis.set_xlabel(label)
    if show_labels:
        axis.set_yticklabels(frame["label"].astype(str))
    else:
        axis.tick_params(axis="y", labelleft=False, length=0)
    axis.grid(axis="x", color="#D8D8D8", linewidth=0.8, linestyle=(0, (3, 2)))
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(1.1)


def main() -> int:
    args = parse_args()
    requested = args.metric or [metric for metric, _ in DEFAULT_METRICS]
    metric_labels = {metric: label for metric, label in DEFAULT_METRICS}
    frame = prepare_frame(read_metrics(args.input), requested)
    configure_style(args.font)
    fig, axes = plt.subplots(1, len(requested), figsize=(4.2 * len(requested), max(3.5, 0.48 * len(frame))), sharey=True, squeeze=False)
    for index, (axis, metric) in enumerate(zip(axes[0], requested, strict=True)):
        draw_metric(axis, frame, metric, metric_labels.get(metric, metric.replace("_", " ")), index == 0)
    save_figure(fig, args.output_dir, "llps_ablation")
    frame.to_csv(args.output_dir / "llps_ablation_plot_data.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
