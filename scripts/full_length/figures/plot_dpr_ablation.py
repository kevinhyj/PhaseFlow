#!/usr/bin/env python3
"""Render the DPR ablation figure from the released multi-scale metric summary."""

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


ROWS = (("s1111_full_retrain", "PhaseFlow", "full"), ("s1011_no_biophys", "w/o BioPhys", "ablation"), ("s0111_no_esm2", "w/o ESM2", "ablation"), ("s1110_no_phaseflow_bridge", "w/o peptide module", "peptide"))
COLORS = {"full": "#F585BF", "ablation": "#728AB9", "peptide": "#C9857B"}
METRICS = (("auprc", "AUPRC"), ("segment_f1", "IoU@0.25 F1"), ("spearman", "Per-protein Spearman"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV exported by the DPR benchmark evaluation.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    return parser.parse_args()


def metric_columns(metric: str) -> tuple[str, list[str]]:
    if metric == "auprc":
        prefix = "phasepro_official_all_scales_global_residue_AUPRC_scale-"
        return prefix + "p257", [prefix + scale for scale in ("p33", "p129", "p257", "mean")]
    if metric == "segment_f1":
        return "phasepro_region_p257_segment_f1_iou_0_25", []
    prefix = "phasepro_official_all_scales_per_protein_Spearman_median_scale-"
    return prefix + "p257", [prefix + scale for scale in ("p33", "p129", "p257", "mean")]


def prepare_frame(source: pd.DataFrame) -> pd.DataFrame:
    if "arm_id" not in source:
        raise ValueError("metrics input is missing arm_id")
    source = source.drop_duplicates("arm_id", keep="last").set_index("arm_id", drop=False)
    missing_arms = [arm_id for arm_id, _, _ in ROWS if arm_id not in source.index]
    if missing_arms:
        raise ValueError(f"metrics input is missing DPR arms: {missing_arms}")
    rows: list[dict[str, object]] = []
    for arm_id, label, group in ROWS:
        item = source.loc[arm_id]
        record: dict[str, object] = {"arm_id": arm_id, "label": label, "group": group}
        for metric, _ in METRICS:
            value_column, scale_columns = metric_columns(metric)
            required = [value_column, *scale_columns]
            missing = [column for column in required if column not in source.columns]
            if missing:
                raise ValueError(f"metrics input is missing {metric} columns: {missing}")
            value = float(item[value_column])
            scales = pd.to_numeric(item[scale_columns], errors="raise").to_numpy(float) if scale_columns else np.array([value])
            record[metric] = value
            record[f"{metric}_scale_low"] = float(scales.min())
            record[f"{metric}_scale_high"] = float(scales.max())
        rows.append(record)
    return pd.DataFrame(rows)


def draw_metric(axis: plt.Axes, frame: pd.DataFrame, metric: str, label: str, show_labels: bool) -> None:
    values = frame[metric].to_numpy(float)
    lows = frame[f"{metric}_scale_low"].to_numpy(float)
    highs = frame[f"{metric}_scale_high"].to_numpy(float)
    y = np.arange(len(frame))
    axis.axvline(values[0], color="#8A8A8A", linewidth=1.1, linestyle=(0, (3, 2)), alpha=0.7, zorder=0)
    for index, (value, low, high, group) in enumerate(zip(values, lows, highs, frame["group"], strict=True)):
        color = COLORS[str(group)]
        if np.isclose(low, high):
            axis.plot(value, index, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1, zorder=3)
        else:
            axis.errorbar(value, index, xerr=[[value - low], [high - value]], fmt="o", color=color, ecolor=color, markersize=8, markeredgecolor="white", markeredgewidth=1, elinewidth=1.7, capsize=3, zorder=3)
    span = max(float(highs.max() - lows.min()), 0.02)
    axis.set_xlim(float(lows.min() - span * 0.12), float(highs.max() + span * 0.12))
    axis.set_ylim(-0.5, len(frame) - 0.5)
    axis.set_yticks(y)
    axis.invert_yaxis()
    axis.set_xlabel(label)
    if show_labels:
        axis.set_yticklabels(frame["label"])
    else:
        axis.tick_params(axis="y", labelleft=False, length=0)
    axis.grid(axis="x", color="#D8D8D8", linewidth=0.8, linestyle=(0, (3, 2)))
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(1.1)


def main() -> int:
    args = parse_args()
    frame = prepare_frame(read_metrics(args.input))
    configure_style(args.font)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(12.6, 4.2), sharey=True, squeeze=False)
    for index, (axis, (metric, label)) in enumerate(zip(axes[0], METRICS, strict=True)):
        draw_metric(axis, frame, metric, label, index == 0)
    save_figure(fig, args.output_dir, "dpr_ablation")
    frame.to_csv(args.output_dir / "dpr_ablation_plot_data.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
