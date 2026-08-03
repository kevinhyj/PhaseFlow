"""Render the final six-panel DPR benchmark comparison."""


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS = ("auroc", "auprc", "iou_at_0_25", "precision", "f1", "recall")
LABELS = {"auroc": "AUROC", "auprc": "AUPRC", "iou_at_0_25": "IoU@0.25", "precision": "Precision", "f1": "F1", "recall": "Recall"}
COLORS = {"PSTP": "#D2A68C", "PSPHunter": "#D89BB7", "catGRANULE": "#A8C7C6", "PhaseFlow": "#56669E"}
RESULTS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = RESULTS_ROOT / "benchmark/dpr/dpr_requested_table.csv"
RELEASE_COLUMNS = {
    "Model": "model",
    "residue AUROC": "auroc",
    "residue AUPRC": "auprc",
    "IoU@0.25 segment F1": "iou_at_0_25",
    "region precision": "precision",
    "Dice": "f1",
    "region recall": "recall",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV with one model per row and DPR benchmark metrics (default: released DPR table).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-column", default="model")
    parser.add_argument("--metric", action="append", help="Metric column to plot; repeat to override the six-paper-panel set.")
    parser.add_argument("--font", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"metrics input does not exist: {args.input}")
    frame = pd.read_csv(args.input).rename(columns=RELEASE_COLUMNS)
    metrics = tuple(dict.fromkeys(args.metric or DEFAULT_METRICS))
    missing = [column for column in (args.label_column, *metrics) if column not in frame]
    if missing:
        raise ValueError(f"metrics input is missing columns: {missing}")
    if frame.empty:
        raise ValueError("metrics input contains no rows")
    frame = frame.dropna(subset=[args.label_column, *metrics]).copy()
    if frame.empty:
        raise ValueError("metrics input has no complete rows for the requested metrics")
    if args.font is not None:
        from matplotlib.font_manager import FontProperties, fontManager

        if not args.font.is_file():
            raise FileNotFoundError(f"font does not exist: {args.font}")
        fontManager.addfont(str(args.font))
        plt.rcParams["font.sans-serif"] = [FontProperties(fname=str(args.font)).get_name()]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    columns = min(3, len(metrics))
    rows = (len(metrics) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 1.98 * rows), squeeze=False)
    for axis, metric in zip(axes.flat, metrics, strict=True):
        values = pd.to_numeric(frame[metric], errors="raise")
        labels = frame[args.label_column].astype(str)
        for position, (label, value) in enumerate(zip(labels, values, strict=True)):
            axis.barh(position * 0.32, value, height=0.16, color=COLORS.get(label, "#9AA4AE"), alpha=1.0 if label == "PhaseFlow" else 0.70)
        span = max(float(values.max() - values.min()), 0.02)
        axis.set_xlim(max(0.0, float(values.min() - span * 0.18)), float(values.max() + span * 0.18))
        axis.set_ylim(-0.14, (len(frame) - 1) * 0.32 + 0.14)
        axis.set_yticks([index * 0.32 for index in range(len(frame))], labels)
        axis.tick_params(axis="y", length=0)
        axis.set_title(LABELS.get(metric, metric.replace("_", " ")), loc="left", fontsize=10, fontweight="bold", color="#333333", pad=3)
        axis.grid(axis="x", alpha=0.3, linestyle="--")
        axis.spines[["top", "right", "left"]].set_visible(False)
    for axis in axes.flat[len(metrics):]:
        axis.set_visible(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.loc[:, [args.label_column, *metrics]].to_csv(args.output_dir / "dpr_benchmark_plot_data.csv", index=False)
    for suffix, kwargs in (("png", {"dpi": 600, "facecolor": "white"}), ("pdf", {"facecolor": "white"}), ("svg", {"facecolor": "none", "transparent": True})):
        figure.savefig(args.output_dir / f"dpr_benchmark.{suffix}", bbox_inches="tight", pad_inches=0.3, **kwargs)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
