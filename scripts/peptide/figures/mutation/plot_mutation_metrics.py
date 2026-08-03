"""Plot mutation-effect AUROC and AUPRC comparisons."""


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {"model", "mean_auroc", "mean_auprc"}
MODEL_COLORS = {
    "PhaseFlow": "#56669E",
    "DeePhase": "#79AAA3",
    "PSPHunter": "#D89BB7",
    "PSPredictor": "#8FA184",
    "PSTP": "#D2A68C",
    "PhaseMotif": "#5F8E87",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with model, mean_auroc, and mean_auprc columns.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(metrics.columns))
    if missing:
        raise ValueError(f"mutation metric table is missing columns: {missing}")
    metrics = metrics.loc[:, ["model", "mean_auroc", "mean_auprc"]].copy()
    metrics["model"] = metrics["model"].astype(str)
    for column in ("mean_auroc", "mean_auprc"):
        metrics[column] = pd.to_numeric(metrics[column], errors="raise")
    if metrics.empty:
        raise ValueError("mutation metric table is empty")
    return metrics


def plot_metric(axis: plt.Axes, metrics: pd.DataFrame, *, column: str, label: str, baseline: float | None) -> pd.DataFrame:
    ranked = metrics.sort_values(column, ascending=False, kind="stable").reset_index(drop=True)
    values = ranked[column].tolist()
    minimum, maximum = min(values), max(values)
    spread = maximum - minimum if maximum > minimum else max(maximum * 0.1, 0.01)
    axis.set_xlim(max(0.0, minimum - spread * 0.06), maximum + spread * 0.18)

    positions = list(range(len(ranked)))
    for position, row in enumerate(ranked.itertuples(index=False)):
        color = MODEL_COLORS.get(row.model, "#8B959E")
        is_phaseflow = row.model == "PhaseFlow"
        axis.barh(position, getattr(row, column), color=color, alpha=1.0 if is_phaseflow else 0.70, edgecolor=color if is_phaseflow else "none", linewidth=1.0, height=0.56)
        axis.text(getattr(row, column) + 0.003, position, f"{getattr(row, column):.3f}", va="center", fontsize=7)

    axis.set_yticks(positions, ranked["model"])
    axis.invert_yaxis()
    if baseline is not None and axis.get_xlim()[0] < baseline < axis.get_xlim()[1]:
        axis.axvline(baseline, color="#8B959E", linewidth=0.8, linestyle="--", alpha=0.7)
    axis.set_xlabel(label)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.4, linestyle="--", linewidth=0.5)
    axis.set_axisbelow(True)
    return ranked.loc[:, ["model", column]].rename(columns={"model": f"{column[5:]}_model", column: f"{column[5:]}_value"})


def main() -> int:
    args = parse_args()
    metrics = load_metrics(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 1.95), constrained_layout=True)
    auroc = plot_metric(axes[0], metrics, column="mean_auroc", label="AUROC", baseline=0.5)
    auprc = plot_metric(axes[1], metrics, column="mean_auprc", label="AUPRC", baseline=None)
    pd.concat([auroc, auprc], axis=1).to_csv(args.output_dir / "mutation_metrics_plot_data.csv", index=False)

    for suffix, facecolor, transparent in (("svg", "none", True), ("pdf", "white", False), ("png", "white", False)):
        figure.savefig(args.output_dir / f"mutation_metrics.{suffix}", dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.3, facecolor=facecolor, transparent=transparent)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
