"""Plot a multi-mutation dose response from mutation-effect scores."""


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"mutation", "mutation_count", "official_score"}
COUNT_COLORS = {
    0: "#73777C",
    1: "#728AB9",
    2: "#C9857B",
    3: "#B85A61",
}
TEXT_COLOR = "#2A2A2A"
GRID_COLOR = "#D6D6D6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with mutation, mutation_count, and official_score columns.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for publication assets and plot data.")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(scores.columns))
    if missing:
        raise ValueError(f"mutation score table is missing columns: {missing}")
    scores = scores.loc[:, ["mutation", "mutation_count", "official_score"]].copy()
    scores["mutation"] = scores["mutation"].astype(str)
    scores["mutation_count"] = pd.to_numeric(scores["mutation_count"], errors="raise").astype(int)
    scores["official_score"] = pd.to_numeric(scores["official_score"], errors="raise")
    if scores.empty:
        raise ValueError("mutation score table is empty")
    if not scores["mutation"].eq("WT").any():
        raise ValueError("mutation score table must contain a WT row")
    return scores


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("mutation_count", as_index=False)
        .agg(
            n=("official_score", "size"),
            mean_score=("official_score", "mean"),
            sd_score=("official_score", lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
        )
        .sort_values("mutation_count")
    )


def render(scores: pd.DataFrame, summary: pd.DataFrame, *, dpi: int) -> plt.Figure:
    figure = plt.figure(figsize=(7.2, 5.10), dpi=dpi)
    axis = figure.add_axes([0.13, 0.16, 0.74, 0.78])
    axis.set_facecolor("none")

    trend_color = "#AAB7CC"
    count_values = summary["mutation_count"].to_numpy(float)
    mean_values = summary["mean_score"].to_numpy(float)
    axis.plot(count_values, mean_values, color=trend_color, linewidth=2.4, zorder=2, solid_capstyle="round")

    for row in summary.itertuples(index=False):
        if int(row.n) > 1:
            axis.errorbar(
                float(row.mutation_count),
                float(row.mean_score),
                yerr=float(row.sd_score),
                color=trend_color,
                linewidth=2.2,
                linestyle="none",
                marker="o",
                markersize=6.6,
                markerfacecolor=trend_color,
                markeredgecolor="white",
                capsize=4.0,
                capthick=1.6,
                elinewidth=1.8,
                zorder=5,
            )

    offsets = {
        "WT": 0.0,
        "W334G": -0.075,
        "W385G": 0.0,
        "W412G": 0.075,
        "W334G/W385G": -0.075,
        "W334G/W412G": 0.0,
        "W385G/W412G": 0.075,
        "W334G/W385G/W412G": 0.0,
    }
    point_positions: dict[str, tuple[float, float]] = {}
    for row in scores.sort_values(["mutation_count", "official_score"]).itertuples(index=False):
        x_value = float(row.mutation_count) + offsets.get(str(row.mutation), 0.0)
        y_value = float(row.official_score)
        point_positions[str(row.mutation)] = (x_value, y_value)
        axis.scatter(
            x_value,
            y_value,
            s=58,
            color=COUNT_COLORS.get(int(row.mutation_count), "#73777C"),
            edgecolor="white",
            linewidth=0.75,
            zorder=6,
        )

    wild_type_score = float(scores.loc[scores["mutation"].eq("WT"), "official_score"].iloc[0])
    axis.axhline(wild_type_score, color="#888888", linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)

    annotations = {
        "WT": (0.16, 0.0015, "left"),
        "W334G": (0.00, 0.00105, "center"),
        "W385G": (-0.08, 0.0000, "right"),
        "W412G": (0.07, 0.0001, "left"),
        "W334G/W385G": (-0.16, 0.0000, "right"),
        "W334G/W412G": (0.10, 0.0005, "left"),
        "W385G/W412G": (0.08, 0.0004, "left"),
        "W334G/W385G/W412G": (0.00, -0.00125, "center"),
    }
    for label, (dx, dy, alignment) in annotations.items():
        if label not in point_positions:
            continue
        x_value, y_value = point_positions[label]
        axis.text(x_value + dx, y_value + dy, label, ha=alignment, va="center", fontsize=9.8, color=TEXT_COLOR, zorder=7)

    axis.set_xlim(-0.15, max(4.08, float(scores["mutation_count"].max()) + 1.08))
    axis.set_ylim(float(scores["official_score"].min()) - 0.0045, wild_type_score + 0.0045)
    axis.set_xticks(sorted(scores["mutation_count"].unique()))
    axis.set_xticklabels(["0 (WT)" if value == 0 else str(value) for value in sorted(scores["mutation_count"].unique())], fontsize=12.5, color=TEXT_COLOR)
    axis.set_xlabel("TDP-43 LCD Trp-to-Gly substitution count", fontsize=12.8, color=TEXT_COLOR, labelpad=9)
    axis.set_ylabel("PhaseFlow score", fontsize=13.5, color=TEXT_COLOR, labelpad=11)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.3f}"))
    axis.tick_params(axis="y", labelsize=12.0)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.9)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.2)
    axis.tick_params(axis="both", colors="#111111", width=1.2, length=6, pad=7)
    return figure


def main() -> int:
    args = parse_args()
    scores = load_scores(args.input)
    summary = summarize_scores(scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "multi_mutation_dose_plot_data.csv", index=False)

    figure = render(scores, summary, dpi=int(args.dpi))
    for suffix, facecolor, transparent in (("svg", "none", True), ("pdf", "white", False), ("png", "white", False)):
        figure.savefig(args.output_dir / f"multi_mutation_dose.{suffix}", dpi=int(args.dpi), facecolor=facecolor, transparent=transparent)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
