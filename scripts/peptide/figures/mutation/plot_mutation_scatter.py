"""Plot mutation-effect scatter panels for PhaseFlow and comparators."""


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = (
    ("PhaseFlow", "phaseflow_score", False, "#56669E"),
    ("PSPHunter", "psphunter_effect", True, "#D89BB7"),
    ("PhaseMotif", "phasemotif_effect", True, "#5F8E87"),
    ("PSTP", "pstp_scan_score_mean_effect", False, "#D2A68C"),
    ("DeePhase", "deephase_effect", False, "#79AAA3"),
    ("PSPredictor", "pspredictor_effect", True, "#8FA184"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with experimental_effect and model score columns.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def correlation(values: pd.DataFrame) -> tuple[float, float]:
    pearson = float(values["experimental_effect"].corr(values["predicted_effect"], method="pearson"))
    spearman = float(values["experimental_effect"].corr(values["predicted_effect"], method="spearman"))
    return pearson, spearman


def main() -> int:
    args = parse_args()
    source = pd.read_csv(args.input)
    required = {"experimental_effect", *(column for _, column, _, _ in MODELS)}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"mutation effect table is missing columns: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7.6), constrained_layout=True)
    records: list[dict[str, float | int | str]] = []
    for axis, (name, column, invert, color) in zip(axes.flat, MODELS, strict=True):
        pair = source.loc[:, ["experimental_effect", column]].rename(columns={column: "predicted_effect"}).apply(pd.to_numeric, errors="coerce").dropna()
        pair = pair.loc[np.isfinite(pair["experimental_effect"]) & np.isfinite(pair["predicted_effect"])].copy()
        if invert:
            pair["predicted_effect"] = -pair["predicted_effect"]
        if len(pair) < 2:
            raise ValueError(f"{name} requires at least two finite mutation-effect pairs")

        axis.scatter(pair["experimental_effect"], pair["predicted_effect"], s=15, alpha=0.4, color=color, edgecolors="none")
        slope, intercept = np.polyfit(pair["experimental_effect"], pair["predicted_effect"], 1)
        x_values = np.linspace(float(pair["experimental_effect"].min()), float(pair["experimental_effect"].max()), 100)
        axis.plot(x_values, slope * x_values + intercept, "--", linewidth=1.2, alpha=0.7, color="#8B959E")
        axis.axhline(0, color="#A0A0A0", linewidth=0.8, linestyle="--", alpha=0.5)
        axis.axvline(0, color="#A0A0A0", linewidth=0.8, linestyle="--", alpha=0.5)
        pearson, spearman = correlation(pair)
        axis.set_title(name)
        axis.text(0.02, 0.98, f"rho={spearman:.3f}, r={pearson:.3f}, N={len(pair)}", transform=axis.transAxes, va="top", fontsize=8, bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D6D6D6", "alpha": 0.9})
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.4, linestyle="--", linewidth=0.5)
        axis.set_axisbelow(True)
        records.append({"model": name, "n": len(pair), "pearson_r": pearson, "spearman_rho": spearman})

    figure.supxlabel("Experimental mutation effect")
    figure.supylabel("Predicted effect score")
    pd.DataFrame(records).to_csv(args.output_dir / "mutation_scatter_plot_data.csv", index=False)
    for suffix, facecolor, transparent in (("svg", "none", True), ("pdf", "white", False), ("png", "white", False)):
        figure.savefig(args.output_dir / f"mutation_scatter.{suffix}", dpi=int(args.dpi), facecolor=facecolor, transparent=transparent)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
