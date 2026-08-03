"""Render the final three-panel LLPS embedding-ablation figure."""


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARM_ORDER = ("no_starling_reference", "no_disorder_no_starling", "no_physchem_no_starling", "no_protenix_no_starling", "no_esm2_no_starling", "all5_full_control", "no_pseudo_no_starling")
GROUPS = {"no_starling_reference": "phaseflow", "no_disorder_no_starling": "wo", "no_physchem_no_starling": "wo", "no_protenix_no_starling": "wo", "no_esm2_no_starling": "wo", "all5_full_control": "starling", "no_pseudo_no_starling": "pseudo"}
LABELS = {"no_starling_reference": "PhaseFlow", "all5_full_control": "Add STARLING", "no_disorder_no_starling": "w/o disorder", "no_physchem_no_starling": "w/o physchem", "no_pseudo_no_starling": "w/o pseudo", "no_protenix_no_starling": "w/o Protenix", "no_esm2_no_starling": "w/o ESM2"}
COLORS = {"phaseflow": "#F585BF", "wo": "#728AB9", "starling": "#9A9A9A", "pseudo": "#C9857B"}
METRICS = (("auprc", "AUPRC", (0.50, 0.80)), ("auroc", "AUROC", (0.79, 0.90)), ("mcc_at_0.5", "MCC@0.5", (0.00, 0.62)))
RESULTS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOOTSTRAP_INPUT = RESULTS_ROOT / "ablation/llps/llps_embedding_ablation_bootstrap_ci.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-input", type=Path, default=DEFAULT_BOOTSTRAP_INPUT, help="Long-form bootstrap confidence-interval CSV (default: released LLPS ablation table).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"bootstrap input does not exist: {path}")
    source = pd.read_csv(path)
    required = {"arm_id", "metric", "value", "ci_low", "ci_high"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"bootstrap input is missing columns: {missing}")
    pivoted: dict[str, dict[str, object]] = {}
    for row in source.itertuples(index=False):
        arm_id = str(row.arm_id)
        record = pivoted.setdefault(arm_id, {"arm_id": arm_id, "label": getattr(row, "label", arm_id)})
        record[f"{row.metric}_value"] = float(row.value)
        record[f"{row.metric}_ci_low"] = float(row.ci_low)
        record[f"{row.metric}_ci_high"] = float(row.ci_high)
    missing_arms = [arm for arm in ARM_ORDER if arm not in pivoted]
    if missing_arms:
        raise ValueError(f"bootstrap input is missing LLPS arms: {missing_arms}")
    frame = pd.DataFrame([pivoted[arm] for arm in ARM_ORDER])
    required_metrics = [f"{metric}_{field}" for metric, _, _ in METRICS for field in ("value", "ci_low", "ci_high")]
    missing_metrics = [column for column in required_metrics if column not in frame]
    if missing_metrics:
        raise ValueError(f"bootstrap input is missing metrics: {missing_metrics}")
    frame["label"] = [LABELS.get(arm, str(label).replace("\n", " ")) for arm, label in zip(frame["arm_id"], frame["label"], strict=True)]
    frame["group"] = frame["arm_id"].map(GROUPS)
    return frame


def configure_style(font: Path | None) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    if font is not None:
        from matplotlib.font_manager import FontProperties, fontManager

        if not font.is_file():
            raise FileNotFoundError(f"font does not exist: {font}")
        fontManager.addfont(str(font))
        plt.rcParams["font.sans-serif"] = [FontProperties(fname=str(font)).get_name()]


def render(frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 6.8), sharey=True)
    figure.patch.set_alpha(0)
    figure.subplots_adjust(wspace=0.16, left=0.29, right=0.99, top=0.97, bottom=0.17)
    positions = np.arange(len(frame))
    for index, (axis, (metric, label, base)) in enumerate(zip(axes, METRICS, strict=True)):
        values, lows, highs = (frame[f"{metric}_{field}"].to_numpy(float) for field in ("value", "ci_low", "ci_high"))
        span = max(float(highs.max() - lows.min()), 1e-6)
        pad = max(0.01, span * 0.08)
        lower, upper = min(base[0], float(lows.min() - pad)), max(base[1], float(highs.max() + pad))
        reference = float(frame.loc[frame["arm_id"].eq("no_starling_reference"), f"{metric}_value"].iloc[0])
        axis.axvline(reference, color="#8A8A8A", alpha=0.55, linewidth=1.3, linestyle=(0, (3.0, 2.0)), zorder=1)
        for y, value, low, high, group in zip(positions, values, lows, highs, frame["group"], strict=True):
            color = COLORS[str(group)]
            axis.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="o", color=color, ecolor=color, markeredgecolor="white", markeredgewidth=1.2, markersize=11, elinewidth=2.5, capsize=5, capthick=2.5, zorder=4)
        pseudo = np.flatnonzero(frame["group"].eq("pseudo"))
        if len(pseudo) and pseudo[0] > 0:
            axis.axhline(int(pseudo[0]) - 0.5, color="#666666", linewidth=1.1, linestyle=(0, (3.0, 2.0)), alpha=0.75, zorder=2)
        axis.set_xlim(lower, upper)
        axis.set_ylim(-0.5, len(frame) - 0.5)
        axis.set_yticks(positions)
        axis.invert_yaxis()
        axis.set_xticks(np.linspace(lower, upper, 5) if metric not in {"mcc_at_0.5"} else (0.0, 0.2, 0.4, 0.6))
        axis.grid(axis="x", color="#AAAAAA", linewidth=1.5, linestyle=(0, (3.0, 2.0)), alpha=0.7)
        axis.set_axisbelow(True)
        axis.set_xlabel(label, fontsize=14, labelpad=8)
        for spine in axis.spines.values():
            spine.set_color("#111111")
            spine.set_linewidth(2.0)
        if index == 0:
            axis.set_yticklabels(frame["label"], fontsize=13)
        else:
            axis.tick_params(axis="y", length=0, labelleft=False)
    return figure


def main() -> int:
    args = parse_args()
    configure_style(args.font)
    frame = load_frame(args.bootstrap_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "llps_embedding_ablation_plot_data.csv", index=False)
    figure = render(frame)
    for suffix, kwargs in (("png", {"dpi": 600, "facecolor": "white"}), ("pdf", {"dpi": 600, "facecolor": "white"}), ("svg", {"dpi": 600, "facecolor": "none", "transparent": True})):
        figure.savefig(args.output_dir / f"llps_embedding_ablation.{suffix}", bbox_inches="tight", pad_inches=0.24, **kwargs)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
