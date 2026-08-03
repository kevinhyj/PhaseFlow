"""Render the final three-panel DPR component-ablation figure."""


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROWS = (("s1111_full_retrain", "PhaseFlow", "full"), ("s1011_no_biophys", "w/o BioPhys", "ablation"), ("s0111_no_esm2", "w/o ESM2", "ablation"), ("s1110_no_phaseflow_bridge", "w/o peptide module", "peptide"))
COLORS = {"full": "#F585BF", "ablation": "#728AB9", "peptide": "#C9857B"}
METRICS = (("auprc", "AUPRC", "phasepro_official_all_scales_global_residue_AUPRC_scale-p257", ("phasepro_official_all_scales_global_residue_AUPRC_scale-p33", "phasepro_official_all_scales_global_residue_AUPRC_scale-p129", "phasepro_official_all_scales_global_residue_AUPRC_scale-p257", "phasepro_official_all_scales_global_residue_AUPRC_scale-mean"), (0.655, 0.718), (0.66, 0.68, 0.70, 0.72)), ("iou025_segment_f1", "IoU@0.25 F1", "phasepro_region_p257_segment_f1_iou_0_25", (), (0.53, 0.61), (0.54, 0.57, 0.60)), ("protein_spearman", "Per-protein Spearman", "phasepro_official_all_scales_per_protein_Spearman_median_scale-p257", ("phasepro_official_all_scales_per_protein_Spearman_median_scale-p33", "phasepro_official_all_scales_per_protein_Spearman_median_scale-p129", "phasepro_official_all_scales_per_protein_Spearman_median_scale-p257", "phasepro_official_all_scales_per_protein_Spearman_median_scale-mean"), (0.40, 0.55), (0.42, 0.46, 0.50, 0.54)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Wide DPR metric summary CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path, help="Optional TrueType/OpenType font file.")
    return parser.parse_args()


def configure_style(font: Path | None) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    if font is not None:
        from matplotlib.font_manager import FontProperties, fontManager

        if not font.is_file():
            raise FileNotFoundError(f"font does not exist: {font}")
        fontManager.addfont(str(font))
        plt.rcParams["font.sans-serif"] = [FontProperties(fname=str(font)).get_name()]


def build_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"metrics input does not exist: {path}")
    source = pd.read_csv(path, dtype={"bitmask": str})
    if "arm_id" not in source:
        raise ValueError("metrics input is missing arm_id")
    records: list[dict[str, object]] = []
    for arm_id, label, group in ROWS:
        candidates = source.loc[source["arm_id"].eq(arm_id)].copy()
        if candidates.empty:
            raise ValueError(f"metrics input is missing DPR arm: {arm_id}")
        required = [column for _, _, value, scales, _, _ in METRICS for column in (value, *scales)]
        missing = [column for column in required if column not in candidates]
        if missing:
            raise ValueError(f"metrics input is missing columns: {sorted(set(missing))}")
        item = candidates.iloc[candidates[required].notna().sum(axis=1).argmax()]
        record: dict[str, object] = {"arm_id": arm_id, "label": label, "group": group}
        for key, _, value_column, scale_columns, _, _ in METRICS:
            value = float(item[value_column])
            scales = pd.to_numeric(item[list(scale_columns)], errors="raise").to_numpy(float) if scale_columns else np.array([value])
            record.update({f"{key}_value": value, f"{key}_low": float(scales.min()), f"{key}_high": float(scales.max())})
        records.append(record)
    return pd.DataFrame(records)


def render(frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 5.2), sharey=True)
    figure.patch.set_alpha(0)
    figure.subplots_adjust(wspace=0.16, left=0.255, right=0.99, top=0.97, bottom=0.17)
    positions = np.arange(len(frame))
    for index, (axis, (key, label, _, _, base, ticks)) in enumerate(zip(axes, METRICS, strict=True)):
        values, lows, highs = (frame[f"{key}_{field}"].to_numpy(float) for field in ("value", "low", "high"))
        span = max(float(highs.max() - lows.min()), 1e-6)
        lower, upper = min(base[0], float(lows.min() - max(0.01, span * 0.08))), max(base[1], float(highs.max() + max(0.01, span * 0.08)))
        axis.axvline(values[0], color="#8A8A8A", alpha=0.55, linewidth=1.3, linestyle=(0, (3.0, 2.0)), zorder=1)
        for y, value, low, high, group in zip(positions, values, lows, highs, frame["group"], strict=True):
            kwargs = {"fmt": "o", "color": COLORS[str(group)], "ecolor": COLORS[str(group)], "markeredgecolor": "white", "markeredgewidth": 1.2, "markersize": 14 if group == "full" else 9.5, "elinewidth": 3 if group == "full" else 2, "capsize": 6 if group == "full" else 5, "capthick": 2.5, "zorder": 4}
            if np.isclose(low, high):
                axis.plot(value, y, "o", color=COLORS[str(group)], markeredgecolor="white", markeredgewidth=1.2, markersize=kwargs["markersize"], zorder=4)
            else:
                axis.errorbar(value, y, xerr=[[value - low], [high - value]], **kwargs)
        axis.set_xlim(lower, upper)
        axis.set_ylim(-0.5, len(frame) - 0.5)
        axis.set_yticks(positions)
        axis.invert_yaxis()
        axis.set_xticks(ticks)
        axis.grid(axis="x", color="#AAAAAA", linewidth=1.5, linestyle=(0, (3.0, 2.0)), alpha=0.7)
        axis.set_axisbelow(True)
        axis.set_xlabel(label, fontsize=16.1, labelpad=8)
        for spine in axis.spines.values():
            spine.set_color("#111111")
            spine.set_linewidth(2.0)
        if index == 0:
            axis.set_yticklabels(frame["label"], fontsize=14.95)
        else:
            axis.tick_params(axis="y", length=0, labelleft=False)
    return figure


def main() -> int:
    args = parse_args()
    configure_style(args.font)
    frame = build_frame(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "dpr_ablation_summary_plot_data.csv", index=False)
    figure = render(frame)
    for suffix, kwargs in (("png", {"dpi": 600, "facecolor": "white"}), ("pdf", {"dpi": 600, "facecolor": "white"}), ("svg", {"dpi": 600, "facecolor": "none", "transparent": True})):
        figure.savefig(args.output_dir / f"dpr_ablation_summary.{suffix}", bbox_inches="tight", pad_inches=0.4, **kwargs)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
