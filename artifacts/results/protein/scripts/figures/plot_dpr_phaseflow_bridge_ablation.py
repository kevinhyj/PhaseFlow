"""Render the final DPR PhaseFlow-bridge ablation panel."""


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARMS = (("s1111_full_retrain", "Full model", "control"), ("s1110_no_phaseflow_bridge", "w/o PhaseFlow", "module"), ("s1001_esm2_phaseflow_bridge", "ESM2 + PhaseFlow", "baseline"), ("s0101_biophys_phaseflow_bridge", "BioPhys + PhaseFlow", "baseline"), ("s0000_null_negative_control", "Null control", "baseline"))
COLORS = {"control": "#F585BF", "module": "#728AB9", "embedding": "#50B9AE", "baseline": "#F5DDB5"}
SCALE_COLUMNS = ("phasepro_p33_global_residue_AUPRC", "phasepro_p129_global_residue_AUPRC", "phasepro_p257_global_residue_AUPRC", "phasepro_mean_global_residue_AUPRC")
RESULTS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = RESULTS_ROOT / "ablation/dpr/dpr_stream_ablation_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="DPR stream-ablation summary CSV (default: released DPR ablation table).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    return parser.parse_args()


def transform(values: float | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values <= 0.62, values - 0.50, 0.12 + (values - 0.62) * 4.0)


def load_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"metrics input does not exist: {path}")
    source = pd.read_csv(path, dtype={"bitmask": str})
    required = {"arm_id", *SCALE_COLUMNS}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"metrics input is missing columns: {missing}")
    source = source.set_index("arm_id", drop=False)
    records = []
    for arm_id, label, role in ARMS:
        if arm_id not in source.index:
            raise ValueError(f"metrics input is missing DPR arm: {arm_id}")
        row = source.loc[arm_id]
        values = pd.to_numeric(row[list(SCALE_COLUMNS)], errors="raise").to_numpy(float)
        records.append({"arm_id": arm_id, "label": label, "role": role, "color": COLORS[role], "p257": float(row["phasepro_p257_global_residue_AUPRC"]), "scale_min": float(values.min()), "scale_max": float(values.max())})
    return pd.DataFrame(records).sort_values("p257", ascending=False, kind="mergesort").reset_index(drop=True)


def render(frame: pd.DataFrame, font: Path | None) -> plt.Figure:
    if font is not None:
        from matplotlib.font_manager import FontProperties, fontManager

        if not font.is_file():
            raise FileNotFoundError(f"font does not exist: {font}")
        fontManager.addfont(str(font))
        plt.rcParams["font.sans-serif"] = [FontProperties(fname=str(font)).get_name()]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.linewidth": 1.8, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axis = plt.subplots(figsize=(5.8, 3.5))
    figure.subplots_adjust(left=0.33, right=0.97, top=0.93, bottom=0.15)
    positions = list(range(len(frame)))[::-1]
    full = float(frame.loc[frame["role"].eq("control"), "p257"].iloc[0])
    axis.axvline(transform(full), color=COLORS["control"], lw=1.2, alpha=0.38, zorder=1)
    for y, row in zip(positions, frame.itertuples(index=False), strict=True):
        center, low, high = transform([row.p257, row.scale_min, row.scale_max])
        axis.errorbar(center, y, xerr=[[center - low], [high - center]], fmt="o", markersize=10, markerfacecolor=row.color, markeredgecolor="white", markeredgewidth=1.1, ecolor=row.color, elinewidth=2.2, capsize=5.0, capthick=1.8, zorder=3)
    ticks = (0.50, 0.62, 0.68, 0.72)
    axis.set_yticks(positions, frame["label"])
    axis.set_xlabel("AUPRC", fontsize=14, color="#111827")
    axis.set_xlim(transform(0.50), transform(0.74))
    axis.set_xticks(transform(np.array(ticks)), [f"{value:.2f}" for value in ticks])
    axis.tick_params(axis="x", labelsize=13, colors="#111827", width=1.5, length=4.0, pad=6)
    axis.tick_params(axis="y", labelsize=12, colors="#111827", width=1.5, length=4.0)
    axis.grid(axis="x", color="#e4e8f0", linewidth=1.3)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(2.2)
        spine.set_color("#111827")
    return figure


def main() -> int:
    args = parse_args()
    frame = load_frame(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "dpr_phaseflow_bridge_ablation_plot_data.csv", index=False)
    figure = render(frame, args.font)
    for suffix, kwargs in (("png", {"dpi": 600, "facecolor": "white"}), ("pdf", {"facecolor": "white"}), ("svg", {"facecolor": "none", "transparent": True})):
        figure.savefig(args.output_dir / f"dpr_phaseflow_bridge_ablation.{suffix}", **kwargs)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
