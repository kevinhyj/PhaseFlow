"""Render PhasePro DPR score profiles for a specified ranked protein panel."""


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True, help="NPZ archive with one PhaseFlow score vector per protein ID.")
    parser.add_argument("--proteins", type=Path, required=True, help="CSV or Parquet protein metadata with protein_id and optional gene_name.")
    parser.add_argument("--regions", type=Path, required=True, help="CSV or Parquet PhasePro intervals with protein_id, start_0based, and end_exclusive.")
    parser.add_argument("--per-protein", type=Path, required=True, help="CSV ranking table containing protein_id, length, positive_count, auprc, auroc, and spearman.")
    parser.add_argument("--protein-id", action="append", default=[], help="Protein ID in panel order; repeat up to twelve times. Defaults to descending AUPRC.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--font", type=Path)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"table does not exist: {path}")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_selected(per_protein: pd.DataFrame, proteins: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    required = {"protein_id", "length", "positive_count", "auprc", "auroc", "spearman"}
    missing = sorted(required - set(per_protein.columns))
    if missing:
        raise ValueError(f"per-protein table is missing columns: {missing}")
    scored = per_protein.copy()
    order = requested or scored.sort_values("auprc", ascending=False, kind="mergesort")["protein_id"].astype(str).tolist()
    order = order[:12]
    if not order:
        raise ValueError("per-protein table contains no selectable proteins")
    available = scored.assign(protein_id=scored["protein_id"].astype(str)).set_index("protein_id")
    missing_ids = [protein_id for protein_id in order if protein_id not in available.index]
    if missing_ids:
        raise ValueError(f"per-protein table is missing requested proteins: {missing_ids}")
    selected = available.loc[order].reset_index()
    metadata = proteins.copy()
    if "protein_id" not in metadata:
        raise ValueError("protein metadata is missing protein_id")
    metadata["protein_id"] = metadata["protein_id"].astype(str)
    columns = [column for column in ("protein_id", "gene_name") if column in metadata]
    selected = selected.merge(metadata[columns], on="protein_id", how="left")
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    selected["display_name"] = selected.get("gene_name", selected["protein_id"]).fillna(selected["protein_id"]).astype(str).str.split(",").str[0].str.strip()
    selected["positive_fraction"] = selected["positive_count"].astype(float) / selected["length"].astype(float)
    return selected


def gold_mask(regions: pd.DataFrame, protein_id: str, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=np.int8)
    for row in regions.loc[regions["protein_id"].astype(str).eq(protein_id)].itertuples(index=False):
        start, end = max(0, int(row.start_0based)), min(length, int(row.end_exclusive))
        if end > start:
            mask[start:end] = 1
    return mask


def spans(mask: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(np.diff(np.r_[0, mask, 0]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[0, mask, 0]) == -1)
    return list(zip(starts, ends, strict=True))


def configure_style(font: Path | None) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    if font is not None:
        from matplotlib.font_manager import FontProperties, fontManager

        if not font.is_file():
            raise FileNotFoundError(f"font does not exist: {font}")
        fontManager.addfont(str(font))
        plt.rcParams["font.sans-serif"] = [FontProperties(fname=str(font)).get_name()]


def main() -> int:
    args = parse_args()
    proteins, regions, per_protein = (read_table(path) for path in (args.proteins, args.regions, args.per_protein))
    required_regions = {"protein_id", "start_0based", "end_exclusive"}
    if missing := sorted(required_regions - set(regions.columns)):
        raise ValueError(f"region table is missing columns: {missing}")
    selected = load_selected(per_protein, proteins, args.protein_id)
    if not args.profiles.is_file():
        raise FileNotFoundError(f"profile archive does not exist: {args.profiles}")
    configure_style(args.font)
    rows, columns = int(np.ceil(len(selected) / 4)), min(4, len(selected))
    figure, axes = plt.subplots(rows, columns, figsize=(19.2, 4.6 * rows), squeeze=False)
    records: list[dict[str, object]] = []
    with np.load(args.profiles, allow_pickle=False) as profiles:
        for axis, item in zip(axes.flat, selected.itertuples(index=False), strict=True):
            if item.protein_id not in profiles:
                raise ValueError(f"profile archive is missing protein ID: {item.protein_id}")
            scores = np.asarray(profiles[item.protein_id], dtype=float).squeeze()
            if scores.ndim != 1:
                raise ValueError(f"profile {item.protein_id} is not one-dimensional")
            gold = gold_mask(regions, item.protein_id, len(scores))
            position = np.arange(1, len(scores) + 1)
            for start, end in spans(gold):
                axis.axvspan(start + 1, end, color="#F5B5BF", alpha=0.24, linewidth=0, zorder=0)
            predicted = scores >= args.threshold
            axis.fill_between(position, -0.07, -0.02, where=predicted, color="#56669E", alpha=0.30, step="mid", linewidth=0)
            axis.axhline(args.threshold, color="#8A8A8A", linewidth=0.6, linestyle=(0, (2.0, 2.0)))
            axis.plot(position, scores, color="#56669E", linewidth=1.0)
            axis.set_title(f"{item.rank}. {item.display_name} (AUPRC={item.auprc:.3f})", loc="left")
            axis.set_xlim(1, len(scores))
            axis.set_ylim(-0.08, 1.02)
            axis.set_xlabel("Residue position")
            axis.set_ylabel("DPR score")
            axis.grid(axis="x", color="#E4E8F0", linewidth=0.5)
            for residue, (score, label) in enumerate(zip(scores, gold, strict=True), start=1):
                records.append({"rank": int(item.rank), "protein_id": item.protein_id, "display_name": item.display_name, "residue_index_1based": residue, "phaseflow_dpr_score": float(score), "phasepro_dpr_label": int(label), "predicted_dpr_at_threshold_0_5": int(score >= args.threshold), "length": int(item.length), "positive_count": int(item.positive_count), "positive_fraction": float(item.positive_fraction), "auprc": float(item.auprc), "auroc": float(item.auroc), "spearman": float(item.spearman)})
    for axis in axes.flat[len(selected):]:
        axis.set_visible(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "phasepro_dpr_top12_selected.csv", index=False)
    pd.DataFrame(records).to_csv(args.output_dir / "phasepro_dpr_top12_plot_data.csv", index=False)
    for suffix, kwargs in (("png", {"dpi": 600, "facecolor": "white"}), ("pdf", {"dpi": 600, "facecolor": "white"}), ("svg", {"dpi": 600, "facecolor": "white"})):
        figure.savefig(args.output_dir / f"phasepro_dpr_top12.{suffix}", bbox_inches="tight", pad_inches=0.3, **kwargs)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
