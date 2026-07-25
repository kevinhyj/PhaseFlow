#!/usr/bin/env python3
"""Plot DPR residue profiles, optionally with released region annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.full_length.figures.common import configure_style, plt, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True, help="NPZ archive with one score profile per protein identifier.")
    parser.add_argument("--proteins", type=Path, help="Optional CSV or Parquet protein metadata containing protein_id and optionally gene_name.")
    parser.add_argument("--regions", type=Path, help="Optional CSV or Parquet DPR intervals with protein_id, start_0based, and end_exclusive.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-key", action="append", default=[], help="Archive key to include; defaults to the first twelve profiles.")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--font", type=Path)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"metadata table does not exist: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def profile_items(path: Path, requested: list[str]) -> list[tuple[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"profile archive does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        keys = requested or list(archive.files)
        items: list[tuple[str, np.ndarray]] = []
        for key in keys:
            if key not in archive:
                raise ValueError(f"profile key is missing: {key}")
            values = np.asarray(archive[key]).squeeze()
            if values.ndim == 1:
                items.append((key, values.astype(float, copy=False)))
    if not items:
        raise ValueError("the archive does not contain one-dimensional residue profiles")
    return items[:12]


def gold_mask(regions: pd.DataFrame, protein_id: str, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=np.int8)
    if regions.empty:
        return mask
    for row in regions.loc[regions["protein_id"].astype(str).eq(protein_id)].itertuples(index=False):
        start = max(0, int(getattr(row, "start_0based")))
        end = min(length, int(getattr(row, "end_exclusive")))
        if end > start:
            mask[start:end] = 1
    return mask


def main() -> int:
    args = parse_args()
    items = profile_items(args.profiles, args.profile_key)
    proteins = read_table(args.proteins) if args.proteins else pd.DataFrame(columns=["protein_id"])
    regions = read_table(args.regions) if args.regions else pd.DataFrame(columns=["protein_id", "start_0based", "end_exclusive"])
    if not regions.empty and not {"protein_id", "start_0based", "end_exclusive"}.issubset(regions.columns):
        raise ValueError("regions must contain protein_id, start_0based, and end_exclusive")
    display_names = proteins.set_index("protein_id")["gene_name"].to_dict() if "gene_name" in proteins else {}
    columns = max(1, int(args.columns))
    rows = int(np.ceil(len(items) / columns))
    configure_style(args.font)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 2.6 * rows), squeeze=False)
    plot_rows: list[dict[str, object]] = []
    for axis, (protein_id, profile) in zip(axes.flat, items):
        x = np.arange(1, len(profile) + 1)
        gold = gold_mask(regions, protein_id, len(profile))
        for start, end in _spans(gold):
            axis.axvspan(start + 1, end, color="#F5B5BF", alpha=0.25, linewidth=0, zorder=0)
        axis.fill_between(x, -0.07, -0.02, where=profile >= float(args.threshold), color="#F585BF", alpha=0.32, step="mid", linewidth=0)
        axis.axhline(float(args.threshold), color="#6B7280", linewidth=0.8, linestyle=(0, (3, 2)))
        axis.plot(x, profile, color="#56669E", linewidth=1.3)
        title = str(display_names.get(protein_id) or protein_id).split(",")[0].strip()
        axis.set_title(title)
        axis.set_xlabel("Residue")
        axis.set_ylabel("DPR score")
        axis.set_ylim(-0.08, 1.02)
        axis.grid(axis="x", color="#D8D8D8", linewidth=0.7)
        axis.set_axisbelow(True)
        plot_rows.extend({"protein_id": protein_id, "display_name": title, "residue_index_1based": int(index), "dpr_score": float(score), "gold_dpr_label": int(label), "predicted_dpr_at_threshold": int(score >= float(args.threshold))} for index, (score, label) in enumerate(zip(profile, gold, strict=True), start=1))
    for axis in axes.flat[len(items):]:
        axis.set_visible(False)
    save_figure(fig, args.output_dir, "dpr_examples")
    pd.DataFrame(plot_rows).to_csv(args.output_dir / "dpr_examples_plot_data.csv", index=False)
    return 0


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(np.diff(np.r_[0, mask, 0]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[0, mask, 0]) == -1)
    return list(zip(starts, ends, strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
