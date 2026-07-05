#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.full_length.evaluation.analyze_dpr_v6_threshold_curves import (  # noqa: E402
    build_truths,
    per_protein_metrics,
    threshold_free_metrics,
    threshold_metrics,
    to_jsonable,
)


@dataclass(frozen=True)
class ProfileBundle:
    name: str
    profiles: dict[str, np.ndarray]
    threshold: float
    threshold_source: str
    threshold_free: dict[str, Any]
    selected_threshold_metrics: dict[str, Any]
    fixed_05_metrics: dict[str, Any]
    per_protein: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final PhasePro comparison using Plan D validation-selected thresholds.")
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--dpr-profile", type=Path, required=True)
    parser.add_argument(
        "--pstp-profile",
        type=Path,
        default=ROOT / "external_artifacts/pstp_official_benchmark_v1/profiles/pstp_nophasepro/selected_family_p33_profiles.npz",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "artifacts/data/processed/evaluation/phasepro_official_v1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--best-n", type=int, default=12)
    parser.add_argument("--worst-n", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "curves_best_12").mkdir(parents=True, exist_ok=True)
    (out / "curves_worst_12").mkdir(parents=True, exist_ok=True)

    validation = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    best_dpr = validation["primary_selection"]["best_dpr"]
    pstp_val = validation["primary_selection"].get("pstp_on_same_validation_subset", {})
    dpr_threshold = float(best_dpr["val_threshold"])
    pstp_threshold = float(pstp_val.get("val_threshold", 0.5))

    proteins = pd.read_parquet(args.data_root / "proteins.parquet")
    regions = pd.read_parquet(args.data_root / "regions.parquet")
    truths = build_truths(proteins, regions)
    dpr_profiles = load_profiles(args.dpr_profile, proteins)
    pstp_profiles = load_profiles(args.pstp_profile, proteins)

    dpr = analyze_profiles(
        name="DPR_v6_plan_d_selected",
        profiles=dpr_profiles,
        truths=truths,
        threshold=dpr_threshold,
        threshold_source="Plan D non-PhasePro validation selected MCC threshold",
        out_dir=out / "dpr_v6",
    )
    pstp = analyze_profiles(
        name="PSTP_nophasepro",
        profiles=pstp_profiles,
        truths=truths,
        threshold=pstp_threshold,
        threshold_source="Plan D non-PhasePro validation selected MCC threshold",
        out_dir=out / "pstp_nophasepro",
    )
    comparison = comparison_table([dpr, pstp])
    comparison.to_csv(out / "phasepro_external_threshold_comparison.csv", index=False)
    threshold_free = pd.DataFrame([{"model": dpr.name, **dpr.threshold_free}, {"model": pstp.name, **pstp.threshold_free}])
    threshold_free.to_csv(out / "phasepro_threshold_free_comparison.csv", index=False)

    selected = select_best_worst(dpr.per_protein, pstp.per_protein, best_n=int(args.best_n), worst_n=int(args.worst_n))
    selected.to_csv(out / "selected_best_worst_by_dpr_p33_spearman.csv", index=False)
    best_ids = selected.loc[selected["group"].eq("best_12"), "protein_id"].tolist()
    worst_ids = selected.loc[selected["group"].eq("worst_12"), "protein_id"].tolist()
    make_group_outputs("best_12", best_ids, dpr, pstp, truths, out / "curves_best_12")
    make_group_outputs("worst_12", worst_ids, dpr, pstp, truths, out / "curves_worst_12")

    payload = {
        "status": "PASS",
        "protocol": {
            "selection_split": "Plan D mixed HQ non-PhasePro validation",
            "final_split": "official PhasePro",
            "phasepro_used_for_threshold_or_checkpoint_selection": False,
            "dpr_threshold": dpr_threshold,
            "pstp_threshold": pstp_threshold,
        },
        "selected_dpr_from_validation": best_dpr,
        "pstp_validation_threshold_row": pstp_val,
        "dpr_profile": str(args.dpr_profile.resolve()),
        "pstp_profile": str(args.pstp_profile.resolve()),
        "comparison": comparison.to_dict(orient="records"),
        "files": {
            "comparison_csv": str((out / "phasepro_external_threshold_comparison.csv").resolve()),
            "threshold_free_csv": str((out / "phasepro_threshold_free_comparison.csv").resolve()),
            "selection_csv": str((out / "selected_best_worst_by_dpr_p33_spearman.csv").resolve()),
            "best_12_png": str((out / "curves_best_12" / "best_12_merged_curves.png").resolve()),
            "best_12_pdf": str((out / "curves_best_12" / "best_12_merged_curves.pdf").resolve()),
            "worst_12_png": str((out / "curves_worst_12" / "worst_12_merged_curves.png").resolve()),
            "worst_12_pdf": str((out / "curves_worst_12" / "worst_12_merged_curves.pdf").resolve()),
        },
    }
    (out / "phasepro_external_threshold_summary.json").write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "phasepro_external_threshold_report.md").write_text(render_report(payload, comparison, threshold_free, selected), encoding="utf-8")
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), flush=True)
    return 0


def load_profiles(path: Path, proteins: pd.DataFrame) -> dict[str, np.ndarray]:
    expected_lengths = dict(zip(proteins["protein_id"].astype(str), proteins["sequence_length"].astype(int), strict=False))
    z = np.load(path, allow_pickle=False)
    profiles: dict[str, np.ndarray] = {}
    missing = sorted(set(expected_lengths) - set(z.files))
    extra = sorted(set(z.files) - set(expected_lengths))
    if missing or extra:
        raise RuntimeError(f"Profile key mismatch for {path}: missing={missing[:10]} extra={extra[:10]}")
    for pid in sorted(expected_lengths):
        arr = np.asarray(z[pid], dtype=np.float32).reshape(-1)
        if len(arr) != expected_lengths[pid]:
            raise RuntimeError(f"Profile length mismatch for {pid}: got {len(arr)} expected {expected_lengths[pid]}")
        profiles[pid] = np.clip(arr, 0.0, 1.0)
    return profiles


def analyze_profiles(
    *,
    name: str,
    profiles: dict[str, np.ndarray],
    truths: dict[str, dict[str, Any]],
    threshold: float,
    threshold_source: str,
    out_dir: Path,
) -> ProfileBundle:
    out_dir.mkdir(parents=True, exist_ok=True)
    per = per_protein_metrics(profiles, truths)
    per.to_csv(out_dir / "per_protein_p33.csv", index=False)
    tf = threshold_free_metrics(profiles, truths, per)
    selected = threshold_metrics(profiles, truths, threshold=float(threshold))
    selected["threshold"] = float(threshold)
    selected["threshold_source"] = threshold_source
    fixed = threshold_metrics(profiles, truths, threshold=0.5)
    fixed["threshold"] = 0.5
    payload = {
        "model": name,
        "threshold_free": tf,
        "external_validation_threshold": selected,
        "fixed_0.5": fixed,
    }
    (out_dir / "metrics_p33.json").write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ProfileBundle(
        name=name,
        profiles=profiles,
        threshold=float(threshold),
        threshold_source=threshold_source,
        threshold_free=tf,
        selected_threshold_metrics=selected,
        fixed_05_metrics=fixed,
        per_protein=per,
    )


def comparison_table(models: list[ProfileBundle]) -> pd.DataFrame:
    rows = []
    for model in models:
        for policy, metrics in [
            ("external_validation_threshold", model.selected_threshold_metrics),
            ("fixed_0.5", model.fixed_05_metrics),
        ]:
            row = {
                "model": model.name,
                "threshold_policy": policy,
                "threshold": float(metrics["threshold"]),
                "AUROC": model.threshold_free["global_residue_AUROC"],
                "AUPRC": model.threshold_free["global_residue_AUPRC"],
                "Spearman": model.threshold_free["global_residue_Spearman"],
                "median_pp_Spearman": model.threshold_free["per_protein_Spearman_median"],
                "pairwise": model.threshold_free["same_protein_pairwise"],
            }
            for key in [
                "precision",
                "recall",
                "F1",
                "MCC",
                "IoU",
                "empty_rate",
                "near_full_rate",
                "mean_predicted_fraction",
                "region_overlap",
                "official_region_total",
                "fully_covered_regions",
                "partially_covered_regions",
                "missed_regions",
                "recovered_positive_fraction",
            ]:
                row[key] = metrics.get(key, math.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def select_best_worst(dpr_per: pd.DataFrame, pstp_per: pd.DataFrame, *, best_n: int, worst_n: int) -> pd.DataFrame:
    pstp_small = pstp_per[["protein_id", "spearman", "auroc", "auprc", "pos_minus_neg_mean"]].rename(
        columns={
            "spearman": "pstp_spearman",
            "auroc": "pstp_auroc",
            "auprc": "pstp_auprc",
            "pos_minus_neg_mean": "pstp_pos_minus_neg_mean",
        }
    )
    merged = dpr_per.merge(pstp_small, on="protein_id", how="left").rename(
        columns={
            "spearman": "dpr_spearman",
            "auroc": "dpr_auroc",
            "auprc": "dpr_auprc",
            "pos_minus_neg_mean": "dpr_pos_minus_neg_mean",
        }
    )
    valid = merged.loc[merged["dpr_spearman"].notna()].copy()
    best = valid.sort_values(["dpr_spearman", "dpr_auroc"], ascending=[False, False]).head(best_n).copy()
    worst = valid.sort_values(["dpr_spearman", "dpr_auroc"], ascending=[True, True]).head(worst_n).copy()
    best.insert(0, "group", "best_12")
    worst.insert(0, "group", "worst_12")
    return pd.concat([best, worst], ignore_index=True)


def make_group_outputs(group: str, protein_ids: list[str], dpr: ProfileBundle, pstp: ProfileBundle, truths: dict[str, dict[str, Any]], out_dir: Path) -> None:
    rows = []
    for pid in protein_ids:
        drow = dpr.per_protein.loc[dpr.per_protein["protein_id"].eq(pid)].iloc[0].to_dict()
        prow = pstp.per_protein.loc[pstp.per_protein["protein_id"].eq(pid)].iloc[0].to_dict()
        rows.append(
            {
                "protein_id": pid,
                "length": int(drow["length"]),
                "positive_count": int(drow["positive_count"]),
                "regions": drow["regions"],
                "dpr_spearman": drow["spearman"],
                "pstp_spearman": prow["spearman"],
                "dpr_auroc": drow["auroc"],
                "pstp_auroc": prow["auroc"],
            }
        )
        plot_one(pid, dpr, pstp, truths, out_dir / f"{pid}_curves.png")
    pd.DataFrame(rows).to_csv(out_dir / f"{group}_protein_metrics.csv", index=False)
    plot_grid(group, protein_ids, dpr, pstp, truths, out_dir / f"{group}_merged_curves.png", out_dir / f"{group}_merged_curves.pdf")


def plot_one(pid: str, dpr: ProfileBundle, pstp: ProfileBundle, truths: dict[str, dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    draw_profile_axes(ax, pid, dpr, pstp, truths)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grid(group: str, protein_ids: list[str], dpr: ProfileBundle, pstp: ProfileBundle, truths: dict[str, dict[str, Any]], png: Path, pdf: Path) -> None:
    cols = 3
    rows = int(math.ceil(len(protein_ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.4 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, pid in zip(axes_arr, protein_ids, strict=False):
        draw_profile_axes(ax, pid, dpr, pstp, truths)
    for ax in axes_arr[len(protein_ids) :]:
        ax.axis("off")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=10)
    fig.suptitle(f"{group} PhasePro curves with Plan D validation thresholds", y=1.01, fontsize=14)
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def draw_profile_axes(ax: Any, pid: str, dpr: ProfileBundle, pstp: ProfileBundle, truths: dict[str, dict[str, Any]]) -> None:
    dpr_score = np.asarray(dpr.profiles[pid], dtype=float)
    pstp_score = np.asarray(pstp.profiles[pid], dtype=float)
    x = np.arange(1, len(dpr_score) + 1)
    used_truth_label = False
    for region in truths[pid]["regions"]:
        ax.axvspan(
            int(region["start"]) + 1,
            int(region["end"]),
            color="#d7d7d7",
            alpha=0.55,
            linewidth=0,
            label="truth" if not used_truth_label else None,
        )
        used_truth_label = True
    ax.plot(x, dpr_score, color="#1f77b4", linewidth=1.25, label="DPR v6 p33")
    ax.plot(x, pstp_score, color="#ff7f0e", linewidth=1.05, alpha=0.9, label="PSTP p33")
    ax.axhline(float(dpr.threshold), color="#1f77b4", linestyle="--", linewidth=0.8, alpha=0.8, label="DPR val threshold")
    ax.axhline(float(pstp.threshold), color="#ff7f0e", linestyle="--", linewidth=0.8, alpha=0.8, label="PSTP val threshold")
    dpr_rho = dpr.per_protein.loc[dpr.per_protein["protein_id"].eq(pid), "spearman"].iloc[0]
    pstp_rho = pstp.per_protein.loc[pstp.per_protein["protein_id"].eq(pid), "spearman"].iloc[0]
    ax.set_title(f"{pid}  DPR rho={fmt_nan(dpr_rho)}  PSTP rho={fmt_nan(pstp_rho)}", fontsize=9)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(1, max(1, len(dpr_score)))
    ax.set_xlabel("residue", fontsize=8)
    ax.set_ylabel("score", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, axis="y", color="#eeeeee", linewidth=0.6)


def render_report(payload: dict[str, Any], comparison: pd.DataFrame, threshold_free: pd.DataFrame, selected: pd.DataFrame) -> str:
    lines = [
        "# Plan D Threshold PhasePro Final",
        "",
        "Protocol: checkpoint and thresholds were selected on non-PhasePro Plan D validation; PhasePro was used only for final evaluation.",
        "",
        "## Comparison",
        "",
        markdown_table(
            comparison,
            [
                "model",
                "threshold_policy",
                "threshold",
                "AUROC",
                "AUPRC",
                "Spearman",
                "median_pp_Spearman",
                "pairwise",
                "precision",
                "recall",
                "F1",
                "MCC",
                "IoU",
                "mean_predicted_fraction",
                "region_overlap",
            ],
        ),
        "",
        "## Threshold-Free",
        "",
        markdown_table(
            threshold_free,
            [
                "model",
                "global_residue_AUROC",
                "global_residue_AUPRC",
                "global_residue_Spearman",
                "per_protein_Spearman_median",
                "same_protein_pairwise",
            ],
        ),
        "",
        "## Selected Proteins",
        "",
        markdown_table(
            selected,
            ["group", "protein_id", "length", "regions", "dpr_spearman", "pstp_spearman", "dpr_auroc", "pstp_auroc"],
        ),
        "",
        "## Files",
        "",
    ]
    for key, path in payload["files"].items():
        lines.append(f"- {key}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    sub = df.loc[:, [col for col in columns if col in df.columns]].copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    rows = ["| " + " | ".join(sub.columns) + " |", "| " + " | ".join(["---"] * len(sub.columns)) + " |"]
    for row in sub.itertuples(index=False):
        rows.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(rows)


def fmt_nan(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(x) else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
