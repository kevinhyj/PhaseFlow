
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

from scripts.protein.analysis.analyze_dpr_thresholds import (  # noqa: E402
    build_truths,
    concat_labels_scores,
    per_protein_metrics,
    threshold_free_metrics,
    threshold_metrics,
    threshold_vector_stats,
    to_jsonable,
)
from scripts.protein.analysis.dpr_plan import (  # noqa: E402
    build_residue_truths,
    load_candidate_index,
    restrict_to_common_profiles,
)


DEFAULT_DPR_PHASEPRO = (
    ROOT
    / "artifacts/data/protein/dpr/evaluation/phasepro/dpr_profiles.npz"
)
DEFAULT_DPR_VAL = (
    ROOT
    / "artifacts/data/protein/dpr/evaluation/validation/dpr_profiles.npz"
)
DEFAULT_PSTP_PHASEPRO = ROOT / "artifacts/data/protein/dpr/evaluation/phasepro/pstp_profiles.npz"
DEFAULT_PSTP_VAL = ROOT / "artifacts/data/protein/dpr/evaluation/validation/pstp_profiles.npz"
DEFAULT_VALIDATION_SUMMARY = ROOT / "artifacts/data/protein/dpr/evaluation/validation/selection_summary.json"
DEFAULT_PHASEPRO_DATA = ROOT / "artifacts/data/protein/dpr/evaluation/phasepro"
DEFAULT_PLAN_D_VAL = ROOT / "artifacts/data/protein/dpr/evaluation/validation/candidate_index.parquet"
DEFAULT_OUT = ROOT / "runs/dpr_threshold_policy"


@dataclass(frozen=True)
class ModelBundle:
    name: str
    display_label: str
    phasepro_profiles: dict[str, np.ndarray]
    val_profiles: dict[str, np.ndarray]
    phasepro_per: pd.DataFrame
    phasepro_threshold_free: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DPR v6 and PSTP under fair fixed/validation threshold policies.")
    parser.add_argument("--dpr-phasepro-profile", type=Path, default=DEFAULT_DPR_PHASEPRO)
    parser.add_argument("--dpr-val-profile", type=Path, default=DEFAULT_DPR_VAL)
    parser.add_argument("--pstp-phasepro-profile", type=Path, default=DEFAULT_PSTP_PHASEPRO)
    parser.add_argument("--pstp-val-profile", type=Path, default=DEFAULT_PSTP_VAL)
    parser.add_argument("--dpr-display-label", default="DPR v6")
    parser.add_argument("--pstp-display-label", default="PSTP")
    parser.add_argument("--validation-summary", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    parser.add_argument("--phasepro-data-root", type=Path, default=DEFAULT_PHASEPRO_DATA)
    parser.add_argument("--plan-d-val-index", type=Path, default=DEFAULT_PLAN_D_VAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixed-thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--grid-step", type=float, default=0.001)
    parser.add_argument("--best-n", type=int, default=12)
    parser.add_argument("--worst-n", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    fixed_thresholds = parse_thresholds(args.fixed_thresholds)
    if 0.5 not in fixed_thresholds:
        fixed_thresholds.append(0.5)
    fixed_thresholds = sorted(set(float(x) for x in fixed_thresholds))
    grid = make_threshold_grid(float(args.grid_step), fixed_thresholds)

    validation_summary = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    dpr_val_threshold = float(validation_summary["primary_selection"]["best_dpr"]["val_threshold"])
    pstp_val_threshold = float(validation_summary["primary_selection"]["pstp_on_same_validation_subset"]["val_threshold"])

    proteins = pd.read_parquet(args.phasepro_data_root / "proteins.parquet")
    regions = pd.read_parquet(args.phasepro_data_root / "regions.parquet")
    phasepro_truths = build_truths(proteins, regions)
    expected_phasepro_lengths = {pid: len(item["label"]) for pid, item in phasepro_truths.items()}

    plan_d_candidate = load_candidate_index(args.plan_d_val_index)
    plan_d_truths, plan_d_audit = build_residue_truths(plan_d_candidate)

    dpr_phasepro_profiles = load_profiles(args.dpr_phasepro_profile, expected_lengths=expected_phasepro_lengths)
    pstp_phasepro_profiles = load_profiles(args.pstp_phasepro_profile, expected_lengths=expected_phasepro_lengths)
    dpr_val_profiles_raw = load_profiles(args.dpr_val_profile)
    pstp_val_profiles_raw = load_profiles(args.pstp_val_profile)
    dpr_val_profiles, dpr_val_truths, dpr_val_coverage = restrict_to_common_profiles(dpr_val_profiles_raw, plan_d_truths)
    pstp_val_profiles, pstp_val_truths, pstp_val_coverage = restrict_to_common_profiles(pstp_val_profiles_raw, plan_d_truths)
    common_val_ids = sorted(set(dpr_val_profiles) & set(pstp_val_profiles) & set(plan_d_truths))
    dpr_val_profiles = {pid: dpr_val_profiles[pid] for pid in common_val_ids}
    pstp_val_profiles = {pid: pstp_val_profiles[pid] for pid in common_val_ids}
    common_val_truths = {pid: plan_d_truths[pid] for pid in common_val_ids}

    dpr = make_bundle("DPR_v6_plan_d_selected", str(args.dpr_display_label), dpr_phasepro_profiles, dpr_val_profiles, phasepro_truths)
    pstp = make_bundle("PSTP_nophasepro", str(args.pstp_display_label), pstp_phasepro_profiles, pstp_val_profiles, phasepro_truths)

    val_dpr_grid = threshold_curve_on_grid(dpr.val_profiles, common_val_truths, grid)
    val_pstp_grid = threshold_curve_on_grid(pstp.val_profiles, common_val_truths, grid)
    phasepro_dpr_grid = threshold_curve_on_grid(dpr.phasepro_profiles, phasepro_truths, grid)
    phasepro_pstp_grid = threshold_curve_on_grid(pstp.phasepro_profiles, phasepro_truths, grid)

    val_grid_compare = merge_grid_curves(val_dpr_grid, val_pstp_grid)
    phasepro_grid_compare = merge_grid_curves(phasepro_dpr_grid, phasepro_pstp_grid)
    val_grid_compare.to_csv(out / "plan_d_validation_common_fixed_threshold_grid.csv", index=False)
    phasepro_grid_compare.to_csv(out / "phasepro_common_fixed_threshold_grid.csv", index=False)

    common_val_mcc = select_common_validation_threshold(val_grid_compare, objective="MCC")
    common_val_f1 = select_common_validation_threshold(val_grid_compare, objective="F1")

    policies: list[dict[str, Any]] = []
    for threshold in fixed_thresholds:
        policies.append(
            {
                "threshold_policy": f"same_fixed_{format_threshold_for_name(threshold)}",
                "fair_policy_type": "same predefined fixed threshold for both models",
                "dpr_threshold": float(threshold),
                "pstp_threshold": float(threshold),
                "phasepro_used_for_threshold_selection": False,
            }
        )
    policies.extend(
        [
            {
                "threshold_policy": "common_plan_d_validation_mcc",
                "fair_policy_type": "single common threshold selected on non-PhasePro Plan D validation",
                "dpr_threshold": float(common_val_mcc["threshold"]),
                "pstp_threshold": float(common_val_mcc["threshold"]),
                "validation_selection_objective": "mean(DPR_MCC, PSTP_MCC)",
                "validation_selection_row": common_val_mcc,
                "phasepro_used_for_threshold_selection": False,
            },
            {
                "threshold_policy": "common_plan_d_validation_f1",
                "fair_policy_type": "single common threshold selected on non-PhasePro Plan D validation",
                "dpr_threshold": float(common_val_f1["threshold"]),
                "pstp_threshold": float(common_val_f1["threshold"]),
                "validation_selection_objective": "mean(DPR_F1, PSTP_F1)",
                "validation_selection_row": common_val_f1,
                "phasepro_used_for_threshold_selection": False,
            },
            {
                "threshold_policy": "per_model_plan_d_validation_mcc",
                "fair_policy_type": "each model threshold selected on the same non-PhasePro Plan D validation split",
                "dpr_threshold": float(dpr_val_threshold),
                "pstp_threshold": float(pstp_val_threshold),
                "validation_selection_objective": "per-model MCC",
                "phasepro_used_for_threshold_selection": False,
            },
        ]
    )

    comparison = build_policy_comparison(policies, dpr, pstp, phasepro_truths)
    comparison.to_csv(out / "phasepro_fair_threshold_policy_comparison.csv", index=False)

    threshold_free = pd.DataFrame(
        [
            {"model": dpr.name, **dpr.phasepro_threshold_free},
            {"model": pstp.name, **pstp.phasepro_threshold_free},
        ]
    )
    threshold_free.to_csv(out / "phasepro_threshold_free_comparison.csv", index=False)

    selected = select_best_worst(dpr.phasepro_per, pstp.phasepro_per, best_n=int(args.best_n), worst_n=int(args.worst_n))
    selected.to_csv(out / "selected_best_worst_by_dpr_spearman.csv", index=False)

    plot_policy_bars(comparison, out / "phasepro_fair_threshold_policy_metrics.png", out / "phasepro_fair_threshold_policy_metrics.pdf")
    plot_threshold_sensitivity(
        phasepro_grid_compare,
        out / "phasepro_common_fixed_threshold_sensitivity.png",
        out / "phasepro_common_fixed_threshold_sensitivity.pdf",
        marker_thresholds={
            "fixed 0.5": 0.5,
            "common PlanD MCC": float(common_val_mcc["threshold"]),
            "common PlanD F1": float(common_val_f1["threshold"]),
            "DPR PlanD MCC": dpr_val_threshold,
            "PSTP PlanD MCC": pstp_val_threshold,
        },
    )
    plot_validation_threshold_selection(
        val_grid_compare,
        out / "plan_d_validation_common_threshold_selection.png",
        out / "plan_d_validation_common_threshold_selection.pdf",
        marker_thresholds={
            "common PlanD MCC": float(common_val_mcc["threshold"]),
            "common PlanD F1": float(common_val_f1["threshold"]),
            "fixed 0.5": 0.5,
        },
    )

    best_ids = selected.loc[selected["group"].eq("best_12"), "protein_id"].astype(str).tolist()
    worst_ids = selected.loc[selected["group"].eq("worst_12"), "protein_id"].astype(str).tolist()
    curve_policy_specs = [
        ("fixed_0p5", {"DPR": 0.5, "PSTP": 0.5}),
        ("common_plan_d_validation_mcc", {"DPR": float(common_val_mcc["threshold"]), "PSTP": float(common_val_mcc["threshold"])}),
        ("per_model_plan_d_validation_mcc", {"DPR": dpr_val_threshold, "PSTP": pstp_val_threshold}),
    ]
    for policy_name, thresholds in curve_policy_specs:
        make_group_curves(policy_name, "best_12", best_ids, dpr, pstp, phasepro_truths, thresholds, out / f"curves_{policy_name}_best_12")
        make_group_curves(policy_name, "worst_12", worst_ids, dpr, pstp, phasepro_truths, thresholds, out / f"curves_{policy_name}_worst_12")

    summary = {
        "status": "PASS",
        "protocol": {
            "fair_main_policies": [
                "same predefined fixed thresholds for both models",
                "single common threshold selected on non-PhasePro Plan D validation",
                "per-model thresholds selected on the same non-PhasePro Plan D validation split",
            ],
            "phasepro_used_for_checkpoint_or_threshold_selection": False,
            "phasepro_threshold_sweep_note": "The PhasePro sweep plot is diagnostic sensitivity only; it is not used to select a reported fair threshold.",
            "validation_common_subset_proteins": int(len(common_val_ids)),
            "plan_d_validation_truth_audit": plan_d_audit,
            "dpr_val_coverage": dpr_val_coverage,
            "pstp_val_coverage": pstp_val_coverage,
        },
        "selected_thresholds": {
            "dpr_plan_d_validation_mcc": dpr_val_threshold,
            "pstp_plan_d_validation_mcc": pstp_val_threshold,
            "common_plan_d_validation_mcc": common_val_mcc,
            "common_plan_d_validation_f1": common_val_f1,
            "fixed_thresholds": fixed_thresholds,
        },
        "inputs": {
            "dpr_phasepro_profile": str(args.dpr_phasepro_profile.resolve()),
            "dpr_validation_profile": str(args.dpr_val_profile.resolve()),
            "dpr_display_label": str(args.dpr_display_label),
            "pstp_phasepro_profile": str(args.pstp_phasepro_profile.resolve()),
            "pstp_validation_profile": str(args.pstp_val_profile.resolve()),
            "pstp_display_label": str(args.pstp_display_label),
            "validation_summary": str(args.validation_summary.resolve()),
        },
        "files": {
            "report": str((out / "phasepro_fair_threshold_policy_report.md").resolve()),
            "summary_json": str((out / "phasepro_fair_threshold_policy_summary.json").resolve()),
            "policy_comparison_csv": str((out / "phasepro_fair_threshold_policy_comparison.csv").resolve()),
            "selection_csv": str((out / "selected_best_worst_by_dpr_spearman.csv").resolve()),
            "phasepro_grid_csv": str((out / "phasepro_common_fixed_threshold_grid.csv").resolve()),
            "validation_grid_csv": str((out / "plan_d_validation_common_fixed_threshold_grid.csv").resolve()),
            "policy_bar_png": str((out / "phasepro_fair_threshold_policy_metrics.png").resolve()),
            "phasepro_threshold_sensitivity_png": str((out / "phasepro_common_fixed_threshold_sensitivity.png").resolve()),
            "validation_threshold_selection_png": str((out / "plan_d_validation_common_threshold_selection.png").resolve()),
            "fixed_0p5_best_12_png": str((out / "curves_fixed_0p5_best_12" / "best_12_merged_curves.png").resolve()),
            "fixed_0p5_worst_12_png": str((out / "curves_fixed_0p5_worst_12" / "worst_12_merged_curves.png").resolve()),
            "common_val_mcc_best_12_png": str((out / "curves_common_plan_d_validation_mcc_best_12" / "best_12_merged_curves.png").resolve()),
            "common_val_mcc_worst_12_png": str((out / "curves_common_plan_d_validation_mcc_worst_12" / "worst_12_merged_curves.png").resolve()),
        },
        "comparison": comparison.to_dict(orient="records"),
        "threshold_free": threshold_free.to_dict(orient="records"),
    }
    (out / "phasepro_fair_threshold_policy_summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "phasepro_fair_threshold_policy_report.md").write_text(
        render_report(summary, comparison, threshold_free),
        encoding="utf-8",
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return 0


def parse_thresholds(text: str) -> list[float]:
    values: list[float] = []
    for part in str(text).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"threshold out of [0, 1]: {value}")
        values.append(value)
    return values


def make_threshold_grid(step: float, fixed_thresholds: list[float]) -> np.ndarray:
    if step <= 0.0 or step > 1.0:
        raise ValueError(f"invalid grid step: {step}")
    n = int(round(1.0 / step))
    grid = np.linspace(0.0, 1.0, n + 1, dtype=float)
    grid = np.unique(np.r_[grid, np.asarray(fixed_thresholds, dtype=float)])
    return np.sort(grid)[::-1]


def load_profiles(path: Path, expected_lengths: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    profiles: dict[str, np.ndarray] = {}
    keys = sorted(str(key) for key in z.files)
    if expected_lengths is not None:
        missing = sorted(set(expected_lengths) - set(keys))
        extra = sorted(set(keys) - set(expected_lengths))
        if missing or extra:
            raise RuntimeError(f"Profile key mismatch for {path}: missing={missing[:10]} extra={extra[:10]}")
    for key in keys:
        arr = np.asarray(z[key], dtype=np.float32).reshape(-1)
        if not np.isfinite(arr).all():
            raise RuntimeError(f"Non-finite profile values for {key} in {path}")
        if expected_lengths is not None and len(arr) != int(expected_lengths[key]):
            raise RuntimeError(f"Profile length mismatch for {key}: got {len(arr)} expected {expected_lengths[key]}")
        profiles[key] = np.clip(arr, 0.0, 1.0)
    return profiles


def make_bundle(
    name: str,
    display_label: str,
    phasepro_profiles: dict[str, np.ndarray],
    val_profiles: dict[str, np.ndarray],
    phasepro_truths: dict[str, dict[str, Any]],
) -> ModelBundle:
    per = per_protein_metrics(phasepro_profiles, phasepro_truths)
    return ModelBundle(
        name=name,
        display_label=display_label,
        phasepro_profiles=phasepro_profiles,
        val_profiles=val_profiles,
        phasepro_per=per,
        phasepro_threshold_free=threshold_free_metrics(phasepro_profiles, phasepro_truths, per),
    )


def threshold_curve_on_grid(
    profiles: dict[str, np.ndarray],
    truths: dict[str, dict[str, Any]],
    thresholds: np.ndarray,
) -> pd.DataFrame:
    y, score = concat_labels_scores(profiles, truths)
    thresholds = np.asarray(thresholds, dtype=float)
    thresholds = thresholds[(thresholds >= 0.0) & (thresholds <= 1.0)]
    thresholds = np.sort(np.unique(thresholds))[::-1]
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_y = y[order].astype(np.int64)
    counts = np.searchsorted(-sorted_score, -thresholds, side="right").astype(np.int64)
    cumsum = np.cumsum(sorted_y)
    tp = np.zeros_like(counts, dtype=float)
    has_pred = counts > 0
    tp[has_pred] = cumsum[counts[has_pred] - 1].astype(float)
    pred_n = counts.astype(float)
    fp = pred_n - tp
    pos_total = float(sorted_y.sum())
    neg_total = float(len(sorted_y) - sorted_y.sum())
    fn = pos_total - tp
    tn = neg_total - fp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros_like(tp), where=denom > 0)
    stats = threshold_vector_stats(thresholds, profiles, truths)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "MCC": mcc,
            "IoU": iou,
            "TP": tp.astype(int),
            "FP": fp.astype(int),
            "FN": fn.astype(int),
            "TN": tn.astype(int),
            **stats,
        }
    ).reset_index(drop=True)


def merge_grid_curves(dpr_curve: pd.DataFrame, pstp_curve: pd.DataFrame) -> pd.DataFrame:
    cols = ["threshold", "precision", "recall", "F1", "MCC", "IoU", "mean_predicted_fraction", "region_overlap"]
    dpr = dpr_curve.loc[:, cols].add_prefix("dpr_").rename(columns={"dpr_threshold": "threshold"})
    pstp = pstp_curve.loc[:, cols].add_prefix("pstp_").rename(columns={"pstp_threshold": "threshold"})
    merged = dpr.merge(pstp, on="threshold", how="inner")
    for metric in ["precision", "recall", "F1", "MCC", "IoU", "mean_predicted_fraction", "region_overlap"]:
        merged[f"mean_{metric}"] = 0.5 * (merged[f"dpr_{metric}"] + merged[f"pstp_{metric}"])
        merged[f"delta_dpr_minus_pstp_{metric}"] = merged[f"dpr_{metric}"] - merged[f"pstp_{metric}"]
    return merged


def select_common_validation_threshold(grid_compare: pd.DataFrame, *, objective: str) -> dict[str, Any]:
    objective = objective.upper()
    if objective not in {"MCC", "F1"}:
        raise ValueError(objective)
    metric = f"mean_{objective}"
    ordered = grid_compare.sort_values(
        [metric, "mean_IoU", "mean_F1", "mean_precision", "threshold"],
        ascending=[False, False, False, False, False],
    )
    row = ordered.iloc[0].to_dict()
    row["selection_objective"] = f"Plan D validation common-threshold {metric}"
    return row


def build_policy_comparison(
    policies: list[dict[str, Any]],
    dpr: ModelBundle,
    pstp: ModelBundle,
    truths: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for model, threshold_key in [(dpr, "dpr_threshold"), (pstp, "pstp_threshold")]:
            threshold = float(policy[threshold_key])
            metrics = threshold_metrics(model.phasepro_profiles, truths, threshold=threshold)
            row = {
                "model": model.name,
                "threshold_policy": policy["threshold_policy"],
                "fair_policy_type": policy["fair_policy_type"],
                "threshold": threshold,
                "phasepro_used_for_threshold_selection": bool(policy.get("phasepro_used_for_threshold_selection", False)),
                "AUROC": model.phasepro_threshold_free["global_residue_AUROC"],
                "AUPRC": model.phasepro_threshold_free["global_residue_AUPRC"],
                "Spearman": model.phasepro_threshold_free["global_residue_Spearman"],
                "median_pp_Spearman": model.phasepro_threshold_free["per_protein_Spearman_median"],
                "pairwise": model.phasepro_threshold_free["same_protein_pairwise"],
            }
            row.update(metrics)
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


def plot_policy_bars(comparison: pd.DataFrame, png: Path, pdf: Path) -> None:
    policies = [
        "same_fixed_0p4",
        "same_fixed_0p5",
        "same_fixed_0p6",
        "common_plan_d_validation_mcc",
        "common_plan_d_validation_f1",
        "per_model_plan_d_validation_mcc",
    ]
    available = [policy for policy in policies if policy in set(comparison["threshold_policy"])]
    metrics = ["MCC", "F1", "IoU", "precision", "recall", "mean_predicted_fraction"]
    labels = {
        "same_fixed_0p4": "same 0.4",
        "same_fixed_0p5": "same 0.5",
        "same_fixed_0p6": "same 0.6",
        "common_plan_d_validation_mcc": "common val MCC",
        "common_plan_d_validation_f1": "common val F1",
        "per_model_plan_d_validation_mcc": "own val MCC",
    }
    colors = {"DPR_v6_plan_d_selected": "#1f77b4", "PSTP_nophasepro": "#ff7f0e"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2), constrained_layout=True)
    axes_arr = axes.reshape(-1)
    x = np.arange(len(available), dtype=float)
    width = 0.38
    for ax, metric in zip(axes_arr, metrics, strict=True):
        for offset, model_name in [(-width / 2, "DPR_v6_plan_d_selected"), (width / 2, "PSTP_nophasepro")]:
            values = []
            for policy in available:
                row = comparison.loc[
                    comparison["threshold_policy"].eq(policy) & comparison["model"].eq(model_name),
                    metric,
                ]
                values.append(float(row.iloc[0]) if len(row) else math.nan)
            ax.bar(x + offset, values, width=width, label=model_name.replace("_", " "), color=colors[model_name], alpha=0.9)
        ax.set_title(metric, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(policy, policy) for policy in available], rotation=25, ha="right", fontsize=8)
        ax.grid(True, axis="y", color="#eeeeee", linewidth=0.7)
    handles, names = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.suptitle("PhasePro fair threshold policy comparison", y=1.04, fontsize=14)
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sensitivity(
    grid_compare: pd.DataFrame,
    png: Path,
    pdf: Path,
    *,
    marker_thresholds: dict[str, float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True)
    specs = [
        ("MCC", "dpr_MCC", "pstp_MCC"),
        ("F1", "dpr_F1", "pstp_F1"),
        ("IoU", "dpr_IoU", "pstp_IoU"),
        ("mean predicted fraction", "dpr_mean_predicted_fraction", "pstp_mean_predicted_fraction"),
    ]
    for ax, (title, dcol, pcol) in zip(axes.reshape(-1), specs, strict=True):
        ax.plot(grid_compare["threshold"], grid_compare[dcol], color="#1f77b4", linewidth=1.3, label="DPR")
        ax.plot(grid_compare["threshold"], grid_compare[pcol], color="#ff7f0e", linewidth=1.3, label="PSTP")
        for label, threshold in marker_thresholds.items():
            color, linestyle = marker_style(label)
            ax.axvline(float(threshold), color=color, linestyle=linestyle, linewidth=0.9, alpha=0.85, label=label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("common/fixed threshold")
        ax.grid(True, color="#eeeeee", linewidth=0.7)
    handles, labels = axes.reshape(-1)[0].get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=False):
        unique.setdefault(label, handle)
    fig.legend(unique.values(), unique.keys(), loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("PhasePro fixed-threshold sensitivity (diagnostic, not threshold selection)", y=1.04, fontsize=14)
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_validation_threshold_selection(
    grid_compare: pd.DataFrame,
    png: Path,
    pdf: Path,
    *,
    marker_thresholds: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.3), constrained_layout=True)
    ax.plot(grid_compare["threshold"], grid_compare["dpr_MCC"], color="#1f77b4", linewidth=1.2, label="DPR validation MCC")
    ax.plot(grid_compare["threshold"], grid_compare["pstp_MCC"], color="#ff7f0e", linewidth=1.2, label="PSTP validation MCC")
    ax.plot(grid_compare["threshold"], grid_compare["mean_MCC"], color="#2ca02c", linewidth=1.5, label="mean MCC")
    for label, threshold in marker_thresholds.items():
        color, linestyle = marker_style(label)
        ax.axvline(float(threshold), color=color, linestyle=linestyle, linewidth=1.0, alpha=0.85, label=label)
    ax.set_title("Plan D validation common threshold selection", fontsize=13)
    ax.set_xlabel("common threshold")
    ax.set_ylabel("MCC")
    ax.grid(True, color="#eeeeee", linewidth=0.7)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def marker_style(label: str) -> tuple[str, str]:
    if "fixed" in label:
        return "#222222", ":"
    if "common PlanD MCC" in label:
        return "#2ca02c", "--"
    if "common PlanD F1" in label:
        return "#9467bd", "--"
    if label.startswith("DPR"):
        return "#1f77b4", "-."
    if label.startswith("PSTP"):
        return "#ff7f0e", "-."
    return "#555555", "--"


def make_group_curves(
    policy_name: str,
    group: str,
    protein_ids: list[str],
    dpr: ModelBundle,
    pstp: ModelBundle,
    truths: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pid in protein_ids:
        drow = dpr.phasepro_per.loc[dpr.phasepro_per["protein_id"].eq(pid)].iloc[0].to_dict()
        prow = pstp.phasepro_per.loc[pstp.phasepro_per["protein_id"].eq(pid)].iloc[0].to_dict()
        rows.append(
            {
                "protein_id": pid,
                "length": int(drow["length"]),
                "regions": drow["regions"],
                "dpr_spearman": drow["spearman"],
                "pstp_spearman": prow["spearman"],
                "dpr_auroc": drow["auroc"],
                "pstp_auroc": prow["auroc"],
            }
        )
        plot_one_curve(policy_name, pid, dpr, pstp, truths, thresholds, out_dir / f"{pid}_curves.png")
    pd.DataFrame(rows).to_csv(out_dir / f"{group}_protein_metrics.csv", index=False)
    plot_grid_curves(policy_name, group, protein_ids, dpr, pstp, truths, thresholds, out_dir / f"{group}_merged_curves.png", out_dir / f"{group}_merged_curves.pdf")


def plot_one_curve(
    policy_name: str,
    pid: str,
    dpr: ModelBundle,
    pstp: ModelBundle,
    truths: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    draw_profile_axes(ax, policy_name, pid, dpr, pstp, truths, thresholds)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grid_curves(
    policy_name: str,
    group: str,
    protein_ids: list[str],
    dpr: ModelBundle,
    pstp: ModelBundle,
    truths: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
    png: Path,
    pdf: Path,
) -> None:
    cols = 3
    rows = int(math.ceil(len(protein_ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.4 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, pid in zip(axes_arr, protein_ids, strict=False):
        draw_profile_axes(ax, policy_name, pid, dpr, pstp, truths, thresholds)
    for ax in axes_arr[len(protein_ids) :]:
        ax.axis("off")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=5, frameon=False, fontsize=10)
    fig.suptitle(f"{group} PhasePro curves ({policy_name})", y=1.02, fontsize=14)
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def draw_profile_axes(
    ax: Any,
    policy_name: str,
    pid: str,
    dpr: ModelBundle,
    pstp: ModelBundle,
    truths: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> None:
    dpr_score = np.asarray(dpr.phasepro_profiles[pid], dtype=float)
    pstp_score = np.asarray(pstp.phasepro_profiles[pid], dtype=float)
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
    ax.plot(x, dpr_score, color="#1f77b4", linewidth=1.25, label=dpr.display_label)
    ax.plot(x, pstp_score, color="#ff7f0e", linewidth=1.05, alpha=0.9, label=pstp.display_label)
    dpr_t = float(thresholds["DPR"])
    pstp_t = float(thresholds["PSTP"])
    if abs(dpr_t - pstp_t) < 1e-12:
        ax.axhline(dpr_t, color="#222222", linestyle="--", linewidth=0.9, alpha=0.8, label=f"threshold {dpr_t:.3f}")
    else:
        ax.axhline(dpr_t, color="#1f77b4", linestyle="--", linewidth=0.8, alpha=0.8, label=f"DPR threshold {dpr_t:.3f}")
        ax.axhline(pstp_t, color="#ff7f0e", linestyle="--", linewidth=0.8, alpha=0.8, label=f"PSTP threshold {pstp_t:.3f}")
    dpr_rho = dpr.phasepro_per.loc[dpr.phasepro_per["protein_id"].eq(pid), "spearman"].iloc[0]
    pstp_rho = pstp.phasepro_per.loc[pstp.phasepro_per["protein_id"].eq(pid), "spearman"].iloc[0]
    ax.set_title(f"{pid}  DPR rho={fmt_nan(dpr_rho)}  PSTP rho={fmt_nan(pstp_rho)}", fontsize=9)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(1, max(1, len(dpr_score)))
    ax.set_xlabel("residue", fontsize=8)
    ax.set_ylabel("score", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, axis="y", color="#eeeeee", linewidth=0.6)


def render_report(summary: dict[str, Any], comparison: pd.DataFrame, threshold_free: pd.DataFrame) -> str:
    policy_rows = comparison.loc[
        comparison["threshold_policy"].isin(
            [
                "same_fixed_0p5",
                "common_plan_d_validation_mcc",
                "common_plan_d_validation_f1",
                "per_model_plan_d_validation_mcc",
            ]
        )
    ].copy()
    fixed_rows = comparison.loc[comparison["threshold_policy"].str.startswith("same_fixed_")].copy()
    lines = [
        "# DPR v6 vs PSTP Fair Threshold Policies",
        "",
        "PhasePro is used here only for final metric computation and diagnostic threshold-sensitivity plots. The reported fair thresholds are fixed in advance or selected on non-PhasePro Plan D validation.",
        "",
        "## Main Fair Policies",
        "",
        markdown_table(
            policy_rows,
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
        "## Same Fixed Thresholds",
        "",
        markdown_table(
            fixed_rows,
            ["model", "threshold_policy", "threshold", "precision", "recall", "F1", "MCC", "IoU", "mean_predicted_fraction", "region_overlap"],
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
        "## Selected Validation Thresholds",
        "",
        f"- DPR own Plan D MCC threshold: `{summary['selected_thresholds']['dpr_plan_d_validation_mcc']:.6f}`",
        f"- PSTP own Plan D MCC threshold: `{summary['selected_thresholds']['pstp_plan_d_validation_mcc']:.6f}`",
        f"- common Plan D mean-MCC threshold: `{float(summary['selected_thresholds']['common_plan_d_validation_mcc']['threshold']):.6f}`",
        f"- common Plan D mean-F1 threshold: `{float(summary['selected_thresholds']['common_plan_d_validation_f1']['threshold']):.6f}`",
        "",
        "## Files",
        "",
    ]
    for key, path in summary["files"].items():
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


def format_threshold_for_name(value: float) -> str:
    return f"{float(value):.3g}".replace(".", "p")


def fmt_nan(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(x) else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
