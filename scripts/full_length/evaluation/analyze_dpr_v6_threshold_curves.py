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
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_DPR = (
    ROOT
    / "artifacts/benchmarks/phasepro_fair_single_matrix_20260617_eval_v2/profiles/"
    / "d1_flat_seed174_raw_planc_rankp257_p257_lr5e-6_50u_seed202606188_u0050_raw_phasepro/raw_p33_profiles.npz"
)
DEFAULT_PSTP = ROOT / "artifacts/benchmarks/plan_d_external_val_rankp257_single_20260617/profiles/pstp_nophasepro/p33_profiles.npz"
DEFAULT_DATA = ROOT / "data/processed/evaluation/phasepro_official_v1"
DEFAULT_OUT = ROOT / "artifacts/benchmarks/phasepro_threshold_tuning_p33"


@dataclass(frozen=True)
class ModelBundle:
    name: str
    profiles: dict[str, np.ndarray]
    threshold_free: dict[str, Any]
    per_protein: pd.DataFrame
    fixed_05: dict[str, Any]
    mcc_oracle: dict[str, Any]
    f1_oracle: dict[str, Any]
    cv_mcc: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune p33 residue threshold and plot DPR v6/PSTP PhasePro curves.")
    parser.add_argument("--dpr-profile", type=Path, default=DEFAULT_DPR)
    parser.add_argument("--pstp-profile", type=Path, default=DEFAULT_PSTP)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--best-n", type=int, default=12)
    parser.add_argument("--worst-n", type=int, default=12)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=20260615)
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    proteins = pd.read_parquet(args.data_root / "proteins.parquet")
    regions = pd.read_parquet(args.data_root / "regions.parquet")
    truths = build_truths(proteins, regions)

    dpr_profiles = load_profiles(args.dpr_profile, proteins)
    pstp_profiles = load_profiles(args.pstp_profile, proteins)

    dpr = analyze_model(
        "DPR_v6_d1_flat_u0250_EMA",
        dpr_profiles,
        truths,
        out / "dpr_v6",
        fixed_threshold=float(args.fixed_threshold),
        cv_folds=int(args.cv_folds),
        cv_seed=int(args.cv_seed),
    )
    pstp = analyze_model(
        "PSTP_nophasepro_fair",
        pstp_profiles,
        truths,
        out / "pstp_nophasepro",
        fixed_threshold=float(args.fixed_threshold),
        cv_folds=int(args.cv_folds),
        cv_seed=int(args.cv_seed),
    )

    compare = comparison_table([dpr, pstp])
    compare.to_csv(out / "dpr_v6_vs_pstp_threshold_comparison.csv", index=False)
    threshold_free = threshold_free_table([dpr, pstp])
    threshold_free.to_csv(out / "threshold_free_comparison.csv", index=False)

    selected = select_best_worst(dpr.per_protein, pstp.per_protein, best_n=int(args.best_n), worst_n=int(args.worst_n))
    selected.to_csv(out / "selected_best_worst_by_dpr_p33_spearman.csv", index=False)

    best_ids = selected.loc[selected["group"].eq("best_12"), "protein_id"].tolist()
    worst_ids = selected.loc[selected["group"].eq("worst_12"), "protein_id"].tolist()
    make_group_outputs(
        "best_12",
        best_ids,
        dpr,
        pstp,
        truths,
        proteins,
        out / "curves_best_12",
    )
    make_group_outputs(
        "worst_12",
        worst_ids,
        dpr,
        pstp,
        truths,
        proteins,
        out / "curves_worst_12",
    )

    summary = {
        "status": "PASS",
        "truth_policy": "pstp_notebook: label[int(start_raw):int(min(end_raw, sequence_length))] = 1",
        "dpr_profile": str(args.dpr_profile.resolve()),
        "pstp_profile": str(args.pstp_profile.resolve()),
        "data_root": str(args.data_root.resolve()),
        "output_root": str(out),
        "fixed_threshold": float(args.fixed_threshold),
        "threshold_selection": {
            "primary": "mcc_oracle",
            "objective": "full PhasePro p33 residue-level MCC",
            "tie_break": "IoU, F1, precision, then higher threshold",
            "note": "Post-hoc oracle threshold selected on the evaluation set; use CV rows as a leakage-aware sanity check.",
        },
        "models": {
            dpr.name: model_summary(dpr),
            pstp.name: model_summary(pstp),
        },
        "files": {
            "comparison_csv": str((out / "dpr_v6_vs_pstp_threshold_comparison.csv").resolve()),
            "threshold_free_csv": str((out / "threshold_free_comparison.csv").resolve()),
            "selection_csv": str((out / "selected_best_worst_by_dpr_p33_spearman.csv").resolve()),
            "best_12_png": str((out / "curves_best_12" / "best_12_merged_curves.png").resolve()),
            "best_12_pdf": str((out / "curves_best_12" / "best_12_merged_curves.pdf").resolve()),
            "worst_12_png": str((out / "curves_worst_12" / "worst_12_merged_curves.png").resolve()),
            "worst_12_pdf": str((out / "curves_worst_12" / "worst_12_merged_curves.pdf").resolve()),
        },
    }
    (out / "threshold_tuning_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    (out / "threshold_tuning_report.md").write_text(render_report(summary, compare, threshold_free, selected), encoding="utf-8")
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
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
            raise RuntimeError(f"Profile length mismatch for {pid} in {path}: got {len(arr)} expected {expected_lengths[pid]}")
        if not np.isfinite(arr).all():
            raise RuntimeError(f"Non-finite profile values for {pid} in {path}")
        profiles[pid] = np.clip(arr, 0.0, 1.0)
    return profiles


def build_truths(proteins: pd.DataFrame, regions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    truths: dict[str, dict[str, Any]] = {}
    protein_meta = proteins.set_index("protein_id", drop=False)
    for row in proteins.itertuples(index=False):
        pid = str(row.protein_id)
        truths[pid] = {
            "label": np.zeros(int(row.sequence_length), dtype=np.int8),
            "regions": [],
            "sequence": str(row.sequence),
            "gene_name": str(row.gene_name),
            "protein_name": str(row.protein_name),
        }
    for row in regions.itertuples(index=False):
        pid = str(row.protein_id)
        start = int(row.pstp_notebook_start_0based)
        end = int(row.pstp_notebook_end_exclusive)
        length = int(protein_meta.loc[pid, "sequence_length"])
        if start < 0 or end > length or end <= start:
            raise RuntimeError(f"Invalid PSTP-notebook truth span {pid}:{start}-{end} length={length}")
        truths[pid]["label"][start:end] = 1
        truths[pid]["regions"].append(
            {
                "region_id": str(row.region_id),
                "start": start,
                "end": end,
                "start_1based": int(row.start_raw),
                "end_1based": int(end),
            }
        )
    return truths


def analyze_model(
    name: str,
    profiles: dict[str, np.ndarray],
    truths: dict[str, dict[str, Any]],
    out_dir: Path,
    *,
    fixed_threshold: float,
    cv_folds: int,
    cv_seed: int,
) -> ModelBundle:
    out_dir.mkdir(parents=True, exist_ok=True)
    per = per_protein_metrics(profiles, truths)
    per.to_csv(out_dir / "per_protein_p33.csv", index=False)
    tf = threshold_free_metrics(profiles, truths, per)
    fixed = threshold_metrics(profiles, truths, threshold=fixed_threshold)
    fixed["threshold"] = fixed_threshold
    curve = threshold_curve(profiles, truths, extra_thresholds=[fixed_threshold, 1.0])
    curve.to_csv(out_dir / "threshold_sweep.csv", index=False)
    mcc_selected = select_best_threshold(curve, objective="MCC")
    mcc = threshold_metrics(profiles, truths, threshold=float(mcc_selected["threshold"]))
    mcc.update(
        {
            "threshold": float(mcc_selected["threshold"]),
            "selection_objective": "full-set p33 residue-level MCC",
            "selection_row": mcc_selected,
        }
    )
    f1_selected = select_best_threshold(curve, objective="F1")
    f1 = threshold_metrics(profiles, truths, threshold=float(f1_selected["threshold"]))
    f1.update(
        {
            "threshold": float(f1_selected["threshold"]),
            "selection_objective": "diagnostic full-set p33 residue-level F1",
            "selection_row": f1_selected,
        }
    )
    cv = grouped_cv_threshold(profiles, truths, folds=cv_folds, seed=cv_seed, objective="MCC")
    bundle = ModelBundle(
        name=name,
        profiles=profiles,
        threshold_free=tf,
        per_protein=per,
        fixed_05=fixed,
        mcc_oracle=mcc,
        f1_oracle=f1,
        cv_mcc=cv,
    )
    payload = model_summary(bundle)
    (out_dir / "threshold_metrics_summary.json").write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    pd.DataFrame([flatten_metrics(name, "fixed_0.5", fixed), flatten_metrics(name, "mcc_oracle", mcc), flatten_metrics(name, "f1_oracle", f1)]).to_csv(
        out_dir / "selected_threshold_metrics.csv", index=False
    )
    pd.DataFrame(cv["folds"]).to_csv(out_dir / "mcc_grouped_cv_folds.csv", index=False)
    return bundle


def per_protein_metrics(profiles: dict[str, np.ndarray], truths: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pid in sorted(profiles):
        y = np.asarray(truths[pid]["label"], dtype=int)
        score = np.asarray(profiles[pid], dtype=float)
        valid = len(np.unique(y)) == 2 and int((y == 0).sum()) >= 20
        rows.append(
            {
                "protein_id": pid,
                "length": int(len(score)),
                "positive_count": int(y.sum()),
                "positive_fraction": float(y.mean()),
                "region_count": int(len(truths[pid]["regions"])),
                "regions": format_regions(truths[pid]["regions"]),
                "spearman": safe_spearman(y, score) if valid else math.nan,
                "auroc": safe_auc(y, score),
                "auprc": safe_ap(y, score),
                "pred_fraction_0p5": float((score >= 0.5).mean()),
                "pos_mean": float(score[y == 1].mean()) if int(y.sum()) else math.nan,
                "neg_mean": float(score[y == 0].mean()) if int((y == 0).sum()) else math.nan,
                "pos_minus_neg_mean": float(score[y == 1].mean() - score[y == 0].mean()) if int(y.sum()) and int((y == 0).sum()) else math.nan,
                "max_score": float(score.max()) if len(score) else math.nan,
                "mean_score": float(score.mean()) if len(score) else math.nan,
                "std_score": float(score.std()) if len(score) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def threshold_free_metrics(profiles: dict[str, np.ndarray], truths: dict[str, dict[str, Any]], per: pd.DataFrame) -> dict[str, Any]:
    y, score = concat_labels_scores(profiles, truths)
    return {
        "global_residue_AUROC": safe_auc(y, score),
        "global_residue_AUPRC": safe_ap(y, score),
        "global_residue_Spearman": safe_spearman(y, score),
        "per_protein_Spearman_mean": float(per["spearman"].mean(skipna=True)),
        "per_protein_Spearman_median": float(per["spearman"].median(skipna=True)),
        "per_protein_Spearman_valid_count": int(per["spearman"].notna().sum()),
        "per_protein_Spearman_invalid_count": int(per["spearman"].isna().sum()),
        "per_protein_AUROC_mean": float(per["auroc"].mean(skipna=True)),
        "per_protein_AUROC_median": float(per["auroc"].median(skipna=True)),
        "same_protein_pairwise": pairwise_accuracy_from_per(per),
        "residue_n": int(len(y)),
        "positive_residue_n": int(y.sum()),
        "positive_residue_fraction": float(y.mean()),
    }


def threshold_curve(
    profiles: dict[str, np.ndarray],
    truths: dict[str, dict[str, Any]],
    *,
    extra_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    y, score = concat_labels_scores(profiles, truths)
    thresholds = np.unique(score.astype(float))
    if extra_thresholds:
        thresholds = np.unique(np.r_[thresholds, np.asarray(extra_thresholds, dtype=float)])
    thresholds = thresholds[(thresholds >= 0.0) & (thresholds <= 1.0)]
    thresholds = np.sort(thresholds)[::-1]
    if thresholds.size == 0:
        return pd.DataFrame()

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


def threshold_vector_stats(
    thresholds: np.ndarray,
    profiles: dict[str, np.ndarray],
    truths: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    thresholds = np.asarray(thresholds, dtype=float)
    protein_n = max(1, len(profiles))
    empty = np.zeros(len(thresholds), dtype=float)
    near_full = np.zeros(len(thresholds), dtype=float)
    pred_fracs: list[np.ndarray] = []
    fully = np.zeros(len(thresholds), dtype=int)
    partial = np.zeros(len(thresholds), dtype=int)
    missed = np.zeros(len(thresholds), dtype=int)
    total_regions = 0
    positive_scores: list[np.ndarray] = []
    for pid, score in profiles.items():
        score = np.asarray(score, dtype=float)
        desc = np.sort(score)[::-1]
        counts = np.searchsorted(-desc, -thresholds, side="right")
        frac = counts.astype(float) / max(1, len(score))
        pred_fracs.append(frac)
        empty += counts == 0
        near_full += frac >= 0.80
        y = np.asarray(truths[pid]["label"], dtype=bool)
        positive_scores.append(score[y])
        for region in truths[pid]["regions"]:
            start = int(region["start"])
            end = int(region["end"])
            region_scores = np.sort(score[start:end])[::-1]
            region_len = int(end - start)
            covered = np.searchsorted(-region_scores, -thresholds, side="right")
            fully += covered == region_len
            partial += (covered > 0) & (covered < region_len)
            missed += covered == 0
            total_regions += 1
    frac_matrix = np.vstack(pred_fracs) if pred_fracs else np.zeros((0, len(thresholds)), dtype=float)
    pos = np.concatenate(positive_scores) if positive_scores else np.asarray([], dtype=float)
    if pos.size:
        pos_desc = np.sort(pos)[::-1]
        recovered = np.searchsorted(-pos_desc, -thresholds, side="right").astype(float) / float(pos.size)
    else:
        recovered = np.full(len(thresholds), np.nan)
    return {
        "empty_rate": empty / protein_n,
        "near_full_rate": near_full / protein_n,
        "mean_predicted_fraction": frac_matrix.mean(axis=0) if frac_matrix.size else np.full(len(thresholds), np.nan),
        "median_predicted_fraction": np.median(frac_matrix, axis=0) if frac_matrix.size else np.full(len(thresholds), np.nan),
        "region_overlap": fully + partial,
        "official_region_total": np.full(len(thresholds), int(total_regions), dtype=int),
        "fully_covered_regions": fully,
        "partially_covered_regions": partial,
        "missed_regions": missed,
        "recovered_positive_fraction": recovered,
    }


def select_best_threshold(curve: pd.DataFrame, *, objective: str) -> dict[str, Any]:
    if objective == "MCC":
        ordered = curve.sort_values(["MCC", "IoU", "F1", "precision", "threshold"], ascending=[False, False, False, False, False])
    elif objective == "F1":
        ordered = curve.sort_values(["F1", "MCC", "IoU", "precision", "threshold"], ascending=[False, False, False, False, False])
    else:
        raise ValueError(objective)
    return ordered.iloc[0].to_dict()


def threshold_metrics(profiles: dict[str, np.ndarray], truths: dict[str, dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    y_all, score_all = concat_labels_scores(profiles, truths)
    pred_all = score_all >= float(threshold)
    tp = int(((pred_all == 1) & (y_all == 1)).sum())
    fp = int(((pred_all == 1) & (y_all == 0)).sum())
    fn = int(((pred_all == 0) & (y_all == 1)).sum())
    tn = int(((pred_all == 0) & (y_all == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    mcc = float(matthews_corrcoef(y_all.astype(int), pred_all.astype(int))) if len(np.unique(pred_all)) > 1 else 0.0

    pred_fracs: list[float] = []
    empty = 0
    near_full = 0
    fully = 0
    partially = 0
    missed = 0
    total_regions = 0
    recovered_positive = 0
    total_positive = 0
    for pid, score in profiles.items():
        score = np.asarray(score, dtype=float)
        y = np.asarray(truths[pid]["label"], dtype=bool)
        mask = score >= float(threshold)
        frac = float(mask.mean()) if len(mask) else 0.0
        pred_fracs.append(frac)
        empty += int(mask.sum() == 0)
        near_full += int(frac >= 0.80)
        recovered_positive += int((mask & y).sum())
        total_positive += int(y.sum())
        for region in truths[pid]["regions"]:
            total_regions += 1
            covered = int(mask[int(region["start"]) : int(region["end"])].sum())
            region_len = int(region["end"]) - int(region["start"])
            if covered == region_len and region_len > 0:
                fully += 1
            elif covered > 0:
                partially += 1
            else:
                missed += 1
    return {
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "MCC": float(mcc),
        "IoU": float(iou),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "empty_rate": float(empty / max(1, len(profiles))),
        "near_full_rate": float(near_full / max(1, len(profiles))),
        "mean_predicted_fraction": float(np.mean(pred_fracs)) if pred_fracs else math.nan,
        "median_predicted_fraction": float(np.median(pred_fracs)) if pred_fracs else math.nan,
        "region_overlap": int(fully + partially),
        "official_region_total": int(total_regions),
        "fully_covered_regions": int(fully),
        "partially_covered_regions": int(partially),
        "missed_regions": int(missed),
        "recovered_positive_fraction": float(recovered_positive / max(1, total_positive)),
    }


def grouped_cv_threshold(
    profiles: dict[str, np.ndarray],
    truths: dict[str, dict[str, Any]],
    *,
    folds: int,
    seed: int,
    objective: str,
) -> dict[str, Any]:
    pids = sorted(profiles)
    fold_map = make_group_folds(pids, profiles, truths, folds=folds, seed=seed)
    fold_rows: list[dict[str, Any]] = []
    pred_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for fold in range(folds):
        held = {pid for pid, value in fold_map.items() if value == fold}
        train = set(pids) - held
        selected = select_best_threshold(
            threshold_curve(filter_profiles(profiles, train), filter_truths(truths, train)),
            objective=objective,
        )
        threshold = float(selected["threshold"])
        held_profiles = filter_profiles(profiles, held)
        held_truths = filter_truths(truths, held)
        metrics = threshold_metrics(held_profiles, held_truths, threshold=threshold)
        fold_rows.append({"fold": fold, "threshold": threshold, **metrics})
        y, score = concat_labels_scores(held_profiles, held_truths)
        y_parts.append(y)
        pred_parts.append((score >= threshold).astype(int))
    y_all = np.concatenate(y_parts).astype(int)
    pred_all = np.concatenate(pred_parts).astype(int)
    tp = int(((pred_all == 1) & (y_all == 1)).sum())
    fp = int(((pred_all == 1) & (y_all == 0)).sum())
    fn = int(((pred_all == 0) & (y_all == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "fold_count": int(folds),
        "seed": int(seed),
        "objective": objective,
        "threshold_mean": float(np.mean([row["threshold"] for row in fold_rows])),
        "threshold_std": float(np.std([row["threshold"] for row in fold_rows], ddof=0)),
        "threshold_min": float(np.min([row["threshold"] for row in fold_rows])),
        "threshold_max": float(np.max([row["threshold"] for row in fold_rows])),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "MCC": float(matthews_corrcoef(y_all, pred_all)) if len(np.unique(pred_all)) > 1 else 0.0,
        "IoU": float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0,
        "folds": fold_rows,
    }


def make_group_folds(
    pids: list[str],
    profiles: dict[str, np.ndarray],
    truths: dict[str, dict[str, Any]],
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    rows = []
    for pid in pids:
        y = np.asarray(truths[pid]["label"], dtype=int)
        rows.append((pid, float(y.mean()), int(len(profiles[pid])), float(rng.random())))
    rows.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
    return {pid: i % folds for i, (pid, _pos, _length, _rand) in enumerate(rows)}


def filter_profiles(profiles: dict[str, np.ndarray], keep: set[str]) -> dict[str, np.ndarray]:
    return {pid: profiles[pid] for pid in keep}


def filter_truths(truths: dict[str, dict[str, Any]], keep: set[str]) -> dict[str, dict[str, Any]]:
    return {pid: truths[pid] for pid in keep}


def comparison_table(models: list[ModelBundle]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in models:
        for policy, metrics in [
            ("fixed_0.5", model.fixed_05),
            ("mcc_oracle", model.mcc_oracle),
            ("f1_oracle", model.f1_oracle),
        ]:
            row = flatten_metrics(model.name, policy, metrics)
            row.update(
                {
                    "AUROC": model.threshold_free["global_residue_AUROC"],
                    "AUPRC": model.threshold_free["global_residue_AUPRC"],
                    "Spearman": model.threshold_free["global_residue_Spearman"],
                    "median_pp_Spearman": model.threshold_free["per_protein_Spearman_median"],
                    "pairwise": model.threshold_free["same_protein_pairwise"],
                }
            )
            rows.append(row)
        cv = model.cv_mcc
        rows.append(
            {
                "model": model.name,
                "threshold_policy": "mcc_grouped_5fold_cv",
                "threshold": cv["threshold_mean"],
                "threshold_std": cv["threshold_std"],
                "AUROC": model.threshold_free["global_residue_AUROC"],
                "AUPRC": model.threshold_free["global_residue_AUPRC"],
                "Spearman": model.threshold_free["global_residue_Spearman"],
                "median_pp_Spearman": model.threshold_free["per_protein_Spearman_median"],
                "pairwise": model.threshold_free["same_protein_pairwise"],
                "precision": cv["precision"],
                "recall": cv["recall"],
                "F1": cv["F1"],
                "MCC": cv["MCC"],
                "IoU": cv["IoU"],
            }
        )
    return pd.DataFrame(rows)


def threshold_free_table(models: list[ModelBundle]) -> pd.DataFrame:
    rows = []
    for model in models:
        rows.append({"model": model.name, **model.threshold_free})
    return pd.DataFrame(rows)


def flatten_metrics(model: str, policy: str, metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "threshold",
        "precision",
        "recall",
        "F1",
        "MCC",
        "IoU",
        "TP",
        "FP",
        "FN",
        "TN",
        "empty_rate",
        "near_full_rate",
        "mean_predicted_fraction",
        "median_predicted_fraction",
        "region_overlap",
        "official_region_total",
        "fully_covered_regions",
        "partially_covered_regions",
        "missed_regions",
        "recovered_positive_fraction",
    ]
    return {"model": model, "threshold_policy": policy, **{key: metrics.get(key, math.nan) for key in keys}}


def select_best_worst(dpr_per: pd.DataFrame, pstp_per: pd.DataFrame, *, best_n: int, worst_n: int) -> pd.DataFrame:
    pstp_small = pstp_per[["protein_id", "spearman", "auroc", "auprc", "pred_fraction_0p5", "pos_minus_neg_mean"]].rename(
        columns={
            "spearman": "pstp_spearman",
            "auroc": "pstp_auroc",
            "auprc": "pstp_auprc",
            "pred_fraction_0p5": "pstp_pred_fraction_0p5",
            "pos_minus_neg_mean": "pstp_pos_minus_neg_mean",
        }
    )
    merged = dpr_per.merge(pstp_small, on="protein_id", how="left").rename(
        columns={
            "spearman": "dpr_spearman",
            "auroc": "dpr_auroc",
            "auprc": "dpr_auprc",
            "pred_fraction_0p5": "dpr_pred_fraction_0p5",
            "pos_minus_neg_mean": "dpr_pos_minus_neg_mean",
        }
    )
    valid = merged.loc[merged["dpr_spearman"].notna()].copy()
    best = valid.sort_values(["dpr_spearman", "dpr_auroc"], ascending=[False, False]).head(best_n).copy()
    worst = valid.sort_values(["dpr_spearman", "dpr_auroc"], ascending=[True, True]).head(worst_n).copy()
    best.insert(0, "group", "best_12")
    worst.insert(0, "group", "worst_12")
    return pd.concat([best, worst], ignore_index=True)


def make_group_outputs(
    group: str,
    protein_ids: list[str],
    dpr: ModelBundle,
    pstp: ModelBundle,
    truths: dict[str, dict[str, Any]],
    proteins: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
                "dpr_pred_fraction_0p5": drow["pred_fraction_0p5"],
                "pstp_pred_fraction_0p5": prow["pred_fraction_0p5"],
            }
        )
        plot_one(pid, dpr, pstp, truths, out_dir / f"{pid}_curves.png")
    pd.DataFrame(rows).to_csv(out_dir / f"{group}_protein_metrics.csv", index=False)
    plot_grid(group, protein_ids, dpr, pstp, truths, out_dir / f"{group}_merged_curves.png", out_dir / f"{group}_merged_curves.pdf")


def plot_one(pid: str, dpr: ModelBundle, pstp: ModelBundle, truths: dict[str, dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    draw_profile_axes(ax, pid, dpr, pstp, truths)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grid(group: str, protein_ids: list[str], dpr: ModelBundle, pstp: ModelBundle, truths: dict[str, dict[str, Any]], png: Path, pdf: Path) -> None:
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
    fig.suptitle(f"{group} DPR v6 p33 PhasePro curves", y=1.01, fontsize=14)
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def draw_profile_axes(ax: Any, pid: str, dpr: ModelBundle, pstp: ModelBundle, truths: dict[str, dict[str, Any]]) -> None:
    dpr_score = np.asarray(dpr.profiles[pid], dtype=float)
    pstp_score = np.asarray(pstp.profiles[pid], dtype=float)
    x = np.arange(1, len(dpr_score) + 1)
    for region in truths[pid]["regions"]:
        ax.axvspan(int(region["start"]) + 1, int(region["end"]), color="#d7d7d7", alpha=0.55, linewidth=0, label="truth" if not ax.get_legend_handles_labels()[1] else None)
    ax.plot(x, dpr_score, color="#1f77b4", linewidth=1.25, label="DPR v6 p33")
    ax.plot(x, pstp_score, color="#ff7f0e", linewidth=1.05, alpha=0.9, label="PSTP fair p33")
    ax.axhline(float(dpr.mcc_oracle["threshold"]), color="#1f77b4", linestyle="--", linewidth=0.8, alpha=0.8, label="DPR MCC threshold")
    ax.axhline(float(pstp.mcc_oracle["threshold"]), color="#ff7f0e", linestyle="--", linewidth=0.8, alpha=0.8, label="PSTP MCC threshold")
    dpr_rho = dpr.per_protein.loc[dpr.per_protein["protein_id"].eq(pid), "spearman"].iloc[0]
    pstp_rho = pstp.per_protein.loc[pstp.per_protein["protein_id"].eq(pid), "spearman"].iloc[0]
    ax.set_title(f"{pid}  DPR rho={fmt_nan(dpr_rho)}  PSTP rho={fmt_nan(pstp_rho)}", fontsize=9)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(1, max(1, len(dpr_score)))
    ax.set_xlabel("residue", fontsize=8)
    ax.set_ylabel("score", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, axis="y", color="#eeeeee", linewidth=0.6)


def concat_labels_scores(profiles: dict[str, np.ndarray], truths: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    y = []
    score = []
    for pid in sorted(profiles):
        yy = np.asarray(truths[pid]["label"], dtype=int)
        ss = np.asarray(profiles[pid], dtype=float)
        if len(yy) != len(ss):
            raise RuntimeError(f"Length mismatch for {pid}: label={len(yy)} score={len(ss)}")
        y.append(yy)
        score.append(ss)
    return np.concatenate(y), np.concatenate(score)


def pairwise_accuracy_from_per(per: pd.DataFrame) -> float:
    valid = per["auroc"].dropna()
    return float(valid.mean()) if len(valid) else math.nan


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if int(y.sum()) == 0:
        return math.nan
    return float(average_precision_score(y, score))


def safe_spearman(y: np.ndarray, score: np.ndarray) -> float:
    yy = np.asarray(y, dtype=float).reshape(-1)
    ss = np.asarray(score, dtype=float).reshape(-1)
    if yy.size != ss.size or yy.size == 0:
        return math.nan
    if not np.isfinite(yy).all() or not np.isfinite(ss).all():
        return math.nan
    ry = average_ranks(yy)
    rs = average_ranks(ss)
    ry -= float(ry.mean())
    rs -= float(rs.mean())
    denom = float(np.sqrt(np.sum(ry * ry) * np.sum(rs * rs)))
    if denom == 0.0:
        return math.nan
    return float(np.sum(ry * rs) / denom)


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    n = int(values.size)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(n, dtype=np.float64)
    if n == 0:
        return ranks
    group_starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], n]
    group_sizes = group_ends - group_starts
    group_ranks = 0.5 * (group_starts + 1 + group_ends)
    ranks[order] = np.repeat(group_ranks, group_sizes)
    return ranks


def format_regions(regions: list[dict[str, Any]]) -> str:
    return ";".join(f"{int(region['start_1based'])}-{int(region['end_1based'])}" for region in regions)


def model_summary(model: ModelBundle) -> dict[str, Any]:
    return {
        "threshold_free": model.threshold_free,
        "fixed_0.5": model.fixed_05,
        "mcc_oracle": model.mcc_oracle,
        "f1_oracle": model.f1_oracle,
        "mcc_grouped_5fold_cv": {k: v for k, v in model.cv_mcc.items() if k != "folds"},
    }


def render_report(summary: dict[str, Any], compare: pd.DataFrame, threshold_free: pd.DataFrame, selected: pd.DataFrame) -> str:
    lines = [
        "# DPR v6 p33 Threshold Tuning vs PSTP",
        "",
        "## Protocol",
        "",
        f"- truth policy: `{summary['truth_policy']}`",
        "- primary tuned threshold: full-set p33 residue-level MCC oracle",
        "- diagnostic threshold: full-set p33 residue-level F1 oracle",
        "- caveat: tuned thresholds are post-hoc on PhasePro; grouped CV rows are included as a leakage-aware sanity check.",
        "",
        "## Threshold Comparison",
        "",
        markdown_table(
            compare,
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
        "## Threshold-Free Comparison",
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
            [
                "group",
                "protein_id",
                "length",
                "regions",
                "dpr_spearman",
                "pstp_spearman",
                "dpr_auroc",
                "pstp_auroc",
            ],
        ),
        "",
        "## Files",
        "",
    ]
    for key, path in summary["files"].items():
        lines.append(f"- {key}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    rows = []
    sub = df.loc[:, [col for col in columns if col in df.columns]].copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    header = "| " + " | ".join(sub.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(sub.columns)) + " |"
    rows.extend([header, sep])
    for row in sub.itertuples(index=False):
        rows.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(rows)


def fmt_nan(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(x) else f"{x:.3f}"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
