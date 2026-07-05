#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseflow.full_length.postprocess import scores_to_regions  # noqa: E402
from scripts.full_length.evaluation.analyze_dpr_v6_threshold_curves import build_truths  # noqa: E402
from scripts.full_length.evaluation.dpr_v6_plan_d_common import build_residue_truths, load_candidate_index  # noqa: E402


FINAL_ROOT = ROOT / "artifacts"
OUT_ROOT = FINAL_ROOT / "benchmarks" / "final_overall_benchmark_20260617"
PHASEPRO_ROOT = ROOT / "artifacts" / "data" / "processed" / "evaluation" / "phasepro_official_v1"
PLAN_D_VAL = ROOT / "artifacts" / "data" / "processed" / "stage2" / "dpr_v8r1a" / "indices" / "sampler_plans" / "plan_d_mixed_hq_val_candidate_index.parquet"
PPMC_SCORE_TABLE = (
    ROOT
    / "external_artifacts"
    / "final_llps"
    / "benchmark"
    / "fair_full_ppmc"
    / "phaseflow_llps_calibrated"
    / "results"
    / "phaseflow_llps_final_calibrated"
    / "ppmc_all_sequences"
    / "combined"
    / "ppmc_panel_scores_with_phaseflow.csv"
)
PHASEFLOW_LLPS_REFERENCE_CKPT = ROOT / "external_artifacts" / "final_llps" / "checkpoints" / "best_single_model.pt"
PHASEFLOW_LLPS_REFERENCE_CALIBRATED_CKPT = ROOT / "external_artifacts" / "final_llps" / "checkpoints" / "best_single_model_calibrated.pt"
FINAL_CKPT = FINAL_ROOT / "model" / "checkpoints" / "update_000050.pt"
DPR_PHASEPRO = (
    FINAL_ROOT
    / "benchmarks"
    / "phasepro_fair_single_matrix_20260617_eval_v2"
    / "profiles"
    / "d1_flat_seed174_raw_planc_rankp257_p257_lr5e-6_50u_seed202606188_u0050_raw_phasepro"
    / "raw_p257_profiles.npz"
)
DPR_PLAN_D = (
    FINAL_ROOT
    / "benchmarks"
    / "plan_d_external_val_rankp257_single_20260617"
    / "profiles"
    / "d1_flat_seed174_raw_planc_rankp257_p257_lr5e-6_50u_seed202606188_u0050_raw"
    / "p257_profiles.npz"
)
PSTP_PHASEPRO = ROOT / "external_artifacts" / "pstp_official_benchmark_v1" / "profiles" / "pstp_nophasepro" / "selected_family_p257_profiles.npz"
PSTP_PLAN_D = FINAL_ROOT / "benchmarks" / "plan_d_external_val_rankp257_single_20260617" / "profiles" / "pstp_nophasepro" / "p257_profiles.npz"
BASELINE_DIR = ROOT / "external_artifacts" / "dpr_baselines" / "profiles"
PPMC_NEGATIVE_FASTA = BASELINE_DIR.parent / "inputs" / "ppmc_np_nd_negative_sequences.fasta"
PSTP_PHASEPRO_H5 = BASELINE_DIR / "pstp_scan_phasepro_profiles.h5"
PSTP_PPMC_NEGATIVE_H5 = BASELINE_DIR / "pstp_scan_ppmc_negative_profiles.h5"


LLPS_MODEL_MAP = [
    ("PSTP", "pstp", "pstp_score", "PSTP protein-level LLPS score from PPMC benchmark table"),
    ("DeePhase", "deephase", "deephase_score", "DeePhase protein-level LLPS score from PPMC benchmark table"),
    ("PSPredictor", "pspredictor", "pspredictor_score", "PSPredictor protein-level LLPS score from PPMC benchmark table"),
    ("PSPHunter", "psphunter", "psphunter_probability", "PSPHunter protein-level LLPS probability from PPMC benchmark table"),
    ("FuzDrop", "fuzdrop", "fuzdrop_score", "FuzDrop protein-level LLPS score from PPMC benchmark table"),
    ("PhaseFlow", "phaseflow", "phaseflow_region_global_score", "frozen PhaseFlow LLPS region_global score inside final DPR v6 stack"),
]

DPR_MODEL_ORDER = ["PSTP-Scan", "FuzDrop", "PSPHunter", "catGRANULE/PLAAC", "PhaseFlow"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final LLPS and DPR benchmark tables from cached final profiles.")
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--phasepro-root", type=Path, default=PHASEPRO_ROOT)
    parser.add_argument("--plan-d-val-index", type=Path, default=PLAN_D_VAL)
    parser.add_argument("--ppmc-score-table", type=Path, default=PPMC_SCORE_TABLE)
    parser.add_argument("--checkpoint", type=Path, default=FINAL_CKPT)
    parser.add_argument("--dpr-phasepro-profile", type=Path, default=DPR_PHASEPRO)
    parser.add_argument("--dpr-negative-profile", type=Path, default=DPR_PLAN_D)
    parser.add_argument("--pstp-phasepro-profile", type=Path, default=PSTP_PHASEPRO)
    parser.add_argument("--pstp-negative-profile", type=Path, default=PSTP_PLAN_D)
    parser.add_argument("--baseline-profile-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--ppmc-negative-fasta", type=Path, default=PPMC_NEGATIVE_FASTA)
    parser.add_argument("--pstp-phasepro-h5", type=Path, default=PSTP_PHASEPRO_H5)
    parser.add_argument("--pstp-ppmc-negative-h5", type=Path, default=PSTP_PPMC_NEGATIVE_H5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--merge-gap", type=int, default=5)
    parser.add_argument("--min-region-len", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_root.resolve()
    for sub in ["llps", "dpr", "reports", "manifests"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    proteins = pd.read_parquet(args.phasepro_root / "proteins.parquet")
    regions = pd.read_parquet(args.phasepro_root / "regions.parquet")
    phasepro_truths = build_truths(proteins, regions)
    phasepro_records = records_from_phasepro(proteins)

    candidate = load_candidate_index(args.plan_d_val_index)
    plan_d_truths, plan_d_truth_audit = build_residue_truths(candidate)
    plan_d_negative_records = records_from_plan_d(candidate, plan_d_truths, negative_only=True)
    ppmc_negative_records = records_from_fasta(args.ppmc_negative_fasta)

    llps_metrics = build_llps_metrics(args.ppmc_score_table)
    llps_table = requested_llps_table(llps_metrics)
    llps_metrics.to_csv(out / "llps" / "llps_all_panel_metrics.csv", index=False)
    llps_table.to_csv(out / "llps" / "llps_requested_table.csv", index=False)

    dpr_outputs = build_dpr_outputs(
        args=args,
        out=out,
        phasepro_truths=phasepro_truths,
        phasepro_records=phasepro_records,
        ppmc_negative_records=ppmc_negative_records,
        plan_d_negative_records=plan_d_negative_records,
    )

    structure_audit = build_structure_audit(args.checkpoint)
    summary = {
        "status": "PASS",
        "output_root": str(out),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "protocol": {
            "llps_head": "frozen PhaseFlow LLPS head; DPR fine-tuning did not train the LLPS backbone, PhaseFlow bridge, or frozen bridge inputs",
            "llps_phaseflow_reference_checkpoint": str(PHASEFLOW_LLPS_REFERENCE_CKPT.resolve()),
            "llps_phaseflow_reference_checkpoint_sha256": file_sha256(PHASEFLOW_LLPS_REFERENCE_CKPT),
            "llps_phaseflow_reference_calibrated_checkpoint": str(PHASEFLOW_LLPS_REFERENCE_CALIBRATED_CKPT.resolve()),
            "llps_phaseflow_reference_calibrated_checkpoint_sha256": file_sha256(PHASEFLOW_LLPS_REFERENCE_CALIBRATED_CKPT),
            "dpr_head": "DPR v6 p257 raw profile from final update_000050 checkpoint",
            "threshold_policy": "same fixed threshold for continuous DPR profiles; PSPHunter uses native binary output",
            "threshold": float(args.threshold),
            "phasepro_used_for_checkpoint_or_threshold_selection": False,
            "dpr_negative_eval": "PPMC NP/ND negative proteins for available residue-level profiles",
            "dpr_negative_plan_d_audit": "PlanD non-PhasePro N2/N3 negative proteins are reported separately where PPMC packed features are unavailable for PhaseFlow DPR v6.",
            "fuzdrop_dpr": "unavailable locally for full residue-level PhasePro DPR profile",
        },
        "inputs": {
            "phasepro_root": str(args.phasepro_root.resolve()),
            "plan_d_val_index": str(args.plan_d_val_index.resolve()),
            "ppmc_score_table": str(args.ppmc_score_table.resolve()),
            "dpr_phasepro_profile": str(args.dpr_phasepro_profile.resolve()),
            "dpr_negative_profile": str(args.dpr_negative_profile.resolve()),
            "pstp_phasepro_profile": str(args.pstp_phasepro_profile.resolve()),
            "pstp_negative_profile": str(args.pstp_negative_profile.resolve()),
            "baseline_profile_dir": str(args.baseline_profile_dir.resolve()),
            "ppmc_negative_fasta": str(args.ppmc_negative_fasta.resolve()),
            "pstp_phasepro_h5": str(args.pstp_phasepro_h5.resolve()),
            "pstp_ppmc_negative_h5": str(args.pstp_ppmc_negative_h5.resolve()),
        },
        "truth_audit": {
            "phasepro_proteins": int(len(phasepro_truths)),
            "phasepro_true_regions": int(sum(len(v["regions"]) for v in phasepro_truths.values())),
            "plan_d": plan_d_truth_audit,
            "plan_d_negative_control_proteins": int(len(plan_d_negative_records)),
            "ppmc_np_nd_negative_control_proteins": int(len(ppmc_negative_records)),
        },
        "structure_audit": structure_audit,
        "files": {
            "llps_requested_table": str((out / "llps" / "llps_requested_table.csv").resolve()),
            "dpr_requested_table": str((out / "dpr" / "dpr_requested_table.csv").resolve()),
            "report": str((out / "reports" / "final_overall_benchmark_report.md").resolve()),
            "summary_json": str((out / "final_overall_benchmark_summary.json").resolve()),
        },
        "llps_requested": llps_table.to_dict(orient="records"),
        "dpr_requested": dpr_outputs["requested"].to_dict(orient="records"),
        "dpr_profile_availability": dpr_outputs["availability"].to_dict(orient="records"),
        "dpr_ppmc_negative": dpr_outputs["negative"].to_dict(orient="records"),
        "dpr_plan_d_negative_audit": dpr_outputs["plan_d_negative"].to_dict(orient="records"),
    }

    (out / "final_overall_benchmark_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "reports" / "final_overall_benchmark_report.md").write_text(render_report(summary, llps_table, dpr_outputs["requested"]), encoding="utf-8")
    write_manifest(out)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return 0


def build_llps_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    dataset_labels = {
        "ppmc_any_llps_vs_nd": "Any PPMC LLPS positive versus DisProt disordered negatives",
        "ppmc_any_llps_vs_np": "Any PPMC LLPS positive versus structured PDB negatives",
        "ppmc_any_llps_vs_np_nd": "Any PPMC LLPS positive versus NP+ND negatives",
        "ppmc_client_vs_np_nd": "Exclusive clients versus NP+ND negatives",
        "ppmc_driver_high_confidence_vs_np_nd": "High-confidence drivers versus NP+ND negatives",
        "ppmc_driver_vs_np_nd": "Exclusive drivers versus NP+ND negatives",
    }
    for dataset_id, sub in frame.groupby("panel_id", sort=True):
        for display, model, score_col, note in LLPS_MODEL_MAP:
            rows.append(binary_metrics(sub, dataset_id=str(dataset_id), dataset_label=dataset_labels.get(str(dataset_id), str(dataset_id)), model=model, model_label=display, score_column=score_col, note=note))
    return pd.DataFrame(rows)


def binary_metrics(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_label: str,
    model: str,
    model_label: str,
    score_column: str,
    note: str,
) -> dict[str, Any]:
    labels = pd.to_numeric(frame["llps_label"], errors="coerce")
    scores = pd.to_numeric(frame[score_column], errors="coerce") if score_column in frame.columns else pd.Series(np.nan, index=frame.index)
    valid = labels.notna() & scores.notna()
    yy = labels.loc[valid].astype(int).to_numpy()
    ss = scores.loc[valid].astype(float).to_numpy()
    pred = (ss >= 0.5).astype(int) if len(ss) else np.asarray([], dtype=int)
    out = {
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "model": model,
        "Model": model_label,
        "score_column": score_column,
        "n": int(len(frame)),
        "positive_n": int((labels == 1).sum()),
        "negative_n": int((labels == 0).sum()),
        "available_n": int(valid.sum()),
        "missing_n": int(len(frame) - valid.sum()),
        "coverage": float(valid.mean()) if len(frame) else math.nan,
        "AUROC": math.nan,
        "AUPRC": math.nan,
        "Accuracy": math.nan,
        "Precision": math.nan,
        "Recall": math.nan,
        "F1": math.nan,
        "MCC": math.nan,
        "Recall@FPR5%": math.nan,
        "FPR@Recall90%": math.nan,
        "ECE_10bin": math.nan,
        "Brier": math.nan,
        "note": note,
    }
    if len(yy) and len(np.unique(yy)) == 2 and len(np.unique(ss)) > 1:
        out["AUROC"] = float(roc_auc_score(yy, ss))
        out["AUPRC"] = float(average_precision_score(yy, ss))
        fpr, tpr, _ = roc_curve(yy, ss)
        recalls = tpr[fpr <= 0.05]
        out["Recall@FPR5%"] = float(np.max(recalls)) if len(recalls) else 0.0
        fprs = fpr[tpr >= 0.90]
        out["FPR@Recall90%"] = float(np.min(fprs)) if len(fprs) else math.nan
    if len(yy):
        clipped = np.clip(ss, 0.0, 1.0)
        out["Accuracy"] = float(np.mean(pred == yy))
        out["Precision"] = float(precision_score(yy, pred, zero_division=0))
        out["Recall"] = float(recall_score(yy, pred, zero_division=0))
        out["F1"] = float(f1_score(yy, pred, zero_division=0))
        out["MCC"] = float(matthews_corrcoef(yy, pred))
        out["ECE_10bin"] = float(expected_calibration_error(yy, clipped, bins=10))
        out["Brier"] = float(brier_score_loss(yy, clipped))
    return out


def requested_llps_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for display, model, score_col, _note in LLPS_MODEL_MAP:
        overall = first_metric_row(metrics, model, "ppmc_any_llps_vs_np_nd")
        nd = first_metric_row(metrics, model, "ppmc_any_llps_vs_nd")
        if overall is None:
            continue
        rows.append(
            {
                "Model": display,
                "PPMC overall AUPRC": overall["AUPRC"],
                "AUROC": overall["AUROC"],
                "MCC": overall["MCC"],
                "F1": overall["F1"],
                "Precision": overall["Precision"],
                "Recall": overall["Recall"],
                "ND-only AUPRC": nd["AUPRC"] if nd is not None else math.nan,
                "Recall@FPR5%": overall["Recall@FPR5%"],
                "FPR@Recall90%": overall["FPR@Recall90%"],
                "ECE": overall["ECE_10bin"],
                "Brier": overall["Brier"],
                "coverage": overall["coverage"],
                "score_column": score_col,
            }
        )
    return pd.DataFrame(rows)


def first_metric_row(metrics: pd.DataFrame, model: str, dataset_id: str) -> dict[str, Any] | None:
    sub = metrics.loc[metrics["model"].eq(model) & metrics["dataset_id"].eq(dataset_id)]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def build_dpr_outputs(
    *,
    args: argparse.Namespace,
    out: Path,
    phasepro_truths: dict[str, dict[str, Any]],
    phasepro_records: dict[str, dict[str, Any]],
    ppmc_negative_records: dict[str, dict[str, Any]],
    plan_d_negative_records: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    expected_phasepro = {pid: int(record["length"]) for pid, record in phasepro_records.items()}
    expected_plan_d_negative = {pid: int(record["length"]) for pid, record in plan_d_negative_records.items()}
    model_phasepro: dict[str, dict[str, dict[str, Any]]] = {
        "PhaseFlow": records_with_profiles(phasepro_records, load_npz_profiles(args.dpr_phasepro_profile, expected_phasepro), native_binary=False),
        "PSTP-Scan": records_with_profiles(phasepro_records, load_npz_profiles(args.pstp_phasepro_profile, expected_phasepro), native_binary=False),
    }
    model_negative: dict[str, dict[str, dict[str, Any]]] = {
        "PSTP-Scan": load_pstp_h5_profiles(args.pstp_ppmc_negative_h5, ppmc_negative_records),
    }
    model_plan_d_negative: dict[str, dict[str, dict[str, Any]]] = {
        "PhaseFlow": records_with_profiles(
            plan_d_negative_records,
            load_npz_profiles(args.dpr_negative_profile, expected_plan_d_negative, require_all=True),
            native_binary=False,
        ),
        "PSTP-Scan": records_with_profiles(
            plan_d_negative_records,
            load_npz_profiles(args.pstp_negative_profile, expected_plan_d_negative, require_all=False),
            native_binary=False,
        ),
    }
    model_phasepro.update(load_external_phasepro(args.baseline_profile_dir, phasepro_records))
    model_negative.update(load_external_negative(args.baseline_profile_dir, ppmc_negative_records))

    residue_rows: list[dict[str, Any]] = []
    enrichment_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    negative_per_protein_rows: list[dict[str, Any]] = []
    plan_d_negative_rows: list[dict[str, Any]] = []
    plan_d_negative_per_protein_rows: list[dict[str, Any]] = []
    region_prediction_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []

    phasepro_truth_regions = {pid: item["regions"] for pid, item in phasepro_truths.items()}
    for model in DPR_MODEL_ORDER:
        availability_rows.append(
            {
                "Model": model,
                "phasepro_profiled_proteins": int(len(model_phasepro.get(model, {}))),
                "phasepro_expected_proteins": int(len(phasepro_records)),
                "ppmc_negative_profiled_proteins": int(len(model_negative.get(model, {}))),
                "ppmc_negative_expected_proteins": int(len(ppmc_negative_records)),
                "plan_d_negative_profiled_proteins": int(len(model_plan_d_negative.get(model, {}))),
                "plan_d_negative_expected_proteins": int(len(plan_d_negative_records)),
            }
        )
        if model == "FuzDrop":
            continue
        phase_pred = model_phasepro.get(model, {})
        if phase_pred:
            phase_truths_for_model = {pid: phasepro_truths[pid] for pid in phase_pred if pid in phasepro_truths}
            phase_regions_for_model = {pid: phasepro_truth_regions[pid] for pid in phase_pred if pid in phasepro_truth_regions}
            residue, top_rows = residue_level_metrics(model, phase_pred, phase_truths_for_model)
            residue_rows.append(residue)
            enrichment_rows.extend(top_rows)
            pred_regions = predict_regions_by_model(
                model,
                phase_pred,
                threshold=float(args.threshold),
                smooth_window=int(args.smooth_window),
                merge_gap=int(args.merge_gap),
                min_region_len=int(args.min_region_len),
            )
            region_rows.append(
                region_metrics_from_regions(
                    model,
                    pred_regions,
                    phase_pred,
                    phase_regions_for_model,
                    threshold=float(args.threshold),
                    smooth_window=int(args.smooth_window),
                    merge_gap=int(args.merge_gap),
                    min_region_len=int(args.min_region_len),
                )
            )
            region_prediction_rows.extend(flatten_region_predictions(model, pred_regions))
        neg_pred = model_negative.get(model, {})
        if neg_pred:
            neg_regions = predict_regions_by_model(
                model,
                neg_pred,
                threshold=float(args.threshold),
                smooth_window=int(args.smooth_window),
                merge_gap=int(args.merge_gap),
                min_region_len=int(args.min_region_len),
            )
            metrics, rows = negative_region_metrics(
                model,
                neg_regions,
                neg_pred,
                dataset="ppmc_np_nd_negative",
                expected_protein_n=len(ppmc_negative_records),
                threshold=float(args.threshold),
                smooth_window=int(args.smooth_window),
                merge_gap=int(args.merge_gap),
                min_region_len=int(args.min_region_len),
            )
            negative_rows.append(metrics)
            negative_per_protein_rows.extend(rows)
        plan_d_neg_pred = model_plan_d_negative.get(model, {})
        if plan_d_neg_pred:
            plan_d_neg_regions = predict_regions_by_model(
                model,
                plan_d_neg_pred,
                threshold=float(args.threshold),
                smooth_window=int(args.smooth_window),
                merge_gap=int(args.merge_gap),
                min_region_len=int(args.min_region_len),
            )
            metrics, rows = negative_region_metrics(
                model,
                plan_d_neg_regions,
                plan_d_neg_pred,
                dataset="plan_d_n2_n3_negative",
                expected_protein_n=len(plan_d_negative_records),
                threshold=float(args.threshold),
                smooth_window=int(args.smooth_window),
                merge_gap=int(args.merge_gap),
                min_region_len=int(args.min_region_len),
            )
            plan_d_negative_rows.append(metrics)
            plan_d_negative_per_protein_rows.extend(rows)

    residue_df = pd.DataFrame(residue_rows)
    enrichment_df = pd.DataFrame(enrichment_rows)
    region_df = pd.DataFrame(region_rows)
    negative_df = pd.DataFrame(negative_rows)
    negative_per_df = pd.DataFrame(negative_per_protein_rows)
    plan_d_negative_df = pd.DataFrame(plan_d_negative_rows)
    plan_d_negative_per_df = pd.DataFrame(plan_d_negative_per_protein_rows)
    region_pred_df = pd.DataFrame(region_prediction_rows)
    availability_df = pd.DataFrame(availability_rows)
    requested = requested_dpr_table(residue_df, region_df, negative_df)
    for name, frame in [
        ("dpr_phasepro_residue_metrics.csv", residue_df),
        ("dpr_phasepro_positive_residue_enrichment.csv", enrichment_df),
        ("dpr_phasepro_region_metrics.csv", region_df),
        ("dpr_phasepro_region_predictions.csv", region_pred_df),
        ("dpr_ppmc_negative_region_metrics.csv", negative_df),
        ("dpr_ppmc_negative_region_per_protein.csv", negative_per_df),
        ("dpr_plan_d_negative_region_metrics.csv", plan_d_negative_df),
        ("dpr_plan_d_negative_region_per_protein.csv", plan_d_negative_per_df),
        ("dpr_profile_availability.csv", availability_df),
        ("dpr_requested_table.csv", requested),
    ]:
        frame.to_csv(out / "dpr" / name, index=False)
    return {
        "residue": residue_df,
        "enrichment": enrichment_df,
        "region": region_df,
        "negative": negative_df,
        "plan_d_negative": plan_d_negative_df,
        "availability": availability_df,
        "requested": requested,
    }


def load_external_phasepro(root: Path, records: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "PSPHunter": load_psphunter_profiles(root / "psphunter_phasepro_driving_regions.txt", records),
        "catGRANULE/PLAAC": load_jsonl_profiles(root / "catgranule2_phasepro_profiles.jsonl.gz", records),
    }


def load_external_negative(root: Path, records: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "PSPHunter": load_psphunter_profiles(root / "psphunter_ppmc_negative_driving_regions.txt", records),
        "catGRANULE/PLAAC": load_jsonl_profiles(root / "catgranule2_ppmc_negative_profiles.jsonl.gz", records),
    }


def records_from_phasepro(proteins: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in proteins.itertuples(index=False):
        pid = str(row.protein_id)
        out[pid] = {
            "protein_id": pid,
            "sequence": str(row.sequence),
            "length": int(row.sequence_length),
        }
    return out


def records_from_plan_d(candidate: pd.DataFrame, truths: dict[str, dict[str, Any]], *, negative_only: bool) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pid, item in truths.items():
        label = np.asarray(item["label"], dtype=np.int8)
        if negative_only and int(label.sum()) != 0:
            continue
        out[pid] = {
            "protein_id": pid,
            "sequence": "",
            "length": int(len(label)),
        }
    return out


def records_from_fasta(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequence = "".join(chunks)
                    records[current_id] = {"protein_id": current_id, "sequence": sequence, "length": len(sequence)}
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        sequence = "".join(chunks)
        records[current_id] = {"protein_id": current_id, "sequence": sequence, "length": len(sequence)}
    return records


def load_npz_profiles(path: Path, expected_lengths: dict[str, int], *, require_all: bool = True) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    keys = set(str(k) for k in z.files)
    missing = sorted(set(expected_lengths) - keys)
    extra = sorted(keys - set(expected_lengths))
    if missing and require_all:
        raise RuntimeError(f"Profile key mismatch for {path}: missing={missing[:10]}")
    profiles: dict[str, np.ndarray] = {}
    for pid, length in sorted(expected_lengths.items()):
        if pid not in keys:
            continue
        arr = np.asarray(z[pid], dtype=np.float32).reshape(-1)
        if len(arr) != int(length):
            raise RuntimeError(f"Profile length mismatch for {pid} in {path}: got {len(arr)} expected {length}")
        if not np.isfinite(arr).all():
            raise RuntimeError(f"Non-finite profile values for {pid} in {path}")
        profiles[pid] = np.clip(arr, 0.0, 1.0)
    if extra and len(expected_lengths) <= 200:
        # PhasePro should be exact. PlanD negative profiles can have positives/weak bags too.
        pass
    return profiles


def load_pstp_h5_profiles(path: Path, records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    import h5py

    out: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        for pid, record in records.items():
            if pid not in handle or "pstp_scan_score" not in handle[pid]:
                continue
            out[pid] = {
                **record,
                "dpr_scores": fit_length(np.asarray(handle[pid]["pstp_scan_score"], dtype=np.float32), int(record["length"])),
                "native_binary_regions": False,
            }
    return out


def records_with_profiles(records: dict[str, dict[str, Any]], profiles: dict[str, np.ndarray], *, native_binary: bool) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pid, record in records.items():
        if pid not in profiles:
            continue
        out[pid] = {
            **record,
            "dpr_scores": fit_length(np.asarray(profiles[pid], dtype=np.float32), int(record["length"])),
            "native_binary_regions": bool(native_binary),
        }
    return out


def load_jsonl_profiles(path: Path, records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    opener = gzip.open if path.suffix == ".gz" else open
    out: dict[str, dict[str, Any]] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            pid = str(payload.get("protein_id"))
            if pid not in records:
                continue
            out[pid] = {
                **records[pid],
                "dpr_scores": fit_length(np.asarray(payload.get("score", []), dtype=np.float32), int(records[pid]["length"])),
                "native_binary_regions": False,
            }
    return out


def load_psphunter_profiles(path: Path, records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    current_id: str | None = None
    flags: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#Sequecing ID:") or line.startswith("#Sequencing ID:"):
                if current_id is not None and current_id in records:
                    out[current_id] = {
                        **records[current_id],
                        "dpr_scores": fit_length(np.asarray(flags, dtype=np.float32), int(records[current_id]["length"])),
                        "native_binary_regions": True,
                    }
                current_id = line.split(":", 1)[1].strip()
                flags = []
                continue
            if line.startswith("#") or line.lower().startswith("pos"):
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                try:
                    flags.append(float(parts[3]))
                except ValueError:
                    flags.append(0.0)
    if current_id is not None and current_id in records:
        out[current_id] = {
            **records[current_id],
            "dpr_scores": fit_length(np.asarray(flags, dtype=np.float32), int(records[current_id]["length"])),
            "native_binary_regions": True,
        }
    return out


def residue_level_metrics(
    model: str,
    predictions: dict[str, dict[str, Any]],
    truths: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ys: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    top_rows: list[dict[str, Any]] = []
    for pid, item in predictions.items():
        if pid not in truths:
            continue
        score = np.asarray(item["dpr_scores"], dtype=np.float32)
        y = np.asarray(truths[pid]["label"], dtype=np.int8)
        if len(y) != len(score):
            score = fit_length(score, len(y))
        ys.append(y.astype(int))
        scores.append(score.astype(float))
        for frac in (0.05, 0.10):
            top_rows.append(top_enrichment_row(model, pid, y, score, frac))
    y_all = np.concatenate(ys) if ys else np.asarray([], dtype=int)
    score_all = np.concatenate(scores) if scores else np.asarray([], dtype=float)
    out = {
        "model": model,
        "dataset": "phasepro_full",
        "protein_n": int(len(predictions)),
        "residue_n": int(len(y_all)),
        "positive_residue_n": int(y_all.sum()) if len(y_all) else 0,
        "positive_residue_fraction": float(y_all.mean()) if len(y_all) else math.nan,
        "residue_auroc": math.nan,
        "residue_auprc": math.nan,
        "spearman": math.nan,
        "top5_mean_positive_residue_enrichment": math.nan,
        "top10_mean_positive_residue_enrichment": math.nan,
    }
    if len(y_all) and len(np.unique(y_all)) == 2 and len(np.unique(score_all)) > 1:
        out["residue_auroc"] = float(roc_auc_score(y_all, score_all))
        out["residue_auprc"] = float(average_precision_score(y_all, score_all))
        out["spearman"] = float(pd.Series(score_all).corr(pd.Series(y_all), method="spearman"))
    top_frame = pd.DataFrame(top_rows)
    if not top_frame.empty:
        out["top5_mean_positive_residue_enrichment"] = float(top_frame.loc[top_frame["top_fraction"].eq(0.05), "positive_residue_enrichment"].mean(skipna=True))
        out["top10_mean_positive_residue_enrichment"] = float(top_frame.loc[top_frame["top_fraction"].eq(0.10), "positive_residue_enrichment"].mean(skipna=True))
    return out, top_rows


def top_enrichment_row(model: str, pid: str, y: np.ndarray, score: np.ndarray, fraction: float) -> dict[str, Any]:
    length = len(score)
    k = max(1, int(math.ceil(float(fraction) * max(1, length))))
    order = np.argsort(-score, kind="stable")[:k]
    base = float(y.mean()) if length else math.nan
    top = float(y[order].mean()) if len(order) else math.nan
    return {
        "model": model,
        "protein_id": pid,
        "top_fraction": float(fraction),
        "top_k": int(k),
        "background_positive_fraction": base,
        "top_positive_fraction": top,
        "positive_residue_enrichment": float(top / base) if base and math.isfinite(base) else math.nan,
    }


def predict_regions_by_model(
    model: str,
    predictions: dict[str, dict[str, Any]],
    *,
    threshold: float,
    smooth_window: int,
    merge_gap: int,
    min_region_len: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for pid, item in predictions.items():
        score = np.asarray(item["dpr_scores"], dtype=np.float32)
        if item.get("native_binary_regions"):
            regions = binary_mask_to_regions(score >= 0.5)
        else:
            regions = scores_to_regions(score, threshold=threshold, smooth_window=smooth_window, merge_gap=merge_gap, min_region_len=min_region_len)
        out[pid] = regions
    return out


def region_metrics_from_regions(
    model: str,
    pred_by_id: dict[str, list[dict[str, Any]]],
    predictions: dict[str, dict[str, Any]],
    truth_by_id: dict[str, list[dict[str, Any]]],
    *,
    threshold: float,
    smooth_window: int,
    merge_gap: int,
    min_region_len: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": model,
        "dataset": "phasepro_full",
        "protein_n": int(len(predictions)),
        "threshold": math.nan if model == "PSPHunter" else float(threshold),
        "threshold_label": "native" if model == "PSPHunter" else f"{threshold:.3f}",
        "smooth_window": 1 if model == "PSPHunter" else int(smooth_window),
        "merge_gap": 0 if model == "PSPHunter" else int(merge_gap),
        "min_region_len": 1 if model == "PSPHunter" else int(min_region_len),
        "true_region_n": int(sum(len(v) for v in truth_by_id.values())),
        "pred_region_n": int(sum(len(v) for v in pred_by_id.values())),
    }
    total_true_len = 0
    total_pred_len = 0
    best_ious: list[float] = []
    best_dices: list[float] = []
    boundary_errors: list[float] = []
    for pid, truth in truth_by_id.items():
        pred = pred_by_id.get(pid, [])
        total_true_len += sum(interval_len(t) for t in truth)
        total_pred_len += sum(interval_len(p) for p in pred)
        for true_region in truth:
            ious = [interval_iou(p, true_region) for p in pred]
            dices = [interval_dice(p, true_region) for p in pred]
            best_idx = int(np.argmax(ious)) if ious else -1
            best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0
            best_ious.append(best_iou)
            best_dices.append(float(dices[best_idx]) if best_idx >= 0 else 0.0)
            if best_idx >= 0 and best_iou > 0:
                matched = pred[best_idx]
                boundary_errors.append(0.5 * (abs(int(matched["start"]) - int(true_region["start"])) + abs(int(matched["end"]) - int(true_region["end"]))))
    out["mean_best_iou"] = float(np.mean(best_ious)) if best_ious else math.nan
    out["mean_best_dice"] = float(np.mean(best_dices)) if best_dices else math.nan
    out["boundary_mae_matched_iou_gt0"] = float(np.mean(boundary_errors)) if boundary_errors else math.nan
    out["predicted_length_ratio"] = float(total_pred_len / total_true_len) if total_true_len else math.nan
    for thr in (0.1, 0.25, 0.5):
        precision, recall, f1 = precision_recall_f1_at_iou(pred_by_id, truth_by_id, thr)
        label = str(thr).replace(".", "_")
        out[f"region_precision_iou_{label}"] = precision
        out[f"region_recall_iou_{label}"] = recall
        out[f"segment_f1_iou_{label}"] = f1
    return out


def negative_region_metrics(
    model: str,
    pred_by_id: dict[str, list[dict[str, Any]]],
    predictions: dict[str, dict[str, Any]],
    *,
    dataset: str,
    expected_protein_n: int,
    threshold: float,
    smooth_window: int,
    merge_gap: int,
    min_region_len: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for pid, item in predictions.items():
        score = np.asarray(item["dpr_scores"], dtype=np.float32)
        pred = pred_by_id.get(pid, [])
        pred_len = int(sum(interval_len(p) for p in pred))
        rows.append(
            {
                "model": model,
                "protein_id": pid,
                "length": int(item["length"]),
                "predicted_region_n": int(len(pred)),
                "predicted_dpr_length": pred_len,
                "predicted_dpr_fraction": float(pred_len / max(1, int(item["length"]))),
                "max_residue_score": float(np.max(score)) if len(score) else math.nan,
                "mean_residue_score": float(np.mean(score)) if len(score) else math.nan,
                "pred_regions_1based": intervals_to_text(pred),
            }
        )
    frame = pd.DataFrame(rows)
    metrics = {
        "model": model,
        "dataset": str(dataset),
        "protein_n": int(len(frame)),
        "expected_protein_n": int(expected_protein_n),
        "coverage": float(len(frame) / max(1, int(expected_protein_n))),
        "threshold": math.nan if model == "PSPHunter" else float(threshold),
        "threshold_label": "native" if model == "PSPHunter" else f"{threshold:.3f}",
        "smooth_window": 1 if model == "PSPHunter" else int(smooth_window),
        "merge_gap": 0 if model == "PSPHunter" else int(merge_gap),
        "min_region_len": 1 if model == "PSPHunter" else int(min_region_len),
        "fraction_negative_proteins_with_predicted_dpr": float(frame["predicted_region_n"].gt(0).mean()) if not frame.empty else math.nan,
        "mean_predicted_dpr_length_in_negatives": float(frame["predicted_dpr_length"].mean()) if not frame.empty else math.nan,
        "mean_predicted_dpr_fraction_in_negatives": float(frame["predicted_dpr_fraction"].mean()) if not frame.empty else math.nan,
        "mean_max_residue_score_in_negatives": float(frame["max_residue_score"].mean()) if not frame.empty else math.nan,
        "global_max_residue_score_in_negatives": float(frame["max_residue_score"].max()) if not frame.empty else math.nan,
    }
    return metrics, rows


def requested_dpr_table(residue: pd.DataFrame, region: pd.DataFrame, negative: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in DPR_MODEL_ORDER:
        if model == "FuzDrop":
            rows.append(dpr_empty_row(model, "not_available: no local full PhasePro residue-level FuzDrop DPR profile asset/runner was found"))
            continue
        r = first_model_row(residue, model)
        g = first_model_row(region, model)
        n = first_model_row(negative, model)
        if r is None or g is None:
            rows.append(dpr_empty_row(model, "not_available: profile not found or no overlap with PhasePro truth"))
            continue
        rows.append(
            {
                "Model": model,
                "residue AUROC": r.get("residue_auroc", math.nan),
                "residue AUPRC": r.get("residue_auprc", math.nan),
                "Spearman": r.get("spearman", math.nan),
                "top5 enrichment": r.get("top5_mean_positive_residue_enrichment", math.nan),
                "top10 enrichment": r.get("top10_mean_positive_residue_enrichment", math.nan),
                "region recall": g.get("region_recall_iou_0_25", math.nan),
                "region precision": g.get("region_precision_iou_0_25", math.nan),
                "IoU@0.1 recall": g.get("region_recall_iou_0_1", math.nan),
                "IoU@0.25 segment F1": g.get("segment_f1_iou_0_25", math.nan),
                "IoU@0.5 segment F1": g.get("segment_f1_iou_0_5", math.nan),
                "Dice": g.get("mean_best_dice", math.nan),
                "boundary MAE": g.get("boundary_mae_matched_iou_gt0", math.nan),
                "predicted length ratio": g.get("predicted_length_ratio", math.nan),
                "negative false DPR": n.get("fraction_negative_proteins_with_predicted_dpr", math.nan) if n is not None else math.nan,
                "mean negative DPR length": n.get("mean_predicted_dpr_length_in_negatives", math.nan) if n is not None else math.nan,
                "max residue score in negatives": n.get("global_max_residue_score_in_negatives", math.nan) if n is not None else math.nan,
                "negative coverage": n.get("coverage", math.nan) if n is not None else math.nan,
                "threshold_label": g.get("threshold_label", ""),
                "note": model_note(model),
            }
        )
    return pd.DataFrame(rows)


def dpr_empty_row(model: str, note: str) -> dict[str, Any]:
    return {
        "Model": model,
        "residue AUROC": math.nan,
        "residue AUPRC": math.nan,
        "Spearman": math.nan,
        "top5 enrichment": math.nan,
        "top10 enrichment": math.nan,
        "region recall": math.nan,
        "region precision": math.nan,
        "IoU@0.1 recall": math.nan,
        "IoU@0.25 segment F1": math.nan,
        "IoU@0.5 segment F1": math.nan,
        "Dice": math.nan,
        "boundary MAE": math.nan,
        "predicted length ratio": math.nan,
        "negative false DPR": math.nan,
        "mean negative DPR length": math.nan,
        "max residue score in negatives": math.nan,
        "negative coverage": math.nan,
        "threshold_label": "",
        "note": note,
    }


def first_model_row(frame: pd.DataFrame, model: str) -> dict[str, Any] | None:
    if frame.empty or "model" not in frame:
        return None
    sub = frame.loc[frame["model"].eq(model)]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def model_note(model: str) -> str:
    if model == "PhaseFlow":
        return "final DPR v6 raw p257 profile, update_000050; fixed threshold; PPMC negative unavailable because PPMC NP/ND proteins are absent from v6 offline packed features"
    if model == "PSTP-Scan":
        return "PSTP no-PhasePro selected-family p257 profile for PhasePro; offline PSTP-Scan H5 for PPMC negative; fixed threshold"
    if model == "PSPHunter":
        return "offline PSPHunter native binary driving-region output; no threshold tuning"
    if model == "catGRANULE/PLAAC":
        return "catGRANULE2 residue profile; PLAAC separate profile unavailable, legacy combined row kept"
    return ""


def flatten_region_predictions(model: str, pred_by_id: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid, regions in sorted(pred_by_id.items()):
        for idx, region in enumerate(regions, start=1):
            rows.append(
                {
                    "model": model,
                    "protein_id": pid,
                    "pred_region_index": idx,
                    "start_0based": int(region["start"]),
                    "end_0based_inclusive": int(region["end"]),
                    "start_1based": int(region["start"]) + 1,
                    "end_1based": int(region["end"]) + 1,
                    "length": interval_len(region),
                    "score": float(region.get("score", math.nan)),
                }
            )
    return rows


def precision_recall_f1_at_iou(
    pred_by_id: dict[str, list[dict[str, Any]]],
    truth_by_id: dict[str, list[dict[str, Any]]],
    iou_threshold: float,
) -> tuple[float, float, float]:
    tp = 0
    pred_total = 0
    true_total = 0
    for pid, truth in truth_by_id.items():
        pred = pred_by_id.get(pid, [])
        pred_total += len(pred)
        true_total += len(truth)
        matched_pred: set[int] = set()
        for true_region in truth:
            best_idx = -1
            best_iou = 0.0
            for idx, pred_region in enumerate(pred):
                if idx in matched_pred:
                    continue
                iou = interval_iou(pred_region, true_region)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                tp += 1
                matched_pred.add(best_idx)
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / true_total if true_total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def binary_mask_to_regions(mask: np.ndarray) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.astype(bool)):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            regions.append({"start": int(start), "end": int(idx - 1), "score": 1.0})
            start = None
    if start is not None:
        regions.append({"start": int(start), "end": int(len(mask) - 1), "score": 1.0})
    return regions


def interval_len(region: dict[str, Any]) -> int:
    return max(0, int(region["end"]) - int(region["start"]) + 1)


def interval_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    start = max(int(a["start"]), int(b["start"]))
    end = min(int(a["end"]), int(b["end"]))
    intersection = max(0, end - start + 1)
    union = interval_len(a) + interval_len(b) - intersection
    return float(intersection / union) if union else 0.0


def interval_dice(a: dict[str, Any], b: dict[str, Any]) -> float:
    start = max(int(a["start"]), int(b["start"]))
    end = min(int(a["end"]), int(b["end"]))
    intersection = max(0, end - start + 1)
    denom = interval_len(a) + interval_len(b)
    return float(2.0 * intersection / denom) if denom else 0.0


def intervals_to_text(regions: list[dict[str, Any]]) -> str:
    return ";".join(f"{int(r['start']) + 1}-{int(r['end']) + 1}" for r in regions)


def fit_length(arr: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if len(arr) == length:
        return np.clip(arr, 0.0, 1.0)
    if len(arr) > length:
        return np.clip(arr[:length], 0.0, 1.0)
    padded = np.zeros(length, dtype=np.float32)
    padded[: len(arr)] = arr
    return np.clip(padded, 0.0, 1.0)


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, *, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        mask = (scores >= lo) & (scores <= hi if idx == bins - 1 else scores < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return float(ece)


def build_structure_audit(checkpoint: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("model", payload.get("state_dict", payload)))
    reference = torch.load(PHASEFLOW_LLPS_REFERENCE_CKPT, map_location="cpu", weights_only=False)
    reference_state = reference.get("model", reference.get("model_state_dict", reference.get("state_dict", reference)))
    legacy_llps_prefix = "phase" + "gt."
    final_llps_state = {name[len(legacy_llps_prefix) :]: tensor for name, tensor in state.items() if name.startswith(legacy_llps_prefix)}
    llps_reference_match = state_dict_tensors_equal(final_llps_state, reference_state)
    groups: dict[str, int] = {}
    for name, tensor in state.items():
        if not hasattr(tensor, "numel"):
            continue
        group = name.split(".", 1)[0]
        groups[group] = groups.get(group, 0) + int(tensor.numel())
    trainable_like = sum(count for group, count in groups.items() if group in {"v6", "v6_feature_projectors"})
    legacy_llps_group = "phase" + "gt"
    protected_like = sum(count for group, count in groups.items() if group in {legacy_llps_group, "phaseflow", "phaseflow_bridge"})
    trainable_state = payload.get("dpr_v6_trainable_state_dict", {})
    dpr_v6_state = payload.get("dpr_v6_state_dict", {})
    return {
        "state_dict_parameter_groups": groups,
        "dpr_head_state_parameters": int(trainable_like),
        "dpr_v6_state_parameters": int(sum(int(tensor.numel()) for tensor in dpr_v6_state.values() if hasattr(tensor, "numel"))),
        "dpr_v6_trainable_state_parameters": int(sum(int(tensor.numel()) for tensor in trainable_state.values() if hasattr(tensor, "numel"))),
        "protected_phaseflow_llps_phaseflow_bridge_state_parameters": int(protected_like),
        "phaseflow_llps_reference_checkpoint": str(PHASEFLOW_LLPS_REFERENCE_CKPT.resolve()),
        "phaseflow_llps_reference_checkpoint_sha256": file_sha256(PHASEFLOW_LLPS_REFERENCE_CKPT),
        "final_phaseflow_llps_matches_reference_checkpoint": bool(llps_reference_match),
        "final_phaseflow_llps_parameter_tensors": int(len(final_llps_state)),
        "reference_phaseflow_llps_parameter_tensors": int(len(reference_state)),
        "source_freeze_rule": "phaseflow/full_length/models/dpr_v6.py freezes the PhaseFlow LLPS backbone, PhaseFlow bridge, and bridge inputs; only v6./v6_feature_projectors are trainable.",
    }


def state_dict_tensors_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    import torch

    if set(left) != set(right):
        return False
    for key in left:
        a = left[key]
        b = right[key]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            continue
        if tuple(a.shape) != tuple(b.shape):
            return False
        if not torch.equal(a.detach().cpu(), b.detach().cpu()):
            return False
    return True


def render_report(summary: dict[str, Any], llps: pd.DataFrame, dpr: pd.DataFrame) -> str:
    availability = pd.DataFrame(summary.get("dpr_profile_availability", []))
    plan_d_negative = pd.DataFrame(summary.get("dpr_plan_d_negative_audit", []))
    structure = dict(summary.get("structure_audit", {}))
    lines = [
        "# final overall benchmark 报告",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- checkpoint_sha256: `{summary['checkpoint_sha256']}`",
        "- LLPS: 使用 final DPR v6 内冻结的 PhaseFlow LLPS head，对应 `phaseflow_region_global_score`。",
        f"- PhaseFlow LLPS 权重核对: final checkpoint 内冻结 LLPS backbone 与参考 checkpoint 逐张量一致 = `{structure.get('final_phaseflow_llps_matches_reference_checkpoint')}`。",
        "- DPR: 使用 final update_000050 raw p257 profile；连续 profile 用固定阈值 0.5，PSPHunter 用原生二值区间。",
        "- PhasePro 只用于最终评估，不用于 checkpoint 或阈值选择；DPR negative false-DPR 主表使用 PPMC NP/ND 负蛋白中release artifact 中已有 residue profile 的模型。",
        "- PhaseFlow final v6 目前没有 PPMC NP/ND 的 offline packed features，因此不能公平生成 PPMC negative DPR profile；已单独输出 PlanD 非 PhasePro N2/N3 negative audit。",
        "",
        "## LLPS",
        "",
        markdown_table(llps),
        "",
        "## DPR",
        "",
        markdown_table(dpr),
        "",
        "## DPR Profile Coverage",
        "",
        markdown_table(availability),
        "",
        "## PlanD Negative Audit",
        "",
        markdown_table(plan_d_negative),
        "",
        "## 输出文件",
        "",
        f"- `llps/llps_requested_table.csv`",
        f"- `dpr/dpr_requested_table.csv`",
        f"- `dpr/dpr_phasepro_residue_metrics.csv`",
        f"- `dpr/dpr_phasepro_region_metrics.csv`",
        f"- `dpr/dpr_ppmc_negative_region_metrics.csv`",
        f"- `dpr/dpr_plan_d_negative_region_metrics.csv`",
        f"- `dpr/dpr_profile_availability.csv`",
        f"- `final_overall_benchmark_summary.json`",
        "",
        "## 注意",
        "",
        "- FuzDrop 没有release artifact 中可复用的全长 residue-level DPR profile/runner，本次 DPR 表不填充，避免把蛋白级 LLPS score 误作为 region benchmark。",
        "- `catGRANULE/PLAAC` 行由 catGRANULE2 profile 填充；PLAAC 单独 profile 不可用，沿用历史合并行名。",
    ]
    return "\n".join(lines) + "\n"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root: Path) -> None:
    rows = []
    manifest_path = root / "manifests" / "benchmark_file_manifest.csv"
    for path in sorted(root.rglob("*")):
        if path.is_file():
            if path.resolve() == manifest_path.resolve():
                continue
            rel = path.relative_to(root)
            rows.append({"path": str(rel), "size": path.stat().st_size, "sha256": file_sha256(path)})
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
