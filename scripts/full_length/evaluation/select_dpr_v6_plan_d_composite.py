#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.full_length.evaluation.dpr_v6_plan_d_common import (
    build_residue_truths,
    evaluate_profiles_for_threshold_selection,
    load_candidate_index,
)
from scripts.full_length.evaluation.analyze_dpr_v6_threshold_curves import to_jsonable


DEFAULT_INPUT_ROOTS = [
    ROOT / "artifacts/benchmarks/plan_d_external_val_seed_finalonly",
    ROOT / "artifacts/benchmarks/plan_d_external_val_fair_single_matrix_20260617",
]
DEFAULT_OUT = ROOT / "artifacts/benchmarks/plan_d_fair_single_matrix_20260617_selection"
DEFAULT_PLAN_D_VAL = ROOT / "data/processed/stage2/dpr_v8r1a/indices/sampler_plans/plan_d_mixed_hq_val_candidate_index.parquet"

FREE_METRICS = [
    "val_AUROC",
    "val_AUPRC",
    "val_Spearman",
    "val_per_protein_Spearman_median",
]
THRESHOLDED_METRICS = ["val_selected_MCC", "val_selected_F1"]
PROFILE_SCALES = ("p33", "p129", "p257", "mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one DPR v6 model using only non-PhasePro PlanD validation. "
            "All rows are recomputed on the common subset covered by PSTP so the "
            "DPR/PSTP validation policies are comparable."
        )
    )
    parser.add_argument("--input-root", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--candidate-index", type=Path, default=DEFAULT_PLAN_D_VAL)
    parser.add_argument("--min-common-proteins", type=int, default=200)
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_roots = [path.resolve() for path in (args.input_root or DEFAULT_INPUT_ROOTS)]
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    truths, truth_audit = build_residue_truths(load_candidate_index(args.candidate_index))
    frame = load_metric_rows(input_roots)
    frame = deduplicate_rows(frame)

    pstp_profiles_by_scale = find_pstp_profiles(frame)
    common_ids_by_scale = {
        scale: sorted(set(pstp_profiles_by_scale[scale]) & set(truths))
        for scale in PROFILE_SCALES
    }
    missing_scales = [scale for scale, ids in common_ids_by_scale.items() if len(ids) < int(args.min_common_proteins)]
    if missing_scales:
        details = {scale: len(common_ids_by_scale[scale]) for scale in missing_scales}
        raise RuntimeError(f"PSTP common PlanD subset is too small: {details}")

    recomputed = recompute_metrics_on_common_subset(
        frame=frame,
        truths=truths,
        common_ids_by_scale=common_ids_by_scale,
        fixed_threshold=float(args.fixed_threshold),
    )
    recomputed.to_csv(out / "combined_common_subset_validation_model_metrics.csv", index=False)

    dpr = recomputed.loc[~recomputed["model_name"].astype(str).str.startswith("PSTP")].copy()
    pstp = recomputed.loc[recomputed["model_name"].astype(str).str.startswith("PSTP")].copy()
    if dpr.empty:
        raise RuntimeError("No DPR validation rows found after common-subset recompute")
    if pstp.empty:
        raise RuntimeError("No PSTP validation rows found after common-subset recompute")

    dpr_ranked = add_composite_scores(dpr)
    pstp_ranked = add_composite_scores(pstp)
    eligible = dpr_ranked.loc[dpr_ranked["common_proteins"].ge(int(args.min_common_proteins))].copy()
    if eligible.empty:
        raise RuntimeError(f"No DPR rows with common_proteins >= {args.min_common_proteins}")

    best_dpr = order_by_composite(eligible).iloc[0].to_dict()
    best_pstp = order_by_composite(pstp_ranked).iloc[0].to_dict()

    dpr_top = order_by_composite(dpr_ranked).head(max(1, int(args.top_n))).copy()
    pstp_top = order_by_composite(pstp_ranked).head(min(max(1, int(args.top_n)), len(pstp_ranked))).copy()
    dpr_top.to_csv(out / "top_dpr_plan_d_composite_rows.csv", index=False)
    pstp_top.to_csv(out / "top_pstp_plan_d_composite_rows.csv", index=False)

    payload = {
        "status": "PASS",
        "protocol": {
            "selection_split": "Plan D mixed HQ non-PhasePro validation",
            "phasepro_used_for_selection": False,
            "phaseflow_backbone_contract": "frozen PhaseFlow LLPS and PhaseFlow bridge backbones; trainable DPR v6 head/projectors only",
            "single_model_only": True,
            "same_validation_subset_policy": (
                "All DPR and PSTP rows were recomputed on the same PlanD residue-level protein subset "
                "covered by PSTP for each scale; PhasePro was not loaded."
            ),
            "selection_rule": (
                "predeclared composite = 0.7*mean_percentile(AUROC,AUPRC,Spearman,median per-protein Spearman) "
                "+ 0.3*mean_percentile(MCC,F1), common_proteins>=min_common_proteins"
            ),
            "threshold_policy": "each model threshold selected on PlanD validation by MCC; fixed thresholds remain predefined diagnostics",
            "min_common_proteins": int(args.min_common_proteins),
            "fixed_threshold": float(args.fixed_threshold),
            "scale_policy": "all output scales allowed for both DPR and PSTP, selected by the same composite rule",
        },
        "candidate_index": str(args.candidate_index.resolve()),
        "truth_audit": truth_audit,
        "validation_common_subset_sizes": {scale: int(len(ids)) for scale, ids in common_ids_by_scale.items()},
        "input_roots": [str(path) for path in input_roots],
        "primary_selection": {
            "best_dpr": jsonable(best_dpr),
            "pstp_on_same_validation_subset": jsonable(best_pstp),
        },
        "files": {
            "combined_metrics_csv": str((out / "combined_common_subset_validation_model_metrics.csv").resolve()),
            "top_dpr_csv": str((out / "top_dpr_plan_d_composite_rows.csv").resolve()),
            "top_pstp_csv": str((out / "top_pstp_plan_d_composite_rows.csv").resolve()),
            "report": str((out / "validation_selection_report_all_scales.md").resolve()),
            "summary_json": str((out / "validation_selection_summary_all_scales.json").resolve()),
        },
    }
    (out / "validation_selection_summary_all_scales.json").write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "validation_selection_report_all_scales.md").write_text(render_report(payload, dpr_top, pstp_top), encoding="utf-8")
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), flush=True)
    return 0


def load_metric_rows(input_roots: list[Path]) -> pd.DataFrame:
    paths: list[Path] = []
    for root in input_roots:
        candidates = [root / "validation_model_metrics.csv"]
        candidates.extend(sorted(root.glob("seed_*/validation_model_metrics.csv")))
        candidates.extend(sorted(root.glob("shard_*/validation_model_metrics.csv")))
        for path in candidates:
            if path.exists() and path not in paths:
                paths.append(path)
    if not paths:
        raise RuntimeError(f"No validation_model_metrics.csv found below {[str(x) for x in input_roots]}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path.resolve())
        frame["source_output_root"] = str(path.parent.resolve())
        frame["validation_profile"] = frame.apply(profile_path_for_row, axis=1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def profile_path_for_row(row: pd.Series) -> str:
    root = Path(str(row["source_output_root"]))
    model_name = str(row["model_name"])
    scale = str(row["scale"])
    if model_name.startswith("PSTP"):
        path = root / "profiles" / "pstp_nophasepro" / f"{scale}_profiles.npz"
    else:
        path = root / "profiles" / model_name / f"{scale}_profiles.npz"
    return str(path.resolve())


def deduplicate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for col in ("model_name", "checkpoint_sha256", "variant", "scale", "validation_profile"):
        normalized[col] = normalized[col].fillna("").astype(str)
    normalized["_dedup_key"] = normalized.apply(dedup_key_for_row, axis=1)
    normalized = normalized.sort_values(["_dedup_key", "source_csv", "model_name", "scale"]).drop_duplicates("_dedup_key", keep="first")
    return normalized.drop(columns=["_dedup_key"]).reset_index(drop=True)


def dedup_key_for_row(row: pd.Series) -> str:
    model_name = str(row["model_name"])
    scale = str(row["scale"])
    if model_name.startswith("PSTP"):
        return f"PSTP::{scale}"
    sha = str(row.get("checkpoint_sha256", ""))
    variant = str(row.get("variant", ""))
    if sha:
        return f"DPR::{sha}::{variant}::{scale}"
    return f"DPR_PROFILE::{row['validation_profile']}"


def find_pstp_profiles(frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    pstp = frame.loc[frame["model_name"].astype(str).str.startswith("PSTP")].copy()
    profiles_by_scale: dict[str, dict[str, np.ndarray]] = {}
    for scale in PROFILE_SCALES:
        rows = pstp.loc[pstp["scale"].astype(str).eq(scale)]
        if rows.empty:
            raise RuntimeError(f"Missing PSTP validation row for scale {scale}")
        path = Path(str(rows.iloc[0]["validation_profile"]))
        profiles_by_scale[scale] = load_profiles(path)
    return profiles_by_scale


def recompute_metrics_on_common_subset(
    *,
    frame: pd.DataFrame,
    truths: dict[str, dict[str, Any]],
    common_ids_by_scale: dict[str, list[str]],
    fixed_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cache: dict[Path, dict[str, np.ndarray]] = {}
    for _idx, series in frame.iterrows():
        raw = series.to_dict()
        scale = str(raw["scale"])
        if scale not in PROFILE_SCALES:
            continue
        path = Path(str(raw["validation_profile"]))
        if not path.exists():
            raise RuntimeError(f"Missing validation profile: {path}")
        profiles = cache.setdefault(path, load_profiles(path))
        common_ids = [pid for pid in common_ids_by_scale[scale] if pid in profiles and pid in truths]
        if not common_ids:
            continue
        subset_profiles = {pid: profiles[pid] for pid in common_ids}
        subset_truths = {pid: truths[pid] for pid in common_ids}
        for pid, profile in subset_profiles.items():
            expected = len(subset_truths[pid]["label"])
            if int(np.asarray(profile).reshape(-1).shape[0]) != expected:
                raise RuntimeError(f"Profile length mismatch for {path}:{pid}: got {len(profile)} expected {expected}")
        result = evaluate_profiles_for_threshold_selection(
            subset_profiles,
            subset_truths,
            fixed_threshold=fixed_threshold,
            objective="MCC",
        )
        out_row = dict(raw)
        out_row.update(
            {
                "common_proteins": int(result["coverage"]["common_proteins"]),
                "val_threshold": float(result["selected"]["threshold"]),
                "val_selected_precision": float(result["selected"]["precision"]),
                "val_selected_recall": float(result["selected"]["recall"]),
                "val_selected_F1": float(result["selected"]["F1"]),
                "val_selected_MCC": float(result["selected"]["MCC"]),
                "val_selected_IoU": float(result["selected"]["IoU"]),
                "val_fixed_0.5_F1": float(result["fixed"]["F1"]),
                "val_fixed_0.5_MCC": float(result["fixed"]["MCC"]),
                "val_AUROC": float(result["threshold_free"]["global_residue_AUROC"]),
                "val_AUPRC": float(result["threshold_free"]["global_residue_AUPRC"]),
                "val_Spearman": float(result["threshold_free"]["global_residue_Spearman"]),
                "val_per_protein_AUROC_mean": float(result["threshold_free"]["per_protein_AUROC_mean"]),
                "val_per_protein_Spearman_median": float(result["threshold_free"]["per_protein_Spearman_median"]),
                "val_mean_predicted_fraction_at_threshold": float(result["selected"]["mean_predicted_fraction"]),
                "selection_subset": "PlanD_PSTP_common_by_scale",
                "phasepro_used_for_selection": False,
            }
        )
        rows.append(out_row)
    return pd.DataFrame(rows)


def add_composite_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    for col in FREE_METRICS + THRESHOLDED_METRICS + ["common_proteins"]:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")
    for col in FREE_METRICS + THRESHOLDED_METRICS:
        scored[f"{col}_pct"] = scored[col].rank(method="average", pct=True)
    scored["plan_d_free_mean_pct"] = scored[[f"{col}_pct" for col in FREE_METRICS]].mean(axis=1)
    scored["plan_d_thresholded_mean_pct"] = scored[[f"{col}_pct" for col in THRESHOLDED_METRICS]].mean(axis=1)
    scored["plan_d_reselect_composite"] = 0.7 * scored["plan_d_free_mean_pct"] + 0.3 * scored["plan_d_thresholded_mean_pct"]
    return scored


def order_by_composite(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "plan_d_reselect_composite",
            "val_AUROC",
            "val_AUPRC",
            "val_Spearman",
            "val_per_protein_Spearman_median",
            "val_selected_MCC",
            "val_selected_F1",
            "common_proteins",
            "model_name",
            "scale",
        ],
        ascending=[False, False, False, False, False, False, False, False, True, True],
    ).reset_index(drop=True)


def load_profiles(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    profiles: dict[str, np.ndarray] = {}
    for key in sorted(str(k) for k in z.files):
        arr = np.asarray(z[key], dtype=np.float32).reshape(-1)
        if not np.isfinite(arr).all():
            raise RuntimeError(f"Non-finite values in {path}:{key}")
        profiles[key] = np.clip(arr, 0.0, 1.0)
    return profiles


def render_report(payload: dict[str, Any], dpr_top: pd.DataFrame, pstp_top: pd.DataFrame) -> str:
    best = payload["primary_selection"]["best_dpr"]
    pstp = payload["primary_selection"]["pstp_on_same_validation_subset"]
    lines = [
        "# DPR v6 PlanD Composite Selection",
        "",
        "Protocol: checkpoint/scale/threshold selection used only non-PhasePro PlanD validation. PhasePro was not loaded here.",
        "All validation rows were recomputed on the PlanD protein subset also covered by PSTP.",
        "",
        "## Selected DPR",
        "",
        f"- model: `{best['model_name']}`",
        f"- checkpoint: `{best['checkpoint']}`",
        f"- variant/scale: `{best['variant']}` / `{best['scale']}`",
        f"- validation profile: `{best['validation_profile']}`",
        f"- threshold: `{float(best['val_threshold']):.6f}`",
        f"- composite: `{float(best['plan_d_reselect_composite']):.6f}`",
        f"- validation MCC/F1: `{float(best['val_selected_MCC']):.6f}` / `{float(best['val_selected_F1']):.6f}`",
        f"- validation AUROC/AUPRC/Spearman: `{float(best['val_AUROC']):.6f}` / `{float(best['val_AUPRC']):.6f}` / `{float(best['val_Spearman']):.6f}`",
        "",
        "## Selected PSTP",
        "",
        f"- scale: `{pstp['scale']}`",
        f"- validation profile: `{pstp['validation_profile']}`",
        f"- threshold: `{float(pstp['val_threshold']):.6f}`",
        f"- composite: `{float(pstp['plan_d_reselect_composite']):.6f}`",
        f"- validation MCC/F1: `{float(pstp['val_selected_MCC']):.6f}` / `{float(pstp['val_selected_F1']):.6f}`",
        f"- validation AUROC/AUPRC/Spearman: `{float(pstp['val_AUROC']):.6f}` / `{float(pstp['val_AUPRC']):.6f}` / `{float(pstp['val_Spearman']):.6f}`",
        "",
        "## Top DPR Rows",
        "",
        markdown_table(dpr_top, ["model_name", "variant", "scale", "checkpoint_step", "plan_d_reselect_composite", *FREE_METRICS, *THRESHOLDED_METRICS, "val_threshold"]),
        "",
        "## Top PSTP Rows",
        "",
        markdown_table(pstp_top, ["model_name", "scale", "plan_d_reselect_composite", *FREE_METRICS, *THRESHOLDED_METRICS, "val_threshold"]),
        "",
        "## Files",
        "",
    ]
    for key, value in payload["files"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return ""
    sub = df.loc[:, [col for col in columns if col in df.columns]].copy()
    for col in sub.columns:
        if pd.api.types.is_numeric_dtype(sub[col]):
            sub[col] = sub[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    rows = ["| " + " | ".join(sub.columns) + " |", "| " + " | ".join(["---"] * len(sub.columns)) + " |"]
    for row in sub.itertuples(index=False):
        rows.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(rows)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        return jsonable(value.item())
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


if __name__ == "__main__":
    raise SystemExit(main())
