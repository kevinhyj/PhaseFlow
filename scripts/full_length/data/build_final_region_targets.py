from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final stratified DPR region targets from teacher profiles.")
    parser.add_argument("--teacher-scores", default="artifacts/data/pseudo_labels/round0_external/teacher_scores.h5")
    parser.add_argument("--out", default="artifacts/data/processed/final_region_targets.h5")
    parser.add_argument("--report", default="artifacts/data/processed/final_region_targets_report.json")
    parser.add_argument("--feature-dir", action="append", default=[])
    parser.add_argument(
        "--policy",
        choices=("stratified", "pstp_scan"),
        default="stratified",
        help="Target construction policy. pstp_scan uses only PSTP-Scan residue profiles for DPR locate supervision.",
    )
    parser.add_argument("--use-phaseflow", action="store_true", help="Opt in to PhaseFlow teacher scores for ablation runs.")
    parser.add_argument("--phaseflow-pos", type=float, default=0.62)
    parser.add_argument("--phasemotif-pos", type=float, default=0.70)
    parser.add_argument("--pstp-pos", type=float, default=0.75)
    parser.add_argument("--catgranule-pos", type=float, default=0.70)
    parser.add_argument("--psphunter-key-pos", type=float, default=0.65)
    parser.add_argument("--consensus-pos", type=float, default=0.55)
    parser.add_argument("--confidence-pos", type=float, default=0.55)
    parser.add_argument("--consensus-neg", type=float, default=0.25)
    parser.add_argument("--phaseflow-neg", type=float, default=0.20)
    parser.add_argument("--phasemotif-neg", type=float, default=0.15)
    parser.add_argument("--pstp-neg", type=float, default=0.20)
    parser.add_argument("--catgranule-neg", type=float, default=0.58)
    parser.add_argument("--psphunter-neg", type=float, default=0.05)
    parser.add_argument("--disorder-hard-neg", type=float, default=0.55)
    parser.add_argument("--min-pos-len", type=int, default=6)
    parser.add_argument("--min-neg-len", type=int, default=10)
    parser.add_argument("--merge-gap", type=int, default=3)
    parser.add_argument("--boundary-radius", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    teacher_path = Path(args.teacher_scores)
    out_path = Path(args.out)
    report_path = Path(args.report)
    feature_dirs = [Path(path) for path in args.feature_dir]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "teacher_scores": str(teacher_path),
        "out": str(out_path),
        "feature_dirs": [str(path) for path in feature_dirs],
        "proteins": 0,
        "residues": 0,
        "positive_residues": 0,
        "negative_residues": 0,
        "key_positive_residues": 0,
        "boundary_positive_residues": 0,
        "positive_spans": 0,
        "negative_spans": 0,
        "with_disorder": 0,
        "policy": str(args.policy),
        "use_phaseflow": bool(args.use_phaseflow),
    }
    with h5py.File(teacher_path, "r") as src, h5py.File(out_path, "w") as dst:
        dst.attrs["source"] = str(teacher_path)
        dst.attrs["policy"] = _target_policy_name(args)
        dst.attrs["use_phaseflow"] = int(bool(args.use_phaseflow) and str(args.policy) == "stratified")
        for protein_id in sorted(src.keys()):
            group = src[protein_id]
            target = build_targets_for_group(str(protein_id), group, feature_dirs, args)
            out_group = dst.create_group(str(protein_id))
            for key, value in target.items():
                if isinstance(value, np.ndarray):
                    out_group.create_dataset(key, data=value.astype(np.float32, copy=False), compression="gzip")
            out_group.attrs["positive_spans_json"] = json.dumps(target["positive_spans"])
            out_group.attrs["negative_spans_json"] = json.dumps(target["negative_spans"])
            out_group.attrs["length"] = int(target["length"])
            out_group.attrs["has_disorder"] = int(bool(target["has_disorder"]))
            out_group.attrs["target_policy"] = _target_policy_name(args)
            report["proteins"] += 1
            report["residues"] += int(target["length"])
            target_values = np.asarray(target["region_teacher_target"], dtype=np.float32)
            target_weights = np.asarray(target["region_teacher_weight"], dtype=np.float32)
            supervised = np.isfinite(target_values) & (target_weights > 0)
            report["positive_residues"] += int(np.sum(supervised & (target_values >= 0.5)))
            report["negative_residues"] += int(np.sum(supervised & (target_values < 0.5)))
            report["key_positive_residues"] += int(np.sum(target["region_key_target"] == 1.0))
            report["boundary_positive_residues"] += int(np.sum(target["region_boundary_target"] == 1.0))
            report["positive_spans"] += len(target["positive_spans"])
            report["negative_spans"] += len(target["negative_spans"])
            report["with_disorder"] += int(bool(target["has_disorder"]))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


def build_targets_for_group(
    protein_id: str,
    group: h5py.Group,
    feature_dirs: list[Path],
    args: argparse.Namespace,
) -> dict[str, object]:
    if str(args.policy) == "pstp_scan":
        return build_pstp_scan_targets_for_group(protein_id, group, feature_dirs, args)
    length = _profile_length(group)
    phasemotif = _read_profile(group, "phasemotif_score", length, np.nan)
    psphunter = _read_profile(group, "psphunter_key_score", length, np.nan)
    pstp = _read_profile(group, "pstp_scan_score", length, np.nan)
    catgranule = _read_profile(group, "catgranule_score", length, np.nan)
    if bool(args.use_phaseflow):
        consensus = _read_profile(group, "teacher_consensus", length, np.nan)
        confidence = _read_profile(group, "teacher_confidence", length, 0.0)
        phaseflow = _read_profile(group, "phaseflow_score", length, np.nan)
    else:
        consensus, confidence = _consensus_without_phaseflow(
            length,
            [
                (pstp, 1.2),
                (phasemotif, 1.0),
                (catgranule, 1.0),
                (psphunter, 1.0),
            ],
        )
        phaseflow = np.full(length, np.nan, dtype=np.float32)
    disorder = _read_disorder_profile(protein_id, feature_dirs, length)

    phaseflow_pos = phaseflow >= float(args.phaseflow_pos) if bool(args.use_phaseflow) else np.zeros(length, dtype=bool)
    phasemotif_pos = phasemotif >= float(args.phasemotif_pos)
    pstp_pos = pstp >= float(args.pstp_pos)
    catgranule_pos = catgranule >= float(args.catgranule_pos)
    key_pos = psphunter >= float(args.psphunter_key_pos)
    consensus_pos = (consensus >= float(args.consensus_pos)) & (confidence >= float(args.confidence_pos))
    vote_count = (
        phaseflow_pos.astype(np.int16)
        + phasemotif_pos.astype(np.int16)
        + pstp_pos.astype(np.int16)
        + catgranule_pos.astype(np.int16)
    )
    positive_seed = (
        ((vote_count >= 2) & (consensus >= 0.45))
        | (consensus_pos & (phasemotif_pos | key_pos | phaseflow_pos))
    )
    if bool(args.use_phaseflow):
        positive_seed |= phaseflow_pos & (phasemotif_pos | key_pos | pstp_pos)
    positive_spans = _spans_from_mask(positive_seed, min_len=int(args.min_pos_len), merge_gap=int(args.merge_gap))
    positive_mask = _mask_from_spans(length, positive_spans)

    low_all = (
        (np.nan_to_num(consensus, nan=0.0) <= float(args.consensus_neg))
        & (np.nan_to_num(phasemotif, nan=0.0) <= float(args.phasemotif_neg))
        & (np.nan_to_num(pstp, nan=0.0) <= float(args.pstp_neg))
        & (np.nan_to_num(catgranule, nan=0.0) <= float(args.catgranule_neg))
        & (np.nan_to_num(psphunter, nan=0.0) <= float(args.psphunter_neg))
    )
    if bool(args.use_phaseflow):
        low_all &= np.nan_to_num(phaseflow, nan=0.0) <= float(args.phaseflow_neg)
    if disorder is not None:
        hard_negative_seed = low_all & (disorder >= float(args.disorder_hard_neg))
    else:
        hard_negative_seed = low_all & (confidence >= 0.35)
    hard_negative_seed &= ~positive_mask
    negative_spans = _spans_from_mask(hard_negative_seed, min_len=int(args.min_neg_len), merge_gap=int(args.merge_gap))
    negative_mask = _mask_from_spans(length, negative_spans)

    region_target = np.full(length, np.nan, dtype=np.float32)
    region_weight = np.zeros(length, dtype=np.float32)
    pos_weight = 0.45 + 0.12 * vote_count.astype(np.float32) + 0.30 * np.nan_to_num(confidence, nan=0.0)
    pos_weight = np.clip(pos_weight, 0.35, 1.0)
    region_target[positive_mask] = 1.0
    region_weight[positive_mask] = pos_weight[positive_mask]
    region_target[negative_mask] = 0.0
    region_weight[negative_mask] = 0.45 if disorder is not None else 0.30

    key_target = np.full(length, np.nan, dtype=np.float32)
    key_weight = np.zeros(length, dtype=np.float32)
    key_positive = key_pos & (positive_mask | (consensus >= 0.45))
    key_target[key_positive] = 1.0
    key_weight[key_positive] = np.clip(np.nan_to_num(psphunter[key_positive], nan=0.0), 0.35, 1.0)
    key_target[negative_mask] = 0.0
    key_weight[negative_mask] = np.maximum(key_weight[negative_mask], 0.25)

    boundary_target = np.full(length, np.nan, dtype=np.float32)
    boundary_weight = np.zeros(length, dtype=np.float32)
    for start, end in positive_spans:
        _write_window(boundary_target, boundary_weight, start, int(args.boundary_radius), 1.0, 0.8)
        _write_window(boundary_target, boundary_weight, end, int(args.boundary_radius), 1.0, 0.8)
        inner_start = min(end, start + int(args.boundary_radius) + 1)
        inner_end = max(start, end - int(args.boundary_radius) - 1)
        if inner_end >= inner_start:
            boundary_target[inner_start : inner_end + 1] = 0.0
            boundary_weight[inner_start : inner_end + 1] = np.maximum(boundary_weight[inner_start : inner_end + 1], 0.25)
    boundary_target[negative_mask] = 0.0
    boundary_weight[negative_mask] = np.maximum(boundary_weight[negative_mask], 0.25)

    contrast_target = np.full(length, np.nan, dtype=np.float32)
    contrast_weight = np.zeros(length, dtype=np.float32)
    contrast_target[positive_mask] = 1.0
    contrast_weight[positive_mask] = np.maximum(region_weight[positive_mask], 0.5)
    contrast_target[negative_mask] = 0.0
    contrast_weight[negative_mask] = np.maximum(region_weight[negative_mask], 0.35)

    return {
        "length": length,
        "has_disorder": disorder is not None,
        "positive_spans": [{"start": start, "end": end} for start, end in positive_spans],
        "negative_spans": [{"start": start, "end": end} for start, end in negative_spans],
        "region_teacher_target": region_target,
        "region_teacher_weight": region_weight,
        "region_key_target": key_target,
        "region_key_weight": key_weight,
        "region_boundary_target": boundary_target,
        "region_boundary_weight": boundary_weight,
        "region_contrast_target": contrast_target,
        "region_contrast_weight": contrast_weight,
    }


def build_pstp_scan_targets_for_group(
    protein_id: str,
    group: h5py.Group,
    feature_dirs: list[Path],
    args: argparse.Namespace,
) -> dict[str, object]:
    length = _profile_length(group)
    pstp = _read_profile(group, "pstp_scan_score", length, np.nan)
    disorder = _read_disorder_profile(protein_id, feature_dirs, length)

    pstp_valid = np.isfinite(pstp)
    positive_seed = pstp_valid & (pstp >= float(args.pstp_pos))
    positive_spans = _spans_from_mask(positive_seed, min_len=int(args.min_pos_len), merge_gap=int(args.merge_gap))
    positive_mask = _mask_from_spans(length, positive_spans)

    negative_seed = pstp_valid & (pstp <= float(args.pstp_neg)) & ~positive_mask
    if disorder is not None:
        negative_seed &= disorder >= float(args.disorder_hard_neg)
    negative_spans = _spans_from_mask(negative_seed, min_len=int(args.min_neg_len), merge_gap=int(args.merge_gap))
    negative_mask = _mask_from_spans(length, negative_spans)

    clipped_pstp = np.clip(np.nan_to_num(pstp, nan=0.0), 0.0, 1.0).astype(np.float32, copy=False)
    region_target = np.full(length, np.nan, dtype=np.float32)
    region_weight = np.zeros(length, dtype=np.float32)
    region_target[positive_mask] = clipped_pstp[positive_mask]
    region_weight[positive_mask] = np.clip(0.35 + 0.65 * clipped_pstp[positive_mask], 0.35, 1.0)
    region_target[negative_mask] = clipped_pstp[negative_mask]
    region_weight[negative_mask] = 0.45 if disorder is not None else 0.30

    key_target = np.full(length, np.nan, dtype=np.float32)
    key_weight = np.zeros(length, dtype=np.float32)

    boundary_target = np.full(length, np.nan, dtype=np.float32)
    boundary_weight = np.zeros(length, dtype=np.float32)
    for start, end in positive_spans:
        _write_window(boundary_target, boundary_weight, start, int(args.boundary_radius), 1.0, 0.8)
        _write_window(boundary_target, boundary_weight, end, int(args.boundary_radius), 1.0, 0.8)
        inner_start = min(end, start + int(args.boundary_radius) + 1)
        inner_end = max(start, end - int(args.boundary_radius) - 1)
        if inner_end >= inner_start:
            boundary_target[inner_start : inner_end + 1] = 0.0
            boundary_weight[inner_start : inner_end + 1] = np.maximum(boundary_weight[inner_start : inner_end + 1], 0.25)
    boundary_target[negative_mask] = 0.0
    boundary_weight[negative_mask] = np.maximum(boundary_weight[negative_mask], 0.25)

    contrast_target = np.full(length, np.nan, dtype=np.float32)
    contrast_weight = np.zeros(length, dtype=np.float32)
    contrast_target[positive_mask] = 1.0
    contrast_weight[positive_mask] = np.maximum(region_weight[positive_mask], 0.5)
    contrast_target[negative_mask] = 0.0
    contrast_weight[negative_mask] = np.maximum(region_weight[negative_mask], 0.35)

    positive_span_rows = []
    for start, end in positive_spans:
        span_scores = clipped_pstp[start : end + 1]
        confidence = float(np.mean(span_scores)) if span_scores.size else 1.0
        positive_span_rows.append({"start": start, "end": end, "confidence": confidence, "sample_weight": confidence})

    negative_span_rows = []
    for start, end in negative_spans:
        span_scores = clipped_pstp[start : end + 1]
        confidence = float(1.0 - np.mean(span_scores)) if span_scores.size else 0.5
        negative_span_rows.append({"start": start, "end": end, "confidence": confidence, "sample_weight": 0.45})

    return {
        "length": length,
        "has_disorder": disorder is not None,
        "positive_spans": positive_span_rows,
        "negative_spans": negative_span_rows,
        "region_teacher_target": region_target,
        "region_teacher_weight": region_weight,
        "region_key_target": key_target,
        "region_key_weight": key_weight,
        "region_boundary_target": boundary_target,
        "region_boundary_weight": boundary_weight,
        "region_contrast_target": contrast_target,
        "region_contrast_weight": contrast_weight,
    }


def _target_policy_name(args: argparse.Namespace) -> str:
    if str(args.policy) == "pstp_scan":
        return "pstp_scan_only_multiscale_window_no_gold"
    if bool(args.use_phaseflow):
        return "stratified_phaseflow_phasemotif_psphunter_hard_negative_boundary"
    return "stratified_phasemotif_pstp_catgranule_psphunter_hard_negative_boundary_no_phaseflow"


def _profile_length(group: h5py.Group) -> int:
    for key in ("teacher_consensus", "phaseflow_score", "pstp_scan_score", "catgranule_score"):
        if key in group:
            return int(group[key].shape[0])
    raise ValueError(f"Cannot infer profile length for {group.name}")


def _read_profile(group: h5py.Group, key: str, length: int, fill: float) -> np.ndarray:
    out = np.full(length, fill, dtype=np.float32)
    if key not in group:
        return out
    value = np.asarray(group[key], dtype=np.float32)
    copy_len = min(length, int(value.shape[0]))
    out[:copy_len] = value[:copy_len]
    return out


def _consensus_without_phaseflow(length: int, profiles: list[tuple[np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray]:
    consensus = np.full(length, np.nan, dtype=np.float32)
    confidence = np.zeros(length, dtype=np.float32)
    score_sum = np.zeros(length, dtype=np.float64)
    weight_sum = np.zeros(length, dtype=np.float64)
    total_possible = float(sum(weight for _, weight in profiles if weight > 0.0))
    if total_possible <= 0.0:
        return consensus, confidence
    for profile, weight in profiles:
        if weight <= 0.0:
            continue
        valid = np.isfinite(profile)
        score_sum[valid] += np.clip(profile[valid], 0.0, 1.0).astype(np.float64) * float(weight)
        weight_sum[valid] += float(weight)
    valid = weight_sum > 0
    if not np.any(valid):
        return consensus, confidence
    mean = np.zeros(length, dtype=np.float64)
    mean[valid] = score_sum[valid] / weight_sum[valid]
    var_sum = np.zeros(length, dtype=np.float64)
    for profile, weight in profiles:
        if weight <= 0.0:
            continue
        profile_valid = np.isfinite(profile)
        clipped = np.clip(profile[profile_valid], 0.0, 1.0).astype(np.float64)
        var_sum[profile_valid] += float(weight) * (clipped - mean[profile_valid]) ** 2
    variance = np.zeros(length, dtype=np.float64)
    variance[valid] = var_sum[valid] / weight_sum[valid]
    coverage = np.zeros(length, dtype=np.float64)
    coverage[valid] = np.clip(weight_sum[valid] / max(total_possible, 1.0e-6), 0.0, 1.0)
    agreement = np.ones(length, dtype=np.float64)
    agreement[valid] = np.clip(1.0 - 2.0 * np.sqrt(np.clip(variance[valid], 0.0, 0.25)), 0.0, 1.0)
    consensus[valid] = np.clip(mean[valid], 0.0, 1.0).astype(np.float32)
    confidence[valid] = (coverage[valid] * agreement[valid]).astype(np.float32)
    return consensus, confidence


def _read_disorder_profile(protein_id: str, feature_dirs: list[Path], length: int) -> np.ndarray | None:
    for directory in feature_dirs:
        path = directory / f"{protein_id}.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as handle:
            if "disorder" not in handle:
                return None
            disorder = np.asarray(handle["disorder"], dtype=np.float32)
            if disorder.ndim != 2 or disorder.shape[0] == 0:
                return None
            copy_len = min(length, int(disorder.shape[0]))
            out = np.zeros(length, dtype=np.float32)
            out[:copy_len] = np.max(disorder[:copy_len, : min(3, disorder.shape[1])], axis=1)
            return out
    return None


def _spans_from_mask(mask: np.ndarray, *, min_len: int, merge_gap: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + merge_gap + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [(start, end) for start, end in merged if end - start + 1 >= min_len]


def _mask_from_spans(length: int, spans: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in spans:
        mask[max(0, start) : min(length - 1, end) + 1] = True
    return mask


def _write_window(target: np.ndarray, weight: np.ndarray, center: int, radius: int, value: float, item_weight: float) -> None:
    start = max(0, center - radius)
    end = min(target.shape[0] - 1, center + radius)
    target[start : end + 1] = float(value)
    weight[start : end + 1] = np.maximum(weight[start : end + 1], float(item_weight))


if __name__ == "__main__":
    raise SystemExit(main())
