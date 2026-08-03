
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.protein.analysis.analyze_dpr_thresholds import (
    per_protein_metrics,
    select_best_threshold,
    threshold_curve,
    threshold_free_metrics,
    threshold_metrics,
    to_jsonable,
)


POSITIVE_REGION_TIERS = {"S1_CAUSAL_REGION", "S2_VALIDATED_REGION", "REGION_S1_S2"}
NEGATIVE_TIERS = {"N2_DISORDERED_NEGATIVE", "N3_STRUCTURED_NEGATIVE"}
WEAK_BAG_TIERS = {"W1_SELF_DRIVER_BAG", "W2_CONTEXT_DRIVER_BAG"}


def load_candidate_index(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = {
        "sampler_tier",
        "protein_id",
        "sequence_sha256",
        "supervision_id",
        "region_start",
        "region_end",
        "sequence_length",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"candidate index missing columns: {missing}")
    frame["sampler_tier"] = frame["sampler_tier"].astype(str)
    frame["protein_id"] = frame["protein_id"].astype(str)
    frame["sequence_sha256"] = frame["sequence_sha256"].astype(str)
    for column in ("region_start", "region_end", "sequence_length"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    return frame


def build_residue_truths(candidate: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build residue-level validation labels from Plan D.

    S1/S2 rows have explicit positive regions. N2/N3 rows are all-negative
    residue labels. W1/W2 rows are bag-level positives only, so they are audited
    but excluded from residue-threshold selection unless the same protein also
    has S/N residue-level evidence.
    """
    grouped: dict[str, dict[str, Any]] = {}
    conflicts: list[str] = []
    for row in candidate.itertuples(index=False):
        pid = str(row.protein_id)
        tier = str(row.sampler_tier)
        length = int(row.sequence_length)
        if length <= 0:
            continue
        item = grouped.setdefault(
            pid,
            {
                "label": np.zeros(length, dtype=np.int8),
                "regions": [],
                "sequence": "",
                "gene_name": pid,
                "protein_name": pid,
                "sequence_sha256": str(row.sequence_sha256),
                "tiers": set(),
                "positive_region_evidence": 0,
                "negative_evidence": 0,
                "weak_bag_evidence": 0,
                "residue_eligible": False,
            },
        )
        if int(item["label"].shape[0]) != length:
            conflicts.append(pid)
            max_len = max(int(item["label"].shape[0]), length)
            old = item["label"]
            new_label = np.zeros(max_len, dtype=np.int8)
            new_label[: len(old)] = old
            item["label"] = new_label
        item["tiers"].add(tier)
        if tier in POSITIVE_REGION_TIERS:
            start = int(row.region_start)
            end = int(row.region_end)
            if start < 0 or end <= start or end > len(item["label"]):
                raise RuntimeError(f"invalid region in validation candidate {pid}:{start}-{end} length={len(item['label'])}")
            item["label"][start:end] = 1
            item["positive_region_evidence"] += 1
            item["regions"].append(
                {
                    "region_id": str(row.supervision_id),
                    "start": start,
                    "end": end,
                    "start_1based": start + 1,
                    "end_1based": end,
                    "tier": tier,
                }
            )
        elif tier in NEGATIVE_TIERS:
            item["negative_evidence"] += 1
        elif tier in WEAK_BAG_TIERS:
            item["weak_bag_evidence"] += 1
    truths: dict[str, dict[str, Any]] = {}
    excluded_weak_only = 0
    for pid, item in sorted(grouped.items()):
        item["tiers"] = sorted(str(x) for x in item["tiers"])
        item["residue_eligible"] = bool(item["positive_region_evidence"] or item["negative_evidence"])
        if item["residue_eligible"]:
            truths[pid] = item
        else:
            excluded_weak_only += 1
    positive = int(sum(1 for item in truths.values() if int(np.asarray(item["label"]).sum()) > 0))
    negative = int(sum(1 for item in truths.values() if int(np.asarray(item["label"]).sum()) == 0))
    audit = {
        "candidate_rows": int(len(candidate)),
        "candidate_unique_proteins": int(candidate["protein_id"].nunique()),
        "truth_unique_proteins": int(len(truths)),
        "positive_region_proteins": positive,
        "negative_residue_proteins": negative,
        "excluded_weak_bag_only_proteins": int(excluded_weak_only),
        "length_conflict_count": int(len(set(conflicts))),
        "length_conflict_examples": sorted(set(conflicts))[:20],
        "tier_counts": {str(k): int(v) for k, v in candidate["sampler_tier"].value_counts().sort_index().items()},
        "residue_count": int(sum(len(item["label"]) for item in truths.values())),
        "positive_residue_count": int(sum(int(np.asarray(item["label"]).sum()) for item in truths.values())),
    }
    audit["positive_residue_fraction"] = float(audit["positive_residue_count"] / max(1, audit["residue_count"]))
    return truths, audit


def restrict_to_common_profiles(
    profiles: dict[str, np.ndarray],
    truths: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, Any]]:
    common = sorted(set(profiles) & set(truths))
    missing_profile = sorted(set(truths) - set(profiles))
    extra_profile = sorted(set(profiles) - set(truths))
    return (
        {pid: np.asarray(profiles[pid], dtype=np.float32) for pid in common},
        {pid: truths[pid] for pid in common},
        {
            "common_proteins": int(len(common)),
            "missing_profile_count": int(len(missing_profile)),
            "missing_profile_examples": missing_profile[:20],
            "extra_profile_count": int(len(extra_profile)),
            "extra_profile_examples": extra_profile[:20],
        },
    )


def evaluate_profiles_for_threshold_selection(
    profiles: dict[str, np.ndarray],
    truths: dict[str, dict[str, Any]],
    *,
    fixed_threshold: float = 0.5,
    objective: str = "MCC",
) -> dict[str, Any]:
    profiles, truths, coverage = restrict_to_common_profiles(profiles, truths)
    if not profiles:
        raise RuntimeError("no common residue-level validation profiles")
    per = per_protein_metrics(profiles, truths)
    tf = threshold_free_metrics(profiles, truths, per)
    fixed = threshold_metrics(profiles, truths, threshold=float(fixed_threshold))
    curve = threshold_curve(profiles, truths, extra_thresholds=[float(fixed_threshold), 1.0])
    selected = select_best_threshold(curve, objective=objective)
    selected_threshold = float(selected["threshold"])
    tuned = threshold_metrics(profiles, truths, threshold=selected_threshold)
    tuned.update(
        {
            "threshold": selected_threshold,
            "selection_objective": f"external Plan D validation residue-level {objective}",
            "selection_row": selected,
        }
    )
    fixed["threshold"] = float(fixed_threshold)
    return {
        "coverage": coverage,
        "per_protein": per,
        "threshold_free": tf,
        "fixed": fixed,
        "threshold_curve": curve,
        "selected": tuned,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_value(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, math.nan))
    except (TypeError, ValueError):
        return math.nan
    return value
