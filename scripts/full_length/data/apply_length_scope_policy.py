#!/usr/bin/env python3
"""Apply the sequence-quality and length-scope policy to processed manifests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

import augment_train_external_sources as aug


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def first_col(df: pd.DataFrame, *names: str, default: Any = "") -> pd.Series:
    for name in names:
        if name in df.columns:
            value = df[name]
            if isinstance(value, pd.DataFrame):
                return value.iloc[:, 0]
            return value
    return pd.Series(default, index=df.index)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def add_scope_fields(
    df: pd.DataFrame,
    *,
    label_col: str,
    role_col: str,
    tier_col: str,
    notes_col: str = "notes",
    evidence_type_col: str = "evidence_type",
    evidence_level_col: str = "evidence_level",
    status_col: str = "leakage_status",
    context: str = "candidate",
) -> pd.DataFrame:
    out = df.copy()
    seq = first_col(out, "sequence").fillna("").astype(str)
    label = first_col(out, label_col, "llps_label", "final_llps_label", default=-100)
    role = first_col(out, role_col, "role_label", "final_role_label")
    tier = first_col(out, tier_col, "label_tier", "label_quality", "final_label_tier")
    notes = first_col(out, notes_col)
    evidence_type = first_col(out, evidence_type_col)
    evidence_level = first_col(out, evidence_level_col)
    scope_rows = [
        aug.length_scope_fields(
            s,
            llps_label=lab,
            role_label=r,
            label_tier=t,
            notes=n,
            evidence_type=et,
            evidence_level=el,
        )
        for s, lab, r, t, n, et, el in zip(seq, label, role, tier, notes, evidence_type, evidence_level)
    ]
    scope = pd.DataFrame(scope_rows, index=out.index)
    for col in ["seq_valid", "bad_seq", "len_bucket", "train_scope", "teacher_scope"]:
        out[col] = scope[col]
    status = first_col(out, status_col, "final_leakage_status", default="clean").fillna("clean").astype(str)
    hard = [aug.hard_label(lab, r, t) for lab, r, t in zip(label, role, tier)]
    out["skip_reason"] = [
        aug.skip_reason(
            bad_seq=bool(bad),
            train_scope=bool(train_scope),
            teacher_scope=bool(teacher_scope),
            leakage=st not in {"clean", "candidate_unfiltered", ""},
            hard_label=bool(hard_flag) and not bool(train_scope),
            context=context,
        )
        for bad, train_scope, teacher_scope, st, hard_flag in zip(
            out["bad_seq"], out["train_scope"], out["teacher_scope"], status, hard
        )
    ]
    return out


def migrate_full_candidate_pool(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = add_scope_fields(
        df,
        label_col="llps_label_candidate",
        role_col="role_label",
        tier_col="label_tier_candidate",
        status_col="leakage_status",
    )
    old_invalid = df["leakage_status"].fillna("").astype(str).eq("removed_invalid_train_sequence")
    bad = df["bad_seq"].fillna(False).astype(bool)
    df.loc[old_invalid & bad, "leakage_status"] = "bad_seq"
    df.loc[old_invalid & bad, "leakage_reason"] = "bad_seq"
    legal_len_oos = old_invalid & ~bad
    df.loc[legal_len_oos, "leakage_status"] = "clean"
    df.loc[legal_len_oos, "leakage_reason"] = ""
    leakage = df["leakage_status"].fillna("").astype(str).ne("clean")
    df["skip_reason"] = "none"
    df.loc[bad, "skip_reason"] = "bad_seq"
    df.loc[~bad & leakage, "skip_reason"] = "leakage"
    df.loc[~bad & ~leakage & ~df["train_scope"].fillna(False).astype(bool), "skip_reason"] = "len_oos"
    atomic_write_csv(df, path)
    return df


def manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "protein_id" not in out:
        acc = first_col(out, "uniprot_acc").fillna("").astype(str)
        source = first_col(out, "source").fillna("").astype(str)
        md5 = first_col(out, "sequence_md5").fillna("").astype(str)
        out["protein_id"] = acc.where(acc.ne(""), source + "_" + md5.str[:12])
    out["length"] = first_col(out, "sequence").fillna("").astype(str).str.len()
    if "llps_label" not in out:
        out["llps_label"] = first_col(out, "llps_label_candidate", default=-100)
    if "label_tier" not in out:
        out["label_tier"] = first_col(out, "label_tier_candidate", "label_quality", default="unknown")
    if "label_quality" not in out:
        out["label_quality"] = out["label_tier"]
    if "negative_type" not in out:
        out["negative_type"] = first_col(out, "negative_type_candidate", default="none")
    if "region_label_tier" not in out:
        out["region_label_tier"] = first_col(out, "region_label_tier_candidate", default="none")
    if "sample_weight" not in out:
        out["sample_weight"] = 0.0
    if "label_confidence" not in out:
        out["label_confidence"] = pd.to_numeric(out["sample_weight"], errors="coerce").fillna(0.0)
    if "split" not in out:
        out["split"] = ""
    cols = [
        "protein_id",
        "canonical_key",
        "sequence",
        "length",
        "llps_label",
        "sample_weight",
        "label_confidence",
        "label_quality",
        "label_tier",
        "negative_type",
        "role_label",
        "source",
        "split",
        "uniprot_acc",
        "gene_name",
        "organism",
        "taxonomy_id",
        "sequence_md5",
        "region_label_tier",
        "seq_valid",
        "bad_seq",
        "len_bucket",
        "train_scope",
        "teacher_scope",
        "skip_reason",
        "leakage_status",
        "leakage_reason",
        "source_record_id",
        "evidence_type",
        "evidence_level",
        "pmid",
        "notes",
    ]
    for col in cols:
        if col not in out:
            out[col] = ""
    return out[cols]


def write_scope_outputs(root: Path, pool: pd.DataFrame) -> dict[str, int]:
    processed = root / "artifacts/data/processed"
    priority = {"gold": 5, "curated": 4, "silver": 3, "pseudo": 2, "unknown": 1}
    legal = pool[pool["seq_valid"].fillna(False).astype(bool)].copy()
    legal["_tier_priority"] = first_col(legal, "label_tier_candidate", default="unknown").map(priority).fillna(0)
    legal["_weight"] = pd.to_numeric(first_col(legal, "sample_weight", default=0.0), errors="coerce").fillna(0.0)
    legal["_length"] = first_col(legal, "sequence").fillna("").astype(str).str.len()
    candidate = (
        legal.sort_values(["canonical_key", "_tier_priority", "_weight", "_length"], ascending=[True, False, False, False])
        .drop_duplicates("canonical_key", keep="first")
        .drop(columns=["_tier_priority", "_weight", "_length"])
        .copy()
    )
    candidate_manifest = manifest_columns(candidate)
    atomic_write_csv(candidate_manifest, processed / "candidate_manifest.csv")

    clean = candidate[candidate["leakage_status"].fillna("").astype(str).eq("clean")].copy()
    teacher = clean[clean["teacher_scope"].fillna(False).astype(bool)].copy()
    teacher = add_scope_fields(
        teacher,
        label_col="llps_label_candidate",
        role_col="role_label",
        tier_col="label_tier_candidate",
        status_col="leakage_status",
        context="teacher",
    )
    atomic_write_csv(manifest_columns(teacher), processed / "teacher_manifest.csv")

    short_manifest = candidate_manifest[pd.to_numeric(candidate_manifest["length"], errors="coerce").fillna(0).lt(100)].copy()
    long_manifest = candidate_manifest[pd.to_numeric(candidate_manifest["length"], errors="coerce").fillna(0).gt(2048)].copy()
    atomic_write_csv(short_manifest, processed / "short_manifest.csv")
    atomic_write_csv(long_manifest, processed / "long_manifest.csv")

    rows: list[dict[str, Any]] = []

    def add_counts(scope: str, values: pd.Series) -> None:
        for value, count in values.value_counts(dropna=False).sort_index().items():
            rows.append({"scope": scope, "value": str(value), "count": int(count)})

    add_counts("len_bucket", candidate_manifest["len_bucket"])
    add_counts("seq_valid", pool["seq_valid"].fillna(False).astype(bool))
    add_counts("bad_seq", pool["bad_seq"].fillna(False).astype(bool))
    add_counts("train_scope", candidate_manifest["train_scope"].fillna(False).astype(bool))
    add_counts("teacher_scope", candidate_manifest["teacher_scope"].fillna(False).astype(bool))
    add_counts("candidate_skip_reason", candidate_manifest["skip_reason"])
    pd.DataFrame(rows).to_csv(processed / "length_filter_report.csv", index=False)
    return {
        "candidate_manifest": len(candidate_manifest),
        "teacher_manifest": len(teacher),
        "short_manifest": len(short_manifest),
        "long_manifest": len(long_manifest),
    }


def migrate_manifest(path: Path, *, label_col: str, role_col: str, tier_col: str, status_col: str, context: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df = add_scope_fields(df, label_col=label_col, role_col=role_col, tier_col=tier_col, status_col=status_col, context=context)
    atomic_write_csv(df, path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    processed = root / "artifacts/data/processed"

    pool = migrate_full_candidate_pool(processed / "full_candidate_pool.csv")
    migrate_manifest(
        processed / "active_train_manifest.csv",
        label_col="llps_label",
        role_col="role_label",
        tier_col="label_tier",
        status_col="leakage_status",
        context="train",
    )
    migrate_manifest(
        processed / "model_train_manifest.csv",
        label_col="llps_label",
        role_col="role_label",
        tier_col="label_tier",
        status_col="leakage_status",
        context="train",
    )
    migrate_manifest(
        processed / "canonical_protein_pool.csv",
        label_col="final_llps_label",
        role_col="final_role_label",
        tier_col="final_label_tier",
        status_col="final_leakage_status",
        context="candidate",
    )
    stats = write_scope_outputs(root, pool)
    print(stats)


if __name__ == "__main__":
    main()
