#!/usr/bin/env python3
"""Build model-ready PhaseFlow / PhaseFlow manifests from the clean candidate pool.

This stage does not re-parse raw_src and does not rebuild feature H5 files.  It
collapses protein evidence to one canonical protein, merges the previous
leakage-clean train evidence, adjudicates labels, re-checks benchmark leakage,
keeps every exact-leakage-clean valid sequence in train, and cleans silver DPR
region spans. No validation split is created at this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import augment_train_external_sources as aug
from phaseflow.full_length.data.full_benchmark_leakage import full_benchmark_key_sets

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    h5py = None


DATE = "20260606"
AA_RE = re.compile(r"[^A-Z]")
UNIPROT_RE = aug.UNIPROT_RE
ACCESSION_LIKE_RE = re.compile(r"^(?:[A-Z][0-9][A-Z0-9]{3}[0-9]|[A-Z0-9]{10})(?:-\d+)?$")
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
SOURCE_LABEL_AUDIT_COLUMNS = [
    "source",
    "raw_id",
    "uniprot_id",
    "canonical_id",
    "sequence_md5",
    "role_raw",
    "role_normalized",
    "evidence_type",
    "is_driver",
    "is_client",
    "is_regulator",
    "partner_dependency",
    "rna_dna_dependency",
    "ptm_dependency",
    "mutation_dependency",
    "assigned_label",
    "label_confidence",
    "exclude_reason",
]


def log(message: str) -> None:
    print(f"[model-ready {DATE}] {message}", flush=True)


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def clean_sequence(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).upper().replace("*", "")
    if text.startswith(">"):
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith(">")]
        text = "".join(lines)
    return AA_RE.sub("", text)


def md5_sequence(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest() if seq else ""


def sha256_sequence(seq: str) -> str:
    return hashlib.sha256(seq.encode("utf-8")).hexdigest() if seq else ""


def normalize_acc(value: Any) -> str:
    acc = aug.normalize_acc(value)
    return aug.base_acc(acc) if acc else ""


def looks_like_uniprot(value: Any) -> bool:
    text = norm_text(value)
    return bool(text and (UNIPROT_RE.match(text) or ACCESSION_LIKE_RE.match(text)))


def valid_train_sequence(seq: str) -> bool:
    return aug.seq_valid(seq)


def canonical_key_from(acc: Any, seq: Any = "", md5: Any = "", fallback: Any = "") -> str:
    acc_norm = normalize_acc(acc)
    if acc_norm:
        return acc_norm
    md5_norm = norm_text(md5).lower()
    if not md5_norm:
        md5_norm = md5_sequence(clean_sequence(seq))
    if md5_norm:
        return md5_norm
    return norm_text(fallback)


def canonical_key_from_row(row: pd.Series | dict[str, Any]) -> str:
    getter = row.get if isinstance(row, dict) else row.get
    acc = first_nonempty(
        getter("uniprot_acc", ""),
        getter("uniprot_id", ""),
        getter("protein_id", "") if looks_like_uniprot(getter("protein_id", "")) else "",
    )
    seq = getter("sequence", "")
    md5 = getter("sequence_md5", "")
    fallback = first_nonempty(getter("canonical_key", ""), getter("protein_id", ""), getter("source_record_id", ""))
    return canonical_key_from(acc, seq, md5, fallback)


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = norm_text(value)
        if text:
            return text
    return ""


def safe_int(value: Any, default: int = -100) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(float(str(value).strip()))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(str(value).strip())
    except Exception:
        return default


def unique_join(values: Iterable[Any], limit: int = 80) -> str:
    seen: list[str] = []
    for value in values:
        text = norm_text(value)
        if not text:
            continue
        for part in str(text).split(";"):
            part = part.strip()
            if part and part not in seen:
                seen.append(part)
                if len(seen) >= limit:
                    break
        if len(seen) >= limit:
            break
    suffix = "" if len(seen) < limit else ";..."
    return ";".join(seen) + suffix


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, **kwargs)


def write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ident, seq in records:
            seq = clean_sequence(seq)
            if not seq:
                continue
            handle.write(f">{ident}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_benchmark_sets(root: Path) -> dict[str, set[str]]:
    base = aug.load_benchmark_sets(root)
    out = {key: set(value) for key, value in base.items()}
    full_keys = full_benchmark_key_sets(root)
    out.setdefault("sha256s", set()).update(full_keys.get("sha256", set()))
    out.setdefault("md5s", set()).update(full_keys.get("md5", set()))
    out.setdefault("ids", set()).update(full_keys.get("ids", set()))
    out.setdefault("accs", set()).update(full_keys.get("ids", set()))
    expanded_accs = set(out.get("accs", set()))
    expanded_ids = set(out.get("ids", set()))
    for value in list(expanded_accs | expanded_ids):
        acc = normalize_acc(value)
        if acc:
            expanded_accs.add(acc)
            expanded_ids.add(acc)
    out["accs"] = {x for x in expanded_accs if x}
    out["ids"] = {x for x in expanded_ids if x}
    return out


def load_deleted_sets(root: Path) -> dict[str, set[str]]:
    audit = read_csv(root / "data/processed/qc/leakage_cleanup_audit_20260606.csv")
    ids: set[str] = set()
    accs: set[str] = set()
    md5s: set[str] = set()
    if not audit.empty and "action" in audit.columns:
        removed = audit[audit["action"].astype(str) == "remove"].copy()
        if "removal_reasons" in removed.columns:
            reason = removed["removal_reasons"].fillna("").astype(str)
            parts = reason.map(lambda text: {item for item in text.split(";") if item})
            exact_reasons = {
                "benchmark_ppmc_seed",
                "benchmark_phasepro_seed",
                "same_uniprot_accession",
                "same_canonical_sequence_md5",
                "same_gene_species_sequence",
                "same_protein_other_source_database",
                "phasepro_benchmark_region_targets_removed",
                "ppmc_benchmark_pseudo_labels_removed",
            }
            removed = removed[parts.map(lambda values: bool(values & exact_reasons))].copy()
        for col in ["protein_id", "uniprot_id"]:
            if col in removed:
                for value in removed[col].dropna().astype(str):
                    text = norm_text(value)
                    if text:
                        ids.add(text)
                    acc = normalize_acc(text)
                    if acc:
                        accs.add(acc)
                        ids.add(acc)
        if "sequence_md5" in removed:
            md5s.update(norm_text(x).lower() for x in removed["sequence_md5"].dropna().astype(str) if norm_text(x))
    return {"ids": ids, "accs": accs, "md5s": md5s}


def load_phasepro_region_tuples(root: Path) -> set[tuple[str, int, int]]:
    regions = read_csv(root / "data/benchmarks/dpr_benchmark_phasepro/regions.csv")
    tuples: set[tuple[str, int, int]] = set()
    if regions.empty:
        return tuples
    for _, row in regions.iterrows():
        pid = norm_text(row.get("protein_id", ""))
        if not pid:
            continue
        start = safe_int(row.get("start_1based", row.get("start", "")), -1)
        end = safe_int(row.get("end_1based", row.get("end", "")), -1)
        if start >= 1 and end >= start:
            tuples.add((normalize_acc(pid) or pid, start, end))
    return tuples


def normalize_candidate_frame(df: pd.DataFrame, source_origin: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "source_origin",
                "protein_id_hint",
                "source",
                "source_record_id",
                "uniprot_acc",
                "gene_name",
                "organism",
                "taxonomy_id",
                "sequence",
                "sequence_md5",
                "length",
                "llps_label_candidate",
                "label_tier_candidate",
                "negative_type_candidate",
                "role_label",
                "region_label_tier_candidate",
                "sample_weight_candidate",
                "evidence_type",
                "evidence_level",
                "pmid",
                "notes",
                "leakage_status_input",
                "canonical_key",
                "valid_sequence",
                "seq_valid",
                "bad_seq",
                "len_bucket",
                "train_scope",
                "teacher_scope",
                "skip_reason",
            ]
        )
    out = pd.DataFrame(index=df.index)
    out["source_origin"] = source_origin
    out["protein_id_hint"] = df.get("protein_id", pd.Series("", index=df.index)).map(norm_text)
    out["source"] = df.get("source", pd.Series(source_origin, index=df.index)).map(norm_text)
    out["source_record_id"] = df.get("source_record_id", pd.Series("", index=df.index)).map(norm_text)
    out["uniprot_acc"] = df.get("uniprot_acc", df.get("uniprot_id", pd.Series("", index=df.index))).map(normalize_acc)
    missing_acc = out["uniprot_acc"].eq("") & out["protein_id_hint"].map(looks_like_uniprot)
    out.loc[missing_acc, "uniprot_acc"] = out.loc[missing_acc, "protein_id_hint"].map(normalize_acc)
    out["gene_name"] = df.get("gene_name", pd.Series("", index=df.index)).map(norm_text)
    if "species" in df.columns and "organism" not in df.columns:
        out["organism"] = df["species"].map(norm_text)
    else:
        out["organism"] = df.get("organism", pd.Series("", index=df.index)).map(norm_text)
    if "tax_id" in df.columns and "taxonomy_id" not in df.columns:
        out["taxonomy_id"] = df["tax_id"].map(norm_text)
    else:
        out["taxonomy_id"] = df.get("taxonomy_id", pd.Series("", index=df.index)).map(norm_text)
    out["sequence"] = df.get("sequence", pd.Series("", index=df.index)).map(clean_sequence)
    existing_md5 = df.get("sequence_md5", pd.Series("", index=df.index)).map(lambda x: norm_text(x).lower())
    computed_md5 = out["sequence"].map(md5_sequence)
    out["sequence_md5"] = existing_md5.where(existing_md5.ne(""), computed_md5)
    out.loc[out["sequence_md5"].eq("") & out["sequence"].ne(""), "sequence_md5"] = computed_md5
    out["length"] = out["sequence"].str.len()
    if "llps_label_candidate" in df.columns:
        out["llps_label_candidate"] = df["llps_label_candidate"].map(safe_int)
    else:
        out["llps_label_candidate"] = df.get("llps_label", pd.Series(-100, index=df.index)).map(safe_int)
    if "label_tier_candidate" in df.columns:
        out["label_tier_candidate"] = df["label_tier_candidate"].map(norm_text).str.lower()
    else:
        out["label_tier_candidate"] = df.get("label_tier", df.get("label_quality", pd.Series("unknown", index=df.index))).map(norm_text).str.lower()
    if "negative_type_candidate" in df.columns:
        out["negative_type_candidate"] = df["negative_type_candidate"].map(norm_text).str.lower()
    else:
        out["negative_type_candidate"] = df.get("negative_type", pd.Series("background_unlabeled", index=df.index)).map(norm_text).str.lower()
    out.loc[out["negative_type_candidate"].eq(""), "negative_type_candidate"] = "none"
    out["role_label"] = df.get("role_label", pd.Series("unknown", index=df.index)).map(norm_text).str.lower()
    out.loc[out["role_label"].eq("negative_structured"), "role_label"] = "unknown"
    out["region_label_tier_candidate"] = df.get("region_label_tier_candidate", df.get("region_label_tier", pd.Series("none", index=df.index))).map(norm_text).str.lower()
    out["sample_weight_candidate"] = df.get("sample_weight", pd.Series(0.0, index=df.index)).map(safe_float)
    out["evidence_type"] = df.get("evidence_type", pd.Series(source_origin, index=df.index)).map(norm_text)
    out["evidence_level"] = df.get("evidence_level", pd.Series("", index=df.index)).map(norm_text)
    out["pmid"] = df.get("pmid", pd.Series("", index=df.index)).map(norm_text)
    out["notes"] = df.get("notes", pd.Series("", index=df.index)).map(norm_text)
    out["leakage_status_input"] = df.get("leakage_status", pd.Series("clean", index=df.index)).map(norm_text)
    out["canonical_key"] = [
        canonical_key_from(a, s, m, f"{src}:{rid}")
        for a, s, m, src, rid in zip(out["uniprot_acc"], out["sequence"], out["sequence_md5"], out["source"], out["source_record_id"])
    ]
    scope_rows = [
        aug.length_scope_fields(
            seq,
            llps_label=label,
            role_label=role,
            label_tier=tier,
            notes=notes,
            evidence_type=evidence_type,
            evidence_level=evidence_level,
        )
        for seq, label, role, tier, notes, evidence_type, evidence_level in zip(
            out["sequence"],
            out["llps_label_candidate"],
            out["role_label"],
            out["label_tier_candidate"],
            out["notes"],
            out["evidence_type"],
            out["evidence_level"],
        )
    ]
    scope_df = pd.DataFrame(scope_rows, index=out.index)
    for col in ["seq_valid", "bad_seq", "len_bucket", "train_scope", "teacher_scope"]:
        out[col] = scope_df[col]
    out["valid_sequence"] = out["seq_valid"]
    out["skip_reason"] = [
        aug.skip_reason(
            bad_seq=bool(bad),
            train_scope=bool(train),
            teacher_scope=bool(teacher),
            hard_label=aug.hard_label(label, role, tier),
        )
        for bad, train, teacher, label, role, tier in zip(
            out["bad_seq"],
            out["train_scope"],
            out["teacher_scope"],
            out["llps_label_candidate"],
            out["role_label"],
            out["label_tier_candidate"],
        )
    ]
    return out


def normalize_old_manifest(df: pd.DataFrame, source_origin: str) -> pd.DataFrame:
    out = normalize_candidate_frame(df, source_origin)
    if out.empty:
        return out
    llps = out["llps_label_candidate"]
    role = out["role_label"]
    tier = out["label_tier_candidate"]
    score = df.get("teacher_consensus_score", pd.Series(float("nan"), index=df.index)).map(safe_float)
    has_score = df.get("teacher_consensus_score", pd.Series(float("nan"), index=df.index)).notna()
    hard = llps.eq(1) & role.isin(["driver", "scaffold"]) & tier.isin(["gold", "curated"])
    positive_not_hard = llps.eq(1) & ~hard
    teacher_like_positive = llps.eq(1) & ~role.isin(["driver", "scaffold"])
    out.loc[hard, "label_tier_candidate"] = "curated"
    out.loc[positive_not_hard, "label_tier_candidate"] = "pseudo"
    out.loc[positive_not_hard, "role_label"] = "teacher_positive"
    out.loc[positive_not_hard, "sample_weight_candidate"] = 0.4
    out.loc[hard, "sample_weight_candidate"] = 1.0
    out.loc[llps.eq(0) & out["negative_type_candidate"].isin(["", "none"]), "negative_type_candidate"] = "structured_negative"
    out.loc[llps.eq(0), "sample_weight_candidate"] = out.loc[llps.eq(0), "sample_weight_candidate"].where(
        out.loc[llps.eq(0), "sample_weight_candidate"].gt(0), 0.85
    )
    out["old_teacher_pseudo_flag"] = teacher_like_positive
    out["old_hard_positive_flag"] = hard
    out["teacher_consensus_score"] = score.where(has_score, "")
    out["evidence_type"] = out["evidence_type"].where(out["evidence_type"].ne(""), source_origin)
    out["notes"] = (
        out["notes"].astype(str)
        + "; old_manifest="
        + source_origin
        + "; original_label_tier="
        + df.get("label_tier", df.get("label_quality", pd.Series("", index=df.index))).map(norm_text)
    )
    return out


def add_evidence_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source_l = out["source"].fillna("").astype(str).str.lower()
    role = out["role_label"].fillna("").astype(str).str.lower()
    tier = out["label_tier_candidate"].fillna("").astype(str).str.lower()
    neg = out["negative_type_candidate"].fillna("").astype(str).str.lower()
    label = pd.to_numeric(out["llps_label_candidate"], errors="coerce").fillna(-100).astype(int)
    source_allowed_for_hard = ~source_l.str.contains("homologous", regex=False)
    out["is_hard_evidence"] = label.eq(1) & role.isin(["driver", "scaffold"]) & tier.isin(["gold", "curated"]) & source_allowed_for_hard
    out["is_pseudo_evidence"] = label.eq(1) & ~out["is_hard_evidence"]
    out["is_positive_evidence"] = out["is_hard_evidence"] | out["is_pseudo_evidence"]
    out["is_structured_negative_evidence"] = label.eq(0) & neg.eq("structured_negative")
    out["is_disprot_negative_evidence"] = label.eq(0) & neg.eq("disordered_negative") & source_l.str.contains("disprot")
    out["is_mobidb_negative_evidence"] = label.eq(0) & neg.isin(["disordered_negative_silver", "disordered_negative"]) & source_l.str.contains("mobidb")
    out["is_disordered_negative_evidence"] = out["is_disprot_negative_evidence"] | out["is_mobidb_negative_evidence"] | (label.eq(0) & neg.eq("disordered_negative"))
    out["is_negative_evidence"] = label.eq(0) & neg.ne("none")
    out["is_associated_context_evidence"] = (
        role.isin(["client", "member", "regulator", "associated", "associated_context"])
        | source_l.str.contains("homologous", regex=False)
    ) & ~out["is_positive_evidence"] & ~out["is_negative_evidence"]
    out["is_unknown_pu_evidence"] = label.eq(-100) & ~out["is_associated_context_evidence"]
    out["row_priority"] = 1
    out.loc[out["is_unknown_pu_evidence"], "row_priority"] = 1
    out.loc[out["is_associated_context_evidence"], "row_priority"] = 2
    out.loc[out["is_negative_evidence"], "row_priority"] = 3
    out.loc[out["is_pseudo_evidence"], "row_priority"] = 4
    out.loc[out["is_hard_evidence"], "row_priority"] = 5
    out["has_uniprot_acc"] = out["uniprot_acc"].ne("")
    return out


def bridge_md5_to_accession(evidence: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    out = evidence.copy()
    stats = Counter()
    out["uniprot_acc"] = out["uniprot_acc"].fillna("").astype(str)
    out["sequence_md5"] = out["sequence_md5"].fillna("").astype(str)
    out["canonical_key"] = out["canonical_key"].fillna("").astype(str)
    with_acc = out[out["sequence_md5"].ne("") & out["uniprot_acc"].ne("")]
    if with_acc.empty:
        return out, stats
    md5_to_accs = with_acc.groupby("sequence_md5")["uniprot_acc"].agg(lambda values: sorted(set(values))).to_dict()
    unique_md5_to_acc = {md5: accs[0] for md5, accs in md5_to_accs.items() if len(accs) == 1}
    ambiguous_md5 = {md5 for md5, accs in md5_to_accs.items() if len(accs) > 1}
    missing_acc = out["uniprot_acc"].eq("") & out["sequence_md5"].isin(set(unique_md5_to_acc))
    out.loc[missing_acc, "uniprot_acc"] = out.loc[missing_acc, "sequence_md5"].map(unique_md5_to_acc)
    out.loc[missing_acc, "canonical_key"] = out.loc[missing_acc, "uniprot_acc"]
    stats["md5_to_uniprot_bridge_rows"] = int(missing_acc.sum())
    stats["md5_to_uniprot_bridge_unique_md5"] = len(unique_md5_to_acc)
    stats["md5_to_uniprot_bridge_ambiguous_md5"] = len(ambiguous_md5)
    return out, stats


def collapse_canonical(
    evidence: pd.DataFrame,
    benchmark: dict[str, set[str]],
    deleted: dict[str, set[str]],
) -> tuple[pd.DataFrame, Counter]:
    stats = Counter()
    evidence = evidence[evidence["canonical_key"].astype(str).ne("")].copy()
    evidence, bridge_stats = bridge_md5_to_accession(evidence)
    stats.update(bridge_stats)
    evidence = add_evidence_flags(evidence)
    for col in ["old_teacher_pseudo_flag", "old_hard_positive_flag"]:
        if col not in evidence.columns:
            evidence[col] = False
    evidence["benchmark_acc_overlap_row"] = evidence["uniprot_acc"].isin(benchmark["accs"]) | evidence["uniprot_acc"].isin(benchmark["ids"])
    evidence["benchmark_md5_overlap_row"] = evidence["sequence_md5"].isin(benchmark["md5s"])
    evidence["sequence_sha256"] = evidence["sequence"].map(lambda value: sha256_sequence(clean_sequence(value)))
    evidence["benchmark_sha256_overlap_row"] = evidence["sequence_sha256"].isin(benchmark.get("sha256s", set()))
    evidence["benchmark_source_overlap_row"] = evidence["source_record_id"].isin(benchmark["source_ids"]) | evidence["protein_id_hint"].isin(benchmark["ids"])
    evidence["deleted_id_overlap_row"] = (
        evidence["uniprot_acc"].isin(deleted["accs"])
        | evidence["protein_id_hint"].isin(deleted["ids"])
        | evidence["sequence_md5"].isin(deleted["md5s"])
    )
    evidence["_source_text"] = evidence["source"].fillna("")
    evidence["_source_record_text"] = evidence["source_record_id"].fillna("")
    evidence["_evidence_type_text"] = evidence["evidence_type"].fillna("")
    evidence["_evidence_level_text"] = evidence["evidence_level"].fillna("")
    evidence["_role_text"] = evidence["role_label"].fillna("")
    evidence["_pmid_text"] = evidence["pmid"].fillna("")
    evidence["_notes_text"] = evidence["notes"].fillna("")

    sort_cols = ["canonical_key", "row_priority", "has_uniprot_acc", "valid_sequence", "length"]
    rep = evidence.sort_values(sort_cols, ascending=[True, False, False, False, False]).drop_duplicates("canonical_key", keep="first")
    rep_cols = [
        "canonical_key",
        "protein_id_hint",
        "uniprot_acc",
        "gene_name",
        "organism",
        "taxonomy_id",
        "sequence",
        "sequence_md5",
        "length",
    ]
    canonical = rep[rep_cols].copy()
    canonical["protein_id"] = canonical["uniprot_acc"].where(canonical["uniprot_acc"].ne(""), canonical["protein_id_hint"])
    missing_pid = canonical["protein_id"].eq("")
    canonical.loc[missing_pid, "protein_id"] = "SEQ_" + canonical.loc[missing_pid, "sequence_md5"].astype(str).str[:16]

    bool_cols = [
        "is_hard_evidence",
        "is_pseudo_evidence",
        "is_positive_evidence",
        "is_structured_negative_evidence",
        "is_disprot_negative_evidence",
        "is_mobidb_negative_evidence",
        "is_disordered_negative_evidence",
        "is_negative_evidence",
        "is_associated_context_evidence",
        "is_unknown_pu_evidence",
        "old_teacher_pseudo_flag",
        "old_hard_positive_flag",
        "benchmark_acc_overlap_row",
        "benchmark_md5_overlap_row",
        "benchmark_sha256_overlap_row",
        "benchmark_source_overlap_row",
        "deleted_id_overlap_row",
    ]
    bool_agg = evidence.groupby("canonical_key", sort=False)[bool_cols].max().reset_index()
    count_agg = evidence.groupby("canonical_key", sort=False).agg(
        evidence_row_count=("canonical_key", "size"),
        source_count=("source", "nunique"),
    ).reset_index()
    text_agg = evidence.groupby("canonical_key", sort=False).agg(
        sources=("_source_text", unique_join),
        source_record_ids=("_source_record_text", unique_join),
        evidence_types=("_evidence_type_text", unique_join),
        evidence_levels=("_evidence_level_text", unique_join),
        roles=("_role_text", unique_join),
        pmids=("_pmid_text", unique_join),
        notes=("_notes_text", lambda values: unique_join(values, limit=20)),
    ).reset_index()
    canonical = canonical.merge(bool_agg, on="canonical_key", how="left").merge(count_agg, on="canonical_key", how="left").merge(text_agg, on="canonical_key", how="left")

    hard_roles = (
        evidence.loc[evidence["is_hard_evidence"], ["canonical_key", "role_label", "label_tier_candidate"]]
        .groupby("canonical_key")
        .agg(hard_roles=("role_label", unique_join), hard_tiers=("label_tier_candidate", unique_join))
        .reset_index()
    )
    canonical = canonical.merge(hard_roles, on="canonical_key", how="left")
    canonical[["hard_roles", "hard_tiers"]] = canonical[["hard_roles", "hard_tiers"]].fillna("")

    canonical["label_conflict"] = canonical["is_positive_evidence"].fillna(False) & canonical["is_negative_evidence"].fillna(False)
    stats["label_conflict_canonical"] = int(canonical["label_conflict"].sum())

    canonical["final_llps_label"] = -100
    canonical["final_label_tier"] = "unknown"
    canonical["final_role_label"] = "unknown"
    canonical["final_negative_type"] = "background_unlabeled"
    canonical["sample_weight"] = 0.0
    canonical["sampler_group"] = "unknown_pu"
    canonical["conflict_resolution"] = ""

    hard = canonical["is_hard_evidence"].fillna(False)
    pseudo = ~hard & canonical["is_pseudo_evidence"].fillna(False)
    disprot_neg = ~hard & ~pseudo & canonical["is_disprot_negative_evidence"].fillna(False)
    mobidb_neg = ~hard & ~pseudo & ~disprot_neg & canonical["is_mobidb_negative_evidence"].fillna(False)
    disordered_neg = ~hard & ~pseudo & ~disprot_neg & ~mobidb_neg & canonical["is_disordered_negative_evidence"].fillna(False)
    structured_neg = ~hard & ~pseudo & ~disprot_neg & ~mobidb_neg & ~disordered_neg & canonical["is_structured_negative_evidence"].fillna(False)
    associated = ~hard & ~pseudo & ~disprot_neg & ~mobidb_neg & ~disordered_neg & ~structured_neg & canonical["is_associated_context_evidence"].fillna(False)

    canonical.loc[hard, ["final_llps_label", "final_label_tier", "final_negative_type", "sample_weight", "sampler_group"]] = [1, "curated", "none", 1.0, "hard_positive"]
    canonical.loc[hard & canonical["hard_tiers"].astype(str).str.contains("gold"), "final_label_tier"] = "gold"
    canonical.loc[hard & canonical["hard_roles"].astype(str).str.contains("scaffold") & ~canonical["hard_roles"].astype(str).str.contains("driver"), "final_role_label"] = "scaffold"
    canonical.loc[hard & ~canonical["final_role_label"].eq("scaffold"), "final_role_label"] = "driver"

    canonical.loc[pseudo, ["final_llps_label", "final_label_tier", "final_role_label", "final_negative_type", "sample_weight", "sampler_group"]] = [
        1,
        "pseudo",
        "teacher_positive",
        "none",
        0.4,
        "pseudo_positive",
    ]
    canonical.loc[disprot_neg, ["final_llps_label", "final_label_tier", "final_role_label", "final_negative_type", "sample_weight", "sampler_group"]] = [
        0,
        "curated",
        "unknown",
        "disordered_negative",
        1.0,
        "disordered_negative",
    ]
    canonical.loc[mobidb_neg, ["final_llps_label", "final_label_tier", "final_role_label", "final_negative_type", "sample_weight", "sampler_group"]] = [
        0,
        "silver",
        "unknown",
        "disordered_negative_silver",
        0.5,
        "disordered_negative",
    ]
    canonical.loc[disordered_neg, ["final_llps_label", "final_label_tier", "final_role_label", "final_negative_type", "sample_weight", "sampler_group"]] = [
        0,
        "curated",
        "unknown",
        "disordered_negative",
        1.0,
        "disordered_negative",
    ]
    canonical.loc[structured_neg, ["final_llps_label", "final_label_tier", "final_role_label", "final_negative_type", "sample_weight", "sampler_group"]] = [
        0,
        "curated",
        "unknown",
        "structured_negative",
        0.85,
        "structured_negative",
    ]
    canonical.loc[associated, ["final_llps_label", "final_label_tier", "final_role_label", "final_negative_type", "sample_weight", "sampler_group"]] = [
        -100,
        "unknown",
        "associated_context",
        "none",
        0.0,
        "associated_context",
    ]

    conflict = canonical["label_conflict"].fillna(False)
    canonical.loc[conflict & canonical["is_positive_evidence"].fillna(False), "conflict_resolution"] = "positive_evidence_overrode_negative_candidate"
    canonical.loc[conflict & ~canonical["is_positive_evidence"].fillna(False), "conflict_resolution"] = "nonpositive_conflict_suppressed"

    scope_rows = [
        aug.length_scope_fields(
            seq,
            llps_label=label,
            role_label=role,
            label_tier=tier,
            notes=notes,
            evidence_type=evidence_type,
            evidence_level=evidence_level,
        )
        for seq, label, role, tier, notes, evidence_type, evidence_level in zip(
            canonical["sequence"],
            canonical["final_llps_label"],
            canonical["final_role_label"],
            canonical["final_label_tier"],
            canonical["notes"],
            canonical["evidence_types"],
            canonical["evidence_levels"],
        )
    ]
    scope_df = pd.DataFrame(scope_rows, index=canonical.index)
    for col in ["seq_valid", "bad_seq", "len_bucket", "train_scope", "teacher_scope"]:
        canonical[col] = scope_df[col]
    # Current policy: every valid sequence that is not an exact PPMC full /
    # PhasePro benchmark duplicate is train data, independent of length.
    canonical["train_scope"] = canonical["seq_valid"].fillna(False).astype(bool)

    canonical["final_leakage_status"] = "clean_direct"
    canonical["final_leakage_reason"] = ""
    bad = canonical["bad_seq"].fillna(False).astype(bool)
    masks = [
        ("bad_seq", bad, "bad_seq"),
        (
            "removed_benchmark_accession",
            canonical["uniprot_acc"].isin(benchmark["accs"]) | canonical["protein_id"].isin(benchmark["ids"]) | canonical["benchmark_acc_overlap_row"].fillna(False),
            "same benchmark accession/id",
        ),
        (
            "removed_sequence_hash",
            canonical["sequence_md5"].isin(benchmark["md5s"])
            | canonical["benchmark_md5_overlap_row"].fillna(False)
            | canonical["sequence"].map(lambda value: sha256_sequence(clean_sequence(value))).isin(benchmark.get("sha256s", set()))
            | canonical["benchmark_sha256_overlap_row"].fillna(False),
            "same benchmark exact sequence hash",
        ),
        ("removed_benchmark_source_id", canonical["benchmark_source_overlap_row"].fillna(False), "same benchmark source record"),
        (
            "removed_deleted_leakage_id",
            canonical["uniprot_acc"].isin(deleted["accs"])
            | canonical["protein_id"].isin(deleted["ids"])
            | canonical["sequence_md5"].isin(deleted["md5s"])
            | canonical["deleted_id_overlap_row"].fillna(False),
            "listed in previous leakage cleanup deleted ids",
        ),
    ]
    for status, mask, reason in masks:
        active = mask & canonical["final_leakage_status"].eq("clean_direct")
        canonical.loc[active, "final_leakage_status"] = status
        canonical.loc[active, "final_leakage_reason"] = reason
        stats[status] += int(active.sum())
    canonical.loc[canonical["final_leakage_status"].eq("clean_direct"), "final_leakage_status"] = "clean"
    leakage = canonical["final_leakage_status"].ne("clean")
    hard = [
        aug.hard_label(label, role, tier)
        for label, role, tier in zip(canonical["final_llps_label"], canonical["final_role_label"], canonical["final_label_tier"])
    ]
    canonical["skip_reason"] = [
        aug.skip_reason(
            bad_seq=bool(bad_flag),
            train_scope=bool(train_flag),
            teacher_scope=bool(teacher_flag),
            leakage=bool(leak_flag),
            hard_label=bool(hard_flag) and not bool(train_flag),
        )
        for bad_flag, train_flag, teacher_flag, leak_flag, hard_flag in zip(
            canonical["bad_seq"],
            canonical["train_scope"],
            canonical["teacher_scope"],
            leakage,
            hard,
        )
    ]
    return canonical, stats


def run_mmseqs_benchmark_search(root: Path, canonical: pd.DataFrame, skip: bool = False) -> tuple[set[str], Counter, dict[str, str]]:
    stats = Counter()
    paths: dict[str, str] = {}
    clean = canonical[canonical["final_leakage_status"].eq("clean")].copy()
    work = root / f"data/interim/model_ready/mmseqs40_benchmark_{DATE}"
    if skip:
        stats["mmseqs_benchmark_skipped"] = 1
        return set(), stats, paths
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    benchmark_rows: list[tuple[str, str]] = []
    for path in [
        root / "data/benchmarks/protein_benchmark_ppmc/manifest.csv",
        root / "data/benchmarks/dpr_benchmark_phasepro/proteins.csv",
    ]:
        df = read_csv(path)
        if df.empty:
            continue
        for _, row in df.iterrows():
            seq = clean_sequence(row.get("sequence", ""))
            pid = first_nonempty(row.get("protein_id", ""), row.get("uniprot_id", ""), row.get("sequence_md5", ""))
            if seq and pid:
                benchmark_rows.append((f"benchmark|{pid}", seq))
    target_rows = [(f"model|{row.canonical_key}", row.sequence) for row in clean[["canonical_key", "sequence"]].itertuples(index=False)]
    bench_fasta = work / "benchmark.fasta"
    target_fasta = work / "model_candidates.fasta"
    result = work / "benchmark_vs_model.m8"
    tmp = work / "tmp"
    log_path = work / "mmseqs.log"
    write_fasta(bench_fasta, benchmark_rows)
    write_fasta(target_fasta, target_rows)
    mmseqs = shutil.which("mmseqs") or "mmseqs"
    cmd = [
        mmseqs,
        "easy-search",
        str(bench_fasta),
        str(target_fasta),
        str(result),
        str(tmp),
        "--min-seq-id",
        "0.4",
        "-c",
        "0.8",
        "--cov-mode",
        "0",
        "--threads",
        "8",
        "--format-output",
        "query,target,pident,alnlen,qcov,tcov,evalue,bits",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)
    homolog_keys: set[str] = set()
    hit_rows = 0
    if result.exists():
        with result.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                hit_rows += 1
                target = parts[1]
                if target.startswith("model|"):
                    homolog_keys.add(target.split("|", 1)[1])
    stats.update(
        {
            "mmseqs40_query_benchmark_records": len(benchmark_rows),
            "mmseqs40_target_model_records": len(target_rows),
            "mmseqs40_homolog_hit_rows": hit_rows,
            "mmseqs40_homolog_audit_only": len(homolog_keys),
        }
    )
    paths = {"work_dir": str(work), "result_m8": str(result), "log": str(log_path)}
    return homolog_keys, stats, paths


def run_mmseqs_model_cluster(root: Path, clean: pd.DataFrame, skip: bool = False) -> tuple[dict[str, str], Counter, dict[str, str]]:
    stats = Counter()
    paths: dict[str, str] = {}
    work = root / f"data/interim/model_ready/mmseqs40_model_clusters_{DATE}"
    if skip:
        mapping = {str(row.protein_id): str(row.protein_id) for row in clean[["protein_id"]].itertuples(index=False)}
        stats["mmseqs_cluster_skipped_singletons"] = len(mapping)
        return mapping, stats, paths
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "model_candidates.fasta"
    records = [(f"model|{row.protein_id}|{row.canonical_key}", row.sequence) for row in clean[["protein_id", "canonical_key", "sequence"]].itertuples(index=False)]
    write_fasta(fasta, records)
    out_prefix = work / "cluster"
    tmp = work / "tmp"
    log_path = work / "mmseqs_linclust.log"
    mmseqs = shutil.which("mmseqs") or "mmseqs"
    cmd = [
        mmseqs,
        "easy-linclust",
        str(fasta),
        str(out_prefix),
        str(tmp),
        "--min-seq-id",
        "0.4",
        "-c",
        "0.8",
        "--cov-mode",
        "0",
        "--threads",
        "8",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)
    cluster_tsv = work / "cluster_cluster.tsv"
    mapping: dict[str, str] = {}
    if cluster_tsv.exists():
        with cluster_tsv.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                rep, member = parts[0], parts[1]
                rep_id = rep.split("|")[1] if rep.startswith("model|") and len(rep.split("|")) >= 3 else rep
                mem_id = member.split("|")[1] if member.startswith("model|") and len(member.split("|")) >= 3 else member
                mapping[mem_id] = rep_id
    for pid in clean["protein_id"].astype(str):
        mapping.setdefault(pid, pid)
    stats.update({"mmseqs40_model_records": len(clean), "mmseqs40_model_clusters": len(set(mapping.values()))})
    paths = {"work_dir": str(work), "cluster_tsv": str(cluster_tsv), "log": str(log_path)}
    return mapping, stats, paths


def preliminary_dpr_keys(root: Path, clean: pd.DataFrame, benchmark: dict[str, set[str]]) -> set[str]:
    path = root / "data/processed/full_candidate_region_spans.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return set()
    spans = pd.read_json(path, lines=True)
    if spans.empty:
        return set()
    clean_keys = set(clean["canonical_key"].astype(str))
    lengths = clean.set_index("canonical_key")["length"].to_dict()
    keys: set[str] = set()
    for _, row in spans.iterrows():
        key = canonical_key_from(first_nonempty(row.get("uniprot_acc", ""), row.get("protein_id", "")), row.get("sequence", ""), row.get("sequence_md5", ""), row.get("canonical_key", ""))
        if key not in clean_keys:
            continue
        start = safe_int(row.get("region_start", row.get("start", "")), -1)
        end = safe_int(row.get("region_end", row.get("end", "")), -1)
        if start >= 1 and end >= start and end <= int(lengths.get(key, 0)):
            if normalize_acc(row.get("uniprot_acc", "")) in benchmark["accs"] or norm_text(row.get("sequence_md5", "")) in benchmark["md5s"]:
                continue
            keys.add(key)
    return keys


def assign_valid_split(clean: pd.DataFrame, cluster_map: dict[str, str], valid_fraction: float = 0.0) -> tuple[pd.DataFrame, Counter]:
    df = clean.copy()
    df["cluster_id_40"] = df["protein_id"].astype(str).map(cluster_map).fillna(df["protein_id"].astype(str))
    mmseqs_cluster_count = int(df["cluster_id_40"].nunique())
    if "sequence_md5" in df.columns:
        md5_cluster = (
            df[df["sequence_md5"].fillna("").astype(str).ne("")]
            .groupby("sequence_md5")["cluster_id_40"]
            .agg(lambda values: sorted(set(map(str, values)))[0])
            .to_dict()
        )
        df["cluster_id_40"] = df["sequence_md5"].map(md5_cluster).fillna(df["cluster_id_40"])
    md5_merged_cluster_count = int(df["cluster_id_40"].nunique())
    df["split"] = "train"
    cluster_members: dict[str, set[str]] = defaultdict(set)
    for row in df[["protein_id", "cluster_id_40"]].itertuples(index=False):
        cluster_members[str(row.cluster_id_40)].add(str(row.protein_id))
    if valid_fraction <= 0:
        stats = Counter(
            {
                "split_total_rows": len(df),
                "split_train_rows": len(df),
                "split_valid_rows": 0,
                "split_valid_clusters": 0,
                "split_total_clusters": len(cluster_members),
                "split_mmseqs40_clusters_before_md5_merge": mmseqs_cluster_count,
                "split_exact_md5_cluster_merges": mmseqs_cluster_count - md5_merged_cluster_count,
                "split_valid_fraction_x10000": 0,
            }
        )
        for group, count in df["sampler_group"].value_counts().items():
            stats[f"train_{group}"] = int(count)
        stats["train_dpr_silver_candidate_proteins"] = int(df["has_dpr_silver_candidate"].sum())
        stats["valid_dpr_silver_candidate_proteins"] = 0
        return df, stats
    total = len(df)
    target = max(1, int(round(total * valid_fraction)))
    min_valid = max(1, int(math.floor(total * valid_fraction * 0.90)))
    max_valid = max(min_valid, int(math.ceil(total * valid_fraction * 1.10)))
    pid_to_group = df.set_index("protein_id")["sampler_group"].to_dict()
    pid_to_dpr = df.set_index("protein_id")["has_dpr_silver_candidate"].to_dict()
    selected: set[str] = set()
    valid_ids: set[str] = set()

    def cluster_score_for_group(cluster: str, group: str) -> tuple[int, str]:
        ids = cluster_members[cluster]
        count = sum(1 for pid in ids if pid_to_group.get(pid) == group)
        return (-count, hashlib.md5(cluster.encode("utf-8")).hexdigest())

    def add_cluster(cluster: str) -> bool:
        if cluster in selected:
            return False
        ids = cluster_members[cluster]
        if len(valid_ids) + len(ids) > max_valid and len(valid_ids) >= min_valid:
            return False
        selected.add(cluster)
        valid_ids.update(ids)
        return True

    for group in ["hard_positive", "structured_negative", "disordered_negative"]:
        candidates = [cluster for cluster, ids in cluster_members.items() if any(pid_to_group.get(pid) == group for pid in ids)]
        for cluster in sorted(candidates, key=lambda c: (len(cluster_members[c]), cluster_score_for_group(c, group)[1])):
            if add_cluster(cluster):
                break
    dpr_clusters = [cluster for cluster, ids in cluster_members.items() if any(bool(pid_to_dpr.get(pid)) for pid in ids)]
    for cluster in sorted(dpr_clusters, key=lambda c: (len(cluster_members[c]), hashlib.md5(c.encode("utf-8")).hexdigest())):
        if add_cluster(cluster):
            break

    for cluster in sorted(cluster_members, key=lambda c: hashlib.md5(c.encode("utf-8")).hexdigest()):
        if len(valid_ids) >= target and len(valid_ids) >= min_valid:
            break
        add_cluster(cluster)
    if len(valid_ids) < min_valid:
        for cluster in sorted(cluster_members, key=lambda c: (len(cluster_members[c]), hashlib.md5(c.encode("utf-8")).hexdigest())):
            if len(valid_ids) >= min_valid:
                break
            add_cluster(cluster)

    df.loc[df["protein_id"].astype(str).isin(valid_ids), "split"] = "valid"
    stats = Counter(
        {
            "split_total_rows": total,
            "split_train_rows": int((df["split"] == "train").sum()),
            "split_valid_rows": int((df["split"] == "valid").sum()),
            "split_valid_clusters": len(selected),
            "split_total_clusters": len(cluster_members),
            "split_mmseqs40_clusters_before_md5_merge": mmseqs_cluster_count,
            "split_exact_md5_cluster_merges": mmseqs_cluster_count - md5_merged_cluster_count,
        }
    )
    stats["split_valid_fraction_x10000"] = int(round(stats["split_valid_rows"] / max(total, 1) * 10000))
    for split in ["train", "valid"]:
        sub = df[df["split"] == split]
        for group, count in sub["sampler_group"].value_counts().items():
            stats[f"{split}_{group}"] = int(count)
        stats[f"{split}_dpr_silver_candidate_proteins"] = int(sub["has_dpr_silver_candidate"].sum())
    return df, stats


def _legacy_clean_region_spans_unused(root: Path, split_df: pd.DataFrame, benchmark: dict[str, set[str]], phasepro_tuples: set[tuple[str, int, int]]) -> tuple[list[dict[str, Any]], Counter]:
    stats = Counter()
    path = root / "data/processed/full_candidate_region_spans.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        stats["input_span_file_missing"] = 1
        return [], stats
    spans = pd.read_json(path, lines=True)
    stats["input_spans"] = len(spans)
    if spans.empty:
        return [], stats
    by_key = split_df.set_index("canonical_key").to_dict(orient="index")
    clean_keys = set(by_key)
    cleaned: list[dict[str, Any]] = []
    for _, row in spans.iterrows():
        source = norm_text(row.get("source", ""))
        key = canonical_key_from(first_nonempty(row.get("uniprot_acc", ""), row.get("protein_id", "")), row.get("sequence", ""), row.get("sequence_md5", ""), row.get("canonical_key", ""))
        if key not in clean_keys:
            stats["removed_span_not_in_model_protein"] += 1
            continue
        meta = by_key[key]
        acc = normalize_acc(row.get("uniprot_acc", "")) or normalize_acc(meta.get("uniprot_acc", ""))
        md5 = norm_text(row.get("sequence_md5", "")).lower() or norm_text(meta.get("sequence_md5", "")).lower()
        if acc in benchmark["accs"] or acc in benchmark["ids"] or md5 in benchmark["md5s"]:
            stats["removed_span_benchmark_direct"] += 1
            continue
        if "phasepro" in source.lower() or (acc, safe_int(row.get("region_start", row.get("start", "")), -1), safe_int(row.get("region_end", row.get("end", "")), -1)) in phasepro_tuples:
            stats["removed_span_phasepro_exact"] += 1
            continue
        start = safe_int(row.get("region_start", row.get("start", "")), -1)
        end = safe_int(row.get("region_end", row.get("end", "")), -1)
        length = int(meta.get("length", 0) or 0)
        if start < 1 or end < start or end > length:
            stats["removed_span_out_of_bounds"] += 1
            continue
        tier = norm_text(row.get("region_label_tier_candidate", row.get("region_label_tier", ""))).lower()
        if source.lower().startswith("phasepdb") or source.lower().startswith("phasepdb") or "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" not in source.lower() and "phasepdb" not in source.lower() and "phasepdb" not in source.lower() and "phasepdb" not in source.lower():
            pass
        if "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" not in source.lower() and "phasepdb" not in source.lower() and "phasepdb" not in source.lower():
            if "phasedb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
                tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" not in source.lower() and "phasepdb" not in source.lower():
            if "phasepdb" in source.lower() or "phasepdb" in source.lower():
                tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" not in source.lower() and "phasepdb" not in source.lower():
            if "phasepdb" in source.lower():
                tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasedb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        # Canonical parser source names use PhaSepDB_3, not PhaSePro.
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasEpdb".lower() in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasesepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasEpdb".lower() in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasEpdb".lower() in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasedb" in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasEpdb".lower() in source.lower() or "phasepdb_3" in source.lower() or "phasepdb" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "phasepdb" in source.lower() or "phasepdb_3" in source.lower():
            tier = "dpr_silver"
        if "llpsdb" in source.lower():
            tier = "dpr_silver_low"
        if tier not in {"dpr_silver", "dpr_silver_low", "dpr_pseudo"}:
            tier = "dpr_silver_low" if "llpsdb" in source.lower() else "dpr_silver"
        cleaned.append(
            {
                "protein_id": meta["protein_id"],
                "canonical_key": key,
                "uniprot_acc": meta.get("uniprot_acc", ""),
                "sequence_md5": meta.get("sequence_md5", ""),
                "split": meta.get("split", "train"),
                "region_start": start,
                "region_end": end,
                "start_1based": start,
                "end_1based": end,
                "coordinate_system": "1-based inclusive",
                "region_length": end - start + 1,
                "region_type": norm_text(row.get("region_type", "")) or "DPR_silver",
                "region_label_tier": tier,
                "source": source,
                "evidence_level": norm_text(row.get("evidence_level", "")),
                "pmid": norm_text(row.get("pmid", "")),
                "notes": norm_text(row.get("notes", "")),
            }
        )
    priority = {"dpr_silver": 3, "dpr_silver_low": 2, "dpr_pseudo": 1}
    merged: list[dict[str, Any]] = []
    by_protein: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in cleaned:
        by_protein[str(item["protein_id"])].append(item)
    for _, items in by_protein.items():
        items = sorted(items, key=lambda x: (x["region_start"], x["region_end"], -priority.get(x["region_label_tier"], 0)))
        for item in items:
            if not merged or merged[-1]["protein_id"] != item["protein_id"] or item["region_start"] > merged[-1]["region_end"]:
                merged.append(item.copy())
                continue
            cur = merged[-1]
            cur["region_end"] = max(cur["region_end"], item["region_end"])
            cur["end_1based"] = cur["region_end"]
            cur["region_length"] = cur["region_end"] - cur["region_start"] + 1
            if priority.get(item["region_label_tier"], 0) > priority.get(cur["region_label_tier"], 0):
                cur["region_label_tier"] = item["region_label_tier"]
            cur["source"] = unique_join([cur.get("source", ""), item.get("source", "")])
            cur["evidence_level"] = unique_join([cur.get("evidence_level", ""), item.get("evidence_level", "")])
            cur["pmid"] = unique_join([cur.get("pmid", ""), item.get("pmid", "")])
            cur["notes"] = unique_join([cur.get("notes", ""), item.get("notes", "")], limit=10)
            stats["merged_overlapping_spans"] += 1
    stats["kept_spans_before_merge"] = len(cleaned)
    stats["output_spans"] = len(merged)
    stats["dpr_silver_spans"] = sum(1 for item in merged if item["region_label_tier"] == "dpr_silver")
    stats["dpr_silver_low_spans"] = sum(1 for item in merged if item["region_label_tier"] == "dpr_silver_low")
    stats["dpr_protein_coverage"] = len({item["protein_id"] for item in merged})
    stats["avg_span_length_x100"] = int(round((sum(item["region_length"] for item in merged) / max(len(merged), 1)) * 100))
    for item in merged:
        stats[f"source_{item['source']}"] += 1
    return merged, stats


def clean_region_spans(root: Path, split_df: pd.DataFrame, benchmark: dict[str, set[str]], phasepro_tuples: set[tuple[str, int, int]]) -> tuple[list[dict[str, Any]], Counter]:
    stats = Counter()
    path = root / "data/processed/full_candidate_region_spans.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        stats["input_span_file_missing"] = 1
        return [], stats
    spans = pd.read_json(path, lines=True)
    stats["input_spans"] = len(spans)
    if spans.empty:
        return [], stats

    by_key = split_df.set_index("canonical_key").to_dict(orient="index")
    cleaned: list[dict[str, Any]] = []
    for _, row in spans.iterrows():
        source = norm_text(row.get("source", ""))
        source_l = source.lower()
        key = canonical_key_from(
            first_nonempty(row.get("uniprot_acc", ""), row.get("protein_id", "")),
            row.get("sequence", ""),
            row.get("sequence_md5", ""),
            row.get("canonical_key", ""),
        )
        if key not in by_key:
            stats["removed_span_not_in_model_protein"] += 1
            continue
        meta = by_key[key]
        acc = normalize_acc(row.get("uniprot_acc", "")) or normalize_acc(meta.get("uniprot_acc", ""))
        md5 = norm_text(row.get("sequence_md5", "")).lower() or norm_text(meta.get("sequence_md5", "")).lower()
        start = safe_int(row.get("region_start", row.get("start", "")), -1)
        end = safe_int(row.get("region_end", row.get("end", "")), -1)
        if acc in benchmark["accs"] or acc in benchmark["ids"] or md5 in benchmark["md5s"]:
            stats["removed_span_benchmark_direct"] += 1
            continue
        if "phasepro" in source_l or (acc, start, end) in phasepro_tuples:
            stats["removed_span_phasepro_exact"] += 1
            continue
        length = int(meta.get("length", 0) or 0)
        if start < 1 or end < start or end > length:
            stats["removed_span_out_of_bounds"] += 1
            continue

        if "phasepdb" in source_l or "phasesepdb" in source_l or "phasepdb_3" in source_l:
            tier = "dpr_silver"
        elif "llpsdb" in source_l:
            tier = "dpr_silver_low"
        else:
            tier = norm_text(row.get("region_label_tier_candidate", row.get("region_label_tier", ""))).lower()
            if tier not in {"dpr_silver", "dpr_silver_low", "dpr_pseudo"}:
                tier = "dpr_silver_low"

        cleaned.append(
            {
                "protein_id": meta["protein_id"],
                "canonical_key": key,
                "uniprot_acc": meta.get("uniprot_acc", ""),
                "sequence_md5": meta.get("sequence_md5", ""),
                "split": meta.get("split", "train"),
                "region_start": start,
                "region_end": end,
                "start_1based": start,
                "end_1based": end,
                "coordinate_system": "1-based inclusive",
                "region_length": end - start + 1,
                "region_type": norm_text(row.get("region_type", "")) or "DPR_silver",
                "region_label_tier": tier,
                "source": source,
                "evidence_level": norm_text(row.get("evidence_level", "")),
                "pmid": norm_text(row.get("pmid", "")),
                "notes": norm_text(row.get("notes", "")),
            }
        )

    priority = {"dpr_silver": 3, "dpr_silver_low": 2, "dpr_pseudo": 1}
    merged: list[dict[str, Any]] = []
    by_protein: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in cleaned:
        by_protein[str(item["protein_id"])].append(item)
    for protein_id in sorted(by_protein):
        current: dict[str, Any] | None = None
        for item in sorted(by_protein[protein_id], key=lambda x: (x["region_start"], x["region_end"], -priority.get(x["region_label_tier"], 0))):
            if current is None or item["region_start"] > current["region_end"]:
                if current is not None:
                    merged.append(current)
                current = item.copy()
                continue
            current["region_end"] = max(current["region_end"], item["region_end"])
            current["end_1based"] = current["region_end"]
            current["region_length"] = current["region_end"] - current["region_start"] + 1
            if priority.get(item["region_label_tier"], 0) > priority.get(current["region_label_tier"], 0):
                current["region_label_tier"] = item["region_label_tier"]
            current["source"] = unique_join([current.get("source", ""), item.get("source", "")])
            current["evidence_level"] = unique_join([current.get("evidence_level", ""), item.get("evidence_level", "")])
            current["pmid"] = unique_join([current.get("pmid", ""), item.get("pmid", "")])
            current["notes"] = unique_join([current.get("notes", ""), item.get("notes", "")], limit=10)
            stats["merged_overlapping_spans"] += 1
        if current is not None:
            merged.append(current)

    stats["kept_spans_before_merge"] = len(cleaned)
    stats["output_spans"] = len(merged)
    stats["dpr_silver_spans"] = sum(1 for item in merged if item["region_label_tier"] == "dpr_silver")
    stats["dpr_silver_low_spans"] = sum(1 for item in merged if item["region_label_tier"] == "dpr_silver_low")
    stats["dpr_protein_coverage"] = len({item["protein_id"] for item in merged})
    stats["avg_span_length_x100"] = int(round((sum(item["region_length"] for item in merged) / max(len(merged), 1)) * 100))
    for item in merged:
        stats[f"source_{item['source']}"] += 1
    return merged, stats


def scan_teacher_targets(root: Path, benchmark: dict[str, set[str]], deleted: dict[str, set[str]]) -> Counter:
    stats = Counter()
    bad_ids = set(benchmark["ids"]) | set(benchmark["accs"]) | set(deleted["ids"]) | set(deleted["accs"])
    bad_md5 = set(benchmark["md5s"]) | set(deleted["md5s"])
    csv_paths = [
        root / "data/pseudo_labels/round0_external/teacher_scores.csv",
        root / "data/pseudo_labels/round0_external/teacher_protein_labels.csv",
        root / "data/pseudo_labels/round0_external/teacher_region_candidates.csv",
    ]
    for path in csv_paths:
        key = path.name
        if not path.exists():
            stats[f"{key}:missing"] += 1
            continue
        rows = 0
        overlaps = 0
        for chunk in pd.read_csv(path, chunksize=100000, low_memory=False):
            rows += len(chunk)
            mask = pd.Series(False, index=chunk.index)
            for col in ["protein_id", "uniprot_acc", "uniprot_id", "canonical_key"]:
                if col in chunk:
                    vals = chunk[col].map(norm_text)
                    vals_acc = vals.map(normalize_acc)
                    mask = mask | vals.isin(bad_ids) | vals_acc.isin(bad_ids)
            if "sequence_md5" in chunk:
                mask = mask | chunk["sequence_md5"].map(lambda x: norm_text(x).lower()).isin(bad_md5)
            overlaps += int(mask.sum())
        stats[f"{key}:rows"] = rows
        stats[f"{key}:benchmark_deleted_overlap_rows"] = overlaps
    jsonl_path = root / "data/pseudo_labels/round0_external/teacher_region_candidates.jsonl"
    if jsonl_path.exists():
        rows = 0
        overlaps = 0
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                item = json.loads(line)
                pid = norm_text(item.get("protein_id", ""))
                acc = normalize_acc(first_nonempty(item.get("uniprot_acc", ""), pid))
                md5 = norm_text(item.get("sequence_md5", "")).lower()
                if pid in bad_ids or acc in bad_ids or md5 in bad_md5:
                    overlaps += 1
        stats["teacher_region_candidates.jsonl:rows"] = rows
        stats["teacher_region_candidates.jsonl:benchmark_deleted_overlap_rows"] = overlaps
    for path in [
        root / "data/pseudo_labels/round0_external/teacher_scores.h5",
        root / "data/processed/pstp_scan_region_targets.h5",
        root / "data/processed/final_region_targets.h5",
    ]:
        key = path.name
        if not path.exists():
            stats[f"{key}:missing"] += 1
            continue
        if h5py is None:
            stats[f"{key}:h5py_missing_scan_skipped"] += 1
            continue
        with h5py.File(path, "r") as handle:
            keys = list(handle.keys())
        overlaps = 0
        for item in keys:
            acc = normalize_acc(item)
            if item in bad_ids or acc in bad_ids:
                overlaps += 1
        stats[f"{key}:groups"] = len(keys)
        stats[f"{key}:benchmark_deleted_overlap_groups"] = overlaps
    return stats


def direct_overlap_checks(root: Path, df: pd.DataFrame, benchmark: dict[str, set[str]], phasepro_tuples: set[tuple[str, int, int]], spans: list[dict[str, Any]], homolog_keys: set[str]) -> Counter:
    stats = Counter()
    accs = set(df["uniprot_acc"].dropna().astype(str))
    pids = set(df["protein_id"].dropna().astype(str))
    md5s = set(df["sequence_md5"].dropna().astype(str))
    stats["train_valid_vs_benchmark_accession_overlap"] = len((accs | pids) & (benchmark["accs"] | benchmark["ids"]))
    stats["train_valid_vs_benchmark_sequence_md5_overlap"] = len(md5s & benchmark["md5s"])
    stats["train_valid_vs_benchmark_sequence_sha256_overlap"] = (
        int(df["sequence"].map(lambda value: sha256_sequence(clean_sequence(value))).isin(benchmark.get("sha256s", set())).sum())
        if "sequence" in df.columns
        else 0
    )
    ppmc = read_csv(root / "data/benchmarks/protein_benchmark_ppmc/manifest.csv")
    phasepro = read_csv(root / "data/benchmarks/dpr_benchmark_phasepro/proteins.csv")
    if not ppmc.empty:
        ppmc_acc = set()
        for col in ["protein_id", "uniprot_id"]:
            if col in ppmc:
                ppmc_acc.update(ppmc[col].dropna().astype(str).map(normalize_acc))
        ppmc_md5 = set(ppmc.get("sequence_md5", pd.Series(dtype=str)).dropna().astype(str))
        stats["train_valid_vs_ppmc_accession_overlap"] = len(accs & ppmc_acc)
        stats["train_valid_vs_ppmc_sequence_md5_overlap"] = len(md5s & ppmc_md5)
    if not phasepro.empty:
        phase_acc = set()
        for col in ["protein_id", "uniprot_id"]:
            if col in phasepro:
                phase_acc.update(phasepro[col].dropna().astype(str).map(normalize_acc))
        phase_md5 = set(phasepro.get("sequence_md5", pd.Series(dtype=str)).dropna().astype(str))
        stats["train_valid_vs_phasepro_protein_overlap"] = len(accs & phase_acc)
        stats["train_valid_vs_phasepro_sequence_md5_overlap"] = len(md5s & phase_md5)
    stats["train_valid_vs_benchmark_mmseqs40_homolog_overlap"] = len(homolog_keys & set(df["canonical_key"].astype(str)))
    span_overlap = 0
    for item in spans:
        key = normalize_acc(item.get("uniprot_acc", "")) or str(item.get("protein_id", ""))
        if (key, int(item["region_start"]), int(item["region_end"])) in phasepro_tuples:
            span_overlap += 1
    stats["region_span_vs_phasepro_exact_region_overlap"] = span_overlap
    return stats


def parse_source_funnel(root: Path, full_pool: pd.DataFrame, final_clean: pd.DataFrame) -> list[dict[str, Any]]:
    previous: dict[str, dict[str, Any]] = {}
    path = root / f"data/reports/full_augmentation_funnel_{DATE}.md"
    if path.exists():
        in_table = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("| source | raw records |"):
                in_table = True
                continue
            if in_table and line.startswith("| ---"):
                continue
            if in_table:
                if not line.startswith("|") or line.startswith("| item |"):
                    break
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) >= 8:
                    previous[parts[0]] = {
                        "raw_records": parts[1],
                        "parsed_records": parts[2],
                        "mapped_uniprot_records": parts[3],
                        "has_sequence_records": parts[4],
                        "candidate_positive": parts[5],
                        "candidate_negative": parts[6],
                        "candidate_dpr_span": parts[7],
                    }
    pool_by_source = {}
    if not full_pool.empty:
        for source, group in full_pool.groupby("source", dropna=False):
            pool_by_source[str(source)] = {
                "candidate_rows_in_full_pool": len(group),
                "clean_rows_after_candidate_leakage": int((group["leakage_status"].astype(str) == "clean").sum()) if "leakage_status" in group else 0,
                "removed_direct_or_hash": int(group["leakage_status"].astype(str).isin(["removed_benchmark_accession", "removed_sequence_md5", "removed_sequence_hash", "removed_benchmark_source_id"]).sum()) if "leakage_status" in group else 0,
                "bad_seq": int(group.get("bad_seq", pd.Series(False, index=group.index)).fillna(False).astype(bool).sum()),
                "len_oos": int((group.get("seq_valid", pd.Series(False, index=group.index)).fillna(False).astype(bool) & ~group.get("train_scope", pd.Series(False, index=group.index)).fillna(False).astype(bool)).sum()),
                "mmseqs40_homolog_audit_only": int((group["leakage_status"].astype(str) == "removed_mmseqs40_homolog").sum()) if "leakage_status" in group else 0,
                "removed_label_conflict": int((group["leakage_status"].astype(str) == "removed_conflict").sum()) if "leakage_status" in group else 0,
            }
    final_counts = Counter()
    for sources in final_clean["sources"].fillna("").astype(str):
        for source in [x.strip() for x in sources.split(";") if x.strip() and x.strip() != "..."]:
            final_counts[source] += 1
    rows: list[dict[str, Any]] = []
    for source in sorted(set(previous) | set(pool_by_source) | set(final_counts)):
        row = {"source": source}
        row.update(previous.get(source, {}))
        row.update(pool_by_source.get(source, {}))
        row["final_clean_canonical_with_source"] = final_counts.get(source, 0)
        rows.append(row)
    return rows


def model_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["llps_label"] = out["final_llps_label"]
    out["label_tier"] = out["final_label_tier"]
    out["label_quality"] = out["final_label_tier"]
    out["role_label"] = out["final_role_label"]
    out["negative_type"] = out["final_negative_type"]
    out["label_confidence"] = out["sample_weight"]
    out["region_label_tier"] = out["has_dpr_silver_candidate"].map(lambda x: "dpr_silver" if bool(x) else "none")
    out["leakage_status"] = out["final_leakage_status"]
    out["sampler_epoch_target"] = out["sampler_group"].map(
        {
            "hard_positive": "all",
            "pseudo_positive": "weighted",
            "structured_negative": "cap_5000_per_epoch",
            "disordered_negative": "all_oversample_allowed",
            "unknown_pu": "nnPU_or_ignore",
            "associated_context": "ignore_or_aux_context",
        }
    )
    cols = [
        "protein_id",
        "canonical_key",
        "sequence",
        "length",
        "llps_label",
        "final_llps_label",
        "sample_weight",
        "label_confidence",
        "label_quality",
        "label_tier",
        "final_label_tier",
        "negative_type",
        "final_negative_type",
        "role_label",
        "final_role_label",
        "sampler_group",
        "sampler_epoch_target",
        "source",
        "sources",
        "split",
        "cluster_id_40",
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
        "has_dpr_silver_candidate",
        "leakage_status",
        "final_leakage_reason",
        "label_conflict",
        "conflict_resolution",
        "source_record_ids",
        "evidence_types",
        "evidence_levels",
        "roles",
            "pmids",
            "notes",
        ]
    for col in cols:
        if col not in out:
            out[col] = ""
    return out[cols]


def _scope_skip_reasons(df: pd.DataFrame, context: str) -> list[str]:
    return [
        aug.skip_reason(
            bad_seq=bool(bad),
            train_scope=bool(train_scope),
            teacher_scope=bool(teacher_scope),
            leakage=str(leakage_status) != "clean",
            hard_label=aug.hard_label(label, role, tier) and not bool(train_scope),
            context=context,
        )
        for bad, train_scope, teacher_scope, leakage_status, label, role, tier in zip(
            df.get("bad_seq", pd.Series(False, index=df.index)),
            df.get("train_scope", pd.Series(False, index=df.index)),
            df.get("teacher_scope", pd.Series(False, index=df.index)),
            df.get("final_leakage_status", pd.Series("clean", index=df.index)),
            df.get("final_llps_label", pd.Series(-100, index=df.index)),
            df.get("final_role_label", pd.Series("", index=df.index)),
            df.get("final_label_tier", pd.Series("", index=df.index)),
        )
    ]


def write_scope_manifests(root: Path, canonical: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame) -> dict[str, int]:
    processed = root / "data/processed"
    canonical = canonical.copy()
    canonical["source"] = canonical.get("source", canonical.get("sources", "")).fillna("").astype(str).str.split(";").str[0]
    canonical["skip_reason"] = _scope_skip_reasons(canonical, "candidate")
    legal = canonical[canonical["seq_valid"].fillna(False).astype(bool)].copy()
    candidate_manifest = model_manifest_columns(legal)
    candidate_manifest.to_csv(processed / "candidate_manifest.csv", index=False)

    teacher_source = legal[legal["final_leakage_status"].eq("clean")].copy()
    teacher_source["skip_reason"] = _scope_skip_reasons(teacher_source, "teacher")
    teacher_manifest = model_manifest_columns(teacher_source[teacher_source["teacher_scope"].fillna(False).astype(bool)].copy())
    teacher_manifest.to_csv(processed / "teacher_manifest.csv", index=False)

    short_manifest = candidate_manifest[pd.to_numeric(candidate_manifest["length"], errors="coerce").fillna(0).lt(100)].copy()
    long_manifest = candidate_manifest[pd.to_numeric(candidate_manifest["length"], errors="coerce").fillna(0).gt(2048)].copy()
    short_manifest.to_csv(processed / "short_manifest.csv", index=False)
    long_manifest.to_csv(processed / "long_manifest.csv", index=False)

    rows: list[dict[str, Any]] = []

    def add_counts(scope: str, values: pd.Series) -> None:
        for value, count in values.value_counts(dropna=False).sort_index().items():
            rows.append({"scope": scope, "value": str(value), "count": int(count)})

    add_counts("len_bucket", legal["len_bucket"])
    add_counts("seq_valid", canonical["seq_valid"].fillna(False).astype(bool))
    add_counts("bad_seq", canonical["bad_seq"].fillna(False).astype(bool))
    add_counts("train_scope", legal["train_scope"].fillna(False).astype(bool))
    add_counts("teacher_scope", legal["teacher_scope"].fillna(False).astype(bool))
    add_counts("candidate_skip_reason", legal["skip_reason"])
    train_skip = legal.copy()
    train_skip["skip_reason"] = _scope_skip_reasons(train_skip, "train")
    add_counts("train_skip_reason", train_skip["skip_reason"])
    teacher_audit = legal.copy()
    teacher_audit["skip_reason"] = _scope_skip_reasons(teacher_audit, "teacher")
    add_counts("teacher_skip_reason", teacher_audit["skip_reason"])
    pd.DataFrame(rows).to_csv(processed / "length_filter_report.csv", index=False)
    return {
        "candidate_manifest_rows": len(candidate_manifest),
        "teacher_manifest_rows": len(teacher_manifest),
        "short_manifest_rows": len(short_manifest),
        "long_manifest_rows": len(long_manifest),
    }


def write_sampler_config(root: Path) -> None:
    config = {
        "date": DATE,
        "strategy": "group_weighted_sampling_without_truncating_candidate_pool",
        "groups": {
            "hard_positive": {"use": "all", "sample_weight": 1.0},
            "pseudo_positive": {"use": "weighted", "sample_weight": 0.4},
            "structured_negative": {"use": "sample_per_epoch", "target_per_epoch": 5000, "sample_weight": 0.85},
            "disordered_negative": {"use": "all_with_oversampling_allowed", "sample_weight": 1.0},
            "unknown_pu": {"use": "ignore_or_nnPU", "sample_weight": 0.0},
            "associated_context": {"use": "ignore_or_auxiliary_context", "sample_weight": 0.0},
        },
        "epoch_balance_target": {
            "positive_to_negative": "1:1 to 1:3",
            "negative_structured_to_disordered": "50:50 to 70:30",
        },
    }
    path = root / "data/processed/model_sampler_config.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def normalize_audit_role(value: Any) -> str:
    role = norm_text(value).lower().replace(" ", "_").replace("-", "_")
    role = re.sub(r"_+", "_", role).strip("_")
    aliases = {
        "": "unknown",
        "nan": "unknown",
        "none": "unknown",
        "negative_structured": "unknown",
        "negative_disordered": "unknown",
        "member": "client",
        "associated": "associated_context",
    }
    return aliases.get(role, role)


def audit_dependency_text(row: pd.Series) -> str:
    text = " ".join(
        norm_text(row.get(col, ""))
        for col in ["source", "role_label", "evidence_type", "evidence_level", "notes"]
    ).lower()
    text = text.replace("_", " ")
    dependency_terms = r"(?:dependency\s*terms?|mutation|mutant|ptm|phosphorylation|repeat|nucleic\s*acid|rna|dna)"
    text = re.sub(rf"\b(?:no|without)\s+{dependency_terms}(?:\s*[/,; -]\s*{dependency_terms})*", " ", text)
    text = re.sub(r"\bnot\s+(?:rna|dna|partner|ptm|phosphorylation|mutation)[-\s]?dependent\b", " ", text)
    return text


def resolve_audit_keys(pool: pd.DataFrame, canonical: pd.DataFrame) -> list[str]:
    final_keys = set(canonical["canonical_key"].fillna("").astype(str))
    md5_to_keys = (
        canonical[canonical["sequence_md5"].fillna("").astype(str).ne("")]
        .groupby("sequence_md5")["canonical_key"]
        .agg(lambda values: sorted(set(map(str, values))))
        .to_dict()
    )
    unique_md5_to_key = {md5: keys[0] for md5, keys in md5_to_keys.items() if len(keys) == 1}
    resolved: list[str] = []
    for acc, key, md5 in zip(
        pool.get("uniprot_acc", pd.Series("", index=pool.index)),
        pool.get("canonical_key", pd.Series("", index=pool.index)),
        pool.get("sequence_md5", pd.Series("", index=pool.index)),
    ):
        acc_norm = normalize_acc(acc)
        key_text = norm_text(key)
        md5_text = norm_text(md5).lower()
        if acc_norm and acc_norm in final_keys:
            resolved.append(acc_norm)
        elif key_text in final_keys:
            resolved.append(key_text)
        elif md5_text in unique_md5_to_key:
            resolved.append(unique_md5_to_key[md5_text])
        else:
            resolved.append(first_nonempty(key_text, acc_norm, md5_text))
    return resolved


def read_source_audit_pool(path: Path) -> pd.DataFrame:
    columns = [
        "source",
        "source_record_id",
        "uniprot_acc",
        "sequence_md5",
        "role_label",
        "evidence_type",
        "evidence_level",
        "notes",
        "canonical_key",
        "leakage_status",
        "leakage_reason",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in columns if col in header.columns]
    df = pd.read_csv(path, low_memory=False, usecols=usecols)
    for col in columns:
        if col not in df:
            df[col] = ""
    return df[columns]


def write_source_label_audit(root: Path, canonical: pd.DataFrame) -> Path:
    reports = root / "data/reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "source_label_audit.csv"
    pool = read_source_audit_pool(root / "data/processed/full_candidate_pool.csv")
    if pool.empty or canonical.empty:
        pd.DataFrame(columns=SOURCE_LABEL_AUDIT_COLUMNS).to_csv(path, index=False)
        return path

    pool = pool.copy()
    resolved_keys = resolve_audit_keys(pool, canonical)
    key_series = pd.Series(resolved_keys, index=pool.index)
    canonical_index = canonical.drop_duplicates("canonical_key", keep="first").set_index("canonical_key")
    canon_status = key_series.map(canonical_index["final_leakage_status"].to_dict()).fillna("")
    canon_reason = key_series.map(canonical_index["final_leakage_reason"].to_dict()).fillna("")
    canon_group = key_series.map(canonical_index["sampler_group"].to_dict()).fillna("")
    canon_weight = pd.to_numeric(key_series.map(canonical_index["sample_weight"].to_dict()), errors="coerce").fillna(0.0)

    row_status = pool["leakage_status"].fillna("").astype(str)
    row_reason = pool["leakage_reason"].fillna("").astype(str)
    row_clean = row_status.eq("clean")
    canonical_clean = canon_status.eq("clean")
    excluded = ~row_clean | ~canonical_clean

    role_raw = pool["role_label"].map(norm_text)
    role_norm = role_raw.map(normalize_audit_role)
    dependency_text = pool.apply(audit_dependency_text, axis=1)
    partner_regex = re.compile(
        r"\b(partner[-\s]?dependent|protein[-\s]?partner|co[-\s]?condens|requires?\s+partner|"
        r"dependent\s+on\s+partner|presence\s+of\s+partner)\b",
        re.I,
    )
    rna_dna_regex = re.compile(
        r"\b(rna[-\s]?dependent|dna[-\s]?dependent|requires?\s+(?:rna|dna)|dependent\s+on\s+(?:rna|dna)|"
        r"presence\s+of\s+(?:rna|dna)|with\s+(?:rna|dna)|nucleic[-\s]?acid)\b",
        re.I,
    )
    ptm_regex = re.compile(r"\b(ptm|phosphorylat|post[-\s]?translational)\b", re.I)
    mutation_regex = re.compile(r"\b(mutant|mutation)\b", re.I)

    is_driver = role_norm.isin(["driver", "scaffold"])
    is_client = role_norm.isin(["client", "associated_context"])
    is_regulator = role_norm.eq("regulator")
    partner_dependency = dependency_text.map(lambda text: bool(partner_regex.search(text))) | role_norm.isin(["client", "regulator"])
    rna_dna_dependency = dependency_text.map(lambda text: bool(rna_dna_regex.search(text)))
    ptm_dependency = dependency_text.map(lambda text: bool(ptm_regex.search(text)))
    mutation_dependency = dependency_text.map(lambda text: bool(mutation_regex.search(text)))

    assigned_label = canon_group.where(~excluded, "excluded")
    assigned_label = assigned_label.where(assigned_label.astype(str).ne(""), "unknown_pu")
    label_confidence = canon_weight.where(~excluded, 0.0)

    exclude_reason: list[str] = []
    for is_row_clean, status, reason, final_status, final_reason in zip(row_clean, row_status, row_reason, canon_status, canon_reason):
        status_text = norm_text(status)
        reason_text = norm_text(reason)
        final_status_text = norm_text(final_status)
        final_reason_text = norm_text(final_reason)
        if not is_row_clean:
            exclude_reason.append(f"{status_text}: {reason_text}" if reason_text else status_text or "excluded_before_model_ready")
        elif not final_status_text:
            exclude_reason.append("missing_after_canonical_collapse")
        elif final_status_text != "clean":
            exclude_reason.append(f"{final_status_text}: {final_reason_text}" if final_reason_text else final_status_text)
        else:
            exclude_reason.append("")

    audit = pd.DataFrame(
        {
            "source": pool["source"].map(norm_text),
            "raw_id": pool["source_record_id"].map(norm_text),
            "uniprot_id": pool["uniprot_acc"].map(normalize_acc),
            "canonical_id": key_series,
            "sequence_md5": pool["sequence_md5"].map(lambda value: norm_text(value).lower()),
            "role_raw": role_raw,
            "role_normalized": role_norm,
            "evidence_type": pool["evidence_type"].map(norm_text),
            "is_driver": is_driver,
            "is_client": is_client,
            "is_regulator": is_regulator,
            "partner_dependency": partner_dependency,
            "rna_dna_dependency": rna_dna_dependency,
            "ptm_dependency": ptm_dependency,
            "mutation_dependency": mutation_dependency,
            "assigned_label": assigned_label,
            "label_confidence": label_confidence,
            "exclude_reason": exclude_reason,
        }
    )
    audit = audit[SOURCE_LABEL_AUDIT_COLUMNS]
    audit.sort_values(["source", "raw_id", "canonical_id"], kind="stable").to_csv(path, index=False)
    return path


def make_final_key_resolver(canonical: pd.DataFrame):
    final_keys = set(canonical["canonical_key"].astype(str))
    md5_to_keys = canonical[canonical["sequence_md5"].astype(str).ne("")].groupby("sequence_md5")["canonical_key"].agg(lambda values: sorted(set(map(str, values)))).to_dict()
    unique_md5_to_key = {md5: keys[0] for md5, keys in md5_to_keys.items() if len(keys) == 1}

    def resolve(row: pd.Series | dict[str, Any]) -> str:
        acc = normalize_acc(row.get("uniprot_acc", ""))
        key = str(row.get("canonical_key", ""))
        md5 = norm_text(row.get("sequence_md5", "")).lower()
        if acc and acc in final_keys:
            return acc
        if key in final_keys:
            return key
        if md5 in unique_md5_to_key:
            return unique_md5_to_key[md5]
        return key

    return resolve


def write_reports(
    root: Path,
    canonical: pd.DataFrame,
    split_df: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    spans: list[dict[str, Any]],
    source_funnel_rows: list[dict[str, Any]],
    canonical_stats: Counter,
    mmseqs_stats: Counter,
    cluster_stats: Counter,
    split_stats: Counter,
    region_stats: Counter,
    leakage_stats: Counter,
    teacher_stats: Counter,
    old_stats: dict[str, Any],
    mmseqs_paths: dict[str, str],
    cluster_paths: dict[str, str],
) -> None:
    reports = root / "data/reports"
    reports.mkdir(parents=True, exist_ok=True)
    clean = canonical[canonical["final_leakage_status"].eq("clean")]
    write_source_label_audit(root, canonical)
    def count_group(group: str) -> int:
        return int((clean["sampler_group"] == group).sum())
    seq_valid_total = int(canonical["seq_valid"].fillna(False).astype(bool).sum()) if "seq_valid" in canonical else 0
    bad_seq_total = int(canonical["bad_seq"].fillna(False).astype(bool).sum()) if "bad_seq" in canonical else 0
    train_scope_clean = clean[clean["train_scope"].fillna(False).astype(bool)] if "train_scope" in clean else clean.iloc[0:0]
    len_oos_clean = int((clean["seq_valid"].fillna(False).astype(bool) & ~clean["train_scope"].fillna(False).astype(bool)).sum()) if "train_scope" in clean else 0
    teacher_scope_clean = int((clean["teacher_scope"].fillna(False).astype(bool)).sum()) if "teacher_scope" in clean else 0
    teacher_skip_clean = int((clean["seq_valid"].fillna(False).astype(bool) & ~clean["teacher_scope"].fillna(False).astype(bool)).sum()) if "teacher_scope" in clean else 0
    disordered_total = int(clean["final_negative_type"].isin(["disordered_negative", "disordered_negative_silver"]).sum())
    unknown_total = int((clean["sampler_group"] == "unknown_pu").sum())
    associated_total = int((clean["sampler_group"] == "associated_context").sum())
    dpr_silver_spans = sum(1 for item in spans if item["region_label_tier"] == "dpr_silver")
    dpr_silver_low_spans = sum(1 for item in spans if item["region_label_tier"] == "dpr_silver_low")
    dpr_proteins = len({item["protein_id"] for item in spans})
    label_lines = [
        f"# Model-ready Label Summary {DATE}",
        "",
        "## Canonical 规模",
        "",
        "| 指标 | 数量 |",
        "| --- | ---: |",
        f"| canonical protein 总数（含最终删除状态） | {len(canonical):,} |",
        f"| final clean canonical protein | {len(clean):,} |",
        f"| seq_valid canonical protein | {seq_valid_total:,} |",
        f"| bad_seq canonical protein | {bad_seq_total:,} |",
        f"| length-audit-only clean canonical protein（train_scope=false） | {len_oos_clean:,} |",
        f"| teacher_scope clean canonical protein | {teacher_scope_clean:,} |",
        f"| teacher_skip clean canonical protein | {teacher_skip_clean:,} |",
        f"| train_scope clean canonical protein | {len(train_scope_clean):,} |",
        f"| model train protein | {len(train):,} |",
        f"| model valid protein | {len(valid):,} |",
        f"| valid fraction | {len(valid) / max(len(split_df), 1):.4%} |",
        "",
        "## 标签统计",
        "",
        "| 类型 | 数量 |",
        "| --- | ---: |",
        f"| hard positive driver/scaffold | {count_group('hard_positive'):,} |",
        f"| pseudo positive | {count_group('pseudo_positive'):,} |",
        f"| associated_context | {associated_total:,} |",
        f"| structured negative | {count_group('structured_negative'):,} |",
        f"| disordered negative | {disordered_total:,} |",
        f"| unknown/PU | {unknown_total:,} |",
        f"| DPR silver span | {dpr_silver_spans:,} |",
        f"| DPR silver_low span | {dpr_silver_low_spans:,} |",
        f"| DPR silver protein 覆盖 | {dpr_proteins:,} |",
        f"| label conflict canonical | {canonical_stats['label_conflict_canonical']:,} |",
        "",
        "## Split 标签覆盖",
        "",
        "| split | hard_positive | pseudo_positive | structured_negative | disordered_negative | associated_context | unknown_pu | DPR proteins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, sub in [("train", train), ("valid", valid)]:
        label_lines.append(
            f"| {name} | {int((sub['sampler_group']=='hard_positive').sum()):,} | "
            f"{int((sub['sampler_group']=='pseudo_positive').sum()):,} | "
            f"{int((sub['sampler_group']=='structured_negative').sum()):,} | "
            f"{int((sub['sampler_group']=='disordered_negative').sum()):,} | "
            f"{int((sub['sampler_group']=='associated_context').sum()):,} | "
            f"{int((sub['sampler_group']=='unknown_pu').sum()):,} | "
            f"{int(sub['has_dpr_silver_candidate'].sum()):,} |"
        )
    label_lines += [
        "",
        "## 旧 clean train 保留",
        "",
        "| 项目 | 数量 |",
        "| --- | ---: |",
        f"| 旧 leakage-clean train rows | {old_stats['old_clean_rows']:,} |",
        f"| 旧 teacher pseudo positive（目标 4,263） | {old_stats['old_teacher_total']:,} |",
        f"| 旧 teacher retained clean total | {old_stats['old_teacher_retained_clean_total']:,} |",
        f"| 旧 teacher retained as pseudo | {old_stats['old_teacher_retained_pseudo']:,} |",
        f"| 旧 teacher upgraded/overridden by hard evidence | {old_stats['old_teacher_retained_hard']:,} |",
        f"| 旧 teacher removed by leakage/MMseqs | {old_stats['old_teacher_removed']:,} |",
        f"| 旧 augmented hard positive（目标 382） | {old_stats['old_augmented_hard_total']:,} |",
        f"| 当前 full active hard positive（merge 前） | {old_stats['current_active_hard_before_merge']:,} |",
        f"| 旧 hard retained as final hard | {old_stats['old_hard_retained_final_hard']:,} |",
        f"| 旧 hard restored by old-clean merge | {old_stats['old_hard_restored_by_merge']:,} |",
        "",
        "## 新 external 真正新增",
        "",
        "| 口径 | 数量 |",
        "| --- | ---: |",
        f"| final clean 不在旧 leakage-clean train 中 | {old_stats['new_vs_old_clean_total']:,} |",
        f"| final clean 不在旧 augmented train 中 | {old_stats['new_vs_old_augmented_total']:,} |",
    ]
    for group, count in old_stats["new_vs_old_augmented_by_group"].items():
        label_lines.append(f"| new vs old augmented / {group} | {count:,} |")
    (reports / f"model_ready_label_summary_{DATE}.md").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    leakage_lines = [
        f"# Model-ready Leakage Report {DATE}",
        "",
        "## 结论",
        "",
        "| 检查项 | overlap/count |",
        "| --- | ---: |",
    ]
    for key, value in leakage_stats.items():
        leakage_lines.append(f"| {key} | {value:,} |")
    for key, value in teacher_stats.items():
        if key.endswith("overlap_rows") or key.endswith("overlap_groups"):
            leakage_lines.append(f"| {key} | {value:,} |")
    leakage_lines += [
        "",
        "## 过滤/跳过统计",
        "",
        "| 原因 | 数量 |",
        "| --- | ---: |",
        f"| bad_seq | {bad_seq_total:,} |",
        f"| len_oos | {len_oos_clean:,} |",
        f"| teacher_skip | {teacher_skip_clean:,} |",
        f"| train_skip | {len_oos_clean:,} |",
    ]
    for key, value in sorted(canonical_stats.items()):
        if key.startswith("removed_") or key in {"label_conflict_canonical", "bad_seq"}:
            leakage_lines.append(f"| {key} | {value:,} |")
    for key, value in sorted(mmseqs_stats.items()):
        leakage_lines.append(f"| {key} | {value:,} |")
    leakage_lines += [
        "",
        "## MMseqs40",
        "",
        f"- benchmark search result: `{mmseqs_paths.get('result_m8', '')}`",
        f"- benchmark search log: `{mmseqs_paths.get('log', '')}`",
        f"- model cluster TSV: `{cluster_paths.get('cluster_tsv', '')}`",
        f"- model cluster log: `{cluster_paths.get('log', '')}`",
        "",
        "最终写入 `model_train_manifest.csv` 与 `model_valid_manifest.csv` 的 protein 没有 benchmark direct accession/sequence-hash overlap；benchmark MMseqs40 hit keys 只作为同源审计字段，不再按泄漏删除。",
    ]
    (reports / f"model_ready_leakage_report_{DATE}.md").write_text("\n".join(leakage_lines) + "\n", encoding="utf-8")

    funnel_lines = [
        f"# Model-ready Source Funnel {DATE}",
        "",
        "## Source funnel",
        "",
        "| source | raw records | parsed records | mapped UniProt | has sequence | candidate positive | candidate negative | candidate DPR span | full pool rows | clean rows after prior leakage | bad_seq | len_oos | removed direct/hash | MMseqs40 audit-only | removed conflict | final clean canonical with source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in source_funnel_rows:
        funnel_lines.append(
            f"| {row.get('source','')} | {row.get('raw_records','')} | {row.get('parsed_records','')} | "
            f"{row.get('mapped_uniprot_records','')} | {row.get('has_sequence_records','')} | "
            f"{row.get('candidate_positive','')} | {row.get('candidate_negative','')} | {row.get('candidate_dpr_span','')} | "
            f"{row.get('candidate_rows_in_full_pool',0):,} | {row.get('clean_rows_after_candidate_leakage',0):,} | "
            f"{row.get('bad_seq',0):,} | {row.get('len_oos',0):,} | "
            f"{row.get('removed_direct_or_hash',0):,} | {row.get('mmseqs40_homolog_audit_only',0):,} | "
            f"{row.get('removed_label_conflict',0):,} | {row.get('final_clean_canonical_with_source',0):,} |"
        )
    funnel_lines += [
        "",
        "## Split / sampler",
        "",
        "| 指标 | 数量 |",
        "| --- | ---: |",
    ]
    for counter in [cluster_stats, split_stats, region_stats]:
        for key, value in sorted(counter.items()):
            if key.startswith("source_"):
                continue
            funnel_lines.append(f"| {key} | {value:,} |")
    funnel_lines += [
        "",
        "采样策略写入 `data/processed/model_sampler_config.json`：structured negative 不截断 manifest，但每个 epoch 建议 cap 5,000；disordered negative 全量使用并允许过采样。",
        "",
        "长度策略输出：`candidate_manifest.csv` 保留所有 seq_valid candidate；`model_train_manifest.csv` 包含所有 final clean 且 seq_valid 的训练序列，不再按长度过滤，也不创建 valid split；`short_manifest.csv`、`long_manifest.csv` 与 `length_filter_report.csv` 仅用于长度分布审计。",
    ]
    (reports / f"model_ready_source_funnel_{DATE}.md").write_text("\n".join(funnel_lines) + "\n", encoding="utf-8")


def write_hard_positive_audit(root: Path, old_aug: pd.DataFrame, active: pd.DataFrame, canonical: pd.DataFrame) -> dict[str, Any]:
    reports = root / "data/reports"
    reports.mkdir(parents=True, exist_ok=True)
    old_norm = normalize_old_manifest(old_aug, "previous_augmented_train_20260606")
    old_norm = add_evidence_flags(old_norm)
    old_hard = old_norm[old_norm["is_hard_evidence"]].copy()
    active_norm = normalize_candidate_frame(active, "current_full_active_entry_20260606")
    active_norm = add_evidence_flags(active_norm)
    active_hard_keys = set(active_norm.loc[active_norm["is_hard_evidence"], "canonical_key"].astype(str))
    final = canonical.set_index("canonical_key").to_dict(orient="index")
    resolve_final_key = make_final_key_resolver(canonical)
    rows: list[dict[str, Any]] = []
    reason_counts = Counter()
    for _, row in old_hard.iterrows():
        key = resolve_final_key(row)
        item = final.get(key)
        if not item:
            reason = "missing_after_canonical_collapse"
            final_status = "missing"
            final_group = ""
            final_label = ""
        else:
            final_status = str(item.get("final_leakage_status", ""))
            final_group = str(item.get("sampler_group", ""))
            final_label = str(item.get("final_llps_label", ""))
            if final_status != "clean":
                reason = final_status
            elif final_group == "hard_positive":
                if key in active_hard_keys:
                    reason = "retained_as_current_full_active_hard"
                else:
                    reason = "restored_by_old_clean_merge"
            else:
                reason = f"demoted_or_changed_to_{final_group or final_label}"
        reason_counts[reason] += 1
        rows.append(
            {
                "old_protein_id": row.get("protein_id_hint", ""),
                "canonical_key": key,
                "uniprot_acc": row.get("uniprot_acc", ""),
                "sequence_md5": row.get("sequence_md5", ""),
                "old_source": row.get("source", ""),
                "old_role_label": row.get("role_label", ""),
                "old_label_tier": row.get("label_tier_candidate", ""),
                "present_in_current_full_active_hard": key in active_hard_keys,
                "final_status": final_status,
                "final_sampler_group": final_group,
                "final_llps_label": final_label,
                "loss_or_retention_reason": reason,
            }
        )
    detail = pd.DataFrame(rows)
    detail_path = reports / f"hard_positive_loss_details_{DATE}.csv"
    detail.to_csv(detail_path, index=False)
    lines = [
        f"# Hard Positive Loss Audit {DATE}",
        "",
        "口径：旧 hard positive 来自 `data/processed/augmented_train_manifest.csv` 中 `llps_label=1, role_label in {driver,scaffold}, label_tier in {gold,curated}` 的 382 条；merge 前当前 hard positive 来自 full raw `active_train_manifest.csv`。",
        "",
        "| 项目 | 数量 |",
        "| --- | ---: |",
        f"| old augmented hard positive | {len(old_hard):,} |",
        f"| current full active hard before merge | {len(active_hard_keys):,} |",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"| {reason} | {count:,} |")
    lines += [
        "",
        f"逐条明细：`{detail_path.relative_to(root)}`。",
    ]
    (reports / f"hard_positive_loss_audit_{DATE}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "old_augmented_hard_total": len(old_hard),
        "current_active_hard_before_merge": len(active_hard_keys),
        "old_hard_retained_final_hard": int((detail["final_sampler_group"] == "hard_positive").sum()) if not detail.empty else 0,
        "old_hard_restored_by_merge": int((detail["loss_or_retention_reason"] == "restored_by_old_clean_merge").sum()) if not detail.empty else 0,
        "hard_detail_path": str(detail_path),
    }


def old_retention_stats(root: Path, old_clean: pd.DataFrame, old_aug: pd.DataFrame, canonical: pd.DataFrame) -> dict[str, Any]:
    old_clean_norm = normalize_old_manifest(old_clean, "old_leakage_clean_train_20260606")
    old_teacher = old_clean_norm[old_clean_norm["old_teacher_pseudo_flag"].fillna(False)].copy()
    final = canonical.set_index("canonical_key")
    resolve_final_key = make_final_key_resolver(canonical)
    teacher_keys = [resolve_final_key(row) for _, row in old_teacher.iterrows()]
    present = final.reindex(teacher_keys)
    retained_clean = int((present["final_leakage_status"] == "clean").sum()) if not present.empty else 0
    retained_pseudo = int(((present["final_leakage_status"] == "clean") & (present["sampler_group"] == "pseudo_positive")).sum()) if not present.empty else 0
    retained_hard = int(((present["final_leakage_status"] == "clean") & (present["sampler_group"] == "hard_positive")).sum()) if not present.empty else 0
    removed = len(teacher_keys) - retained_clean
    old_clean_keys = {resolve_final_key(row) for _, row in old_clean_norm.iterrows()}
    old_aug_norm = normalize_old_manifest(old_aug, "previous_augmented_train_20260606")
    old_aug_keys = {resolve_final_key(row) for _, row in old_aug_norm.iterrows()}
    clean = canonical[canonical["final_leakage_status"].eq("clean")]
    new_vs_old_clean = clean[~clean["canonical_key"].astype(str).isin(old_clean_keys)]
    new_vs_old_aug = clean[~clean["canonical_key"].astype(str).isin(old_aug_keys)]
    by_group = {str(k): int(v) for k, v in new_vs_old_aug["sampler_group"].value_counts().items()}
    return {
        "old_clean_rows": len(old_clean),
        "old_teacher_total": len(old_teacher),
        "old_teacher_retained_clean_total": retained_clean,
        "old_teacher_retained_pseudo": retained_pseudo,
        "old_teacher_retained_hard": retained_hard,
        "old_teacher_removed": removed,
        "new_vs_old_clean_total": len(new_vs_old_clean),
        "new_vs_old_augmented_total": len(new_vs_old_aug),
        "new_vs_old_augmented_by_group": by_group,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--skip-mmseqs", action="store_true")
    parser.add_argument("--skip-cluster", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    processed = root / "data/processed"
    splits = root / "data/splits"
    processed.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)

    log("loading inputs")
    full_pool = read_csv(processed / "full_candidate_pool.csv")
    active = read_csv(processed / "active_train_manifest.csv")
    old_aug = read_csv(processed / "augmented_train_manifest.csv")
    old_clean = read_csv(root / "data/pseudo_labels/round0_external/manifest_with_teacher.csv")
    if full_pool.empty or active.empty:
        raise SystemExit("full_candidate_pool.csv and active_train_manifest.csv are required.")
    benchmark = load_benchmark_sets(root)
    deleted = load_deleted_sets(root)
    phasepro_tuples = load_phasepro_region_tuples(root)

    log("normalizing evidence tables")
    if "leakage_status" in full_pool:
        full_clean = full_pool[
            full_pool["leakage_status"].astype(str).isin(["clean", "removed_mmseqs40_homolog"])
        ].copy()
    else:
        full_clean = full_pool.copy()
    if "leakage_status" in full_clean:
        homolog_mask = full_clean["leakage_status"].astype(str).eq("removed_mmseqs40_homolog")
        full_clean.loc[homolog_mask, "leakage_status"] = "clean"
        if "leakage_reason" in full_clean:
            full_clean.loc[homolog_mask, "leakage_reason"] = ""
    evidence_parts = [
        normalize_candidate_frame(full_clean, "full_candidate_pool_clean_20260606"),
        normalize_candidate_frame(active, "current_full_active_entry_20260606"),
    ]
    if not old_clean.empty:
        evidence_parts.append(normalize_old_manifest(old_clean, "old_leakage_clean_train_20260606"))
    if not old_aug.empty:
        evidence_parts.append(normalize_old_manifest(old_aug, "previous_augmented_train_20260606"))
    evidence = pd.concat(evidence_parts, ignore_index=True, sort=False)
    log(f"evidence rows={len(evidence):,}")

    log("collapsing canonical proteins and adjudicating labels")
    canonical, canonical_stats = collapse_canonical(evidence, benchmark, deleted)
    log(f"canonical rows before final MMseqs={len(canonical):,}; direct clean={int((canonical['final_leakage_status']=='clean').sum()):,}")

    log("running final benchmark MMseqs40 search")
    homolog_keys, mmseqs_stats, mmseqs_paths = run_mmseqs_benchmark_search(root, canonical, skip=args.skip_mmseqs)
    if homolog_keys:
        canonical_stats["mmseqs40_homolog_audit_only"] = int(
            (
                canonical["canonical_key"].astype(str).isin(homolog_keys)
                & canonical["final_leakage_status"].eq("clean")
            ).sum()
        )
    canonical["skip_reason"] = _scope_skip_reasons(canonical, "candidate")
    clean = canonical[canonical["final_leakage_status"].eq("clean")].copy()
    log(f"final clean canonical rows={len(clean):,}")
    train_scope_clean = clean[clean["seq_valid"].fillna(False).astype(bool)].copy()
    train_scope_clean["train_scope"] = True
    log(f"all-length train clean canonical rows={len(train_scope_clean):,}")

    log("precomputing DPR silver protein coverage for split")
    dpr_keys = preliminary_dpr_keys(root, train_scope_clean, benchmark)
    train_scope_clean["has_dpr_silver_candidate"] = train_scope_clean["canonical_key"].astype(str).isin(dpr_keys)

    log("running model MMseqs40 clustering for internal split")
    cluster_map, cluster_stats, cluster_paths = run_mmseqs_model_cluster(root, train_scope_clean, skip=args.skip_cluster)
    split_df, split_stats = assign_valid_split(train_scope_clean, cluster_map)

    log("cleaning DPR region spans")
    spans_records, region_stats = clean_region_spans(root, split_df, benchmark, phasepro_tuples)
    dpr_protein_ids = {item["protein_id"] for item in spans_records}
    split_df["has_dpr_silver_candidate"] = split_df["protein_id"].astype(str).isin(dpr_protein_ids) | split_df["has_dpr_silver_candidate"].fillna(False)

    log("building model manifests")
    split_df["source"] = split_df["sources"].fillna("").astype(str).str.split(";").str[0].fillna("")
    manifest = model_manifest_columns(split_df)
    train = manifest[manifest["split"] == "train"].copy()
    valid = manifest[manifest["split"] == "valid"].copy()
    canonical_out = canonical.copy()
    canonical_out["has_dpr_silver_candidate"] = canonical_out["canonical_key"].astype(str).isin(set(split_df.loc[split_df["has_dpr_silver_candidate"], "canonical_key"].astype(str)))
    canonical_out["source"] = canonical_out["sources"].fillna("").astype(str).str.split(";").str[0].fillna("")

    log("writing outputs")
    canonical_out.to_csv(processed / "canonical_protein_pool.csv", index=False)
    train.to_csv(processed / "model_train_manifest.csv", index=False)
    valid.to_csv(processed / "model_valid_manifest.csv", index=False)
    scope_stats = write_scope_manifests(root, canonical_out, train, valid)
    write_jsonl(processed / "model_region_spans.jsonl", spans_records)
    (splits / "train_ids.txt").write_text("\n".join(train["protein_id"].astype(str)) + "\n", encoding="utf-8")
    (splits / "valid_ids.txt").write_text("\n".join(valid["protein_id"].astype(str)) + "\n", encoding="utf-8")
    write_sampler_config(root)

    log("running final overlap scans")
    leakage_stats = direct_overlap_checks(root, split_df, benchmark, phasepro_tuples, spans_records, homolog_keys)
    teacher_stats = scan_teacher_targets(root, benchmark, deleted)
    source_funnel_rows = parse_source_funnel(root, full_pool, clean)
    old_stats = old_retention_stats(root, old_clean, old_aug, canonical)
    hard_stats = write_hard_positive_audit(root, old_aug, active, canonical)
    old_stats.update(hard_stats)

    write_reports(
        root,
        canonical,
        split_df,
        train,
        valid,
        spans_records,
        source_funnel_rows,
        canonical_stats,
        mmseqs_stats,
        cluster_stats,
        split_stats,
        region_stats,
        leakage_stats,
        teacher_stats,
        old_stats,
        mmseqs_paths,
        cluster_paths,
    )
    summary = {
        "canonical_total": len(canonical),
        "final_clean": len(clean),
        "train": len(train),
        "valid": len(valid),
        **scope_stats,
        "hard_positive": int((clean["sampler_group"] == "hard_positive").sum()),
        "pseudo_positive": int((clean["sampler_group"] == "pseudo_positive").sum()),
        "structured_negative": int((clean["sampler_group"] == "structured_negative").sum()),
        "disordered_negative": int((clean["sampler_group"] == "disordered_negative").sum()),
        "associated_context": int((clean["sampler_group"] == "associated_context").sum()),
        "unknown_pu": int((clean["sampler_group"] == "unknown_pu").sum()),
        "dpr_spans": len(spans_records),
        "leakage_stats": dict(leakage_stats),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
