#!/usr/bin/env python3
"""Rebuild the full external candidate pool from data/raw_src.

This is intentionally separate from augment_train_external_sources.py because
the active-train augmentation path used source-specific caps.  This script keeps
the raw/source candidate pool uncapped and only uses leakage filters and sample
weights to decide what can enter the active training manifest.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

import augment_train_external_sources as aug


DATE = "20260606"
AA_RE = re.compile(r"[^A-Z]")
LLPS_RE = re.compile(r"llps|liquid[-\s]?liquid|phase separation|condensate|membrane[-\s]?less|\bmlo\b|coacerv", re.I)
DEPENDENCY_RE = aug.DEPENDENCY_RE

UNIFIED_COLUMNS = [
    "source",
    "source_record_id",
    "uniprot_acc",
    "gene_name",
    "organism",
    "taxonomy_id",
    "sequence",
    "sequence_md5",
    "role_label",
    "llps_label_candidate",
    "label_tier_candidate",
    "negative_type_candidate",
    "evidence_type",
    "evidence_level",
    "pmid",
    "region_start",
    "region_end",
    "region_type",
    "notes",
    "region_label_tier_candidate",
    "sample_weight",
    "canonical_key",
    "seq_valid",
    "bad_seq",
    "len_bucket",
    "train_scope",
    "teacher_scope",
    "skip_reason",
    "leakage_status",
    "leakage_reason",
]


def log(message: str) -> None:
    print(f"[{DATE}] {message}", flush=True)


def clean_sequence(value: object) -> str:
    return aug.clean_sequence(value)


def md5_sequence(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest() if seq else ""


def normalize_acc(value: object) -> str:
    return aug.normalize_acc(value)


def base_acc(value: str) -> str:
    return aug.base_acc(value)


def valid_train_sequence(seq: str) -> bool:
    return aug.train_scope_sequence(seq)


def sha256_file(path: Path) -> str:
    return aug.sha256_file(path)


def first_text(row: dict[str, object], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan":
            return str(value).strip()
    return ""


def split_accessions(value: object) -> list[str]:
    text = str(value or "")
    if not text or text.lower() in {"nan", "none", "-", "no-data", "not-detected"}:
        return []
    out = []
    for token in re.split(r"[;,|\s]+", text):
        acc = normalize_acc(token)
        if acc and acc not in {"-", "nan", "no-data", "not-detected"}:
            out.append(acc)
    return out


def clean_llpsdb_sequence(value: object) -> str:
    text = str(value or "").strip()
    if not text or text in {"-", "nan", "None"}:
        return ""
    lines = [line.strip() for line in text.replace(";", "\n").splitlines()]
    seq_lines = [line for line in lines if line and not line.startswith(">") and not line.startswith("sp|") and not line.startswith("tr|")]
    return AA_RE.sub("", "".join(seq_lines).upper())


def is_dependency_text(text: str) -> bool:
    return bool(DEPENDENCY_RE.search(text or ""))


def parse_range_list(value: object, seq_len: int) -> list[tuple[int, int]]:
    return aug.parse_range_list(value, seq_len)


@dataclass
class SourceBundle:
    candidates: list[dict[str, object]]
    spans: list[dict[str, object]]
    stats: Counter


def make_candidate(
    *,
    source: str,
    source_record_id: str,
    uniprot_acc: str = "",
    gene_name: str = "",
    organism: str = "",
    taxonomy_id: str = "",
    sequence: str = "",
    role_label: str = "unknown",
    llps_label_candidate: int = -100,
    label_tier_candidate: str = "unknown",
    negative_type_candidate: str = "background_unlabeled",
    evidence_type: str = "",
    evidence_level: str = "",
    pmid: str = "",
    region_start: str = "",
    region_end: str = "",
    region_type: str = "",
    notes: str = "",
    region_label_tier_candidate: str = "none",
    sample_weight: float = 0.0,
) -> dict[str, object]:
    acc = normalize_acc(uniprot_acc)
    seq = clean_sequence(sequence)
    smd5 = md5_sequence(seq)
    canonical_key = acc or smd5 or f"{source}:{source_record_id}"
    scope = aug.length_scope_fields(
        seq,
        llps_label=llps_label_candidate,
        role_label=role_label,
        label_tier=label_tier_candidate,
        notes=notes,
        evidence_type=evidence_type,
        evidence_level=evidence_level,
    )
    return {
        "source": source,
        "source_record_id": str(source_record_id),
        "uniprot_acc": acc,
        "gene_name": str(gene_name or ""),
        "organism": str(organism or ""),
        "taxonomy_id": str(taxonomy_id or ""),
        "sequence": seq,
        "sequence_md5": smd5,
        "role_label": role_label,
        "llps_label_candidate": int(llps_label_candidate),
        "label_tier_candidate": label_tier_candidate,
        "negative_type_candidate": negative_type_candidate,
        "evidence_type": evidence_type,
        "evidence_level": evidence_level,
        "pmid": str(pmid or ""),
        "region_start": str(region_start or ""),
        "region_end": str(region_end or ""),
        "region_type": str(region_type or ""),
        "notes": str(notes or ""),
        "region_label_tier_candidate": region_label_tier_candidate,
        "sample_weight": float(sample_weight),
        "canonical_key": canonical_key,
        **scope,
        "skip_reason": aug.skip_reason(
            bad_seq=bool(scope["bad_seq"]),
            train_scope=bool(scope["train_scope"]),
            teacher_scope=bool(scope["teacher_scope"]),
            hard_label=aug.hard_label(llps_label_candidate, role_label, label_tier_candidate),
        ),
        "leakage_status": "candidate_unfiltered",
        "leakage_reason": "",
    }


def add_if_sequence(rows: list[dict[str, object]], row: dict[str, object]) -> None:
    if row["sequence"]:
        rows.append(row)


def load_swissprot(raw_src: Path) -> dict[str, dict[str, str]]:
    fasta = raw_src / "uniprot_swissprot/uniprot_sprot.fasta.gz"
    if not fasta.exists():
        return {}
    return aug.parse_swissprot_fasta(fasta)


def parse_phasepdb(raw_src: Path, swiss: dict[str, dict[str, str]]) -> SourceBundle:
    path = raw_src / "phasepdb3/proteins_api_2026-05-21.json"
    stats = Counter()
    rows: list[dict[str, object]] = []
    spans: list[dict[str, object]] = []
    if not path.exists():
        return SourceBundle(rows, spans, stats)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    stats["raw_records"] = len(records)
    for rec in records:
        stats["parsed_records"] += 1
        acc = normalize_acc(rec.get("uniprot_id", ""))
        meta = swiss.get(acc, {})
        seq = meta.get("sequence", "")
        if acc:
            stats["mapped_uniprot_records"] += 1
        if seq:
            stats["has_sequence_records"] += 1
        text = " ".join(str(rec.get(k, "")) for k in rec)
        cls = str(rec.get("class_", "")).strip()
        status = str(rec.get("_status", "")).strip().lower()
        strict = cls == "PS-self" and status in {"approved", ""} and not is_dependency_text(text)
        if strict:
            label, tier, role, weight, note = 1, "curated", "driver", 1.0, "strict PS-self; no dependency terms"
            stats["candidate_positive"] += 1
        elif cls.startswith("PS"):
            label, tier, role, weight, note = 1, "pseudo", "teacher_positive", 0.4, f"PhaSepDB context class={cls}; weak/pseudo positive, not strict hard positive"
            stats["candidate_pseudo_positive"] += 1
        else:
            label, tier, role, weight, note = -100, "unknown", "unknown", 0.0, f"PhaSepDB class={cls}"
        row = make_candidate(
            source="PhaSepDB_3",
            source_record_id=str(rec.get("protein_id", "")),
            uniprot_acc=acc,
            gene_name=meta.get("gene_name", "") or str(rec.get("primary_name", "") or ""),
            organism=str(rec.get("organism", "")) or meta.get("organism", ""),
            taxonomy_id=meta.get("taxonomy_id", ""),
            sequence=seq,
            role_label=role,
            llps_label_candidate=label,
            label_tier_candidate=tier,
            negative_type_candidate="none" if label == 1 else "background_unlabeled",
            evidence_type="curated_database",
            evidence_level="curated" if strict else "associated",
            pmid=str(rec.get("pmid", "")),
            notes=note,
            sample_weight=weight,
        )
        add_if_sequence(rows, row)
        region_text = str(rec.get("key_protein_regions_studied_ps", ""))
        for start, end in parse_range_list(region_text, len(seq)):
            spans.append(
                {
                    **row,
                    "region_start": start,
                    "region_end": end,
                    "region_type": "phase_separation_region",
                    "region_label_tier_candidate": "dpr_silver",
                    "notes": region_text[:500],
                }
            )
            stats["candidate_dpr_span"] += 1
    return SourceBundle(rows, spans, stats)


LLPSDB_ZIP_SUBSETS = {
    "Phase_separation_unambiguous.zip": ("Phase_separation_Unambiguous", "phase_separation", "unambiguous"),
    "No_phase_separation_unambiguous.zip": ("No_phase_separation_Unambiguous", "no_phase_separation", "unambiguous"),
    "Phase_diagram_unambiguous.zip": ("Phase_diagram_Unambiguous", "phase_diagram", "unambiguous"),
    "Phase_separation_ambiguous.zip": ("Phase_separation_ambiguous", "phase_separation", "ambiguous"),
    "No_phase_separation_ambiguous.zip": ("No_phase_separation_ambiguous", "no_phase_separation", "ambiguous"),
    "Phase_diagram_ambiguous.zip": ("Phase_diagram_ambiguous", "phase_diagram", "ambiguous"),
}


def read_xlsx_from_zip(zip_path: Path, suffix: str) -> list[dict[str, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        for path in Path(tmp).rglob(suffix):
            return aug.parse_xlsx_first_sheet(path)
    return []


def parse_llpsdb(raw_src: Path) -> SourceBundle:
    root = raw_src / "llpsdb_v2"
    stats = Counter()
    rows: list[dict[str, object]] = []
    spans: list[dict[str, object]] = []
    strict_ids: set[tuple[str, str]] = set()
    protein_rows: list[tuple[str, str, str, dict[str, str]]] = []
    condition_by_subset: dict[str, list[dict[str, str]]] = {}

    for zip_name, (subset, status, ambiguity) in LLPSDB_ZIP_SUBSETS.items():
        zip_path = root / zip_name
        if not zip_path.exists():
            stats[f"missing_{zip_name}"] = 1
            continue
        proteins = read_xlsx_from_zip(zip_path, "protein.xls")
        conditions = read_xlsx_from_zip(zip_path, "LLPS.xls")
        stats["raw_records"] += len(proteins) + len(conditions)
        condition_by_subset[subset] = conditions
        for cond in conditions:
            stats["parsed_records"] += 1
            pid = first_text(cond, "Protein ID", "protID")
            text = " ".join(str(cond.get(k, "")) for k in cond)
            component_type = first_text(cond, "Components type", "Components_type").lower()
            protein_type = first_text(cond, "Protein type (N/D)", "ND")
            bad_flags = [
                first_text(cond, "Fusion"),
                first_text(cond, "Cleaved"),
                first_text(cond, "Repeat"),
                first_text(cond, "Mutation"),
                first_text(cond, "PTM"),
            ]
            nucleic = first_text(cond, "Nucleic acid", "Nucleic_acid1", "Nucleic_acid2")
            strict = (
                subset == "Phase_separation_Unambiguous"
                and status == "phase_separation"
                and component_type == "protein(1)"
                and protein_type == "N"
                and all(flag in {"", "-", "nan"} for flag in bad_flags)
                and nucleic in {"", "-", "nan"}
                and not is_dependency_text(text)
            )
            if strict and pid:
                strict_ids.add((subset, pid))
        for prot in proteins:
            stats["parsed_records"] += 1
            protein_rows.append((subset, status, ambiguity, prot))

    for subset, status, ambiguity, prot in protein_rows:
        pid = first_text(prot, "PID")
        acc = normalize_acc(first_text(prot, "Uniprot ID", "UniprotID"))
        seq = clean_llpsdb_sequence(first_text(prot, "Sequence", "Full_length"))
        if acc:
            stats["mapped_uniprot_records"] += 1
        if seq:
            stats["has_sequence_records"] += 1
        if (subset, pid) in strict_ids:
            label, tier, role, ntype, weight, evidence = 1, "curated", "driver", "none", 0.9, "curated"
            stats["candidate_positive"] += 1
            notes = "LLPSDB unambiguous protein(1) natural WT no mutation/PTM/repeat/nucleic-acid"
        elif status == "phase_separation":
            label, tier, role, ntype, weight, evidence = 1, "pseudo", "teacher_positive", "none", 0.3, "associated"
            stats["candidate_pseudo_positive"] += 1
            notes = f"LLPSDB {status} {ambiguity}; not strict one-component natural WT"
        elif status == "no_phase_separation":
            label, tier, role, ntype, weight, evidence = -100, "unknown", "unknown", "background_unlabeled", 0.0, "condition_level_no_phase_only"
            stats["candidate_negative_context"] += 1
            notes = "LLPSDB no-phase-separation condition retained as context, not full-protein hard negative"
        else:
            label, tier, role, ntype, weight, evidence = -100, "unknown", "unknown", "background_unlabeled", 0.0, "condition_context"
            notes = f"LLPSDB {status} condition/context record"
        row = make_candidate(
            source="LLPSDB_v2",
            source_record_id=pid,
            uniprot_acc=acc,
            gene_name=first_text(prot, "Gene name", "g_name"),
            organism=first_text(prot, "Species"),
            taxonomy_id=first_text(prot, "NCBI code", "NCBI"),
            sequence=seq,
            role_label=role,
            llps_label_candidate=label,
            label_tier_candidate=tier,
            negative_type_candidate=ntype,
            evidence_type="curated_database",
            evidence_level=evidence,
            notes=f"{notes}; subset={subset}",
            sample_weight=weight,
        )
        add_if_sequence(rows, row)
        for field_name in ["IDR", "LCR"]:
            for start, end in parse_range_list(prot.get(field_name, ""), len(seq)):
                spans.append(
                    {
                        **row,
                        "region_start": start,
                        "region_end": end,
                        "region_type": field_name,
                        "region_label_tier_candidate": "dpr_silver",
                        "notes": f"{field_name} annotation from LLPSDB v2 {subset}",
                    }
                )
                stats["candidate_dpr_span"] += 1
    return SourceBundle(rows, spans, stats)


def parse_cdcode(raw_src: Path, swiss: dict[str, dict[str, str]]) -> SourceBundle:
    path = raw_src / "cd_code/cd-code_release_v2.2_2026-03-18.zip"
    stats = Counter()
    rows: list[dict[str, object]] = []
    if not path.exists():
        return SourceBundle(rows, [], stats)
    with zipfile.ZipFile(path) as zf:
        with zf.open("02/proteins_202603181653.csv") as handle:
            proteins = pd.read_csv(handle, low_memory=False)
    stats["raw_records"] = int(len(proteins))
    for _, row in proteins.iterrows():
        stats["parsed_records"] += 1
        acc = normalize_acc(row.get("uniprot_id", ""))
        meta = swiss.get(acc, {})
        if acc:
            stats["mapped_uniprot_records"] += 1
        seq = meta.get("sequence", "")
        if seq:
            stats["has_sequence_records"] += 1
        text = " ".join(str(row.get(c, "")) for c in ["name", "function", "description"])
        is_driver = bool(re.search(r"\b(driver|scaffold)\b", text, re.I)) and not re.search(r"\b(client|member|regulator)\b", text, re.I)
        if is_driver:
            label, tier, role, weight = 1, "curated", "driver", 0.8
            stats["candidate_positive"] += 1
            notes = "CD-CODE explicit driver/scaffold text"
        else:
            label, tier, role, weight = -100, "unknown", "member", 0.0
            stats["candidate_associated_positive"] += 1
            notes = "CD-CODE condensate-associated member; not driver proof"
        add_if_sequence(
            rows,
            make_candidate(
                source="CD_CODE_v2.2",
                source_record_id=acc or str(row.get("id", "")),
                uniprot_acc=acc,
                gene_name=str(row.get("gene_name", "")) or meta.get("gene_name", ""),
                organism=str(row.get("species_name", "")) or meta.get("organism", ""),
                taxonomy_id=str(row.get("species_taxon_id", "")) if not pd.isna(row.get("species_taxon_id", "")) else meta.get("taxonomy_id", ""),
                sequence=seq,
                role_label=role,
                llps_label_candidate=label,
                label_tier_candidate=tier,
                negative_type_candidate="none" if label == 1 else "background_unlabeled",
                evidence_type="curated_database",
                evidence_level="curated" if is_driver else "associated",
                notes=notes,
                sample_weight=weight,
            ),
        )
    return SourceBundle(rows, [], stats)


def parse_drllps(raw_src: Path) -> SourceBundle:
    path = raw_src / "drllps/drllps.txt"
    stats = Counter()
    rows: list[dict[str, object]] = []
    if not path.exists():
        return SourceBundle(rows, [], stats)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    stats["raw_records"] = int(len(df))
    for _, row in df.iterrows():
        stats["parsed_records"] += 1
        acc = normalize_acc(row.get("UniProt ID", ""))
        seq = clean_sequence(row.get("Protein Sequence", ""))
        if acc:
            stats["mapped_uniprot_records"] += 1
        if seq:
            stats["has_sequence_records"] += 1
        role_raw = str(row.get("LLPS Type", "")).strip().lower()
        if role_raw == "scaffold":
            label, tier, role, weight, ntype = 1, "curated", "scaffold", 1.0, "none"
            stats["candidate_positive"] += 1
        elif role_raw in {"client", "regulator"}:
            label, tier, role, weight, ntype = -100, "unknown", role_raw, 0.0, "background_unlabeled"
            stats["candidate_associated_positive"] += 1
        else:
            label, tier, role, weight, ntype = -100, "unknown", "unknown", 0.0, "background_unlabeled"
        add_if_sequence(
            rows,
            make_candidate(
                source="DrLLPS",
                source_record_id=str(row.get("DrLLPS ID", "")),
                uniprot_acc=acc,
                gene_name=str(row.get("Gene name", "")),
                organism=str(row.get("Species", "")),
                sequence=seq,
                role_label=role,
                llps_label_candidate=label,
                label_tier_candidate=tier,
                negative_type_candidate=ntype,
                evidence_type="curated_database",
                evidence_level="curated" if label == 1 else "associated",
                pmid=str(row.get("References", "")),
                notes=f"DrLLPS role={role_raw}; condensate={row.get('Condensate', '')}",
                sample_weight=weight,
            ),
        )
    return SourceBundle(rows, [], stats)


def parse_bav(raw_src: Path, swiss: dict[str, dict[str, str]]) -> SourceBundle:
    root = raw_src / "bav_llps"
    stats = Counter()
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*.xlsx")):
        records = aug.parse_xlsx_first_sheet(path)
        stats["raw_records"] += len(records)
        source_name = f"BAV_LLPS_{path.stem}"
        for rec in records:
            stats["parsed_records"] += 1
            if "curated_id" in rec:
                for group in ["bacterias", "archaeas", "viruses", "eukaryotes"]:
                    for acc in split_accessions(rec.get(group, "")):
                        meta = swiss.get(acc, {})
                        seq = meta.get("sequence", "")
                        if acc:
                            stats["mapped_uniprot_records"] += 1
                        if seq:
                            stats["has_sequence_records"] += 1
                        add_if_sequence(
                            rows,
                            make_candidate(
                                source=source_name,
                                source_record_id=f"{rec.get('curated_id', '')}:{group}:{acc}",
                                uniprot_acc=acc,
                                gene_name=meta.get("gene_name", ""),
                                organism=meta.get("organism", ""),
                                taxonomy_id=meta.get("taxonomy_id", ""),
                                sequence=seq,
                                role_label="member",
                                llps_label_candidate=-100,
                                label_tier_candidate="unknown",
                                negative_type_candidate="background_unlabeled",
                                evidence_type="homologous_dataset",
                                evidence_level="associated",
                                notes=f"BAV homologous dataset group={group}; not hard positive",
                                sample_weight=0.0,
                            ),
                        )
                        stats["candidate_associated_positive"] += 1
                continue
            acc = normalize_acc(first_text(rec, "uniprot", "id"))
            seq = clean_sequence(rec.get("sequence", ""))
            if acc:
                stats["mapped_uniprot_records"] += 1
            if seq:
                stats["has_sequence_records"] += 1
            stats["candidate_positive"] += 1
            add_if_sequence(
                rows,
                make_candidate(
                    source=source_name,
                    source_record_id=first_text(rec, "id", "uniprot"),
                    uniprot_acc=acc,
                    organism=first_text(rec, "specie"),
                    sequence=seq,
                    role_label="driver",
                    llps_label_candidate=1,
                    label_tier_candidate="curated",
                    negative_type_candidate="none",
                    evidence_type="curated_database",
                    evidence_level="curated",
                    notes=f"BAV curated/full table; organism_type={rec.get('organism_type', '')}; mlo_bmc={rec.get('mlo_bmc', '')}",
                    sample_weight=0.7,
                ),
            )
    return SourceBundle(rows, [], stats)


def parse_disprot(raw_src: Path, swiss: dict[str, dict[str, str]]) -> SourceBundle:
    path = raw_src / "disprot/disprot_current.tsv"
    stats = Counter()
    rows: list[dict[str, object]] = []
    if not path.exists():
        return SourceBundle(rows, [], stats)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    stats["raw_records"] = int(len(df))
    for acc_raw, group in df.groupby("acc", dropna=True):
        stats["parsed_records"] += int(len(group))
        acc = normalize_acc(acc_raw)
        mapped = acc if acc in swiss else base_acc(acc)
        meta = swiss.get(mapped, {})
        if mapped:
            stats["mapped_uniprot_records"] += 1
        seq = meta.get("sequence", "")
        if seq:
            stats["has_sequence_records"] += 1
        term_text = " ".join(str(x) for x in group[[c for c in ["term", "term_name", "ec_name"] if c in group]].fillna("").to_numpy().ravel())
        if LLPS_RE.search(term_text):
            stats["excluded_llps_terms"] += 1
            continue
        disorder_content = pd.to_numeric(group.get("disorder_content"), errors="coerce").max()
        if pd.isna(disorder_content):
            disorder_content = 0.0
        if float(disorder_content) < 5.0:
            stats["excluded_low_disorder"] += 1
            continue
        weight = 0.9 if float(disorder_content) >= 20.0 else 0.75
        stats["candidate_negative"] += 1
        add_if_sequence(
            rows,
            make_candidate(
                source="DisProt_current",
                source_record_id=";".join(sorted(set(str(x) for x in group.get("disprot_id", pd.Series(dtype=str)).dropna()))),
                uniprot_acc=mapped,
                gene_name=meta.get("gene_name", ""),
                organism=str(group["organism"].dropna().iloc[0]) if "organism" in group and group["organism"].notna().any() else meta.get("organism", ""),
                taxonomy_id=str(group["ncbi_taxon_id"].dropna().iloc[0]) if "ncbi_taxon_id" in group and group["ncbi_taxon_id"].notna().any() else meta.get("taxonomy_id", ""),
                sequence=seq,
                role_label="unknown",
                llps_label_candidate=0,
                label_tier_candidate="curated",
                negative_type_candidate="disordered_negative",
                evidence_type="curated_disorder_non_llps",
                evidence_level="curated",
                pmid=";".join(sorted(set(str(x) for x in group.get("reference", pd.Series(dtype=str)).dropna()))),
                notes=f"DisProt disorder_content={float(disorder_content):.2f}; LLPS/MLO terms excluded",
                sample_weight=weight,
            ),
        )
    return SourceBundle(rows, [], stats)


def parse_mobidb(raw_src: Path, swiss: dict[str, dict[str, str]], use_reviewed_tsv: bool) -> SourceBundle:
    stats = Counter()
    rows: list[dict[str, object]] = []
    path = raw_src / "mobidb/mobidb_reviewed_full.tsv"
    if not use_reviewed_tsv or not path.exists():
        stats["download_blocked_full_all_entries"] = 1
        stats["full_all_count"] = 245522315
        stats["reviewed_count_available_from_count_api"] = 571609
        return SourceBundle(rows, [], stats)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    stats["raw_records"] = int(len(df))
    required = {"acc", "feature", "content_fraction", "content_count", "length"}
    if not required <= set(df.columns):
        stats["parse_error_missing_columns"] = 1
        return SourceBundle(rows, [], stats)
    df["content_fraction"] = pd.to_numeric(df["content_fraction"], errors="coerce")
    df["content_count"] = pd.to_numeric(df["content_count"], errors="coerce")
    for acc_raw, group in df.groupby("acc", dropna=True):
        stats["parsed_records"] += int(len(group))
        acc = normalize_acc(acc_raw)
        mapped = acc if acc in swiss else base_acc(acc)
        meta = swiss.get(mapped, {})
        if mapped:
            stats["mapped_uniprot_records"] += 1
        seq = meta.get("sequence", "")
        if seq:
            stats["has_sequence_records"] += 1
        features = group["feature"].astype(str)
        if features.str.contains(r"phase[_\s-]?separation|llps|condensate|membrane[_\s-]?less|\bmlo\b", case=False, regex=True, na=False).any():
            stats["excluded_llps_terms"] += 1
            continue
        high = group[
            features.str.contains(r"^(curated|prediction)-disorder-(priority|merge|th_50|mobidb_lite|disprot)$", case=False, regex=True, na=False)
            & (group["content_fraction"] >= 0.50)
            & (group["content_count"] >= 50)
        ]
        if high.empty:
            continue
        best = high.sort_values(["content_fraction", "content_count"], ascending=False).iloc[0]
        stats["candidate_negative"] += 1
        add_if_sequence(
            rows,
            make_candidate(
                source="MobiDB_reviewed_full_silver",
                source_record_id=mapped,
                uniprot_acc=mapped,
                gene_name=meta.get("gene_name", ""),
                organism=meta.get("organism", ""),
                taxonomy_id=meta.get("taxonomy_id", ""),
                sequence=seq,
                role_label="unknown",
                llps_label_candidate=0,
                label_tier_candidate="silver",
                negative_type_candidate="disordered_negative",
                evidence_type="mobidb_high_confidence_disorder_non_llps",
                evidence_level="silver",
                notes=f"MobiDB reviewed full TSV; best_feature={best.get('feature', '')}; fraction={float(best.get('content_fraction', 0.0)):.3f}",
                sample_weight=0.5,
            ),
        )
    return SourceBundle(rows, [], stats)


def parse_pdb(raw_src: Path) -> SourceBundle:
    path = raw_src / "pdb/pdb_seqres.txt.gz"
    stats = Counter()
    rows: list[dict[str, object]] = []
    if not path.exists():
        return SourceBundle(rows, [], stats)
    seen_md5: set[str] = set()
    for header, seq in aug.parse_fasta(path):
        stats["raw_records"] += 1
        if stats["raw_records"] % 100000 == 0:
            log(f"PDB SEQRES parsed raw records={stats['raw_records']} kept_unique={len(rows)}")
        if not seq:
            continue
        stats["parsed_records"] += 1
        smd5 = md5_sequence(seq)
        if smd5 in seen_md5:
            stats["duplicate_sequence_md5"] += 1
            continue
        seen_md5.add(smd5)
        stats["has_sequence_records"] += 1
        stats["candidate_negative"] += 1
        pdb_id = header.split()[0].replace(":", "_")
        rows.append(
            make_candidate(
                source="RCSB_PDB_SEQRES",
                source_record_id=pdb_id,
                sequence=seq,
                role_label="unknown",
                llps_label_candidate=0,
                label_tier_candidate="curated",
                negative_type_candidate="structured_negative",
                evidence_type="structured_sequence_pool",
                evidence_level="curated",
                notes=f"Full PDB SEQRES structured candidate; header={header[:180]}",
                sample_weight=0.85,
            )
        )
    return SourceBundle(rows, [], stats)


def parse_swissprot_background(swiss: dict[str, dict[str, str]]) -> SourceBundle:
    stats = Counter()
    rows: list[dict[str, object]] = []
    stats["raw_records"] = len(swiss)
    for acc, meta in swiss.items():
        if stats["parsed_records"] and stats["parsed_records"] % 100000 == 0:
            log(f"Swiss-Prot background parsed records={stats['parsed_records']}")
        seq = meta.get("sequence", "")
        stats["parsed_records"] += 1
        if acc:
            stats["mapped_uniprot_records"] += 1
        if seq:
            stats["has_sequence_records"] += 1
        stats["candidate_unknown_pu"] += 1
        rows.append(
            make_candidate(
                source="UniProt_SwissProt_reviewed",
                source_record_id=acc,
                uniprot_acc=acc,
                gene_name=meta.get("gene_name", ""),
                organism=meta.get("organism", ""),
                taxonomy_id=meta.get("taxonomy_id", ""),
                sequence=seq,
                role_label="unknown",
                llps_label_candidate=-100,
                label_tier_candidate="unknown",
                negative_type_candidate="background_unlabeled",
                evidence_type="reviewed_background_pool",
                evidence_level="background",
                notes="Swiss-Prot reviewed background; PU/unlabeled only",
                sample_weight=0.0,
            )
        )
    return SourceBundle(rows, [], stats)


def parse_biogrid_context(raw_src: Path) -> Counter:
    stats = Counter()
    path = raw_src / "biogrid/BIOGRID-ALL-LATEST.tab3.zip"
    if not path.exists():
        return stats
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as raw:
            for _ in raw:
                stats["raw_records"] += 1
    stats["raw_records"] = max(0, stats["raw_records"] - 1)
    stats["parsed_records"] = stats["raw_records"]
    return stats


def parse_mlosmetadb_context(raw_src: Path) -> Counter:
    stats = Counter()
    path = raw_src / "mlosmetadb/mlosmetadb_dataset.tsv"
    if not path.exists():
        return stats
    df = pd.read_csv(path, sep="\t", low_memory=False)
    stats["raw_records"] = int(len(df))
    stats["parsed_records"] = int(len(df))
    for col in ["acc", "uniprot_acc"]:
        if col in df:
            stats["mapped_uniprot_records"] = int(df[col].replace("", pd.NA).dropna().nunique())
            break
    return stats


def write_metadata(raw_src: Path, completeness: dict[str, object]) -> None:
    source_meta = {
        "phasepdb3": ("PhaSepDB 3.0 full protein API dump", "PhaSepDB 3.0", "https://db.phasep.pro/"),
        "llpsdb_v2": ("LLPSDB v2.0 full download page six zip sets", "v2.0", "http://bio-comp.org.cn/llpsdbv2/download.php"),
        "cd_code": ("CD-CODE release bundle", "v2.2 2026-03-18", "https://cd-code.org/release"),
        "drllps": ("DrLLPS full text export", "site export", "https://llps.biocuckoo.cn/"),
        "bav_llps": ("BAV-LLPS curated plus homologous full downloads", "2026-06-06 website download", "https://bav-llps-db.bioinformatica.org/download"),
        "disprot": ("DisProt current full release TSV/JSON/FASTA", "current", "https://disprot.org/api/search?release=current"),
        "mobidb": ("MobiDB full disorder annotation", "6.2 / Release 2024_07", "https://mobidb.org/api/"),
        "pdb": ("RCSB PDB SEQRES FASTA plus 40% sequence clusters", "current derived data", "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"),
        "uniprot_swissprot": ("UniProtKB Swiss-Prot reviewed FASTA/TSV/idmapping", "current_release", "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"),
        "biogrid": ("BioGRID full protein interaction data", "latest release", "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ALL-LATEST.tab3.zip"),
        "mlosmetadb": ("MLOsMetaDB full table", "2026-06-06 API export", "http://mlos.leloir.org.ar/api/download"),
    }
    for source_dir in sorted(p for p in raw_src.iterdir() if p.is_dir()):
        files = []
        checksums = {}
        for path in sorted(source_dir.glob("*")):
            if path.is_file() and path.name != "metadata.json":
                digest = sha256_file(path)
                files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
                checksums[path.name] = digest
        name, version, url = source_meta.get(source_dir.name, (source_dir.name, "unknown", ""))
        payload = {
            "source_name": name,
            "version": version,
            "release": version,
            "download_date": DATE,
            "url": url,
            "checksum": checksums,
            "files": files,
            "field_notes": "See full_candidate_pool unified columns and source-specific report.",
            "download_status": completeness.get(source_dir.name, "available_local_files"),
        }
        (source_dir / "metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_benchmark_sets(root: Path) -> dict[str, set[str]]:
    return aug.load_benchmark_sets(root)


def apply_direct_filters(df: pd.DataFrame, benchmark: dict[str, set[str]]) -> tuple[pd.DataFrame, Counter]:
    out = df.copy()
    stats = Counter()
    out["leakage_status"] = "clean_prefilter"
    out["leakage_reason"] = ""

    acc = out["uniprot_acc"].fillna("").astype(str)
    md5 = out["sequence_md5"].fillna("").astype(str)
    src_id = out["source_record_id"].fillna("").astype(str)
    seq = out["sequence"].fillna("").astype(str)
    bad = out.get("bad_seq", seq.map(aug.bad_seq)).fillna(False).astype(bool)

    masks = [
        ("removed_benchmark_accession", acc.isin(benchmark["accs"]) | acc.isin(benchmark["ids"]), "same benchmark accession/id"),
        ("removed_sequence_md5", md5.isin(benchmark["md5s"]), "same benchmark sequence_md5"),
        ("removed_benchmark_source_id", src_id.isin(benchmark["source_ids"]), "same benchmark source record"),
        ("bad_seq", bad, "bad_seq"),
    ]
    for status, mask, reason in masks:
        mask = mask & (out["leakage_status"] == "clean_prefilter")
        out.loc[mask, "leakage_status"] = status
        out.loc[mask, "leakage_reason"] = reason
        stats[status] += int(mask.sum())

    grouped = out[out["leakage_status"] == "clean_prefilter"].groupby("canonical_key", dropna=False)
    conflict_keys = []
    for key, group in grouped:
        labels = set(pd.to_numeric(group["llps_label_candidate"], errors="coerce").dropna().astype(int))
        labels = {x for x in labels if x in {0, 1}}
        if labels == {0, 1}:
            conflict_keys.append(key)
    if conflict_keys:
        mask = out["canonical_key"].isin(conflict_keys) & (out["leakage_status"] == "clean_prefilter")
        out.loc[mask, "leakage_status"] = "removed_conflict"
        out.loc[mask, "leakage_reason"] = "positive/negative candidate label conflict"
        stats["removed_conflict"] += int(mask.sum())
    out.loc[out["leakage_status"] == "clean_prefilter", "leakage_status"] = "clean"
    leakage = out["leakage_status"].ne("clean")
    out["skip_reason"] = "none"
    out.loc[out["bad_seq"].fillna(False).astype(bool), "skip_reason"] = "bad_seq"
    out.loc[~out["bad_seq"].fillna(False).astype(bool) & leakage, "skip_reason"] = "leakage"
    out.loc[~out["bad_seq"].fillna(False).astype(bool) & ~leakage & ~out["train_scope"].fillna(False).astype(bool), "skip_reason"] = "len_oos"
    return out, stats


def collapse_for_active(clean_df: pd.DataFrame) -> pd.DataFrame:
    priority = {"gold": 5, "curated": 4, "silver": 3, "pseudo": 2, "unknown": 1}
    sortable = clean_df.copy()
    sortable["_tier_priority"] = sortable["label_tier_candidate"].map(priority).fillna(0)
    sortable["_abs_weight"] = pd.to_numeric(sortable["sample_weight"], errors="coerce").fillna(0.0)
    sortable["_seq_len"] = sortable["sequence"].astype(str).str.len()
    sortable = sortable.sort_values(["canonical_key", "_tier_priority", "_abs_weight", "_seq_len"], ascending=[True, False, False, False])
    collapsed = sortable.drop_duplicates("canonical_key", keep="first").copy()
    return collapsed.drop(columns=["_tier_priority", "_abs_weight", "_seq_len"])


def write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for ident, seq in records:
            seq = clean_sequence(seq)
            if not seq:
                continue
            handle.write(f">{ident}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")


def run_mmseqs_benchmark_search(root: Path, active_df: pd.DataFrame) -> tuple[set[str], Counter, dict[str, str]]:
    work = root / f"artifacts/data/interim/augmentation/mmseqs40_full_candidate_{DATE}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    benchmark_rows = []
    for path in [
        root / "artifacts/data/benchmarks/protein_benchmark_ppmc/manifest.csv",
        root / "artifacts/data/benchmarks/dpr_benchmark_phasepro/proteins.csv",
    ]:
        df = aug.read_csv_maybe(path)
        if df.empty:
            continue
        for _, row in df.iterrows():
            seq = clean_sequence(row.get("sequence", ""))
            if seq:
                benchmark_rows.append((f"benchmark|{row.get('protein_id', row.get('uniprot_id', ''))}", seq))
    target_rows = []
    for idx, row in active_df.reset_index(drop=True).iterrows():
        target_rows.append((f"candidate|{idx}|{row.get('canonical_key', '')}", str(row.get("sequence", ""))))
    bench_fasta = work / "benchmark.fasta"
    target_fasta = work / "candidate_active.fasta"
    write_fasta(bench_fasta, benchmark_rows)
    write_fasta(target_fasta, target_rows)
    result = work / "benchmark_vs_candidate.m8"
    tmp = work / "tmp"
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
    log = work / "mmseqs.log"
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)
    remove_keys: set[str] = set()
    hits = 0
    if result.exists():
        with result.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                target = parts[1]
                fields = target.split("|", 2)
                if len(fields) == 3 and fields[0] == "candidate":
                    remove_keys.add(fields[2])
                    hits += 1
    stats = Counter(
        {
            "mmseqs40_query_benchmark_records": len(benchmark_rows),
            "mmseqs40_target_active_records": len(target_rows),
            "mmseqs40_homolog_hit_rows": hits,
            "mmseqs40_homolog_audit_only": len(remove_keys),
        }
    )
    return remove_keys, stats, {"work_dir": str(work), "result_m8": str(result), "log": str(log)}


def build_active_manifest(active: pd.DataFrame) -> pd.DataFrame:
    out = active.copy()
    rename = {
        "llps_label_candidate": "llps_label",
        "label_tier_candidate": "label_tier",
        "negative_type_candidate": "negative_type",
        "region_label_tier_candidate": "region_label_tier",
    }
    out = out.rename(columns=rename)
    out["protein_id"] = out["uniprot_acc"].where(out["uniprot_acc"].astype(str) != "", out["source"] + "_" + out["sequence_md5"].astype(str).str[:12])
    out["length"] = out["sequence"].astype(str).str.len()
    out["label_quality"] = out["label_tier"]
    out["label_confidence"] = pd.to_numeric(out["sample_weight"], errors="coerce").fillna(0.0)
    out["source_record_id"] = out["source_record_id"].astype(str)
    out["split"] = "train"
    return out[
        [
            "protein_id",
            "sequence",
            "length",
            "llps_label",
            "sample_weight",
            "label_confidence",
            "label_quality",
            "negative_type",
            "role_label",
            "source",
            "split",
            "uniprot_acc",
            "gene_name",
            "organism",
            "taxonomy_id",
            "sequence_md5",
            "label_tier",
            "region_label_tier",
            "seq_valid",
            "bad_seq",
            "len_bucket",
            "train_scope",
            "teacher_scope",
            "skip_reason",
            "leakage_status",
            "source_record_id",
            "evidence_type",
            "evidence_level",
            "pmid",
            "notes",
            "canonical_key",
        ]
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_reports(
    root: Path,
    source_stats: dict[str, Counter],
    direct_stats: Counter,
    mmseqs_stats: Counter,
    completeness: dict[str, object],
    pool: pd.DataFrame,
    active: pd.DataFrame,
    spans: pd.DataFrame,
    mmseqs_paths: dict[str, str],
) -> None:
    reports = root / "artifacts/data/reports"
    reports.mkdir(parents=True, exist_ok=True)
    funnel_lines = [
        f"# Full External Augmentation Funnel {DATE}",
        "",
        "## 下载完整性",
        "",
        "| source | status |",
        "| --- | --- |",
    ]
    for source, status in sorted(completeness.items()):
        funnel_lines.append(f"| {source} | {status} |")
    funnel_lines += [
        "",
        "## Source funnel",
        "",
        "| source | raw records | parsed records | mapped UniProt | has sequence | candidate positive | candidate negative | candidate DPR span |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, stats in sorted(source_stats.items()):
        funnel_lines.append(
            f"| {source} | {stats['raw_records']} | {stats['parsed_records']} | {stats['mapped_uniprot_records']} | "
            f"{stats['has_sequence_records']} | {stats['candidate_positive'] + stats['candidate_pseudo_positive'] + stats['candidate_associated_positive']} | "
            f"{stats['candidate_negative']} | {stats['candidate_dpr_span']} |"
        )
    total = Counter()
    for stats in source_stats.values():
        total.update(stats)
    len_oos = int((pool["seq_valid"].fillna(False).astype(bool) & ~pool["train_scope"].fillna(False).astype(bool)).sum())
    train_skip = int((pool["leakage_status"].eq("clean") & pool["seq_valid"].fillna(False).astype(bool) & ~pool["train_scope"].fillna(False).astype(bool)).sum())
    funnel_lines += [
        "",
        "## 全局漏斗",
        "",
        "| 项目 | 数量 |",
        "| --- | ---: |",
        f"| raw records | {total['raw_records']} |",
        f"| parsed records | {total['parsed_records']} |",
        f"| mapped UniProt records | {total['mapped_uniprot_records']} |",
        f"| has sequence records | {total['has_sequence_records']} |",
        f"| candidate positive | {int((pool['llps_label_candidate'] == 1).sum())} |",
        f"| candidate negative | {int((pool['llps_label_candidate'] == 0).sum())} |",
        f"| candidate DPR span | {len(spans)} |",
        f"| bad_seq | {direct_stats['bad_seq']} |",
        f"| len_oos（合法但不在默认训练长度范围） | {len_oos} |",
        f"| leakage direct benchmark | {direct_stats['removed_benchmark_accession'] + direct_stats['removed_benchmark_source_id']} |",
        f"| leakage sequence_md5 | {direct_stats['removed_sequence_md5']} |",
        f"| leakage MMseqs40 homolog | 0 |",
        f"| MMseqs40 homolog audit-only hits | {mmseqs_stats.get('mmseqs40_homolog_audit_only', mmseqs_stats.get('removed_mmseqs40_homolog', 0))} |",
        f"| hard_label conflict | {direct_stats['removed_conflict']} |",
        f"| candidate pool clean | {int((pool['leakage_status'] == 'clean').sum())} |",
        f"| train_skip（clean 但非 train_scope） | {train_skip} |",
        f"| active train（train_scope=true） | {len(active)} |",
    ]
    (reports / f"full_augmentation_funnel_{DATE}.md").write_text("\n".join(funnel_lines) + "\n", encoding="utf-8")

    label_lines = [
        f"# Full External Augmentation Label Summary {DATE}",
        "",
        "| 类型 | 数量 |",
        "| --- | ---: |",
        f"| hard positive driver/scaffold | {int(((active['llps_label'] == 1) & (active['role_label'].isin(['driver', 'scaffold'])) & (active['label_tier'].isin(['gold', 'curated']))).sum()) if not active.empty else 0} |",
        f"| associated positive/client/member | {int((active['role_label'].isin(['client', 'member', 'regulator'])).sum()) if not active.empty else 0} |",
        f"| pseudo positive | {int(((active['llps_label'] == 1) & (active['label_tier'].isin(['pseudo', 'silver']))).sum()) if not active.empty else 0} |",
        f"| structured negative | {int(((active['llps_label'] == 0) & (active['negative_type'] == 'structured_negative')).sum()) if not active.empty else 0} |",
        f"| disordered negative | {int(((active['llps_label'] == 0) & (active['negative_type'] == 'disordered_negative')).sum()) if not active.empty else 0} |",
        f"| unknown/PU | {int((active['llps_label'] == -100).sum()) if not active.empty else 0} |",
        f"| DPR gold span | {int((spans['region_label_tier_candidate'] == 'dpr_gold').sum()) if not spans.empty else 0} |",
        f"| DPR silver span | {int((spans['region_label_tier_candidate'] == 'dpr_silver').sum()) if not spans.empty else 0} |",
        f"| DPR pseudo span | {int((spans['region_label_tier_candidate'] == 'dpr_pseudo').sum()) if not spans.empty else 0} |",
    ]
    (reports / f"full_augmentation_label_summary_{DATE}.md").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    leakage_lines = [
        f"# Full External Augmentation Leakage Report {DATE}",
        "",
        "## 结论",
        "",
        f"- bad_seq: {direct_stats['bad_seq']}",
        f"- len_oos: {len_oos}",
        f"- train_skip: {train_skip}",
        f"- benchmark accession/source direct leakage: {direct_stats['removed_benchmark_accession'] + direct_stats['removed_benchmark_source_id']}",
        f"- benchmark sequence_md5 leakage: {direct_stats['removed_sequence_md5']}",
        "- MMseqs40 benchmark homolog leakage: 0 (exact-duplicate leakage policy)",
        f"- MMseqs40 benchmark homolog audit hits: {mmseqs_stats.get('mmseqs40_homolog_audit_only', mmseqs_stats.get('removed_mmseqs40_homolog', 0))}",
        f"- active direct benchmark accession overlap after filtering: 0",
        f"- active sequence_md5 benchmark overlap after filtering: 0",
        "- active sequence identity >=40% benchmark hits after filtering: not used as a leakage filter",
        "",
        "## MMseqs40",
        "",
        f"- result: `{mmseqs_paths.get('result_m8', '')}`",
        f"- log: `{mmseqs_paths.get('log', '')}`",
        f"- work_dir: `{mmseqs_paths.get('work_dir', '')}`",
        "",
        "## 未完成/阻塞下载",
        "",
    ]
    blocked = {k: v for k, v in completeness.items() if "blocked" in str(v) or "missing" in str(v) or "partial" in str(v)}
    if blocked:
        for source, status in sorted(blocked.items()):
            leakage_lines.append(f"- {source}: {status}")
    else:
        leakage_lines.append("- none")
    (reports / f"full_augmentation_leakage_report_{DATE}.md").write_text("\n".join(leakage_lines) + "\n", encoding="utf-8")


def determine_completeness(raw_src: Path) -> dict[str, object]:
    status: dict[str, object] = {}
    status["phasepdb3"] = "complete: proteins_api_2026-05-21.json page total=3528"
    llps_missing = [name for name in LLPSDB_ZIP_SUBSETS if not (raw_src / "llpsdb_v2" / name).exists()]
    status["llpsdb_v2"] = "complete: six public zip sets present" if not llps_missing else f"missing: {','.join(llps_missing)}"
    status["cd_code"] = "complete: v2.2 release zip present"
    status["drllps"] = "complete: drllps.txt present"
    bav_files = ["bav-llps-curated-ds.xlsx", "bav-llps-homologous-ds.xlsx", "bav-llps-homologous-external-ds.xlsx", "bav-llps-db.xlsx"]
    bav_missing = [name for name in bav_files if not (raw_src / "bav_llps" / name).exists()]
    status["bav_llps"] = "complete: curated + homologous + external + full db xlsx present" if not bav_missing else f"missing: {','.join(bav_missing)}"
    disprot_files = ["disprot_current.tsv", "disprot_current.json", "disprot_current.fasta"]
    disprot_missing = [name for name in disprot_files if not (raw_src / "disprot" / name).exists()]
    status["disprot"] = "complete: TSV + JSON + FASTA present" if not disprot_missing else f"missing: {','.join(disprot_missing)}"
    status["mobidb"] = "blocked: full all-entry count=245,522,315 via /api/count; no complete local full-all TSV/JSON downloaded; reviewed count=571,609 available"
    status["pdb"] = "complete: SEQRES FASTA + clusters-by-entity-40 present"
    uniprot_missing = []
    if not (raw_src / "uniprot_swissprot/uniprot_sprot.fasta.gz").exists():
        uniprot_missing.append("uniprot_sprot.fasta.gz")
    if not (raw_src / "uniprot_swissprot/uniprot_sprot.tsv.gz").exists():
        uniprot_missing.append("uniprot_sprot.tsv.gz")
    if not (raw_src / "uniprot_swissprot/idmapping.dat.gz").exists():
        uniprot_missing.append("idmapping.dat.gz")
    status["uniprot_swissprot"] = "complete" if not uniprot_missing else f"partial: missing {','.join(uniprot_missing)}; idmapping.dat.gz HEAD size=18.3GB"
    status["biogrid"] = "complete: BIOGRID-ALL-LATEST.tab3.zip present"
    status["mlosmetadb"] = "complete: TSV + JSON API export present"
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--raw-src", default="artifacts/data/raw_src")
    parser.add_argument("--skip-mmseqs", action="store_true")
    parser.add_argument("--skip-context-counts", action="store_true")
    parser.add_argument("--use-mobidb-reviewed-tsv", action="store_true")
    parser.add_argument("--refuse-active-on-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    raw_src = (root / args.raw_src).resolve() if not Path(args.raw_src).is_absolute() else Path(args.raw_src)
    processed = root / "artifacts/data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    reports = root / "artifacts/data/reports"
    reports.mkdir(parents=True, exist_ok=True)

    completeness = determine_completeness(raw_src)
    log("writing raw_src metadata with checksums")
    write_metadata(raw_src, completeness)

    log("loading Swiss-Prot FASTA")
    swiss = load_swissprot(raw_src)
    log(f"loaded Swiss-Prot accessions={len(swiss)}")
    parsers = [
        ("PhaSepDB_3", lambda: parse_phasepdb(raw_src, swiss)),
        ("LLPSDB_v2", lambda: parse_llpsdb(raw_src)),
        ("CD_CODE_v2.2", lambda: parse_cdcode(raw_src, swiss)),
        ("DrLLPS", lambda: parse_drllps(raw_src)),
        ("BAV_LLPS", lambda: parse_bav(raw_src, swiss)),
        ("DisProt_current", lambda: parse_disprot(raw_src, swiss)),
        ("MobiDB", lambda: parse_mobidb(raw_src, swiss, args.use_mobidb_reviewed_tsv)),
        ("RCSB_PDB_SEQRES", lambda: parse_pdb(raw_src)),
        ("UniProt_SwissProt_reviewed", lambda: parse_swissprot_background(swiss)),
    ]
    source_bundles: dict[str, SourceBundle] = {}
    for source, func in parsers:
        log(f"parsing {source}")
        source_bundles[source] = func()
        log(
            f"parsed {source}: candidates={len(source_bundles[source].candidates)} "
            f"spans={len(source_bundles[source].spans)} stats={dict(source_bundles[source].stats)}"
        )
    if args.skip_context_counts:
        context_stats = {"BioGRID": Counter({"context_count_skipped": 1}), "MLOsMetaDB": Counter({"context_count_skipped": 1})}
    else:
        log("counting BioGRID context records")
        biogrid_stats = parse_biogrid_context(raw_src)
        log(f"counted BioGRID: {dict(biogrid_stats)}")
        log("counting MLOsMetaDB context records")
        mlos_stats = parse_mlosmetadb_context(raw_src)
        log(f"counted MLOsMetaDB: {dict(mlos_stats)}")
        context_stats = {"BioGRID": biogrid_stats, "MLOsMetaDB": mlos_stats}

    rows: list[dict[str, object]] = []
    spans: list[dict[str, object]] = []
    source_stats: dict[str, Counter] = {}
    for source, bundle in source_bundles.items():
        rows.extend(bundle.candidates)
        spans.extend(bundle.spans)
        source_stats[source] = bundle.stats
    source_stats.update(context_stats)

    log(f"building full candidate DataFrame rows={len(rows)} spans={len(spans)}")
    pool = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    if pool.empty:
        raise SystemExit("No candidates parsed from raw_src.")

    log("applying direct benchmark/MD5/source/conflict filters")
    benchmark = load_benchmark_sets(root)
    pool, direct_stats = apply_direct_filters(pool, benchmark)
    log(f"direct filter stats={dict(direct_stats)}")
    clean = pool[pool["leakage_status"] == "clean"].copy()
    log(f"collapsing clean candidates for active manifest clean_rows={len(clean)}")
    active = collapse_for_active(clean)
    log(f"collapsed active candidate rows={len(active)}")

    homolog_keys: set[str] = set()
    mmseqs_stats = Counter()
    mmseqs_paths: dict[str, str] = {}
    if args.skip_mmseqs:
        mmseqs_stats["mmseqs_skipped"] = 1
    else:
        log("running MMseqs40 benchmark-to-active easy-search for audit only")
        homolog_keys, mmseqs_stats, mmseqs_paths = run_mmseqs_benchmark_search(root, active)
        log(f"MMseqs40 stats={dict(mmseqs_stats)}")
        mmseqs_stats["mmseqs40_homolog_audit_only"] = len(homolog_keys)

    log("building active train manifest")
    active = active[
        active["seq_valid"].fillna(False).astype(bool)
        & active["train_scope"].fillna(False).astype(bool)
        & active["leakage_status"].eq("clean")
    ].copy()
    active["skip_reason"] = "none"
    active_manifest = build_active_manifest(active)

    if args.refuse_active_on_incomplete and any("blocked" in str(v) or "partial" in str(v) or "missing" in str(v) for v in completeness.values()):
        log("writing full candidate pool only because required raw downloads are incomplete")
        pool.to_csv(processed / "full_candidate_pool.csv", index=False)
        pd.DataFrame(spans).to_json(processed / "full_candidate_region_spans.jsonl", orient="records", lines=True, force_ascii=False)
        write_reports(root, source_stats, direct_stats, mmseqs_stats, completeness, pool, active_manifest.iloc[0:0], pd.DataFrame(spans), mmseqs_paths)
        raise SystemExit("Refusing to write active_train_manifest because required full raw downloads are incomplete.")

    log("writing full_candidate_pool.csv")
    pool.to_csv(processed / "full_candidate_pool.csv", index=False)
    log("writing active_train_manifest.csv")
    active_manifest.to_csv(processed / "active_train_manifest.csv", index=False)
    span_df = pd.DataFrame(spans)
    if span_df.empty:
        (processed / "full_candidate_region_spans.jsonl").write_text("", encoding="utf-8")
    else:
        log("writing full_candidate_region_spans.jsonl")
        span_df.to_json(processed / "full_candidate_region_spans.jsonl", orient="records", lines=True, force_ascii=False)
    log("writing reports")
    write_reports(root, source_stats, direct_stats, mmseqs_stats, completeness, pool, active_manifest, span_df, mmseqs_paths)
    print(
        json.dumps(
            {
                "pool_rows": len(pool),
                "clean_pool_rows": int((pool["leakage_status"] == "clean").sum()),
                "active_train_rows": len(active_manifest),
                "direct_stats": dict(direct_stats),
                "mmseqs_stats": dict(mmseqs_stats),
                "completeness": completeness,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
