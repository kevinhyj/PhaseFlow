#!/usr/bin/env python3
"""Build leakage-clean augmented train data from external LLPS sources.

This script intentionally writes only augmented outputs.  It does not mutate the
current active train manifest, feature H5 files, or benchmark directories.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import pandas as pd


DATE = "20260606"
CANON_COLUMNS = [
    "source",
    "source_record_id",
    "uniprot_acc",
    "gene_name",
    "organism",
    "taxonomy_id",
    "sequence",
    "sequence_md5",
    "label_candidate",
    "role",
    "evidence_type",
    "evidence_level",
    "pmid",
    "region_start",
    "region_end",
    "region_type",
    "notes",
]
AMINO_ACID_RE = re.compile(r"[^A-Z]")
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
TRAIN_MIN_LEN = 30
TRAIN_MAX_LEN = 2048
TEACHER_MIN_LEN = 30
TEACHER_LONG_MAX = 2700
TEACHER_AUDIT_MAX = 5537
UNIPROT_RE = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:-\d+)?$|^[A-Z0-9]{10}(?:-\d+)?$")
RANGE_RE = re.compile(r"(?<!\d)(\d{1,5})\s*[-:–]\s*(\d{1,5})(?!\d)")
DEPENDENCY_PATTERNS = [
    r"\bpartner[-\s]?dependent\b",
    r"\bco[-\s]?condens",
    r"\brequires?\s+(rna|dna|partner|ptm|phosphorylation|mutation)",
    r"\bdependent\s+on\s+(rna|dna|partner|ptm|phosphorylation|mutation)",
    r"\bin\s+the\s+presence\s+of\s+(rna|dna|partner)",
    r"\bwith\s+(rna|dna)\b",
    r"\brna[-\s]?dependent\b",
    r"\bdna[-\s]?dependent\b",
    r"\bmutant\b",
    r"\bmutation\b",
    r"\bphosphorylat",
    r"\bpost[-\s]?translational\b",
    r"\bptm\b",
    r"\brepeat\s+(expansion|dependent)",
    r"\bsplice\b",
    r"\bsplicing\b",
]
DEPENDENCY_RE = re.compile("|".join(DEPENDENCY_PATTERNS), re.I)
LLPS_TERM_RE = re.compile(r"llps|liquid[-\s]?liquid|phase separation|condensate|membrane[-\s]?less|mlo|coacerv", re.I)


@dataclass
class Candidate:
    source: str
    source_record_id: str
    uniprot_acc: str
    gene_name: str
    organism: str
    taxonomy_id: str
    sequence: str
    label_candidate: str
    role: str
    evidence_type: str
    evidence_level: str
    pmid: str = ""
    region_start: str = ""
    region_end: str = ""
    region_type: str = ""
    notes: str = ""
    llps_label: int = -100
    label_tier: str = "unknown"
    role_label: str = "unknown"
    negative_type: str = "background_unlabeled"
    region_label_tier: str = "none"
    sample_weight: float = 0.0
    sample_origin: str = "augmentation"
    sequence_md5: str = field(init=False)

    def __post_init__(self) -> None:
        self.sequence = clean_sequence(self.sequence)
        self.sequence_md5 = md5_sequence(self.sequence) if self.sequence else ""
        self.uniprot_acc = normalize_acc(self.uniprot_acc)

    def canonical_row(self) -> dict[str, object]:
        return {
            "source": self.source,
            "source_record_id": self.source_record_id,
            "uniprot_acc": self.uniprot_acc,
            "gene_name": self.gene_name,
            "organism": self.organism,
            "taxonomy_id": self.taxonomy_id,
            "sequence": self.sequence,
            "sequence_md5": self.sequence_md5,
            "label_candidate": self.label_candidate,
            "role": self.role,
            "evidence_type": self.evidence_type,
            "evidence_level": self.evidence_level,
            "pmid": self.pmid,
            "region_start": self.region_start,
            "region_end": self.region_end,
            "region_type": self.region_type,
            "notes": self.notes,
        }

    def manifest_row(self, protein_id: str | None = None) -> dict[str, object]:
        pid = protein_id or self.uniprot_acc or f"{self.source}_{self.sequence_md5[:12]}"
        confidence = 1.0 if self.sample_weight >= 0.8 else self.sample_weight
        return {
            "protein_id": pid,
            "sequence": self.sequence,
            "length": len(self.sequence),
            "llps_label": self.llps_label,
            "sample_weight": self.sample_weight,
            "label_confidence": confidence,
            "label_quality": self.label_tier,
            "evidence_level": self.evidence_level,
            "negative_type": self.negative_type,
            "role_label": self.role_label,
            "source": self.source,
            "split": "train",
            "cluster_id_30": "",
            "cluster_id_50": "",
            "teacher_consensus_score": "",
            "teacher_consensus_teachers": "",
            "uniprot_acc": self.uniprot_acc,
            "gene_name": self.gene_name,
            "organism": self.organism,
            "taxonomy_id": self.taxonomy_id,
            "sequence_md5": self.sequence_md5,
            "label_tier": self.label_tier,
            "region_label_tier": self.region_label_tier,
            "leakage_status": "clean",
            "source_record_id": self.source_record_id,
            "evidence_type": self.evidence_type,
            "pmid": self.pmid,
            "notes": self.notes,
            "sample_origin": self.sample_origin,
        }


def clean_sequence(seq: object) -> str:
    if seq is None or pd.isna(seq):
        return ""
    text = str(seq).upper().replace("*", "")
    if text.startswith(">"):
        parts = text.splitlines()
        text = "".join(parts[1:])
    return AMINO_ACID_RE.sub("", text)


def normalize_acc(acc: object) -> str:
    if acc is None or pd.isna(acc):
        return ""
    acc = str(acc).strip()
    if not acc or acc in {"-", "nan", "None"}:
        return ""
    if "|" in acc:
        fields = acc.split("|")
        if len(fields) >= 2 and UNIPROT_RE.match(fields[1]):
            return fields[1]
    acc = acc.split(";")[0].split(",")[0].strip()
    return acc


def base_acc(acc: str) -> str:
    return normalize_acc(acc).split("-", 1)[0]


def md5_sequence(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_maybe(path: Path, sep: str = ",") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep=sep, low_memory=False)


def resolve_raw_external_dir(root: Path, raw_external_dir: str | None = None) -> Path:
    if raw_external_dir:
        path = Path(raw_external_dir)
        return path if path.is_absolute() else root / path
    dated = root / f"artifacts/data/raw_external/{DATE}"
    if dated.exists():
        return dated
    flat = root / "artifacts/data/raw_external"
    if flat.exists():
        return flat
    return dated


def parse_fasta(path: Path) -> Iterable[tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = ""
        chunks: list[str] = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    yield header, clean_sequence("".join(chunks))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header:
            yield header, clean_sequence("".join(chunks))


def parse_swissprot_fasta(path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for header, seq in parse_fasta(path):
        if not seq:
            continue
        parts = header.split("|")
        acc = normalize_acc(parts[1] if len(parts) > 1 else parts[0].split()[0])
        if not acc:
            continue
        gene = ""
        organism = ""
        tax = ""
        gn = re.search(r"\bGN=([^=]+?)(?:\s[A-Z]{2}=|$)", header)
        os_match = re.search(r"\bOS=([^=]+?)(?:\sOX=|$)", header)
        ox = re.search(r"\bOX=(\d+)", header)
        if gn:
            gene = gn.group(1).strip()
        if os_match:
            organism = os_match.group(1).strip()
        if ox:
            tax = ox.group(1)
        records[acc] = {
            "sequence": seq,
            "gene_name": gene,
            "organism": organism,
            "taxonomy_id": tax,
            "sequence_md5": md5_sequence(seq),
        }
    return records


def parse_xlsx_first_sheet(path: Path) -> list[dict[str, str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in item.findall(".//a:t", ns)))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in sheet.findall(".//a:sheetData/a:row", ns):
            values: list[str] = []
            last_col = 0
            for cell in row.findall("a:c", ns):
                ref = cell.get("r", "")
                col_letters = re.sub(r"\d", "", ref)
                col = 0
                for char in col_letters:
                    col = col * 26 + ord(char.upper()) - ord("A") + 1
                while last_col + 1 < col:
                    values.append("")
                    last_col += 1
                value_node = cell.find("a:v", ns)
                value = "" if value_node is None else value_node.text or ""
                if cell.get("t") == "s" and value:
                    value = shared[int(value)]
                values.append(value)
                last_col = col
            rows.append(values)
    if not rows:
        return []
    headers = [h.strip() or f"unnamed_{i}" for i, h in enumerate(rows[0])]
    out = []
    for row in rows[1:]:
        if not any(row):
            continue
        row = row + [""] * (len(headers) - len(row))
        out.append(dict(zip(headers, row[: len(headers)])))
    return out


def parse_range_list(value: object, seq_len: int) -> list[tuple[int, int]]:
    if value is None or pd.isna(value):
        return []
    ranges = []
    for start, end in RANGE_RE.findall(str(value)):
        s, e = int(start), int(end)
        if 1 <= s <= e <= seq_len:
            ranges.append((s, e))
    return ranges


def is_dependency_text(text: str) -> bool:
    return bool(DEPENDENCY_RE.search(text or ""))


def bad_seq(seq: str) -> bool:
    return not bool(seq) or not set(seq) <= VALID_AA


def seq_valid(seq: str) -> bool:
    return not bad_seq(seq)


def len_bucket(length: int) -> str:
    if length < 30:
        return "short_lt30"
    if length < 100:
        return "short_30_100"
    if length <= 2048:
        return "normal_100_2048"
    if length <= 2700:
        return "long_2048_2700"
    if length <= 5537:
        return "very_long_2700_5537"
    return "ultra_long_gt5537"


def train_scope_sequence(seq: str, min_len: int = TRAIN_MIN_LEN, max_len: int = TRAIN_MAX_LEN) -> bool:
    return seq_valid(seq) and min_len <= len(seq) <= max_len


def hard_label(llps_label: object, role_label: object, label_tier: object) -> bool:
    try:
        label = int(float(str(llps_label).strip()))
    except Exception:
        label = -100
    role = str(role_label or "").strip().lower()
    tier = str(label_tier or "").strip().lower()
    return label == 1 and role in {"driver", "scaffold"} and tier in {"gold", "curated"}


def manual_construct(*texts: object) -> bool:
    haystack = " ".join(str(text or "") for text in texts).lower()
    return bool(re.search(r"\b(construct|manual|curated_construct|人工确认)\b", haystack))


def teacher_scope_sequence(
    seq: str,
    *,
    llps_label: object = -100,
    role_label: object = "",
    label_tier: object = "",
    notes: object = "",
    evidence_type: object = "",
    evidence_level: object = "",
) -> bool:
    if bad_seq(seq):
        return False
    length = len(seq)
    if length < TEACHER_MIN_LEN:
        return hard_label(llps_label, role_label, label_tier) or manual_construct(notes, evidence_type, evidence_level)
    if length <= TEACHER_AUDIT_MAX:
        return True
    return False


def length_scope_fields(
    seq: str,
    *,
    llps_label: object = -100,
    role_label: object = "",
    label_tier: object = "",
    notes: object = "",
    evidence_type: object = "",
    evidence_level: object = "",
) -> dict[str, object]:
    length = len(seq or "")
    bad = bad_seq(seq)
    train_scope = train_scope_sequence(seq)
    teacher_scope = teacher_scope_sequence(
        seq,
        llps_label=llps_label,
        role_label=role_label,
        label_tier=label_tier,
        notes=notes,
        evidence_type=evidence_type,
        evidence_level=evidence_level,
    )
    return {
        "seq_valid": not bad,
        "bad_seq": bad,
        "len_bucket": len_bucket(length),
        "train_scope": train_scope,
        "teacher_scope": teacher_scope,
    }


def skip_reason(
    *,
    bad_seq: bool,
    train_scope: bool,
    teacher_scope: bool,
    leakage: bool = False,
    hard_label: bool = False,
    context: str = "candidate",
) -> str:
    if bad_seq:
        return "bad_seq"
    if leakage:
        return "leakage"
    if context == "teacher":
        if hard_label:
            return "hard_label"
        if not teacher_scope:
            return "teacher_skip"
        return "none"
    if context == "train":
        if not train_scope:
            return "train_skip"
        return "none"
    if not train_scope:
        return "len_oos"
    return "none"


def valid_train_sequence(seq: str, min_len: int = TRAIN_MIN_LEN, max_len: int = TRAIN_MAX_LEN) -> bool:
    """Compatibility wrapper: this now means current train scope, not sequence validity."""
    return train_scope_sequence(seq, min_len=min_len, max_len=max_len)


def add_candidate(cands: list[Candidate], cand: Candidate) -> None:
    if seq_valid(cand.sequence):
        cands.append(cand)


def load_benchmark_sets(root: Path) -> dict[str, set[str]]:
    accs: set[str] = set()
    md5s: set[str] = set()
    ids: set[str] = set()
    source_ids: set[str] = set()

    for path in [
        root / "artifacts/data/benchmarks/protein_benchmark_ppmc/manifest.csv",
        root / "artifacts/data/benchmarks/protein_benchmark_ppmc/ppmc_ce_de_c_d_np_nd_raw.tsv",
        root / "artifacts/data/benchmarks/dpr_benchmark_phasepro/proteins.csv",
    ]:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False) if path.suffix == ".tsv" and path.exists() else read_csv_maybe(path)
        if df.empty:
            continue
        for col in ["protein_id", "uniprot_id", "UniProt.Acc", "uniprot_accession_norm"]:
            if col in df:
                ids.update(str(x) for x in df[col].dropna().astype(str) if x)
                accs.update(normalize_acc(x) for x in df[col].dropna().astype(str))
        if "mapped_protein_ids" in df:
            for value in df["mapped_protein_ids"].dropna().astype(str):
                for part in value.split(";"):
                    if part:
                        ids.add(part)
                        accs.add(normalize_acc(part))
        if "sequence_md5" in df:
            md5s.update(str(x) for x in df["sequence_md5"].dropna().astype(str) if x)
        for seq_col in ["sequence", "Full.seq"]:
            if seq_col in df:
                md5s.update(md5_sequence(clean_sequence(x)) for x in df[seq_col].dropna())

    for path in [
        root / "artifacts/data/benchmarks/protein_benchmark_ppmc/source_records.csv",
        root / "artifacts/data/benchmarks/protein_benchmark_ppmc/source_map.csv",
        root / "artifacts/data/benchmarks/dpr_benchmark_phasepro/source_records.csv",
        root / "artifacts/data/benchmarks/dpr_benchmark_phasepro/source_map.csv",
    ]:
        df = read_csv_maybe(path)
        if df.empty:
            continue
        for col in ["source_id", "source_record_id", "protein_id", "uniprot_id"]:
            if col in df:
                source_ids.update(str(x) for x in df[col].dropna().astype(str) if x and x != "nan")

    audit = read_csv_maybe(root / "artifacts/data/processed/qc/leakage_cleanup_audit_20260606.csv")
    if not audit.empty:
        removed = audit[audit.get("action", "") == "remove"] if "action" in audit else audit
        for col in ["protein_id", "uniprot_id"]:
            if col in removed:
                ids.update(str(x) for x in removed[col].dropna().astype(str) if x)
                accs.update(normalize_acc(x) for x in removed[col].dropna().astype(str))
        if "sequence_md5" in removed:
            md5s.update(str(x) for x in removed["sequence_md5"].dropna().astype(str) if x)

    return {
        "accs": {x for x in accs if x},
        "md5s": {x for x in md5s if x},
        "ids": {x for x in ids if x},
        "source_ids": {x for x in source_ids if x},
    }


def load_active_manifest(root: Path) -> pd.DataFrame:
    active = read_csv_maybe(root / "artifacts/data/pseudo_labels/round0_external/manifest_with_teacher.csv")
    proteins = read_csv_maybe(root / "artifacts/data/processed/proteins.csv")
    prot_cols = ["protein_id", "uniprot_id", "gene_name", "species", "tax_id", "sequence_md5"]
    if not active.empty and not proteins.empty:
        active = active.merge(proteins[[c for c in prot_cols if c in proteins.columns]], on="protein_id", how="left")
    if "sequence_md5" not in active:
        active["sequence_md5"] = active["sequence"].map(lambda x: md5_sequence(clean_sequence(x)))
    else:
        active["sequence_md5"] = active["sequence_md5"].fillna(active["sequence"].map(lambda x: md5_sequence(clean_sequence(x))))
    active["uniprot_acc"] = active.get("uniprot_id", active["protein_id"]).map(normalize_acc)
    active["gene_name"] = active.get("gene_name", "").fillna("")
    active["organism"] = active.get("species", "").fillna("")
    active["taxonomy_id"] = active.get("tax_id", "").fillna("")
    active["label_tier"] = active.get("label_quality", "").fillna("").replace({"hard_positive": "curated", "PU": "unknown"})
    pseudo_mask = (
        (active["llps_label"] == 1)
        & active["role_label"].astype(str).isin({"", "unknown", "nan"})
        & active["label_tier"].astype(str).isin({"ambiguous", "unknown"})
    )
    active.loc[pseudo_mask, "label_tier"] = "pseudo"
    active.loc[pseudo_mask, "label_quality"] = "pseudo"
    active.loc[pseudo_mask, "role_label"] = "teacher_positive"
    active.loc[pseudo_mask, "sample_weight"] = pd.to_numeric(active.loc[pseudo_mask, "sample_weight"], errors="coerce").fillna(0.5).clip(upper=0.5)
    active["region_label_tier"] = "none"
    active["leakage_status"] = "clean"
    active["source_record_id"] = ""
    active["evidence_type"] = "active_train"
    active["pmid"] = ""
    active["notes"] = "carried_from_clean_active_train_20260606"
    active["sample_origin"] = "active_train_20260606"
    return active


def load_phasepdb_candidates(root: Path, raw_dir: Path, swiss: dict[str, dict[str, str]]) -> tuple[list[Candidate], list[dict[str, object]]]:
    raw_json = raw_dir / "phasepdb3/proteins_api_2026-05-21.json"
    if raw_json.exists():
        payload = json.loads(raw_json.read_text(encoding="utf-8"))
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        df = pd.DataFrame(rows)
    else:
        path = root / "artifacts/data/interim/parsed_source_tables/phasepdb3/phasepdb3_proteins.csv"
        df = read_csv_maybe(path)
    cands: list[Candidate] = []
    spans: list[dict[str, object]] = []
    if df.empty:
        return cands, spans
    for _, row in df.iterrows():
        acc = normalize_acc(row.get("uniprot_id", ""))
        rec = str(row.get("protein_id", ""))
        text = " ".join(str(row.get(c, "")) for c in df.columns if c not in {"sequence"})
        if str(row.get("class_", "")).strip() != "PS-self":
            continue
        if str(row.get("_status", "")).strip().lower() not in {"approved", ""}:
            continue
        if is_dependency_text(text):
            continue
        if acc not in swiss:
            continue
        meta = swiss[acc]
        cand = Candidate(
            source="PhaSepDB_3",
            source_record_id=rec,
            uniprot_acc=acc,
            gene_name=meta.get("gene_name", ""),
            organism=str(row.get("organism", "")) or meta.get("organism", ""),
            taxonomy_id=meta.get("taxonomy_id", ""),
            sequence=meta["sequence"],
            label_candidate="positive",
            role="driver",
            evidence_type="curated_database",
            evidence_level="curated",
            pmid=str(row.get("pmid", "")),
            notes="strict_PS_self_no_dependency_terms",
            llps_label=1,
            label_tier="curated",
            role_label="driver",
            negative_type="none",
            sample_weight=1.0,
        )
        add_candidate(cands, cand)
        region_text = str(row.get("key_protein_regions_studied_ps", ""))
        for s, e in parse_range_list(region_text, len(cand.sequence)):
            spans.append(
                {
                    "protein_id": acc,
                    "source": "PhaSepDB_3",
                    "source_record_id": rec,
                    "start": s,
                    "end": e,
                    "region_type": "phase_separation_region",
                    "region_label_tier": "dpr_silver",
                    "notes": region_text[:300],
                }
            )
    return cands, spans


def load_llpsdb_candidates(root: Path) -> tuple[list[Candidate], list[dict[str, object]]]:
    proteins = read_csv_maybe(root / "artifacts/data/interim/parsed_source_tables/llpsdb_v2/llpsdb_v2_proteins.csv")
    cond = read_csv_maybe(root / "artifacts/data/interim/parsed_source_tables/llpsdb_v2/llpsdb_v2_conditions.csv")
    cands: list[Candidate] = []
    spans: list[dict[str, object]] = []
    if proteins.empty or cond.empty:
        return cands, spans
    strict_ids: set[str] = set()
    for _, row in cond.iterrows():
        if str(row.get("source_subset", "")) != "Phase_separation_Unambiguous":
            continue
        if str(row.get("condition_label", "")) != "phase_separation":
            continue
        if str(row.get("Components type", row.get("Components_type", ""))).lower().strip() != "protein(1)":
            continue
        if str(row.get("Protein type (N/D)", "")).strip() != "N":
            continue
        bad_flags = [row.get("Fusion", "-"), row.get("Cleaved", "-"), row.get("Repeat", "-"), row.get("Mutation", "-"), row.get("PTM", "-")]
        if any(str(x).strip() not in {"-", "", "nan"} for x in bad_flags):
            continue
        if str(row.get("Nucleic acid", row.get("Nucleic_acid1", "-"))).strip() not in {"-", "", "nan"}:
            continue
        if is_dependency_text(" ".join(str(row.get(c, "")) for c in cond.columns)):
            continue
        strict_ids.add(str(row.get("Protein ID", row.get("protID", ""))).strip())
    for _, row in proteins.iterrows():
        pid = str(row.get("source_protein_id", "")).strip()
        if pid not in strict_ids:
            continue
        seq = clean_sequence(row.get("sequence_clean", row.get("sequence", "")))
        acc = normalize_acc(row.get("uniprot_id", ""))
        cand = Candidate(
            source="LLPSDB_v2",
            source_record_id=pid,
            uniprot_acc=acc,
            gene_name=str(row.get("gene_name", "")),
            organism=str(row.get("organism", "")),
            taxonomy_id=str(row.get("NCBI code", "")) if not pd.isna(row.get("NCBI code", "")) else "",
            sequence=seq,
            label_candidate="positive",
            role="driver",
            evidence_type="curated_database",
            evidence_level="curated",
            notes="phase_separation_unambiguous_protein1_natural_no_mutation_ptm_repeat_nucleic_acid",
            llps_label=1,
            label_tier="curated",
            role_label="driver",
            negative_type="none",
            sample_weight=0.9,
        )
        add_candidate(cands, cand)
        for field_name, region_type in [("IDR", "IDR"), ("LCR", "LCR")]:
            for s, e in parse_range_list(row.get(field_name, ""), len(seq)):
                spans.append(
                    {
                        "protein_id": acc or pid,
                        "source": "LLPSDB_v2",
                        "source_record_id": pid,
                        "start": s,
                        "end": e,
                        "region_type": region_type,
                        "region_label_tier": "dpr_silver",
                        "notes": f"{field_name} annotation from LLPSDB v2 protein table",
                    }
                )
    return cands, spans


def load_drllps_candidates(raw_dir: Path) -> list[Candidate]:
    df = read_csv_maybe(raw_dir / "drllps/drllps.txt", sep="\t")
    cands: list[Candidate] = []
    if df.empty:
        return cands
    for _, row in df.iterrows():
        llps_type = str(row.get("LLPS Type", "")).strip().lower()
        if llps_type != "scaffold":
            continue
        seq = clean_sequence(row.get("Protein Sequence", ""))
        cand = Candidate(
            source="DrLLPS",
            source_record_id=str(row.get("DrLLPS ID", "")),
            uniprot_acc=normalize_acc(row.get("UniProt ID", "")),
            gene_name=str(row.get("Gene name", "")),
            organism=str(row.get("Species", "")),
            taxonomy_id="",
            sequence=seq,
            label_candidate="positive",
            role="scaffold",
            evidence_type="curated_database",
            evidence_level="curated",
            pmid=str(row.get("References", "")),
            notes=f"DrLLPS scaffold; condensate={row.get('Condensate', '')}",
            llps_label=1,
            label_tier="curated",
            role_label="scaffold",
            negative_type="none",
            sample_weight=1.0,
        )
        add_candidate(cands, cand)
    return cands


def load_bav_candidates(raw_dir: Path) -> list[Candidate]:
    path = raw_dir / "bav_llps/bav-llps-curated-ds.xlsx"
    cands: list[Candidate] = []
    if not path.exists():
        return cands
    for row in parse_xlsx_first_sheet(path):
        seq = clean_sequence(row.get("sequence", ""))
        acc = normalize_acc(row.get("uniprot", row.get("id", "")))
        cand = Candidate(
            source="BAV_LLPS_curated",
            source_record_id=str(row.get("id", acc)),
            uniprot_acc=acc,
            gene_name="",
            organism=str(row.get("specie", "")),
            taxonomy_id="",
            sequence=seq,
            label_candidate="positive",
            role="driver",
            evidence_type="curated_database",
            evidence_level="curated",
            notes=f"curated BAV-LLPS only; organism_type={row.get('organism_type', '')}; mlo_bmc={row.get('mlo_bmc', '')}",
            llps_label=1,
            label_tier="curated",
            role_label="driver",
            negative_type="none",
            sample_weight=0.7,
        )
        add_candidate(cands, cand)
    return cands


def load_cdcode_candidates(raw_dir: Path, swiss: dict[str, dict[str, str]]) -> list[Candidate]:
    path = raw_dir / "cd_code/cd-code_release_v2.2_2026-03-18.zip"
    cands: list[Candidate] = []
    if not path.exists():
        return cands
    with zipfile.ZipFile(path) as z:
        with z.open("02/proteins_202603181653.csv") as handle:
            proteins = pd.read_csv(handle, low_memory=False)
    for _, row in proteins.iterrows():
        acc = normalize_acc(row.get("uniprot_id", ""))
        if not acc or acc not in swiss:
            continue
        meta = swiss[acc]
        text = " ".join(str(row.get(c, "")) for c in ["name", "function"])
        is_driver = bool(re.search(r"\b(driver|scaffold)\b", text, re.I)) and not re.search(r"\b(client|member|regulator)\b", text, re.I)
        cand = Candidate(
            source="CD_CODE_v2.2",
            source_record_id=acc,
            uniprot_acc=acc,
            gene_name=str(row.get("gene_name", "")) or meta.get("gene_name", ""),
            organism=str(row.get("species_name", "")) or meta.get("organism", ""),
            taxonomy_id=str(row.get("species_taxon_id", "")) if not pd.isna(row.get("species_taxon_id", "")) else meta.get("taxonomy_id", ""),
            sequence=meta["sequence"],
            label_candidate="positive" if is_driver else "unknown",
            role="driver" if is_driver else "member",
            evidence_type="curated_database",
            evidence_level="curated" if is_driver else "associated",
            notes="explicit driver/scaffold in CD-CODE function" if is_driver else "CD-CODE condensate-associated member; not used as hard positive",
            llps_label=1 if is_driver else -100,
            label_tier="curated" if is_driver else "unknown",
            role_label="driver" if is_driver else "member",
            negative_type="none" if is_driver else "background_unlabeled",
            sample_weight=0.8 if is_driver else 0.0,
        )
        add_candidate(cands, cand)
    return cands


def load_disprot_candidates(raw_dir: Path, swiss: dict[str, dict[str, str]]) -> list[Candidate]:
    df = read_csv_maybe(raw_dir / "disprot/disprot_current.tsv", sep="\t")
    cands: list[Candidate] = []
    if df.empty:
        return cands
    grouped = df.groupby("acc", dropna=True)
    for acc, group in grouped:
        acc = normalize_acc(acc)
        mapped_acc = acc if acc in swiss else base_acc(acc)
        if not mapped_acc or mapped_acc not in swiss:
            continue
        text = " ".join(str(x) for x in group[["term", "term_name", "ec_name"]].fillna("").to_numpy().ravel())
        if LLPS_TERM_RE.search(text):
            continue
        disorder_content = pd.to_numeric(group.get("disorder_content"), errors="coerce").max()
        if pd.isna(disorder_content) or float(disorder_content) < 5.0:
            continue
        meta = swiss[mapped_acc]
        weight = 0.9 if float(disorder_content) >= 20.0 else 0.75
        cand = Candidate(
            source="DisProt_current",
            source_record_id=";".join(sorted(set(str(x) for x in group["disprot_id"].dropna()))),
            uniprot_acc=mapped_acc,
            gene_name=meta.get("gene_name", ""),
            organism=str(group["organism"].dropna().iloc[0]) if group["organism"].notna().any() else meta.get("organism", ""),
            taxonomy_id=str(group["ncbi_taxon_id"].dropna().iloc[0]) if group["ncbi_taxon_id"].notna().any() else meta.get("taxonomy_id", ""),
            sequence=meta["sequence"],
            label_candidate="negative",
            role="unknown",
            evidence_type="curated_disorder_non_llps",
            evidence_level="curated",
            pmid=";".join(sorted(set(str(x) for x in group["reference"].dropna()))),
            notes=f"DisProt disorder_content={float(disorder_content):.2f}; LLPS/MLO terms excluded",
            llps_label=0,
            label_tier="curated",
            role_label="unknown",
            negative_type="disordered_negative",
            sample_weight=weight,
        )
        add_candidate(cands, cand)
    return cands


def load_mobidb_silver_candidates(raw_dir: Path, swiss: dict[str, dict[str, str]]) -> list[Candidate]:
    path = raw_dir / "mobidb/mobidb_swissprot_disorder_batches.tsv"
    df = read_csv_maybe(path, sep="\t")
    cands: list[Candidate] = []
    if df.empty:
        return cands
    for required in ["acc", "feature", "content_fraction", "content_count", "length"]:
        if required not in df.columns:
            return cands
    df["content_fraction"] = pd.to_numeric(df["content_fraction"], errors="coerce")
    df["content_count"] = pd.to_numeric(df["content_count"], errors="coerce")
    grouped = df.groupby("acc", dropna=True)
    for acc, group in grouped:
        acc = normalize_acc(acc)
        mapped_acc = acc if acc in swiss else base_acc(acc)
        if not mapped_acc or mapped_acc not in swiss:
            continue
        feature_text = " ".join(str(x) for x in group["feature"].dropna().astype(str))
        if re.search(r"phase[_\s-]?separation|llps|condensate|membrane[_\s-]?less|\bmlo\b", feature_text, re.I):
            continue
        features = group["feature"].astype(str)
        high_conf = group[
            features.str.contains(r"^(curated|prediction)-disorder-(priority|merge|th_50|mobidb_lite|disprot)$", case=False, regex=True, na=False)
            & (group["content_fraction"] >= 0.50)
            & (group["content_count"] >= 50)
        ].copy()
        if high_conf.empty:
            continue
        best = high_conf.sort_values(["content_fraction", "content_count"], ascending=False).iloc[0]
        meta = swiss[mapped_acc]
        cand = Candidate(
            source="MobiDB_highconf_silver",
            source_record_id=mapped_acc,
            uniprot_acc=mapped_acc,
            gene_name=meta.get("gene_name", ""),
            organism=meta.get("organism", ""),
            taxonomy_id=meta.get("taxonomy_id", ""),
            sequence=meta["sequence"],
            label_candidate="negative",
            role="unknown",
            evidence_type="mobidb_high_confidence_disorder_non_llps",
            evidence_level="silver",
            notes=f"MobiDB high-confidence disorder silver; best_feature={best.get('feature', '')}; fraction={float(best.get('content_fraction', 0.0)):.3f}",
            llps_label=0,
            label_tier="silver",
            role_label="unknown",
            negative_type="disordered_negative",
            sample_weight=0.5,
        )
        add_candidate(cands, cand)
    return cands


def parse_pdb_seqres_candidates(raw_dir: Path, max_pdb: int) -> list[Candidate]:
    path = raw_dir / "pdb/pdb_seqres.txt.gz"
    cands: list[Candidate] = []
    seen_md5: set[str] = set()
    if not path.exists():
        return cands
    for header, seq in parse_fasta(path):
        if len(cands) >= max_pdb:
            break
        if not seq_valid(seq):
            continue
        smd5 = md5_sequence(seq)
        if smd5 in seen_md5:
            continue
        seen_md5.add(smd5)
        pdb_id = header.split()[0].replace(":", "_")
        cand = Candidate(
            source="RCSB_PDB_SEQRES",
            source_record_id=pdb_id,
            uniprot_acc="",
            gene_name="",
            organism="",
            taxonomy_id="",
            sequence=seq,
            label_candidate="negative",
            role="unknown",
            evidence_type="structured_sequence_pool",
            evidence_level="curated",
            notes=f"PDB SEQRES structured negative candidate; header={header[:160]}",
            llps_label=0,
            label_tier="curated",
            role_label="unknown",
            negative_type="structured_negative",
            sample_weight=0.85,
        )
        cands.append(cand)
    return cands


def load_swissprot_pu_candidates(swiss: dict[str, dict[str, str]], max_pu: int) -> list[Candidate]:
    cands: list[Candidate] = []
    for acc in sorted(swiss):
        if len(cands) >= max_pu:
            break
        meta = swiss[acc]
        seq = meta["sequence"]
        if not seq_valid(seq):
            continue
        cands.append(
            Candidate(
                source="UniProt_SwissProt_reviewed",
                source_record_id=acc,
                uniprot_acc=acc,
                gene_name=meta.get("gene_name", ""),
                organism=meta.get("organism", ""),
                taxonomy_id=meta.get("taxonomy_id", ""),
                sequence=seq,
                label_candidate="unknown",
                role="unknown",
                evidence_type="reviewed_background_pool",
                evidence_level="background",
                notes="Swiss-Prot reviewed background; PU/unlabeled only",
                llps_label=-100,
                label_tier="unknown",
                role_label="unknown",
                negative_type="background_unlabeled",
                sample_weight=0.0,
            )
        )
    return cands


def load_biogrid_interactors(raw_dir: Path, seed_accs: set[str]) -> set[str]:
    path = raw_dir / "biogrid/BIOGRID-ALL-LATEST.tab3.zip"
    interactors: set[str] = set()
    if not path.exists() or not seed_accs:
        return interactors
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as raw:
            text = (line.decode("utf-8", errors="replace") for line in raw)
            reader = csv.DictReader(text, delimiter="\t")
            for row in reader:
                a = set()
                b = set()
                for col in ["SWISS-PROT Accessions Interactor A", "TREMBL Accessions Interactor A"]:
                    a.update(normalize_acc(x) for x in str(row.get(col, "")).split("|"))
                for col in ["SWISS-PROT Accessions Interactor B", "TREMBL Accessions Interactor B"]:
                    b.update(normalize_acc(x) for x in str(row.get(col, "")).split("|"))
                a.discard("")
                a.discard("-")
                b.discard("")
                b.discard("-")
                if a & seed_accs:
                    interactors.update(b)
                if b & seed_accs:
                    interactors.update(a)
    interactors.difference_update(seed_accs)
    return interactors


def write_metadata(raw: Path) -> None:
    if not raw.exists():
        return
    metadata = {
        "phasepdb3": {
            "source_name": "PhaSepDB 3.0 API mirror",
            "release/version": "3.0 API dump copied from local 2026-05-21 mirror",
            "url": "local mirror: data/raw/phasepdb3/proteins_api_2026-05-21.json",
            "fields": "protein_id, pmid, class_, evidence summaries, region text, UniProt accession",
        },
        "llpsdb_v2": {
            "source_name": "LLPSDB v2.0",
            "release/version": "v2.0 public download zips",
            "url": "LLPSDB v2.0 download package",
            "fields": "protein.xls and LLPS.xls parsed via existing parsed_source_tables",
        },
        "cd_code": {
            "source_name": "CD-CODE",
            "release/version": "2.2, published 2026-03-18",
            "url": "https://owncloud.mpi-cbg.de/index.php/s/rFmNhVx6Ey1eHjO/download",
            "fields": "proteins_202603181653.csv, protein2cdcode_v2.2.tsv, condensates_202603181647.csv",
        },
        "drllps": {
            "source_name": "DrLLPS",
            "release/version": "local public text mirror",
            "url": "local mirror: data/raw/drllps/drllps.txt",
            "fields": "DrLLPS ID, UniProt ID, gene, species, condensate, LLPS Type, references, sequence",
        },
        "bav_llps": {
            "source_name": "BAV-LLPS curated dataset",
            "release/version": "website download as of 2026-06-06",
            "url": "https://s3.us-east-1.amazonaws.com/codnas.inf.pucp.edu.pe/bav-llps-db/datasource/bav-llps-curated-ds.xlsx",
            "fields": "id, organism_type, uniprot, sequence, disorder/LLPS scores, function, location, specie, protein, mlo_bmc",
        },
        "disprot": {
            "source_name": "DisProt current release",
            "release/version": "current API release as of 2026-06-06",
            "url": "https://disprot.org/api/search?release=current&format={tsv,json,fasta}",
            "fields": "acc, disorder_content, organism, taxon, disprot_id, region, term, evidence, reference",
        },
        "mobidb": {
            "source_name": "MobiDB",
            "release/version": "Version 6.2, Release 2024_07",
            "url": "https://mobidb.org/api/download?acc=P04637&format={json,tsv}; bulk frontend endpoint /api/download_page returned 502 in this environment",
            "fields": "single accession JSON/TSV probe; not used to create hard labels in this run",
        },
        "pdb": {
            "source_name": "RCSB PDB SEQRES FASTA",
            "release/version": "current derived_data snapshot, last modified 2026-05-30 from HTTP headers",
            "url": "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz",
            "fields": "SEQRES FASTA header and amino-acid sequence",
        },
        "biogrid": {
            "source_name": "BioGRID protein interaction data",
            "release/version": "BioGRID 5.0.258 Latest-Release, compiled 2026-05-25",
            "url": "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ALL-LATEST.tab3.zip",
            "fields": "TAB3 interaction table, Swiss-Prot/TREMBL interactor accession columns",
        },
        "uniprot_swissprot": {
            "source_name": "UniProtKB Swiss-Prot reviewed FASTA",
            "release/version": "current_release, file last modified 2026-01-28 from HTTP headers",
            "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz",
            "fields": "reviewed FASTA with accession, gene, organism, taxonomy and canonical sequence",
        },
    }
    for source_dir in raw.iterdir():
        if not source_dir.is_dir():
            continue
        files = []
        for path in sorted(source_dir.glob("*")):
            if path.is_file() and path.name != "metadata.json":
                files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        payload = metadata.get(source_dir.name, {"source_name": source_dir.name, "release/version": "unknown", "url": "", "fields": ""})
        release_version = payload.get("release/version", "unknown")
        field_notes = payload.get("fields", "")
        payload = {
            **payload,
            "release": release_version,
            "version": release_version,
            "download_date": DATE,
            "checksum": {item["file"]: item["sha256"] for item in files},
            "field_notes": field_notes,
            "files": files,
        }
        (source_dir / "metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def direct_filter_candidates(
    candidates: list[Candidate],
    benchmark: dict[str, set[str]],
    active: pd.DataFrame,
    interactors: set[str],
) -> tuple[list[Candidate], Counter]:
    stats: Counter = Counter()
    active_md5_label = {
        str(row.sequence_md5): int(row.llps_label)
        for row in active.itertuples(index=False)
        if str(getattr(row, "sequence_md5", "")) and pd.notna(getattr(row, "llps_label", None))
    }
    active_acc_label = {
        normalize_acc(getattr(row, "uniprot_acc", "")): int(row.llps_label)
        for row in active.itertuples(index=False)
        if normalize_acc(getattr(row, "uniprot_acc", "")) and pd.notna(getattr(row, "llps_label", None))
    }
    clean: list[Candidate] = []
    for cand in candidates:
        if cand.uniprot_acc in benchmark["accs"] or cand.uniprot_acc in benchmark["ids"]:
            stats["removed_same_benchmark_accession"] += 1
            continue
        if cand.sequence_md5 in benchmark["md5s"]:
            stats["removed_same_benchmark_md5"] += 1
            continue
        if cand.source_record_id in benchmark["source_ids"]:
            stats["removed_same_benchmark_source_record"] += 1
            continue
        if cand.llps_label == 0 and cand.uniprot_acc and cand.uniprot_acc in interactors:
            stats["removed_negative_biogrid_interactor"] += 1
            continue
        existing_labels = []
        if cand.sequence_md5 in active_md5_label:
            existing_labels.append(active_md5_label[cand.sequence_md5])
        if cand.uniprot_acc in active_acc_label:
            existing_labels.append(active_acc_label[cand.uniprot_acc])
        if cand.llps_label in {0, 1} and any(x in {0, 1} and x != cand.llps_label for x in existing_labels):
            stats["removed_conflict_with_active_label"] += 1
            continue
        clean.append(cand)
    return clean, stats


def collapse_candidates(candidates: list[Candidate]) -> tuple[list[Candidate], Counter, dict[str, list[Candidate]]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        key = cand.uniprot_acc or cand.sequence_md5
        grouped[key].append(cand)
    chosen: list[Candidate] = []
    stats: Counter = Counter()
    kept_sources: dict[str, list[Candidate]] = {}
    priority = {"curated": 5, "gold": 5, "silver": 3, "pseudo": 2, "unknown": 1}
    for key, rows in grouped.items():
        labels = {r.llps_label for r in rows if r.llps_label in {0, 1}}
        if labels == {0, 1}:
            stats["removed_positive_negative_candidate_conflict"] += len(rows)
            continue
        rows = sorted(rows, key=lambda r: (priority.get(r.label_tier, 0), r.sample_weight, len(r.sequence)), reverse=True)
        chosen.append(rows[0])
        kept_sources[key] = rows
    return chosen, stats, kept_sources


def run_mmseqs_filter(
    root: Path,
    active: pd.DataFrame,
    candidates: list[Candidate],
    benchmark: dict[str, set[str]],
    tag: str,
) -> tuple[set[str], set[str], Counter, dict[str, int | str]]:
    work = root / f"artifacts/data/interim/augmentation/mmseqs40_{DATE}_{tag}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "all_sequences.fasta"
    seq_to_kind: dict[str, str] = {}
    seen: set[str] = set()

    def write_record(handle, kind: str, ident: str, seq: str) -> None:
        seq = clean_sequence(seq)
        if not seq:
            return
        key = f"{kind}|{ident}"
        if key in seen:
            return
        seen.add(key)
        seq_to_kind[key] = kind
        handle.write(f">{key}\n")
        for i in range(0, len(seq), 80):
            handle.write(seq[i : i + 80] + "\n")

    with fasta.open("w", encoding="utf-8") as handle:
        for path in [
            root / "artifacts/data/benchmarks/protein_benchmark_ppmc/manifest.csv",
            root / "artifacts/data/benchmarks/dpr_benchmark_phasepro/proteins.csv",
        ]:
            df = read_csv_maybe(path)
            if df.empty:
                continue
            for _, row in df.iterrows():
                write_record(handle, "benchmark", str(row.get("protein_id", row.get("uniprot_id", ""))), str(row.get("sequence", "")))
        for _, row in active.iterrows():
            write_record(handle, "active", str(row.get("protein_id", "")), str(row.get("sequence", "")))
        for cand in candidates:
            write_record(handle, "candidate", cand.uniprot_acc or cand.sequence_md5, cand.sequence)

    mmseqs = shutil.which("mmseqs") or "mmseqs"
    out_prefix = work / "cluster"
    tmp = work / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    cmd = [mmseqs, "easy-cluster", str(fasta), str(out_prefix), str(tmp), "--min-seq-id", "0.4", "-c", "0.8", "--cov-mode", "0", "--threads", "8"]
    log = work / "mmseqs.log"
    with log.open("w", encoding="utf-8") as log_handle:
        subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, check=True)

    cluster_tsv = work / "cluster_cluster.tsv"
    cluster_members: dict[str, list[str]] = defaultdict(list)
    with cluster_tsv.open("r", encoding="utf-8") as handle:
        for line in handle:
            rep, member = line.rstrip("\n").split("\t")[:2]
            cluster_members[rep].append(member)
    homolog_candidate_keys: set[str] = set()
    homolog_active_keys: set[str] = set()
    final_overlap_clusters = 0
    active_overlap_clusters = 0
    candidate_overlap_clusters = 0
    for members in cluster_members.values():
        kinds = {seq_to_kind.get(m, "") for m in members}
        has_bench = "benchmark" in kinds
        if not has_bench:
            continue
        if "active" in kinds:
            active_overlap_clusters += 1
            homolog_active_keys.update(m.split("|", 1)[1] for m in members if seq_to_kind.get(m) == "active")
        if "candidate" in kinds:
            candidate_overlap_clusters += 1
            homolog_candidate_keys.update(m.split("|", 1)[1] for m in members if seq_to_kind.get(m) == "candidate")
        if "active" in kinds or "candidate" in kinds:
            final_overlap_clusters += 1
    stats = Counter(
        {
            f"{tag}_mmseqs_clusters_total": len(cluster_members),
            f"{tag}_mmseqs_candidate_homolog_removed": 0,
            f"{tag}_mmseqs_active_homolog_removed": 0,
            f"{tag}_mmseqs_candidate_homolog_audit_only": len(homolog_candidate_keys),
            f"{tag}_mmseqs_active_homolog_audit_only": len(homolog_active_keys),
            f"{tag}_mmseqs_active_benchmark_overlap_clusters": active_overlap_clusters,
            f"{tag}_mmseqs_candidate_benchmark_overlap_clusters": candidate_overlap_clusters,
            f"{tag}_mmseqs_final_benchmark_overlap_clusters_before_removal": final_overlap_clusters,
        }
    )
    return homolog_candidate_keys, homolog_active_keys, stats, {
        "cluster_tsv": str(cluster_tsv),
        "fasta": str(fasta),
        "log": str(log),
    }


def build_augmented_manifest(
    active: pd.DataFrame,
    candidates: list[Candidate],
    kept_sources: dict[str, list[Candidate]],
    homolog_keys: set[str],
    active_homolog_keys: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Counter]:
    stats: Counter = Counter()
    active = active.copy()
    active_md5 = set(active["sequence_md5"].dropna().astype(str))
    active_acc = set(active["uniprot_acc"].dropna().astype(str))
    augmented_rows = []
    source_rows = []

    for _, row in active.iterrows():
        if str(row.get("protein_id", "")) in active_homolog_keys:
            stats["active_homolog_cluster40_audit_only"] += 1
        base = row.to_dict()
        for col in Candidate("x", "x", "", "", "", "", "M" * 30, "unknown", "unknown", "x", "unknown").manifest_row().keys():
            base.setdefault(col, "")
        base["leakage_status"] = "clean"
        augmented_rows.append(base)

    for cand in candidates:
        key = cand.uniprot_acc or cand.sequence_md5
        if key in homolog_keys:
            stats["candidate_homolog_cluster40_audit_only"] += 1
        if cand.sequence_md5 in active_md5 or (cand.uniprot_acc and cand.uniprot_acc in active_acc):
            mask = (active["sequence_md5"].astype(str) == cand.sequence_md5) | (active["uniprot_acc"].astype(str) == cand.uniprot_acc)
            idx = active.index[mask]
            if len(idx):
                pid = str(active.loc[idx[0], "protein_id"])
                for src in kept_sources.get(cand.uniprot_acc or cand.sequence_md5, [cand]):
                    source_rows.append(
                        {
                            "protein_id": pid,
                            "source": src.source,
                            "source_record_id": src.source_record_id,
                            "uniprot_acc": src.uniprot_acc,
                            "sequence_md5": src.sequence_md5,
                            "llps_label": src.llps_label,
                            "label_tier": src.label_tier,
                            "role_label": src.role_label,
                            "negative_type": src.negative_type,
                            "evidence_level": src.evidence_level,
                            "leakage_status": "clean_existing_active",
                            "sample_weight": src.sample_weight,
                            "notes": src.notes,
                        }
                    )
                active_label = int(active.loc[idx[0], "llps_label"])
                if active_label == -100 and cand.llps_label in {0, 1}:
                    for row in augmented_rows:
                        if str(row.get("protein_id")) == pid:
                            row.update(cand.manifest_row(pid))
                            row["sample_origin"] = "active_train_upgraded_by_augmentation"
                            row["notes"] = f"{row.get('notes', '')}; upgraded_by={cand.source}:{cand.source_record_id}"
                            stats["active_unknown_upgraded"] += 1
                            if cand.llps_label == 1:
                                stats["active_unknown_upgraded_positive"] += 1
                            else:
                                stats["active_unknown_upgraded_negative"] += 1
                            break
                else:
                    stats["candidate_already_in_active"] += 1
            continue
        row = cand.manifest_row()
        row["sample_origin"] = "new_external_augmentation"
        augmented_rows.append(row)
        stats["new_rows_added"] += 1
        if cand.llps_label == 1 and cand.role_label in {"driver", "scaffold"} and cand.label_tier in {"gold", "curated"}:
            stats["new_hard_positive"] += 1
        elif cand.llps_label == 1:
            stats["new_pseudo_weak_positive"] += 1
        elif cand.llps_label == 0 and cand.negative_type == "structured_negative":
            stats["new_structured_negative"] += 1
        elif cand.llps_label == 0 and cand.negative_type == "disordered_negative":
            stats["new_disordered_negative"] += 1
        elif cand.llps_label == -100:
            stats["new_uncertain_pu"] += 1
        for src in kept_sources.get(cand.uniprot_acc or cand.sequence_md5, [cand]):
            source_rows.append(
                {
                    "protein_id": row["protein_id"],
                    "source": src.source,
                    "source_record_id": src.source_record_id,
                    "uniprot_acc": src.uniprot_acc,
                    "sequence_md5": src.sequence_md5,
                    "llps_label": src.llps_label,
                    "label_tier": src.label_tier,
                    "role_label": src.role_label,
                    "negative_type": src.negative_type,
                    "evidence_level": src.evidence_level,
                    "leakage_status": "clean",
                    "sample_weight": src.sample_weight,
                    "notes": src.notes,
                }
            )
    return pd.DataFrame(augmented_rows), pd.DataFrame(source_rows), stats


def final_overlap_checks(root: Path, final_df: pd.DataFrame, benchmark: dict[str, set[str]], homolog_stats: Counter) -> Counter:
    stats = Counter()
    final_accs = set(final_df["uniprot_acc"].dropna().astype(str)) if "uniprot_acc" in final_df else set()
    final_md5s = set(final_df["sequence_md5"].dropna().astype(str)) if "sequence_md5" in final_df else set()
    ppmc = read_csv_maybe(root / "artifacts/data/benchmarks/protein_benchmark_ppmc/manifest.csv")
    phasepro = read_csv_maybe(root / "artifacts/data/benchmarks/dpr_benchmark_phasepro/proteins.csv")

    def bench_overlap(df: pd.DataFrame, prefix: str) -> None:
        if df.empty:
            stats[f"final_{prefix}_accession_overlap"] = 0
            stats[f"final_{prefix}_md5_overlap"] = 0
            return
        accs = set()
        for col in ["protein_id", "uniprot_id"]:
            if col in df:
                accs.update(normalize_acc(x) for x in df[col].dropna().astype(str))
        md5s = set(df["sequence_md5"].dropna().astype(str)) if "sequence_md5" in df else set()
        if "sequence" in df:
            md5s.update(md5_sequence(clean_sequence(x)) for x in df["sequence"].dropna())
        stats[f"final_{prefix}_accession_overlap"] = len(final_accs & {x for x in accs if x})
        stats[f"final_{prefix}_md5_overlap"] = len(final_md5s & {x for x in md5s if x})

    bench_overlap(ppmc, "ppmc")
    bench_overlap(phasepro, "phasepro")
    stats["final_ppmc_phasepro_direct_accession_overlap"] = len(final_accs & benchmark["accs"])
    stats["final_ppmc_phasepro_direct_md5_overlap"] = len(final_md5s & benchmark["md5s"])
    stats["final_mmseqs40_active_benchmark_overlap_clusters"] = homolog_stats["final_mmseqs_active_benchmark_overlap_clusters"]
    return stats


def assign_internal_valid_split(final_df: pd.DataFrame, cluster_tsv: str | Path, valid_fraction: float = 0.12) -> tuple[pd.DataFrame, Counter]:
    df = final_df.copy()
    if "split" not in df.columns:
        df["split"] = "train"
    df["split"] = "train"
    path = Path(cluster_tsv)
    stats = Counter()
    if not path.exists():
        stats["split_cluster_tsv_missing"] = 1
        return df, stats

    cluster_members: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            if member.startswith("active|"):
                cluster_members[rep].add(member.split("|", 1)[1])
    protein_ids = set(df["protein_id"].dropna().astype(str))
    active_clusters = {rep: ids & protein_ids for rep, ids in cluster_members.items()}
    active_clusters = {rep: ids for rep, ids in active_clusters.items() if ids}
    target = max(1, int(round(len(df) * valid_fraction)))
    selected: set[str] = set()
    valid_ids: set[str] = set()
    for rep, ids in sorted(active_clusters.items(), key=lambda item: hashlib.md5(item[0].encode("utf-8")).hexdigest()):
        if len(valid_ids) >= target and len(valid_ids) / max(len(df), 1) >= 0.10:
            break
        selected.add(rep)
        valid_ids.update(ids)
        if len(valid_ids) / max(len(df), 1) >= 0.15:
            break
    df.loc[df["protein_id"].astype(str).isin(valid_ids), "split"] = "valid"
    stats.update(
        {
            "split_total_rows": len(df),
            "split_train_rows": int((df["split"] == "train").sum()),
            "split_valid_rows": int((df["split"] == "valid").sum()),
            "split_valid_fraction": round(int((df["split"] == "valid").sum()) / max(len(df), 1), 6),
            "split_total_active_clusters": len(active_clusters),
            "split_valid_clusters": len(selected),
        }
    )
    return df, stats


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_reports(
    root: Path,
    counters: dict[str, Counter],
    final_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    spans: list[dict[str, object]],
    mmseqs_paths: dict[str, int | str],
    max_pdb: int,
    max_pu: int,
) -> None:
    reports = root / "artifacts/data/reports"
    reports.mkdir(parents=True, exist_ok=True)
    total = Counter()
    for c in counters.values():
        total.update(c)
    label_counts = final_df["llps_label"].value_counts(dropna=False).to_dict()
    negative_counts = final_df.get("negative_type", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
    source_counts = canonical_df["source"].value_counts(dropna=False).to_dict() if not canonical_df.empty else {}
    dpr_counts = Counter(row.get("region_label_tier", "none") for row in spans)
    new_df = final_df[final_df["sample_origin"].astype(str) == "new_external_augmentation"]
    upgraded_df = final_df[final_df["sample_origin"].astype(str) == "active_train_upgraded_by_augmentation"]
    final_new_hard_positive = int(
        (
            (new_df["llps_label"] == 1)
            & new_df["role_label"].isin({"driver", "scaffold"})
            & new_df["label_tier"].isin({"gold", "curated"})
        ).sum()
    )
    final_new_positive = int((new_df["llps_label"] == 1).sum())
    final_new_counts = {
        "new_rows_kept": int(len(new_df)),
        "hard_positive": final_new_hard_positive,
        "pseudo_weak_positive": final_new_positive - final_new_hard_positive,
        "structured_negative": int(((new_df["llps_label"] == 0) & (new_df["negative_type"] == "structured_negative")).sum()),
        "disordered_negative": int(((new_df["llps_label"] == 0) & (new_df["negative_type"] == "disordered_negative")).sum()),
        "uncertain_pu": int((new_df["llps_label"] == -100).sum()),
        "active_unknown_upgraded_positive": int((upgraded_df["llps_label"] == 1).sum()),
        "active_unknown_upgraded_negative": int((upgraded_df["llps_label"] == 0).sum()),
    }

    leakage = f"""# Augmentation leakage report {DATE}

## 结论

- augmented train 写入前过滤完成。
- final direct UniProt overlap: {total['final_ppmc_phasepro_direct_accession_overlap']}
- final direct sequence MD5 overlap: {total['final_ppmc_phasepro_direct_md5_overlap']}
- final PPMC accession/MD5 overlap: {total['final_ppmc_accession_overlap']} / {total['final_ppmc_md5_overlap']}
- final PhaSePro accession/MD5 overlap: {total['final_phasepro_accession_overlap']} / {total['final_phasepro_md5_overlap']}
- final active-vs-benchmark MMseqs40 overlap clusters (audit-only): {total['final_mmseqs40_active_benchmark_overlap_clusters']}
- candidate-vs-benchmark MMseqs40 audit-only proteins: {total['prefilter_mmseqs_candidate_homolog_audit_only'] + total['candidate_homolog_cluster40_audit_only']}
- active-vs-benchmark MMseqs40 audit-only proteins: {total['prefilter_mmseqs_active_homolog_audit_only'] + total['active_homolog_cluster40_audit_only']}

## 删除统计

| 项目 | 数量 |
| --- | ---: |
| same benchmark accession | {total['removed_same_benchmark_accession']} |
| same benchmark sequence MD5 | {total['removed_same_benchmark_md5']} |
| same benchmark source record | {total['removed_same_benchmark_source_record']} |
| BioGRID LLPS/benchmark 一阶互作 negative | {total['removed_negative_biogrid_interactor']} |
| active label conflict | {total['removed_conflict_with_active_label']} |
| candidate positive/negative conflict | {total['removed_positive_negative_candidate_conflict']} |
| positive/negative conflict total | {total['removed_conflict_with_active_label'] + total['removed_positive_negative_candidate_conflict']} |
| MMseqs40 benchmark homolog removed | 0 |
| MMseqs40 benchmark homolog audit-only | {total['prefilter_homolog_audit_only'] + total['final_homolog_audit_only']} |
| final validation iterative homolog | {total['removed_final_validation_homolog_cluster40']} |

## MMseqs40

- prefilter cluster TSV: `{mmseqs_paths.get('prefilter_cluster_tsv', '')}`
- final validation cluster TSV: `{mmseqs_paths.get('final_cluster_tsv', '')}`
- prefilter total clusters: {total['prefilter_mmseqs_clusters_total']}
- prefilter active benchmark overlap clusters: {total['prefilter_mmseqs_active_benchmark_overlap_clusters']}
- prefilter candidate benchmark overlap clusters: {total['prefilter_mmseqs_candidate_benchmark_overlap_clusters']}
- final validation active benchmark overlap clusters: {total['final_mmseqs_active_benchmark_overlap_clusters']}

## 说明

- PhaSePro protein/region 未作为训练监督来源。
- PPMC benchmark protein 的 accession/MD5/source record 及 cleanup audit 中 exact deleted ids 均作为 blacklist；MMseqs40 homolog 只审计、不删除。
- PDB structured negative 与 Swiss-Prot PU 为确定性上限采样：PDB max={max_pdb}, Swiss-Prot PU max={max_pu}。
"""
    (reports / f"augmentation_leakage_report_{DATE}.md").write_text(leakage, encoding="utf-8")

    summary_lines = [
        f"# Augmentation label summary {DATE}",
        "",
        "## 新增样本",
        "",
        "| 类型 | 数量 |",
        "| --- | ---: |",
        f"| new external rows kept after final validation | {final_new_counts['new_rows_kept']} |",
        f"| hard positive | {final_new_counts['hard_positive']} |",
        f"| pseudo/weak positive | {final_new_counts['pseudo_weak_positive']} |",
        f"| structured negative | {final_new_counts['structured_negative']} |",
        f"| disordered negative | {final_new_counts['disordered_negative']} |",
        f"| uncertain/PU | {final_new_counts['uncertain_pu']} |",
        f"| active unknown upgraded positive | {final_new_counts['active_unknown_upgraded_positive']} |",
        f"| active unknown upgraded negative | {final_new_counts['active_unknown_upgraded_negative']} |",
        "",
        "## DPR spans",
        "",
        "| tier | count |",
        "| --- | ---: |",
    ]
    for k, v in sorted(dpr_counts.items()):
        summary_lines.append(f"| {k} | {v} |")
    summary_lines += [
        "",
        "## final manifest label distribution",
        "",
        "| llps_label | count |",
        "| --- | ---: |",
    ]
    for k, v in sorted(label_counts.items(), key=lambda item: str(item[0])):
        summary_lines.append(f"| {k} | {v} |")
    summary_lines += [
        "",
        "## negative_type distribution",
        "",
        "| negative_type | count |",
        "| --- | ---: |",
    ]
    for k, v in sorted(negative_counts.items(), key=lambda item: str(item[0])):
        summary_lines.append(f"| {k} | {v} |")
    summary_lines += [
        "",
        "## canonical source rows",
        "",
        "| source | rows |",
        "| --- | ---: |",
    ]
    for k, v in sorted(source_counts.items()):
        summary_lines.append(f"| {k} | {v} |")
    summary_lines += [
        "",
        "## 标签策略",
        "",
        "- PhaSepDB: 只取 strict PS-self 且无 partner/RNA/DNA/PTM/mutation/repeat/splicing dependency 语义。",
        "- LLPSDB: 只取 unambiguous phase_separation、protein(1)、natural、无 fusion/cleaved/repeat/mutation/PTM/nucleic acid 条件。",
        "- CD-CODE: 只有 function/name 显式 driver/scaffold 才进入 positive；member 仅保留 unknown/PU。",
        "- DrLLPS: 只有 scaffold 进入 hard positive。",
        "- BAV-LLPS: 只用 curated xlsx，未使用 homologous 数据集。",
        "- DisProt: LLPS/MLO/condensate term 排除后作为 curated disordered negative。",
        "- MobiDB: 本轮只归档 API probe 和版本信息，不进入 high-confidence hard negative。",
    ]
    (reports / f"augmentation_label_summary_{DATE}.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--raw-external-dir", default="", help="Raw external source directory; defaults to data/raw_external/YYYYMMDD or data/raw_external")
    parser.add_argument("--max-pdb-negatives", type=int, default=0)
    parser.add_argument("--max-swissprot-pu", type=int, default=10000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    raw_dir = resolve_raw_external_dir(root, args.raw_external_dir or None)

    write_metadata(raw_dir)
    active = load_active_manifest(root)
    benchmark = load_benchmark_sets(root)
    swiss = parse_swissprot_fasta(raw_dir / "uniprot_swissprot/uniprot_sprot.fasta.gz")

    candidates: list[Candidate] = []
    spans: list[dict[str, object]] = []
    phase_cands, phase_spans = load_phasepdb_candidates(root, raw_dir, swiss)
    llps_cands, llps_spans = load_llpsdb_candidates(root)
    candidates.extend(phase_cands)
    candidates.extend(llps_cands)
    spans.extend(phase_spans)
    spans.extend(llps_spans)
    candidates.extend(load_cdcode_candidates(raw_dir, swiss))
    candidates.extend(load_drllps_candidates(raw_dir))
    candidates.extend(load_bav_candidates(raw_dir))

    positive_seed_accs = {c.uniprot_acc for c in candidates if c.llps_label == 1 and c.uniprot_acc}
    positive_seed_accs.update(set(active.loc[active["llps_label"] == 1, "uniprot_acc"].dropna().astype(str)))
    positive_seed_accs.update(benchmark["accs"])
    interactors = load_biogrid_interactors(raw_dir, positive_seed_accs)

    candidates.extend(load_disprot_candidates(raw_dir, swiss))
    candidates.extend(load_mobidb_silver_candidates(raw_dir, swiss))
    candidates.extend(parse_pdb_seqres_candidates(raw_dir, args.max_pdb_negatives))
    candidates.extend(load_swissprot_pu_candidates(swiss, args.max_swissprot_pu))

    canonical_df = pd.DataFrame([c.canonical_row() for c in candidates], columns=CANON_COLUMNS)
    direct_clean, direct_stats = direct_filter_candidates(candidates, benchmark, active, interactors)
    collapsed, collapse_stats, kept_sources = collapse_candidates(direct_clean)
    homolog_keys, active_homolog_keys, mmseqs_stats, mmseqs_paths = run_mmseqs_filter(root, active, collapsed, benchmark, tag="prefilter")
    mmseqs_stats["prefilter_homolog_audit_only"] = len(homolog_keys) + len(active_homolog_keys)
    final_manifest, source_map, add_stats = build_augmented_manifest(active, collapsed, kept_sources, set(), set())
    final_mmseqs_stats = Counter()
    final_mmseqs_paths: dict[str, int | str] = {}
    for _iteration in range(1, 7):
        _, final_active_homolog_keys, final_mmseqs_stats, final_mmseqs_paths = run_mmseqs_filter(root, final_manifest, [], benchmark, tag="final")
        final_mmseqs_stats["final_homolog_audit_only"] = len(final_active_homolog_keys)
        break
    overlap_stats = final_overlap_checks(root, final_manifest, benchmark, final_mmseqs_stats)
    final_manifest, split_stats = assign_internal_valid_split(final_manifest, final_mmseqs_paths.get("cluster_tsv", ""))

    if overlap_stats["final_ppmc_phasepro_direct_accession_overlap"] or overlap_stats["final_ppmc_phasepro_direct_md5_overlap"]:
        raise SystemExit(f"Refusing to write active augmented train because final benchmark overlap is nonzero: {dict(overlap_stats)}")

    # Remove DPR spans for proteins removed by exact leakage filters and for PhaSePro/benchmark accessions.
    final_ids = set(final_manifest["protein_id"].dropna().astype(str))
    final_accs = set(final_manifest["uniprot_acc"].dropna().astype(str))
    clean_spans = []
    for row in spans:
        pid = str(row.get("protein_id", ""))
        if pid in benchmark["accs"] or pid in benchmark["ids"]:
            continue
        if pid in final_ids or pid in final_accs:
            clean_spans.append(row)

    processed = root / "artifacts/data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    canonical_df.to_csv(processed / "augmentation_canonical_source_table.csv", index=False)
    final_manifest.to_csv(processed / "augmented_train_manifest.csv", index=False)
    source_map.to_csv(processed / "augmentation_source_map.csv", index=False)
    write_jsonl(processed / "augmented_region_spans.jsonl", clean_spans)

    counters = {
        "direct": direct_stats,
        "collapse": collapse_stats,
        "prefilter_mmseqs": mmseqs_stats,
        "final_mmseqs": final_mmseqs_stats,
        "add": add_stats,
        "overlap": overlap_stats,
        "split": split_stats,
    }
    report_paths = {
        **{f"prefilter_{k}": v for k, v in mmseqs_paths.items()},
        **{f"final_{k}": v for k, v in final_mmseqs_paths.items()},
    }
    write_reports(root, counters, final_manifest, canonical_df, clean_spans, report_paths, args.max_pdb_negatives, args.max_swissprot_pu)
    print(json.dumps({k: dict(v) for k, v in counters.items()}, indent=2, ensure_ascii=False))
    print(f"wrote {processed / 'augmented_train_manifest.csv'} rows={len(final_manifest)}")
    print(f"wrote {processed / 'augmented_region_spans.jsonl'} spans={len(clean_spans)}")


if __name__ == "__main__":
    main()
