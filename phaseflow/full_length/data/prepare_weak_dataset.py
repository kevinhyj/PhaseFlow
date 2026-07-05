from __future__ import annotations

import argparse
import json
import gzip
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from phaseflow.full_length.features.plm_embedder import clean_protein_sequence
from phaseflow.full_length.utils import write_json


@dataclass(slots=True)
class WeakRecord:
    protein_id: str
    sequence: str
    llps_label: int
    source: set[str] = field(default_factory=set)
    uniprot_id: str = ""
    gene_name: str = ""
    raw_dataset_codes: str = ""
    negative_type: str = "unknown"
    role_label: str = "unknown"
    label_quality: str = "curated"
    evidence_level: str = "medium"
    dependency: str = "unknown"
    species: str = ""
    tax_id: str = ""
    source_id: str = ""
    label_confidence: float = 0.8
    sample_weight: float = 1.0
    notes: list[str] = field(default_factory=list)
    regions: list[dict[str, Any]] = field(default_factory=list)
    split: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def has_region_label(self) -> bool:
        return bool(self.regions)


def prepare_weak_dataset(
    ppmc_tsv: str | Path,
    phasepro_json: str | Path,
    out_dir: str | Path,
    llpsdb_positive_csv: str | Path | None = None,
    phasepdb_csv: str | Path | None = None,
    cd_code_proteins_csv: str | Path | None = None,
    cd_code_links_csv: str | Path | None = None,
    uniprot_fasta: str | Path | None = None,
    min_length: int = 20,
    max_length: int = 2048,
    max_records: int | None = None,
    seed: int = 7,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, WeakRecord] = {}
    skipped = {
        "short": 0,
        "long": 0,
        "empty_sequence": 0,
        "duplicate_sequence": 0,
        "missing_phasepdb_sequence": 0,
        "missing_cd_code_sequence": 0,
    }
    reviewed_accessions = _load_uniprot_accessions(uniprot_fasta)

    sequence_lookup = _load_uniprot_sequence_lookup(
        uniprot_fasta,
        accessions=_collect_accessions(phasepdb_csv, "uniprot_id")
        | _collect_accessions(cd_code_proteins_csv, "uniprot_id"),
    )

    _load_ppmc(ppmc_tsv, records, min_length, max_length, skipped)
    _load_phasepro(phasepro_json, records, min_length, max_length, skipped)
    if llpsdb_positive_csv is not None:
        _load_llpsdb_positive(llpsdb_positive_csv, records, min_length, max_length, skipped)
    if phasepdb_csv is not None:
        _load_phasepdb_positive(phasepdb_csv, sequence_lookup, records, min_length, max_length, skipped)
    if cd_code_proteins_csv is not None:
        _load_cd_code_positive(
            cd_code_proteins_csv,
            cd_code_links_csv,
            sequence_lookup,
            records,
            min_length,
            max_length,
            skipped,
        )

    selected = _deduplicate_sequences(list(records.values()), skipped)
    if max_records is not None and max_records > 0:
        selected = _limit_records(selected, max_records, seed)
    _assign_splits(selected, seed=seed, train_frac=train_frac, valid_frac=valid_frac)

    manifest_path = out_dir / "manifest.csv"
    regions_path = out_dir / "regions.jsonl"
    master_path = out_dir / "protein_master.jsonl"
    proteins_path = out_dir / "proteins.csv"
    protein_labels_path = out_dir / "protein_labels.csv"
    regions_table_path = out_dir / "regions.csv"
    evidence_path = out_dir / "evidence.csv"
    source_map_path = out_dir / "source_map.csv"
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [_manifest_row(record, regions_path) for record in selected]
    manifest = pd.DataFrame(manifest_rows).sort_values(["split", "protein_id"])
    manifest.to_csv(manifest_path, index=False)

    with regions_path.open("w") as handle:
        for record in sorted(selected, key=lambda item: item.protein_id):
            if record.regions:
                handle.write(json.dumps({"protein_id": record.protein_id, "regions": record.regions}, sort_keys=True) + "\n")

    with master_path.open("w") as handle:
        for record in sorted(selected, key=lambda item: item.protein_id):
            handle.write(json.dumps(_master_row(record), sort_keys=True) + "\n")

    _write_phase1_tables(
        records=sorted(selected, key=lambda item: item.protein_id),
        out_dir=out_dir,
        reviewed_accessions=reviewed_accessions,
        proteins_path=proteins_path,
        protein_labels_path=protein_labels_path,
        regions_table_path=regions_table_path,
        evidence_path=evidence_path,
        source_map_path=source_map_path,
    )

    split_counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        ids = sorted(record.protein_id for record in selected if record.split == split)
        split_counts[split] = len(ids)
        (splits_dir / f"{split}_ids.txt").write_text("".join(f"{protein_id}\n" for protein_id in ids))

    report = _build_report(selected, ppmc_tsv, phasepro_json, manifest_path, regions_path, skipped)
    write_json(out_dir / "weak_dataset_report.json", report)
    _write_markdown_report(out_dir / "weak_dataset_report.md", report)
    return report


def _load_ppmc(
    path: str | Path,
    records: dict[str, WeakRecord],
    min_length: int,
    max_length: int,
    skipped: dict[str, int],
) -> None:
    frame = pd.read_csv(path, sep="\t")
    for _, row in frame.iterrows():
        protein_id = str(row["UniProt.Acc"]).strip()
        sequence = clean_protein_sequence(str(row["Full.seq"]))
        if not _sequence_ok(sequence, min_length, max_length, skipped):
            continue
        codes = str(row["Datasets"]).strip()
        llps_label = 0 if codes in {"NP", "ND"} else 1
        negative_type = {"NP": "structured", "ND": "disordered"}.get(codes, "unknown")
        role_label = _role_from_ppmc_codes(codes)
        records[protein_id] = WeakRecord(
            protein_id=protein_id,
            sequence=sequence,
            llps_label=llps_label,
            source={"PPMC-LLPS-Datasets"},
            uniprot_id=protein_id,
            gene_name=str(row.get("Gene.Name", "")).strip(),
            raw_dataset_codes=codes,
            negative_type=negative_type,
            role_label=role_label,
            label_quality="curated",
            evidence_level="high" if llps_label == 0 else "medium",
            dependency="none" if role_label in {"driver", "negative_structured", "negative_disordered"} else "protein_partner",
            label_confidence=0.95 if llps_label == 0 else 0.85,
            sample_weight=0.95 if llps_label == 0 else 0.85,
            source_id=protein_id,
        )


def _load_phasepro(
    path: str | Path,
    records: dict[str, WeakRecord],
    min_length: int,
    max_length: int,
    skipped: dict[str, int],
) -> None:
    data = json.loads(Path(path).read_text())
    values = data.values() if isinstance(data, dict) else data
    for raw in values:
        if not isinstance(raw, dict):
            continue
        protein_id = str(raw.get("accession") or raw.get("uniprot") or raw.get("id") or "").strip()
        if not protein_id:
            continue
        sequence = clean_protein_sequence(str(raw.get("sequence", "")))
        if not _sequence_ok(sequence, min_length, max_length, skipped):
            continue
        existing = records.get(protein_id)
        if existing is None:
            record = WeakRecord(
                protein_id=protein_id,
                sequence=sequence,
                llps_label=1,
                source={"PhaSePro"},
                uniprot_id=protein_id,
                gene_name=str(raw.get("gene", "")).strip(),
                raw_dataset_codes="PhaSePro",
                role_label="driver",
                label_quality="gold",
                evidence_level="high",
                dependency=_phasepro_dependency(raw),
                species=str(raw.get("organism") or "").strip(),
                tax_id=_normalize_tax_id(raw.get("taxon")),
                source_id=protein_id,
                label_confidence=1.0,
                sample_weight=1.0,
            )
            records[protein_id] = record
        else:
            record = existing
            if record.sequence != sequence:
                record.sequence = sequence
                record.notes.append("sequence_replaced_by_phasepro_for_region_alignment")
            record.llps_label = 1
            record.source.add("PhaSePro")
            record.role_label = "driver"
            record.label_quality = "gold"
            record.evidence_level = "high"
            record.dependency = _phasepro_dependency(raw)
            if not record.species:
                record.species = str(raw.get("organism") or "").strip()
            if not record.tax_id:
                record.tax_id = _normalize_tax_id(raw.get("taxon"))
            record.label_confidence = max(record.label_confidence, 1.0)
            record.sample_weight = max(record.sample_weight, 1.0)
            record.negative_type = "unknown"
        record.regions.extend(_phasepro_regions(record.protein_id, raw, len(record.sequence)))


def _load_llpsdb_positive(
    path: str | Path,
    records: dict[str, WeakRecord],
    min_length: int,
    max_length: int,
    skipped: dict[str, int],
) -> None:
    frame = pd.read_csv(path)
    for _, row in frame.iterrows():
        protein_id = str(row.get("uniprot_id") or row.get("Uniprot ID") or row.get("UniprotID") or "").strip()
        sequence = clean_protein_sequence(str(row.get("sequence_clean") or row.get("Sequence") or row.get("sequence") or ""))
        if not protein_id or not _sequence_ok(sequence, min_length, max_length, skipped):
            continue
        source_subset = str(row.get("source_subset") or "").strip()
        subset_tag = source_subset or "silver_positive_candidate"
        evidence_note = f"llpsdb_v2_subset={subset_tag}"
        existing = records.get(protein_id)
        if existing is None:
            records[protein_id] = WeakRecord(
                protein_id=protein_id,
                sequence=sequence,
                llps_label=1,
                source={"LLPSDB-v2"},
                uniprot_id=protein_id,
                gene_name=str(row.get("gene_name") or row.get("Gene name") or row.get("Gene Name") or "").strip(),
                raw_dataset_codes=f"LLPSDB-v2:{subset_tag}",
                role_label="positive",
                label_quality="curated",
                evidence_level="medium",
                dependency="condition",
                species=str(row.get("organism") or row.get("Species_type") or "").strip(),
                tax_id=_normalize_tax_id(row.get("NCBI code") or row.get("NCBI")),
                source_id=protein_id,
                label_confidence=0.8,
                sample_weight=0.8,
                notes=[evidence_note],
            )
            continue
        existing.source.add("LLPSDB-v2")
        existing.notes.append(evidence_note)
        if existing.sequence != sequence and len(sequence) > len(existing.sequence):
            existing.sequence = sequence
            existing.notes.append("sequence_replaced_by_llpsdb_v2")
        if existing.llps_label != 1:
            existing.llps_label = 1
        existing.label_quality = _more_confident_quality(existing.label_quality, "curated")
        existing.evidence_level = _more_confident_evidence(existing.evidence_level, "medium")
        if existing.dependency in {"", "unknown"}:
            existing.dependency = "condition"
        if not existing.species:
            existing.species = str(row.get("organism") or row.get("Species_type") or "").strip()
        if not existing.tax_id:
            existing.tax_id = _normalize_tax_id(row.get("NCBI code") or row.get("NCBI"))
        existing.label_confidence = max(existing.label_confidence, 0.8)
        existing.sample_weight = max(existing.sample_weight, 0.8)
        if existing.role_label == "unknown":
            existing.role_label = "positive"


def _collect_accessions(path: str | Path | None, column: str) -> set[str]:
    if path is None:
        return set()
    csv_path = Path(path)
    if not csv_path.exists():
        return set()
    try:
        frame = pd.read_csv(csv_path, usecols=[column])
    except ValueError:
        return set()
    if column not in frame.columns:
        return set()
    return {_normalize_accession(value) for value in frame[column].dropna().astype(str) if _normalize_accession(value)}


def _read_delimited_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame.columns) == 1:
        retry = pd.read_csv(path, sep="\t")
        if len(retry.columns) > 1:
            return retry
    return frame


def _load_uniprot_sequence_lookup(path: str | Path | None, accessions: set[str]) -> dict[str, str]:
    if path is None or not accessions:
        return {}
    fasta_path = Path(path)
    if not fasta_path.exists():
        return {}
    wanted = {acc.split("-")[0] for acc in accessions if acc}
    lookup: dict[str, str] = {}
    current_acc = ""
    chunks: list[str] = []
    with _open_text(fasta_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_acc and current_acc in wanted and current_acc not in lookup:
                    lookup[current_acc] = clean_protein_sequence("".join(chunks))
                current_acc = _extract_uniprot_accession(line)
                chunks = []
                continue
            chunks.append(line)
        if current_acc and current_acc in wanted and current_acc not in lookup:
            lookup[current_acc] = clean_protein_sequence("".join(chunks))
    return lookup


def _load_uniprot_accessions(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    fasta_path = Path(path)
    if not fasta_path.exists():
        return set()
    accessions: set[str] = set()
    with _open_text(fasta_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or not line.startswith(">"):
                continue
            accession = _extract_uniprot_accession(line)
            if accession:
                accessions.add(accession)
    return accessions


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def _extract_uniprot_accession(header: str) -> str:
    match = re.search(r"\|([^|]+)\|", header)
    if match:
        return match.group(1).split("-")[0].strip()
    return header[1:].split()[0].split("-")[0].strip()


def _normalize_accession(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text == "nan":
        return ""
    return text.split(";")[0].split("|")[0].split("-")[0].strip()


def _normalize_tax_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text in {"0", "nan", "None"}:
        return ""
    if text.isdigit():
        return text
    return ""


def _load_phasepdb_positive(
    path: str | Path,
    sequence_lookup: dict[str, str],
    records: dict[str, WeakRecord],
    min_length: int,
    max_length: int,
    skipped: dict[str, int],
) -> None:
    frame = pd.read_csv(path)
    if frame.empty:
        return
    for _, row in frame.iterrows():
        protein_id = _normalize_accession(row.get("uniprot_id"))
        if not protein_id:
            continue
        sequence = sequence_lookup.get(protein_id, "")
        if not sequence:
            skipped["missing_phasepdb_sequence"] += 1
            continue
        if not _sequence_ok(sequence, min_length, max_length, skipped):
            continue
        class_ = str(row.get("class_") or "").strip()
        source_weight = 0.9 if class_ == "PS-self" else 0.35
        role_label = "driver" if class_ == "PS-self" else "context_positive"
        evidence_note = f"phasepdb_class={class_ or 'unknown'}"
        existing = records.get(protein_id)
        if existing is None:
            records[protein_id] = WeakRecord(
                protein_id=protein_id,
                sequence=sequence,
                llps_label=1,
                source={"PhaSepDB-3"},
                uniprot_id=protein_id,
                gene_name=str(row.get("primary_name") or "").strip(),
                raw_dataset_codes=f"PhaSepDB-3:{class_ or 'unknown'}",
                role_label=role_label,
                label_quality="gold" if class_ == "PS-self" else "weak",
                evidence_level="high" if class_ == "PS-self" else "low",
                dependency="none" if class_ == "PS-self" else "condition",
                species=str(row.get("organism") or "").strip(),
                tax_id=_normalize_tax_id(row.get("tax_id") or row.get("taxon")),
                source_id=protein_id,
                label_confidence=source_weight,
                sample_weight=source_weight,
                notes=[evidence_note],
            )
            continue
        existing.source.add("PhaSepDB-3")
        existing.notes.append(evidence_note)
        if existing.sequence != sequence and len(sequence) > len(existing.sequence):
            existing.sequence = sequence
        existing.notes.append("sequence_replaced_by_phasepdb")
        existing.llps_label = 1
        existing.label_quality = _more_confident_quality(existing.label_quality, "gold" if class_ == "PS-self" else "weak")
        existing.evidence_level = _more_confident_evidence(existing.evidence_level, "high" if class_ == "PS-self" else "low")
        if class_ == "PS-self":
            existing.dependency = "none"
        elif existing.dependency in {"", "unknown"}:
            existing.dependency = "condition"
        if not existing.species:
            existing.species = str(row.get("organism") or "").strip()
        if not existing.tax_id:
            existing.tax_id = _normalize_tax_id(row.get("tax_id") or row.get("taxon"))
        existing.label_confidence = max(existing.label_confidence, source_weight)
        existing.sample_weight = max(existing.sample_weight, source_weight)
        existing.raw_dataset_codes = (
            existing.raw_dataset_codes + f";PhaSepDB-3:{class_ or 'unknown'}"
            if existing.raw_dataset_codes
            else f"PhaSepDB-3:{class_ or 'unknown'}"
        )
        if class_ == "PS-self" and existing.role_label in {"unknown", "positive", "context_positive"}:
            existing.role_label = "driver"
        elif existing.role_label == "unknown":
            existing.role_label = role_label


def _load_cd_code_positive(
    proteins_csv: str | Path,
    links_csv: str | Path | None,
    sequence_lookup: dict[str, str],
    records: dict[str, WeakRecord],
    min_length: int,
    max_length: int,
    skipped: dict[str, int],
) -> None:
    proteins_path = Path(proteins_csv)
    if not proteins_path.exists():
        return
    proteins = pd.read_csv(proteins_path)
    if proteins.empty:
        return
    link_counts: dict[str, int] = {}
    link_names: dict[str, str] = {}
    if links_csv is not None:
        links_path = Path(links_csv)
        if links_path.exists():
            links = _read_delimited_table(links_path)
            if not links.empty and "uniprotkb_ac" in links.columns:
                link_counts = {
                    str(key): int(value)
                    for key, value in links["uniprotkb_ac"].astype(str).value_counts().items()
                }
                if "condensate_name" in links.columns:
                    link_names = {
                        str(key): ";".join(sorted({str(item) for item in group["condensate_name"].dropna() if str(item).strip()}))
                        for key, group in links.groupby("uniprotkb_ac")
                    }
    proteins = proteins.copy()
    if "uniprot_id" not in proteins.columns:
        return
    proteins["acc"] = proteins["uniprot_id"].map(_normalize_accession)
    protein_meta = proteins.dropna(subset=["acc"]).drop_duplicates(subset=["acc"]).set_index("acc", drop=False)
    for protein_id, link_count in sorted(link_counts.items()):
        protein_id = _normalize_accession(protein_id)
        if not protein_id or protein_id not in protein_meta.index:
            continue
        sequence = sequence_lookup.get(protein_id, "")
        if not sequence:
            skipped["missing_cd_code_sequence"] += 1
            continue
        if not _sequence_ok(sequence, min_length, max_length, skipped):
            continue
        source_weight = min(0.45, 0.2 + 0.02 * min(link_count, 10)) if link_count else 0.2
        evidence_note = f"cd_code_links={link_count}"
        if protein_id in link_names and link_names[protein_id]:
            evidence_note = f"{evidence_note};condensates={link_names[protein_id].split(';')[0]}"
        row = protein_meta.loc[protein_id]
        existing = records.get(protein_id)
        if existing is None:
            records[protein_id] = WeakRecord(
                protein_id=protein_id,
                sequence=sequence,
                llps_label=1,
                source={"CD-CODE"},
                uniprot_id=protein_id,
                gene_name=str(row.get("gene_name") or row.get("name") or "").strip(),
                raw_dataset_codes=f"CD-CODE:links={link_count}",
                role_label="condensate_member",
                label_quality="weak",
                evidence_level="low",
                dependency="condition",
                species=str(row.get("species_name") or row.get("species") or "").strip(),
                tax_id=_normalize_tax_id(row.get("species_taxon_id") or row.get("taxon_id")),
                label_confidence=source_weight,
                sample_weight=source_weight,
                notes=[evidence_note],
                source_id=protein_id,
            )
            continue
        existing.source.add("CD-CODE")
        existing.notes.append(evidence_note)
        if existing.sequence != sequence and len(sequence) > len(existing.sequence):
            existing.sequence = sequence
        existing.notes.append("sequence_replaced_by_cd_code")
        existing.llps_label = 1
        existing.label_quality = _more_confident_quality(existing.label_quality, "weak")
        existing.evidence_level = _more_confident_evidence(existing.evidence_level, "low")
        if existing.dependency in {"", "unknown"}:
            existing.dependency = "condition"
        if not existing.species:
            existing.species = str(row.get("species_name") or row.get("species") or "").strip()
        if not existing.tax_id:
            existing.tax_id = _normalize_tax_id(row.get("species_taxon_id") or row.get("taxon_id"))
        existing.label_confidence = max(existing.label_confidence, source_weight)
        existing.sample_weight = max(existing.sample_weight, source_weight)
        existing.raw_dataset_codes = (
            existing.raw_dataset_codes + f";CD-CODE:links={link_count}"
            if existing.raw_dataset_codes
            else f"CD-CODE:links={link_count}"
        )
        if existing.role_label in {"unknown", "positive"}:
            existing.role_label = "condensate_member"


def _sequence_ok(sequence: str, min_length: int, max_length: int, skipped: dict[str, int]) -> bool:
    if not sequence:
        skipped["empty_sequence"] += 1
        return False
    if len(sequence) < min_length:
        skipped["short"] += 1
        return False
    if max_length > 0 and len(sequence) > max_length:
        skipped["long"] += 1
        return False
    return True


def _role_from_ppmc_codes(codes: str) -> str:
    if codes == "NP":
        return "negative_structured"
    if codes == "ND":
        return "negative_disordered"
    tokens = {token.strip() for token in codes.split(";")}
    has_driver = any(token.startswith("D") for token in tokens)
    has_client = any(token.startswith("C") or token == "CE" for token in tokens)
    if has_driver and has_client:
        return "driver_or_client"
    if has_driver:
        return "driver"
    if has_client:
        return "client"
    return "positive"


def _phasepro_dependency(raw: dict[str, Any]) -> str:
    if str(raw.get("rna_req") or "").strip() not in {"", "N", "No", "none", "Not known"}:
        return "RNA_DNA"
    if str(raw.get("partner_dep") or "").strip() == "Y":
        return "protein_partner"
    if str(raw.get("ptm_dep") or "").strip() == "Y":
        return "PTM"
    if str(raw.get("domain_dep") or "").strip() == "Y":
        return "protein_partner"
    if str(raw.get("interaction") or "").strip():
        return "protein_partner"
    return "none"


def _phasepro_regions(protein_id: str, raw: dict[str, Any], length: int) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    boundaries = str(raw.get("boundaries") or "")
    segment = str(raw.get("segment") or "")
    for start_text, end_text in re.findall(r"(\d+)\s*-\s*(\d+)", boundaries):
        start = max(0, int(start_text) - 1)
        end = min(length - 1, int(end_text) - 1)
        if end < start:
            continue
        regions.append(
            {
                "protein_id": protein_id,
                "start": start,
                "end": end,
                "type": "DPR_candidate",
                "region_type": "DPR_gold",
                "region_label": 1,
                "confidence": 1.0,
                "evidence_level": "high",
                "source": "PhaSePro",
                "assay": "literature",
                "segment": segment,
                "notes": str(raw.get("description") or "").strip(),
            }
        )
    return regions


def _more_confident_quality(current: str, candidate: str) -> str:
    order = {"pseudo": 0, "ambiguous": 1, "weak": 2, "curated": 3, "gold": 4}
    return candidate if order.get(candidate, 0) >= order.get(current, 0) else current


def _more_confident_evidence(current: str, candidate: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return candidate if order.get(candidate, 0) >= order.get(current, 0) else current


def _deduplicate_sequences(records: list[WeakRecord], skipped: dict[str, int]) -> list[WeakRecord]:
    ordered = sorted(
        records,
        key=lambda record: (
            not record.has_region_label,
            -{"pseudo": 0, "ambiguous": 1, "weak": 2, "curated": 3, "gold": 4}.get(record.label_quality, 0),
            -record.llps_label,
            -record.label_confidence,
            -record.sample_weight,
            record.protein_id,
        ),
    )
    seen: set[str] = set()
    selected: list[WeakRecord] = []
    for record in ordered:
        if record.sequence in seen:
            skipped["duplicate_sequence"] += 1
            continue
        seen.add(record.sequence)
        selected.append(record)
    return selected


def _limit_records(records: list[WeakRecord], max_records: int, seed: int) -> list[WeakRecord]:
    grouped: dict[str, list[WeakRecord]] = {}
    for record in records:
        grouped.setdefault(_stratum(record), []).append(record)
    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)
    selected: list[WeakRecord] = []
    strata = sorted(grouped)
    while len(selected) < max_records and strata:
        next_strata: list[str] = []
        for stratum in strata:
            group = grouped[stratum]
            if group and len(selected) < max_records:
                selected.append(group.pop())
            if group:
                next_strata.append(stratum)
        strata = next_strata
    return selected


def _assign_splits(
    records: list[WeakRecord],
    seed: int,
    train_frac: float,
    valid_frac: float,
) -> None:
    grouped: dict[str, list[WeakRecord]] = {}
    for record in records:
        grouped.setdefault(_stratum(record), []).append(record)
    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)
        n_total = len(group)
        n_train = int(round(n_total * train_frac))
        n_valid = int(round(n_total * valid_frac))
        if n_total >= 3:
            n_valid = max(1, n_valid)
            n_test = max(1, n_total - n_train - n_valid)
            n_train = max(1, n_total - n_valid - n_test)
        else:
            n_valid = 1 if n_total == 2 else 0
            n_train = n_total - n_valid
        for index, record in enumerate(group):
            if index < n_train:
                record.split = "train"
            elif index < n_train + n_valid:
                record.split = "valid"
            else:
                record.split = "test"


def _stratum(record: WeakRecord) -> str:
    if record.llps_label == 1:
        return "positive_region" if record.has_region_label else "positive"
    return f"negative_{record.negative_type}"


def _manifest_row(record: WeakRecord, regions_path: Path) -> dict[str, Any]:
    return {
        "protein_id": record.protein_id,
        "uniprot_id": record.uniprot_id,
        "gene_name": record.gene_name,
        "species": record.species,
        "tax_id": record.tax_id,
        "sequence": record.sequence,
        "length": record.length,
        "source": ";".join(sorted(record.source)),
        "source_count": len(record.source),
        "split": record.split,
        "llps_label": record.llps_label,
        "label_quality": record.label_quality,
        "evidence_level": record.evidence_level,
        "dependency": record.dependency,
        "label_confidence": record.label_confidence,
        "sample_weight": record.sample_weight,
        "negative_type": record.negative_type,
        "role_label": record.role_label,
        "has_region_label": int(record.has_region_label),
        "region_count": len(record.regions),
        "region_label_path": str(regions_path) if record.has_region_label else "",
        "raw_dataset_codes": record.raw_dataset_codes,
        "notes": ";".join(record.notes),
    }


def _master_row(record: WeakRecord) -> dict[str, Any]:
    row = _manifest_row(record, Path("regions.jsonl"))
    row["regions"] = record.regions
    return row


def _build_report(
    records: list[WeakRecord],
    ppmc_tsv: str | Path,
    phasepro_json: str | Path,
    manifest_path: Path,
    regions_path: Path,
    skipped: dict[str, int],
) -> dict[str, Any]:
    frame = pd.DataFrame([_manifest_row(record, regions_path) for record in records])
    by_split = frame["split"].value_counts().sort_index().to_dict() if not frame.empty else {}
    by_label = frame["llps_label"].value_counts().sort_index().to_dict() if not frame.empty else {}
    by_source = frame["source"].value_counts().to_dict() if not frame.empty else {}
    return {
        "inputs": {"ppmc_tsv": str(ppmc_tsv), "phasepro_json": str(phasepro_json)},
        "outputs": {"manifest": str(manifest_path), "regions": str(regions_path)},
        "total_records": len(records),
        "by_split": {str(key): int(value) for key, value in by_split.items()},
        "by_llps_label": {str(key): int(value) for key, value in by_label.items()},
        "by_source": {str(key): int(value) for key, value in by_source.items()},
        "mean_sample_weight": float(frame["sample_weight"].mean()) if "sample_weight" in frame.columns and not frame.empty else 0.0,
        "region_labeled_records": int(sum(record.has_region_label for record in records)),
        "region_count": int(sum(len(record.regions) for record in records)),
        "phase1_tables": {
            "proteins": int(len(records)),
            "protein_labels": int(len(records)),
            "regions": int(sum(len(record.regions) for record in records)),
            "evidence": int(sum(max(len(record.source), 1) + len(record.regions) for record in records)),
            "source_map": int(sum(max(len(record.source), 1) for record in records)),
        },
        "skipped": skipped,
    }


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Weak Supervision Dataset Report",
        "",
        f"- total_records: {report['total_records']}",
        f"- region_labeled_records: {report['region_labeled_records']}",
        f"- region_count: {report['region_count']}",
        f"- mean_sample_weight: {report.get('mean_sample_weight', 0.0):.3f}",
        f"- proteins.csv: {report['phase1_tables']['proteins']}",
        f"- protein_labels.csv: {report['phase1_tables']['protein_labels']}",
        f"- regions.csv: {report['phase1_tables']['regions']}",
        f"- evidence.csv: {report['phase1_tables']['evidence']}",
        f"- source_map.csv: {report['phase1_tables']['source_map']}",
        f"- manifest: {report['outputs']['manifest']}",
        f"- regions: {report['outputs']['regions']}",
        "",
        "## Split Counts",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report["by_split"].items())
    lines.append("")
    lines.append("## Label Counts")
    lines.extend(f"- {key}: {value}" for key, value in report["by_llps_label"].items())
    lines.append("")
    lines.append("## Source Counts")
    lines.extend(f"- {key}: {value}" for key, value in report.get("by_source", {}).items())
    lines.append("")
    lines.append("## Phase1 Tables")
    for key, value in report.get("phase1_tables", {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Skipped")
    lines.extend(f"- {key}: {value}" for key, value in report["skipped"].items())
    path.write_text("\n".join(lines) + "\n")


def _write_phase1_tables(
    records: list[WeakRecord],
    out_dir: Path,
    reviewed_accessions: set[str],
    proteins_path: Path,
    protein_labels_path: Path,
    regions_table_path: Path,
    evidence_path: Path,
    source_map_path: Path,
) -> None:
    proteins_rows = [_phase1_protein_row(record, reviewed_accessions) for record in records]
    protein_label_rows = [_phase1_protein_label_row(record) for record in records]
    regions_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    source_map_rows: list[dict[str, Any]] = []

    for record in records:
        sources = sorted(record.source) or ["unknown"]
        for source_index, source in enumerate(sources, start=1):
            source_map_rows.append(
                {
                    "source": source,
                    "source_id": record.source_id or record.protein_id,
                    "protein_id": record.protein_id,
                    "uniprot_id": record.uniprot_id,
                    "mapping_status": "direct_uniprot",
                    "sequence_match": "yes" if record.uniprot_id in reviewed_accessions else "unknown",
                    "notes": ";".join(record.notes),
                }
            )
            evidence_rows.append(
                {
                    "evidence_id": f"{record.protein_id}_P{source_index}",
                    "protein_id": record.protein_id,
                    "region_id": "",
                    "label_scope": "protein",
                    "source": source,
                    "pubmed_id": "",
                    "doi": "",
                    "assay": _phase1_assay(source),
                    "in_vitro": "",
                    "in_vivo": "",
                    "condition": _phase1_condition(record),
                    "confidence": float(record.label_confidence),
                    "notes": ";".join(record.notes),
                }
            )

        for region_index, region in enumerate(record.regions, start=1):
            region_id = f"{record.protein_id}_R{region_index}"
            regions_rows.append(_phase1_region_row(record, region, region_id))
            region_source = str(region.get("source") or next(iter(sources), "unknown"))
            region_notes = str(region.get("notes") or "")
            if not region_notes:
                region_notes = ";".join(record.notes)
            evidence_rows.append(
                {
                    "evidence_id": f"{record.protein_id}_R{region_index}",
                    "protein_id": record.protein_id,
                    "region_id": region_id,
                    "label_scope": "region",
                    "source": region_source,
                    "pubmed_id": "",
                    "doi": "",
                    "assay": str(region.get("assay") or _phase1_assay(region_source)),
                    "in_vitro": "",
                    "in_vivo": "",
                    "condition": _phase1_condition(record),
                    "confidence": float(region.get("confidence", record.label_confidence)),
                    "notes": region_notes,
                }
            )

    pd.DataFrame(proteins_rows).sort_values(["protein_id"]).to_csv(proteins_path, index=False)
    pd.DataFrame(protein_label_rows).sort_values(["protein_id"]).to_csv(protein_labels_path, index=False)
    regions_df = pd.DataFrame(regions_rows)
    if not regions_df.empty:
        regions_df = regions_df.sort_values(["protein_id", "start", "end"])
    regions_df.to_csv(regions_table_path, index=False)
    evidence_df = pd.DataFrame(evidence_rows)
    if not evidence_df.empty:
        evidence_df = evidence_df.sort_values(["protein_id", "label_scope", "evidence_id"])
    evidence_df.to_csv(evidence_path, index=False)
    source_map_df = pd.DataFrame(source_map_rows)
    if not source_map_df.empty:
        source_map_df = source_map_df.sort_values(["source", "protein_id", "source_id"])
    source_map_df.to_csv(source_map_path, index=False)


def _phase1_protein_row(record: WeakRecord, reviewed_accessions: set[str]) -> dict[str, Any]:
    return {
        "protein_id": record.protein_id,
        "uniprot_id": record.uniprot_id,
        "gene_name": record.gene_name,
        "species": record.species,
        "tax_id": record.tax_id,
        "sequence": record.sequence,
        "length": record.length,
        "reviewed": int(record.uniprot_id in reviewed_accessions),
        "source_list": ";".join(sorted(record.source)),
    }


def _phase1_protein_label_row(record: WeakRecord) -> dict[str, Any]:
    return {
        "protein_id": record.protein_id,
        "llps_label": record.llps_label,
        "label_quality": record.label_quality,
        "role_hint": _phase1_role_hint(record),
        "negative_type": _phase1_negative_type(record),
        "dependency": record.dependency,
        "confidence": record.label_confidence,
        "evidence_level": record.evidence_level,
        "source": ";".join(sorted(record.source)),
    }


def _phase1_region_row(record: WeakRecord, region: dict[str, Any], region_id: str) -> dict[str, Any]:
    start_0 = int(region.get("start", 0))
    end_0 = int(region.get("end", start_0))
    region_label = region.get("region_label")
    if region_label is None:
        region_label = 1 if str(region.get("type") or "").startswith("DPR") else "unknown"
    return {
        "protein_id": record.protein_id,
        "region_id": region_id,
        "start": start_0 + 1,
        "end": end_0 + 1,
        "region_type": str(region.get("region_type") or region.get("type") or "DPR_candidate"),
        "region_label": region_label,
        "confidence": float(region.get("confidence", 1.0)),
        "evidence_level": str(region.get("evidence_level") or "candidate"),
        "source": str(region.get("source") or ";".join(sorted(record.source))),
        "assay": str(region.get("assay") or "database"),
        "notes": str(region.get("notes") or region.get("segment") or ""),
    }


def _phase1_assay(source: str) -> str:
    mapping = {
        "PhaSePro": "literature",
        "PhaSepDB-3": "database",
        "LLPSDB-v2": "database",
        "PPMC-LLPS-Datasets": "benchmark",
        "CD-CODE": "condensate_membership",
    }
    return mapping.get(source, "database")


def _phase1_condition(record: WeakRecord) -> str:
    return record.dependency if record.dependency != "unknown" else ""


def _phase1_negative_type(record: WeakRecord) -> str:
    mapping = {
        "structured": "NP_structured",
        "disordered": "ND_disordered",
        "unknown": "",
    }
    return mapping.get(record.negative_type, record.negative_type)


def _phase1_role_hint(record: WeakRecord) -> str:
    mapping = {
        "driver": "driver",
        "client": "client",
        "condensate_member": "member",
        "context_positive": "member",
        "driver_or_client": "unknown",
        "positive": "unknown",
        "negative_structured": "unknown",
        "negative_disordered": "unknown",
        "unknown": "unknown",
    }
    return mapping.get(record.role_label, "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a weak-supervision PhaseFlow dataset from PPMC and PhaSePro.")
    parser.add_argument("--ppmc-tsv", default="data/raw/ppmc_llps_datasets/datasets.tsv")
    parser.add_argument("--phasepro-json", default="data/raw/phasepro/phasepro_full.json")
    parser.add_argument("--out-dir", default="data/processed/weak_supervision")
    parser.add_argument("--llpsdb-positive-csv", default=None, help="Optional LLPSDB v2 silver-positive candidate table.")
    parser.add_argument("--phasepdb-csv", default=None, help="Optional parsed PhaSepDB 3 protein table.")
    parser.add_argument("--cd-code-proteins-csv", default=None, help="Optional parsed CD-CODE protein table.")
    parser.add_argument("--cd-code-links-csv", default=None, help="Optional parsed CD-CODE protein-condensate link table.")
    parser.add_argument("--uniprot-fasta", default=None, help="Optional UniProt Swiss-Prot FASTA used for sequence lookup.")
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    args = parser.parse_args()
    report = prepare_weak_dataset(
        ppmc_tsv=args.ppmc_tsv,
        phasepro_json=args.phasepro_json,
        out_dir=args.out_dir,
        llpsdb_positive_csv=args.llpsdb_positive_csv,
        phasepdb_csv=args.phasepdb_csv,
        cd_code_proteins_csv=args.cd_code_proteins_csv,
        cd_code_links_csv=args.cd_code_links_csv,
        uniprot_fasta=args.uniprot_fasta,
        min_length=args.min_length,
        max_length=args.max_length,
        max_records=args.max_records,
        seed=args.seed,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
