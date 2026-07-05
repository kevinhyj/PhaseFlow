from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.full_length.data import build_phaseflow_strict_dataset as strict
except ImportError as exc:  # pragma: no cover - optional historical rebuild helper
    raise ImportError(
        "build_server_final_dataset.py requires the historical helper "
        "scripts/full_length/data/build_phaseflow_strict_dataset.py, which is "
        "not included in the lightweight GitHub integration."
    ) from exc


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MMSEQS = strict.DEFAULT_MMSEQS


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    interim_root = Path(args.interim_root)
    processed_root = Path(args.processed_root)
    splits_root = Path(args.splits_root)
    pseudo_root = Path(args.pseudo_root)
    feature_root = Path(args.feature_root)

    for path in (
        interim_root,
        interim_root / "cleaned_sources",
        processed_root,
        processed_root / "qc",
        splits_root,
        pseudo_root / "round0_external",
        pseudo_root / "round1_self_training",
        feature_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    swiss = strict.load_swissprot(raw_root / "uniprot_swissprot" / "uniprot_sprot_reviewed_20_2048.fasta.gz")
    candidates: list[strict.Candidate] = []
    regions: list[strict.Region] = []
    source_reports: dict[str, Any] = {}

    source_reports["ppmc"] = strict.load_ppmc(raw_root / "ppmc_llps_datasets" / "datasets.tsv", candidates) or {}
    source_reports["phasepro"] = strict.load_phasepro(raw_root / "phasepro" / "phasepro_full.json", candidates, regions) or {}
    source_reports["llpsdb_v2"] = strict.load_llpsdb(Path(args.llpsdb_csv), candidates) or {}
    source_reports["phasepdb3"] = strict.load_phasepdb(Path(args.phasepdb_csv), swiss, candidates) or {}
    source_reports["cd_code"] = strict.load_cd_code(Path(args.cd_code_proteins_csv), swiss, candidates) or {}
    source_reports["drllps"] = load_drllps(raw_root / "drllps" / "drllps.txt", candidates)
    strict.load_candidate_priors(raw_root / "ppmc_llps_datasets" / "sequential_elements.json", regions)
    strict.load_llpsdb_priors(Path(args.llpsdb_all_csv), regions)

    proteins, labels, evidence, source_map, sequence_conflicts, label_conflicts = strict.resolve_candidates(
        candidates,
        swiss,
        max_pu_records=args.max_pu_records,
        seed=args.seed,
    )
    proteins_by_id = {row["protein_id"]: row for row in proteins}
    labels_by_id = {row["protein_id"]: row for row in labels}

    known_llps_ids = {
        row["protein_id"]
        for row in labels
        if str(row.get("label_tier", "")).lower() in {"gold", "curated", "weak", "ambiguous"}
        or row.get("label_mask") == 1
    }
    pdb_report = load_pdb_np_negatives(
        raw_root / "pdb" / "pdb_seqres.txt.gz",
        proteins,
        labels,
        evidence,
        source_map,
        known_llps_ids=known_llps_ids,
        max_records=args.max_pdb_negatives,
        seed=args.seed,
    )
    source_reports["pdb_seqres_np"] = pdb_report

    proteins.sort(key=lambda row: row["protein_id"])
    labels.sort(key=lambda row: row["protein_id"])
    proteins_by_id = {row["protein_id"]: row for row in proteins}
    labels_by_id = {row["protein_id"]: row for row in labels}
    regions = strict.filter_regions(regions, proteins_by_id)

    write_csv(interim_root / "source_records.csv", [strict.candidate_to_row(item) for item in candidates])
    write_csv(interim_root / "evidence_long.csv", evidence)
    write_csv(interim_root / "source_map_long.csv", source_map)
    write_csv(interim_root / "sequence_conflicts.csv", sequence_conflicts)
    write_csv(interim_root / "label_conflicts.csv", label_conflicts)
    write_csv(interim_root / "cleaned_sources" / "pdb_np_negatives.csv", pdb_report.get("selected_rows", []))
    write_csv(interim_root / "cleaned_sources" / "drllps_cleaned.csv", source_reports["drllps"].get("rows", []))

    fasta_path = interim_root / "strict_sequences.fasta"
    strict.write_fasta(fasta_path, [(row["protein_id"], row["sequence"]) for row in proteins])
    cluster30 = strict.run_or_fallback_clusters(fasta_path, interim_root / "mmseqs30", args.mmseqs, 0.3)
    cluster50 = strict.run_or_fallback_clusters(fasta_path, interim_root / "mmseqs50", args.mmseqs, 0.5)
    split_by_id = strict.assign_splits(proteins, labels_by_id, cluster30, args.seed)

    for row in proteins:
        pid = row["protein_id"]
        row["cluster_id_30"] = cluster30.get(pid, row["sequence_hash"])
        row["cluster_id_50"] = cluster50.get(pid, row["sequence_hash"])
        row["split"] = split_by_id[pid]
    for row in labels:
        pid = row["protein_id"]
        row["cluster_id_30"] = cluster30.get(pid, proteins_by_id[pid]["sequence_hash"])
        row["cluster_id_50"] = cluster50.get(pid, proteins_by_id[pid]["sequence_hash"])
        row["split"] = split_by_id[pid]

    region_rows = make_region_rows_for_canonical(regions)
    region_span_rows = make_region_spans_jsonl(proteins, labels_by_id, regions)
    mil_bag_rows = make_mil_bags(proteins, labels_by_id)
    split_rows = make_split_rows(proteins, labels_by_id, split_by_id)
    protein_master_rows = make_protein_master_rows(proteins, labels_by_id)
    region_master_rows = make_region_master_rows(region_rows)
    evidence_table_rows = make_evidence_table(evidence, region_rows)
    manifest_rows = make_manifest_rows(proteins, labels_by_id, split_by_id)
    source_map_rows = make_source_map_rows(source_map)

    write_jsonl(processed_root / "protein_master.jsonl", protein_master_rows)
    write_jsonl(processed_root / "region_master.jsonl", region_master_rows)
    write_csv(processed_root / "evidence_table.csv", evidence_table_rows)
    write_csv(processed_root / "proteins.csv", proteins)
    write_csv(processed_root / "protein_labels.csv", labels)
    write_csv(processed_root / "regions.csv", region_rows)
    write_csv(processed_root / "source_map.csv", source_map_rows)
    write_csv(processed_root / "server_final_manifest.csv", manifest_rows)
    write_jsonl(processed_root / "region_spans.jsonl", region_span_rows)
    write_jsonl(processed_root / "mil_bags.jsonl", mil_bag_rows)
    write_csv(processed_root / "splits.csv", split_rows)
    write_candidate_priors_h5(processed_root / "candidate_priors.h5", regions, proteins_by_id)

    write_splits(splits_root, proteins, labels_by_id, split_by_id, cluster30, cluster50)
    write_pseudo_inputs(pseudo_root / "round0_external", proteins, labels_by_id, split_by_id, cluster30, cluster50)
    write_round1_placeholders(pseudo_root / "round1_self_training")

    leakage_rows = leakage_audit_rows(proteins, labels_by_id, split_by_id)
    write_csv(splits_root / "leakage_audit.csv", leakage_rows)
    write_csv(processed_root / "qc" / "duplicate_report.csv", duplicate_report_rows(proteins))
    write_csv(processed_root / "qc" / "conflict_report.csv", label_conflicts + sequence_conflicts)
    write_csv(processed_root / "qc" / "split_leakage_report.csv", leakage_rows)
    write_csv(processed_root / "qc" / "teacher_coverage_report.csv", teacher_coverage_report_rows(pseudo_root / "round0_external" / "teacher_scores.h5", proteins))
    write_csv(processed_root / "qc" / "region_gold_coverage_report.csv", region_gold_coverage_report_rows(proteins, region_span_rows))

    report = {
        "generated_on": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "disprot_included": False,
            "drllps_included": True,
            "pdb_seqres_np_included": True,
            "teacher_labels_generated": False,
            "feature_cache_generated": False,
        },
        "paths": {
            "interim_root": str(interim_root),
            "processed_root": str(processed_root),
            "splits_root": str(splits_root),
            "pseudo_root": str(pseudo_root),
            "feature_root": str(feature_root),
        },
        "source_reports": compact_source_reports(source_reports),
        "counts": {
            "proteins": len(proteins),
            "protein_master": len(protein_master_rows),
            "region_master": len(region_master_rows),
            "evidence_table": len(evidence_table_rows),
            "source_map": len(source_map_rows),
            "region_spans": len(region_span_rows),
            "mil_bags": len(mil_bag_rows),
            "sequence_conflicts": len(sequence_conflicts),
            "label_conflicts": len(label_conflicts),
            "leakage_rows": len(leakage_rows),
        },
        "by_label_tier": value_counts(labels, "label_tier"),
        "by_llps_label": value_counts(labels, "llps_label"),
        "by_negative_type": value_counts(labels, "negative_type"),
        "by_split": value_counts([{"split": split} for split in split_by_id.values()], "split"),
        "ready_for_training": False,
        "blocking_items": [
            "Round-0 teacher pseudo labels are not generated yet.",
            "Merged HDF5 feature cache is not generated yet.",
            "Protenix features are not generated yet.",
        ],
    }
    (processed_root / "server_final_dataset_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (interim_root / "processing_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical server-final PhaseFlow dataset tables.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--interim-root", default="data/interim/server_final")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument("--pseudo-root", default="data/pseudo_labels")
    parser.add_argument("--feature-root", default="data/features")
    parser.add_argument("--phasepdb-csv", default="data/interim/parsed_source_tables/phasepdb3/phasepdb3_proteins.csv")
    parser.add_argument("--llpsdb-csv", default="data/interim/parsed_source_tables/llpsdb_v2/llpsdb_v2_silver_positive_candidates.csv")
    parser.add_argument("--llpsdb-all-csv", default="data/interim/parsed_source_tables/llpsdb_v2/llpsdb_v2_proteins.csv")
    parser.add_argument("--cd-code-proteins-csv", default="data/interim/parsed_source_tables/cd_code/cd_code_proteins.csv")
    parser.add_argument("--mmseqs", default=str(DEFAULT_MMSEQS))
    parser.add_argument("--max-pu-records", type=int, default=5000)
    parser.add_argument("--max-pdb-negatives", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260524)
    return parser.parse_args()


def load_drllps(path: Path, candidates: list[strict.Candidate]) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "rows": []}
    frame = pd.read_csv(path, sep="\t")
    cleaned_rows: list[dict[str, Any]] = []
    added = 0
    skipped = 0
    for _, row in frame.iterrows():
        acc = strict.normalize_accession(row.get("UniProt ID"))
        sequence = strict.strict_sequence(row.get("Protein Sequence", ""))
        if not acc or sequence is None or not 30 <= len(sequence) <= 2048:
            skipped += 1
            continue
        role = strict.clean_text(row.get("LLPS Type")).lower()
        if role == "scaffold":
            label = strict.label_dict(1, "weak", 1, 0.55, "scaffold", "unknown", "none", 0.55)
        elif role == "regulator":
            label = strict.label_dict(None, "ambiguous", 0, 0.0, "regulator", "partner_dependent", "none", 0.0)
        else:
            label = strict.label_dict(None, "ambiguous", 0, 0.0, "client", "partner_dependent", "none", 0.0)
        candidates.append(
            strict.Candidate(
                source="DrLLPS",
                source_id=strict.clean_text(row.get("DrLLPS ID") or acc),
                protein_id=acc,
                uniprot_id=acc,
                isoform_id="",
                sequence=sequence,
                gene_name=strict.clean_text(row.get("Gene name")),
                primary_name=strict.clean_text(row.get("Gene name")),
                species=strict.clean_text(row.get("Species")),
                tax_id="",
                reviewed=False,
                label_tier=label["label_tier"],
                llps_label=label["llps_label"],
                label_mask=label["label_mask"],
                sample_weight=label["sample_weight"],
                role_type=label["role_type"],
                dependency_type=label["dependency_type"],
                negative_type=label["negative_type"],
                confidence=label["confidence"],
                notes=[
                    f"drllps_type={strict.clean_text(row.get('LLPS Type'))}",
                    f"condensate={strict.clean_text(row.get('Condensate'))[:200]}",
                    f"references={strict.clean_text(row.get('References'))[:200]}",
                ],
            )
        )
        cleaned_rows.append(
            {
                "source_id": strict.clean_text(row.get("DrLLPS ID")),
                "protein_id": acc,
                "gene_name": strict.clean_text(row.get("Gene name")),
                "species": strict.clean_text(row.get("Species")),
                "llps_type": strict.clean_text(row.get("LLPS Type")),
                "label_tier": label["label_tier"],
                "llps_label": "" if label["llps_label"] is None else label["llps_label"],
                "label_mask": label["label_mask"],
                "sample_weight": label["sample_weight"],
                "sequence_hash": strict.sequence_hash(sequence),
                "length": len(sequence),
            }
        )
        added += 1
    return {
        "status": "available",
        "raw_rows": int(len(frame)),
        "added_candidates": added,
        "skipped": skipped,
        "role_counts": {str(k): int(v) for k, v in frame["LLPS Type"].fillna("unknown").value_counts().items()},
        "rows": cleaned_rows,
    }


def load_pdb_np_negatives(
    path: Path,
    proteins: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    source_map: list[dict[str, Any]],
    *,
    known_llps_ids: set[str],
    max_records: int,
    seed: int,
) -> dict[str, Any]:
    if not path.exists() or max_records <= 0:
        return {"status": "missing_or_disabled", "selected_rows": []}
    candidates: list[dict[str, Any]] = []
    seen_hashes = {row["sequence_hash"] for row in proteins}
    current_header = ""
    chunks: list[str] = []
    with gzip.open(path, "rt", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                maybe_add_pdb_candidate(current_header, chunks, candidates, seen_hashes, known_llps_ids)
                current_header = line
                chunks = []
            elif line:
                chunks.append(line)
        maybe_add_pdb_candidate(current_header, chunks, candidates, seen_hashes, known_llps_ids)
    candidates.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['protein_id']}:{row['sequence_hash']}".encode()).hexdigest())
    selected = candidates[:max_records]
    existing_ids = {row["protein_id"] for row in proteins}
    added = 0
    for row in selected:
        pid = row["protein_id"]
        if pid in existing_ids:
            continue
        existing_ids.add(pid)
        proteins.append(
            {
                "protein_id": pid,
                "uniprot_id": "",
                "isoform_id": "",
                "gene_name": "",
                "primary_name": row["description"],
                "species": "",
                "tax_id": "",
                "sequence": row["sequence"],
                "length": row["length"],
                "sequence_hash": row["sequence_hash"],
                "reviewed": 0,
                "source_list": "PDB_SEQRES",
                "source_count": 1,
            }
        )
        labels.append(
            {
                "protein_id": pid,
                "llps_label": 0,
                "soft_label": 0.0,
                "label_mask": 1,
                "label_tier": "negative_curated",
                "role_type": "negative_structured",
                "dependency_type": "autonomous",
                "negative_type": "structured_negative",
                "confidence": 0.85,
                "sample_weight": 0.85,
                "source_set": "PDB_SEQRES",
                "pseudo_round": "none",
                "teacher_agreement": "",
                "notes": "pdb_seqres_structured_np_negative",
            }
        )
        evidence.append(
            {
                "evidence_id": f"{pid}_PDB1",
                "protein_id": pid,
                "region_id": "",
                "label_scope": "protein",
                "source": "PDB_SEQRES",
                "pubmed_id": "",
                "doi": "",
                "assay": "structured_protein_sequence",
                "in_vitro": "",
                "in_vivo": "",
                "condition": "",
                "confidence": 0.85,
                "notes": row["description"],
            }
        )
        source_map.append(
            {
                "source": "PDB_SEQRES",
                "source_id": row["source_id"],
                "protein_id": pid,
                "uniprot_id": "",
                "sequence_hash": row["sequence_hash"],
            }
        )
        added += 1
    return {
        "status": "available",
        "candidate_sequences": len(candidates),
        "selected": len(selected),
        "added": added,
        "selected_rows": selected,
    }


def maybe_add_pdb_candidate(
    header: str,
    chunks: list[str],
    candidates: list[dict[str, Any]],
    seen_hashes: set[str],
    known_llps_ids: set[str],
) -> None:
    if not header or "mol:protein" not in header.lower():
        return
    seq = strict.strict_sequence("".join(chunks))
    if seq is None or not 50 <= len(seq) <= 2048:
        return
    if len(set(seq)) < 8:
        return
    seq_hash = strict.sequence_hash(seq)
    if seq_hash in seen_hashes:
        return
    source_id = header[1:].split()[0]
    protein_id = "PDB_" + re.sub(r"[^A-Za-z0-9_]", "_", source_id.upper())
    if protein_id in known_llps_ids:
        return
    seen_hashes.add(seq_hash)
    candidates.append(
        {
            "source_id": source_id,
            "protein_id": protein_id,
            "description": header[1:500],
            "sequence": seq,
            "length": len(seq),
            "sequence_hash": seq_hash,
        }
    )


def make_region_rows_for_canonical(regions: list[strict.Region]) -> list[dict[str, Any]]:
    rows = []
    for index, region in enumerate(sorted(regions, key=lambda r: (r.protein_id, r.start, r.end, r.region_type)), start=1):
        region_type = canonical_region_type(region)
        rows.append(
            {
                "region_id": f"{region.protein_id}:{region.start + 1}-{region.end}",
                "protein_id": region.protein_id,
                "start": region.start,
                "end": region.end,
                "start_1based": region.start + 1,
                "end_1based": region.end,
                "coordinate_system": "1-based inclusive",
                "region_type": region_type,
                "region_label": "" if region.region_label is None else region.region_label,
                "label_tier": region.label_tier,
                "confidence": region.confidence,
                "sample_weight": region.sample_weight,
                "source": region.source,
                "assay": "database",
                "pseudo_round": region.pseudo_round,
                "notes": region.notes[:1000],
            }
        )
    return rows


def canonical_region_type(region: strict.Region) -> str:
    if region.region_label == 1 and region.label_tier == "gold":
        return "DPR_gold"
    if region.region_label == 1:
        return "DPR_curated"
    if region.region_type in {"IDR", "LCR", "PrLD"}:
        return "DPR_candidate"
    return str(region.region_type or "DPR_candidate")


def make_region_spans_jsonl(
    proteins: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
    regions: list[strict.Region],
) -> list[dict[str, Any]]:
    by_pid: dict[str, list[strict.Region]] = {}
    for region in regions:
        if region.region_label == 1 and region.label_tier in {"gold", "curated"}:
            by_pid.setdefault(region.protein_id, []).append(region)
    rows: list[dict[str, Any]] = []
    for protein in proteins:
        pid = protein["protein_id"]
        spans = []
        for region in sorted(by_pid.get(pid, []), key=lambda item: (item.start, item.end, item.source)):
            spans.append(
                {
                    "start": int(region.start),
                    "end": int(max(region.start, region.end - 1)),
                    "label_tier": region.label_tier,
                    "confidence": float(region.confidence),
                    "sample_weight": float(region.sample_weight),
                    "source": region.source,
                }
            )
        if not spans:
            continue
        label = labels_by_id[pid]
        rows.append(
            {
                "protein_id": pid,
                "sequence_length": int(protein["length"]),
                "dpr_spans": spans,
                "label_tier": _best_region_tier(spans),
                "sample_weight": max(float(span["sample_weight"]) for span in spans),
                "outside_is_negative": False,
                "negative_spans": [],
                "source": ";".join(sorted({str(span["source"]) for span in spans if span.get("source")})),
                "split": protein.get("split", ""),
                "llps_label": label.get("llps_label", ""),
            }
        )
    return rows


def make_mil_bags(proteins: list[dict[str, Any]], labels_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protein in proteins:
        pid = protein["protein_id"]
        label = labels_by_id[pid]
        bag = _bag_from_label(label)
        if bag is None:
            continue
        rows.append(
            {
                "protein_id": pid,
                "bag_label": bag["bag_label"],
                "bag_weight": bag["bag_weight"],
                "bag_type": bag["bag_type"],
                "feature_path": f"features/{pid}.npz",
                "split": protein.get("split", ""),
                "cluster_id_30": protein.get("cluster_id_30", ""),
                "cluster_id_50": protein.get("cluster_id_50", ""),
            }
        )
    return rows


def make_split_rows(
    proteins: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
    split_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for protein in proteins:
        pid = protein["protein_id"]
        label = labels_by_id[pid]
        rows.append(
            {
                "protein_id": pid,
                "split": split_by_id[pid],
                "cluster_id_30": protein.get("cluster_id_30", ""),
                "cluster_id_50": protein.get("cluster_id_50", ""),
                "label_tier": label.get("label_tier", ""),
                "llps_label": label.get("llps_label", ""),
                "role_type": label.get("role_type", ""),
                "negative_type": label.get("negative_type", ""),
            }
        )
    return rows


def write_candidate_priors_h5(path: Path, regions: list[strict.Region], proteins_by_id: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_pid: dict[str, list[strict.Region]] = {}
    for region in regions:
        if region.region_label is None or region.label_tier == "candidate" or region.region_type in {"IDR", "LCR", "PrLD"}:
            if region.protein_id in proteins_by_id:
                by_pid.setdefault(region.protein_id, []).append(region)
    with h5py.File(path, "w") as handle:
        handle.attrs["description"] = "Candidate priors only; not DPR-positive labels."
        for pid, items in sorted(by_pid.items()):
            spans = []
            scores = []
            types = []
            for region in sorted(items, key=lambda item: (item.start, item.end, item.region_type)):
                start = int(max(0, region.start))
                end = int(min(proteins_by_id[pid]["length"] - 1, max(region.start, region.end - 1)))
                if end < start:
                    continue
                spans.append([start, end])
                scores.append(float(np.clip(region.confidence, 0.0, 1.0)))
                types.append(str(region.region_type or "candidate_prior"))
            group = handle.create_group(pid)
            group.create_dataset("spans", data=np.asarray(spans, dtype=np.int64), compression="gzip")
            group.create_dataset("scores", data=np.asarray(scores, dtype=np.float32), compression="gzip")
            group.attrs["types_json"] = json.dumps(types)


def _best_region_tier(spans: list[dict[str, Any]]) -> str:
    if any(span.get("label_tier") == "gold" for span in spans):
        return "gold"
    if any(span.get("label_tier") == "curated" for span in spans):
        return "curated"
    return str(spans[0].get("label_tier", "unknown")) if spans else "unknown"


def _bag_from_label(label: dict[str, Any]) -> dict[str, Any] | None:
    role = str(label.get("role_type", "")).lower()
    tier = str(label.get("label_tier", "")).lower()
    negative_type = str(label.get("negative_type", "")).lower()
    llps_label = label.get("llps_label")
    weight = float(label.get("sample_weight", 0.0) or 0.0)
    if llps_label == 1 and any(token in role for token in ("driver", "scaffold", "self")):
        return {
            "bag_label": 1,
            "bag_weight": weight if weight > 0 else 0.75,
            "bag_type": "protein_positive_driver",
        }
    if llps_label == 0 and (tier == "negative_curated" or "negative" in role or "negative" in negative_type):
        if "disordered" in role or "disordered" in negative_type:
            bag_type = "negative_disordered"
        else:
            bag_type = "negative_structured"
        return {"bag_label": 0, "bag_weight": weight if weight > 0 else 0.75, "bag_type": bag_type}
    return None


def make_protein_master_rows(proteins: list[dict[str, Any]], labels_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for protein in proteins:
        label = labels_by_id[protein["protein_id"]]
        rows.append(
            {
                "protein_id": protein["protein_id"],
                "sequence": protein["sequence"],
                "sequence_md5": hashlib.md5(protein["sequence"].encode()).hexdigest(),
                "sequence_hash": protein["sequence_hash"],
                "species": protein.get("species", ""),
                "tax_id": protein.get("tax_id", ""),
                "length": protein["length"],
                "source_list": str(protein.get("source_list", "")).split(";") if protein.get("source_list") else [],
                "role_hint": label.get("role_type", "unknown"),
                "llps_label": label.get("llps_label", ""),
                "soft_label": label.get("soft_label", ""),
                "label_quality": label.get("label_tier", ""),
                "evidence_level": label.get("label_tier", ""),
                "negative_type": label.get("negative_type", "none"),
                "cluster_id_30": protein.get("cluster_id_30", ""),
                "cluster_id_50": protein.get("cluster_id_50", ""),
                "split": protein.get("split", ""),
                "sample_weight": label.get("sample_weight", 0.0),
                "label_mask": label.get("label_mask", 0),
            }
        )
    return rows


def make_region_master_rows(region_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in region_rows:
        rows.append(
            {
                "protein_id": row["protein_id"],
                "region_id": row["region_id"],
                "start": row["start_1based"],
                "end": row["end_1based"],
                "tensor_start": row["start"],
                "tensor_end": row["end"],
                "coordinate_system": "1-based inclusive",
                "region_type": row["region_type"],
                "soft_label": row["region_label"],
                "confidence": row["confidence"],
                "source": row["source"],
                "assay": row["assay"],
                "notes": row["notes"],
            }
        )
    return rows


def make_evidence_table(evidence: list[dict[str, Any]], region_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(evidence)
    for row in region_rows:
        rows.append(
            {
                "evidence_id": f"{row['region_id']}:region",
                "protein_id": row["protein_id"],
                "region_id": row["region_id"],
                "label_scope": "region",
                "source": row["source"],
                "pubmed_id": "",
                "doi": "",
                "assay": row["assay"],
                "in_vitro": "",
                "in_vivo": "",
                "condition": "",
                "confidence": row["confidence"],
                "notes": row["notes"],
            }
        )
    return rows


def make_manifest_rows(
    proteins: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
    split_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for protein in proteins:
        pid = protein["protein_id"]
        label = labels_by_id[pid]
        llps_label = label.get("llps_label", "")
        if llps_label == "":
            llps_label = -100
        rows.append(
            {
                "protein_id": pid,
                "sequence": protein["sequence"],
                "length": protein["length"],
                "llps_label": llps_label,
                "sample_weight": label.get("sample_weight", 0.0),
                "label_confidence": label.get("confidence", 0.0),
                "label_quality": label.get("label_tier", ""),
                "evidence_level": label.get("label_tier", ""),
                "negative_type": label.get("negative_type", "none"),
                "role_label": label.get("role_type", "unknown"),
                "source": protein.get("source_list", ""),
                "split": split_by_id[pid],
                "cluster_id_30": protein.get("cluster_id_30", ""),
                "cluster_id_50": protein.get("cluster_id_50", ""),
            }
        )
    return rows


def make_source_map_rows(source_map: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup = {}
    for row in source_map:
        key = (row.get("source", ""), row.get("source_id", ""), row.get("protein_id", ""))
        dedup[key] = row
    return [dedup[key] for key in sorted(dedup)]


def write_splits(
    splits_root: Path,
    proteins: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
    split_by_id: dict[str, str],
    cluster30: dict[str, str],
    cluster50: dict[str, str],
) -> None:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test_internal": [], "benchmark_holdout": []}
    proteins_by_id = {row["protein_id"]: row for row in proteins}
    for pid, split in sorted(split_by_id.items()):
        protein = proteins_by_id[pid]
        label = labels_by_id[pid]
        row = {
            "protein_id": pid,
            "sequence_md5": hashlib.md5(protein["sequence"].encode()).hexdigest(),
            "sequence_hash": protein["sequence_hash"],
            "cluster30": cluster30.get(pid, ""),
            "cluster50": cluster50.get(pid, ""),
            "split": split,
            "label_tier": label.get("label_tier", ""),
            "source": protein.get("source_list", ""),
        }
        rows_by_split.setdefault(split, []).append(row)
    write_csv(splits_root / "split_cluster30_train.csv", rows_by_split.get("train", []))
    write_csv(splits_root / "split_cluster30_val.csv", rows_by_split.get("valid", []))
    frozen = rows_by_split.get("test_internal", []) + rows_by_split.get("benchmark_holdout", [])
    write_csv(splits_root / "split_cluster30_test_frozen.csv", frozen)
    write_csv(splits_root / "split_cluster50_stress.csv", [row for rows in rows_by_split.values() for row in rows])
    aliases = {"train": "train_ids.txt", "valid": "valid_ids.txt"}
    for split, filename in aliases.items():
        (splits_root / filename).write_text("".join(f"{row['protein_id']}\n" for row in rows_by_split.get(split, [])))
    (splits_root / "test_ids.txt").write_text("".join(f"{row['protein_id']}\n" for row in frozen))


def leakage_audit_rows(
    proteins: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
    split_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    by_cluster: dict[str, set[str]] = {}
    region_source_by_id: dict[str, str] = {}
    for row in proteins:
        by_cluster.setdefault(row["cluster_id_30"], set()).add(split_by_id[row["protein_id"]])
    rows = []
    for row in proteins:
        pid = row["protein_id"]
        split = split_by_id[pid]
        cluster_splits = by_cluster[row["cluster_id_30"]]
        in_teacher_input = int(split in {"train", "valid"} and labels_by_id[pid].get("label_tier") not in {"gold", "negative_curated"})
        rows.append(
            {
                "protein_id": pid,
                "sequence_md5": hashlib.md5(row["sequence"].encode()).hexdigest(),
                "cluster30": row["cluster_id_30"],
                "cluster50": row["cluster_id_50"],
                "split": split,
                "source": row.get("source_list", ""),
                "in_external_teacher_training_set": 0,
                "homolog_in_train": int("train" in cluster_splits and split != "train"),
                "homolog_in_pseudo_pool": int(in_teacher_input),
                "region_label_source": region_source_by_id.get(pid, ""),
                "leakage_status": "cluster_split_violation" if len(cluster_splits) > 1 else "ok",
            }
        )
    return rows


def duplicate_report_rows(proteins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[str]] = {}
    for protein in proteins:
        by_hash.setdefault(str(protein["sequence_hash"]), []).append(str(protein["protein_id"]))
    rows = []
    for sequence_hash, ids in sorted(by_hash.items()):
        if len(ids) > 1:
            rows.append({"sequence_hash": sequence_hash, "protein_count": len(ids), "protein_ids": ";".join(sorted(ids))})
    return rows


def teacher_coverage_report_rows(path: Path, proteins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(proteins)
    if not path.exists():
        return [{"teacher_scores_h5": str(path), "proteins": total, "covered": 0, "coverage": 0.0, "status": "missing"}]
    with h5py.File(path, "r") as handle:
        covered = len(handle.keys())
    return [
        {
            "teacher_scores_h5": str(path),
            "proteins": total,
            "covered": covered,
            "coverage": covered / total if total else 0.0,
            "status": "ready" if covered else "empty",
        }
    ]


def region_gold_coverage_report_rows(
    proteins: list[dict[str, Any]],
    region_span_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_split: dict[str, dict[str, int]] = {}
    for protein in proteins:
        split = str(protein.get("split", ""))
        by_split.setdefault(split, {"proteins": 0, "region_proteins": 0, "spans": 0})
        by_split[split]["proteins"] += 1
    for row in region_span_rows:
        split = str(row.get("split", ""))
        by_split.setdefault(split, {"proteins": 0, "region_proteins": 0, "spans": 0})
        by_split[split]["region_proteins"] += 1
        by_split[split]["spans"] += len(row.get("dpr_spans", []))
    return [
        {
            "split": split,
            "proteins": counts["proteins"],
            "region_proteins": counts["region_proteins"],
            "spans": counts["spans"],
            "region_protein_coverage": counts["region_proteins"] / counts["proteins"] if counts["proteins"] else 0.0,
        }
        for split, counts in sorted(by_split.items())
    ]


def write_pseudo_inputs(
    pseudo_dir: Path,
    proteins: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
    split_by_id: dict[str, str],
    cluster30: dict[str, str],
    cluster50: dict[str, str],
) -> None:
    allowed = []
    for protein in proteins:
        pid = protein["protein_id"]
        label = labels_by_id[pid]
        if split_by_id[pid] not in {"train", "valid"}:
            continue
        if label["label_tier"] in {"gold", "negative_curated"}:
            continue
        allowed.append((pid, protein["sequence"]))
    strict.write_fasta(pseudo_dir / "round0_teacher_input.fasta", allowed)
    write_dataframe_table(
        pd.DataFrame(
        columns=[
            "protein_id",
            "psphunter",
            "pspredictor",
            "deephase",
            "fuzdrop",
            "pspire",
            "phasepred",
            "picnic",
            "catgranule2",
            "teacher_mean",
            "teacher_weighted",
            "teacher_std",
            "teacher_agreement",
            "teacher_label",
            "teacher_confidence",
            "split",
            "cluster_id_30",
            "cluster_id_50",
        ]
        ),
        pseudo_dir / "protein_teacher_scores.parquet",
    )
    with h5py.File(pseudo_dir / "residue_teacher_profiles.h5", "w") as handle:
        handle.attrs["status"] = "not_run"
        handle.attrs["records_pending"] = len(allowed)
    with h5py.File(pseudo_dir / "teacher_scores.h5", "w") as handle:
        handle.attrs["status"] = "not_run"
        handle.attrs["records_pending"] = len(allowed)
        handle.attrs["policy"] = "soft_distillation_only_not_model_input"
    (pseudo_dir / "region_teacher_candidates.jsonl").write_text("")
    status = {
        "status": "not_run",
        "teacher_input_records": len(allowed),
        "policy": "cluster_split_locked_before_teacher_inference",
        "excluded_splits": ["test_internal", "benchmark_holdout"],
        "excluded_label_tiers": ["gold", "negative_curated"],
        "note": "Canonical teacher input prepared; actual teacher inference is a separate heavy/tool-dependent step.",
    }
    (pseudo_dir / "teacher_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def write_round1_placeholders(round1_dir: Path) -> None:
    write_dataframe_table(
        pd.DataFrame(
        columns=[
            "protein_id",
            "phaseflow_llps_score",
            "external_teacher_weighted",
            "teacher_support",
            "self_training_label",
            "self_training_confidence",
            "split",
            "cluster_id_30",
            "cluster_id_50",
        ]
        ),
        round1_dir / "protein_self_scores.parquet",
    )
    (round1_dir / "region_self_candidates.jsonl").write_text("")
    status = {
        "status": "not_run",
        "reason": "Round-1 self-training requires a trained Round-0 PhaseFlow checkpoint.",
    }
    (round1_dir / "self_training_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_dataframe_table(frame: pd.DataFrame, parquet_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(parquet_path.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(parquet_path, index=False)
    except ImportError:
        return


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def value_counts(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(column, ""))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def compact_source_reports(source_reports: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for name, report in source_reports.items():
        if not isinstance(report, dict):
            compact[name] = report
            continue
        item = {key: value for key, value in report.items() if key != "rows" and key != "selected_rows"}
        if "rows" in report:
            item["cleaned_rows_written"] = len(report.get("rows") or [])
        if "selected_rows" in report:
            item["selected_rows_written"] = len(report.get("selected_rows") or [])
        compact[name] = item
    return compact


if __name__ == "__main__":
    main()
