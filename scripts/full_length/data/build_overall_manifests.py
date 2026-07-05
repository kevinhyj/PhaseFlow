#!/usr/bin/env python3
"""Build unified PhaseFlow manifests under artifacts/data/overall.

The script performs the strict benchmark exclusion audit first, then writes the
pre-offline train_all protein/window manifests. Offline shard ids are left as
placeholders here and are filled by the later shard-packing step.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = ROOT / "data/processed/stage2/dpr_v1"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmarks/dpr_benchmark_phasepro"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/data/overall"

SAMPLE_INDEXES = {
    "residue_supervised": "sample_indexes/residue_supervised_index.parquet",
    "bag_positive": "sample_indexes/bag_positive_index.parquet",
    "negative": "sample_indexes/negative_index.parquet",
    "unlabeled": "sample_indexes/unlabeled_index.parquet",
}

TIER_PRIORITY = {
    "region_gold_high": 0,
    "region_weak": 1,
    "bag_positive": 2,
    "disordered_negative": 3,
    "structured_negative": 4,
    "unlabeled": 5,
}

BUCKETS = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--mmseqs30-cluster",
        type=Path,
        default=ROOT / "data/interim/server_final/mmseqs30_cluster.tsv",
    )
    parser.add_argument(
        "--old-homology-hits",
        type=Path,
        default=ROOT / "external_artifacts/stage2/dpr_stack_v1/audit/benchmark_homology_hits_stage2.csv",
    )
    parser.add_argument(
        "--region-span-jsonl",
        type=Path,
        default=ROOT / "data/processed/model_region_spans.jsonl",
    )
    return parser.parse_args()


def str_set(values: Iterable[object]) -> set[str]:
    out: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            out.add(text)
    return out


def extract_pmids(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {item for item in re.findall(r"\b\d{6,9}\b", str(value))}


def length_bucket(length: int) -> int:
    for bucket in BUCKETS:
        if int(length) <= bucket:
            return bucket
    return BUCKETS[-1]


def read_benchmark_sets(benchmark_root: Path) -> tuple[dict[str, set[str]], dict[str, pd.DataFrame]]:
    proteins = pd.read_csv(benchmark_root / "proteins.csv")
    regions = pd.read_csv(benchmark_root / "regions.csv")
    source_records = pd.read_csv(benchmark_root / "source_records.csv")
    source_map = pd.read_csv(benchmark_root / "source_map.csv")

    benchmark_ids = set()
    for df in (proteins, source_records, source_map):
        for col in ("protein_id", "uniprot_id", "isoform_id", "source_id"):
            if col in df.columns:
                benchmark_ids.update(str_set(df[col]))
    benchmark_hashes = set()
    for df in (proteins, source_records, source_map):
        for col in ("sequence_hash", "sequence_sha256"):
            if col in df.columns:
                benchmark_hashes.update(str_set(df[col]))
    benchmark_md5 = set()
    for df in (proteins,):
        for col in ("sequence_md5",):
            if col in df.columns:
                benchmark_md5.update(str_set(df[col]))

    benchmark_clusters = set()
    for col in ("cluster_id_30", "cluster_id_40", "cluster_id_50"):
        if col in proteins.columns:
            benchmark_clusters.update(str_set(proteins[col]))

    benchmark_pmids = set()
    for df in (regions, source_records):
        for col in ("notes", "pmid", "publication", "source_publication"):
            if col in df.columns:
                for value in df[col]:
                    benchmark_pmids.update(extract_pmids(value))

    sets = {
        "ids": benchmark_ids,
        "sequence_hashes": benchmark_hashes,
        "sequence_md5": benchmark_md5,
        "clusters": benchmark_clusters,
        "pmids": benchmark_pmids,
    }
    frames = {
        "proteins": proteins,
        "regions": regions,
        "source_records": source_records,
        "source_map": source_map,
    }
    return sets, frames


def read_sample_indexes(data_root: Path) -> pd.DataFrame:
    frames = []
    for pool, rel in SAMPLE_INDEXES.items():
        path = data_root / rel
        df = pd.read_parquet(path)
        df = df.copy()
        df["source_pool"] = pool
        df["source_index_path"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def read_mmseqs_cluster_hits(
    cluster_path: Path,
    benchmark_ids: set[str],
    candidate_ids: set[str],
) -> dict[str, set[str]]:
    hits: dict[str, set[str]] = defaultdict(set)
    if not cluster_path.exists():
        return hits
    rep_to_members: dict[str, set[str]] = defaultdict(set)
    member_to_reps: dict[str, set[str]] = defaultdict(set)
    with cluster_path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            rep_to_members[rep].add(member)
            member_to_reps[member].add(rep)
    benchmark_reps: set[str] = set()
    for bid in benchmark_ids:
        benchmark_reps.update(member_to_reps.get(bid, set()))
        if bid in rep_to_members:
            benchmark_reps.add(bid)
    for rep in benchmark_reps:
        for member in rep_to_members.get(rep, set()):
            if member in candidate_ids:
                hits[member].add(rep)
    return hits


def read_old_homology_hits(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "protein_id" not in df.columns:
        return pd.DataFrame()
    return df


def read_region_publication_hits(region_span_jsonl: Path, benchmark_pmids: set[str]) -> dict[str, set[str]]:
    hits: dict[str, set[str]] = defaultdict(set)
    if not region_span_jsonl.exists() or not benchmark_pmids:
        return hits
    with region_span_jsonl.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            protein_id = str(row.get("protein_id", "")).strip()
            if not protein_id:
                continue
            pmids = extract_pmids(row.get("pmid", "")) | extract_pmids(row.get("notes", ""))
            overlap = pmids & benchmark_pmids
            if overlap:
                hits[protein_id].update(overlap)
    return hits


def build_exclusions(
    sample_df: pd.DataFrame,
    benchmark_sets: dict[str, set[str]],
    *,
    mmseqs_hits: dict[str, set[str]],
    old_homology_hits: pd.DataFrame,
    publication_hits: dict[str, set[str]],
) -> pd.DataFrame:
    evidence: dict[str, dict[str, object]] = {}

    def add(protein_id: str, reason: str, detail: str = "") -> None:
        if not protein_id:
            return
        rec = evidence.setdefault(protein_id, {"protein_id": protein_id, "reasons": set(), "details": []})
        rec["reasons"].add(reason)
        if detail:
            rec["details"].append(detail)

    ids = benchmark_sets["ids"]
    hashes = benchmark_sets["sequence_hashes"]
    clusters = benchmark_sets["clusters"]

    for row in sample_df.itertuples(index=False):
        protein_id = str(getattr(row, "protein_id"))
        canonical = str(getattr(row, "canonical_accession", "") or "")
        seq_hash = str(getattr(row, "sequence_hash", "") or "")
        seq_sha256 = str(getattr(row, "sequence_sha256", "") or "")
        cluster_id = str(getattr(row, "cluster_id", "") or "")
        if protein_id in ids or canonical in ids:
            add(protein_id, "benchmark_id_overlap", protein_id if protein_id in ids else canonical)
        if seq_hash in hashes or seq_sha256 in hashes:
            add(protein_id, "benchmark_sequence_hash_overlap", seq_hash or seq_sha256)
        if bool(getattr(row, "phasepro_exact_overlap", False)):
            add(protein_id, "phasepro_exact_overlap_flag", "source_table")
        if bool(getattr(row, "ppmc_exact_overlap", False)):
            add(protein_id, "ppmc_exact_overlap_flag", "source_table")
        if protein_id in clusters or canonical in clusters or cluster_id in clusters:
            add(protein_id, "benchmark_cluster_id_overlap", cluster_id or protein_id)

    for protein_id, reps in mmseqs_hits.items():
        add(protein_id, "mmseqs30_cluster_overlap", ",".join(sorted(reps))[:512])

    if not old_homology_hits.empty:
        for protein_id, sub in old_homology_hits.groupby(old_homology_hits["protein_id"].astype(str)):
            max_ident = float(pd.to_numeric(sub.get("pident"), errors="coerce").max())
            benchmarks = ",".join(sorted(str_set(sub.get("benchmark_id", []))))[:512]
            add(protein_id, "previous_homology_hit", f"max_pident={max_ident:.1f};benchmarks={benchmarks}")

    for protein_id, pmids in publication_hits.items():
        add(protein_id, "region_source_publication_overlap", ",".join(sorted(pmids)))

    if not evidence:
        return pd.DataFrame(
            columns=[
                "protein_id",
                "sequence_hash",
                "original_split",
                "source_pools",
                "exclude_reasons",
                "exclude_details",
                "benchmark_excluded",
            ]
        )

    rows = []
    sample_small = sample_df.drop_duplicates("protein_id").set_index("protein_id", drop=False)
    pools = sample_df.groupby("protein_id")["source_pool"].apply(lambda s: ";".join(sorted(set(map(str, s)))))
    for protein_id, rec in sorted(evidence.items()):
        if protein_id in sample_small.index:
            source = sample_small.loc[protein_id]
            sequence_hash = str(source.get("sequence_hash", "") or source.get("sequence_sha256", "") or "")
            original_split = str(source.get("split", ""))
        else:
            sequence_hash = ""
            original_split = ""
        rows.append(
            {
                "protein_id": protein_id,
                "sequence_hash": sequence_hash,
                "original_split": original_split,
                "source_pools": str(pools.get(protein_id, "")),
                "exclude_reasons": ";".join(sorted(rec["reasons"])),
                "exclude_details": " | ".join(map(str, rec["details"])),
                "benchmark_excluded": True,
            }
        )
    return pd.DataFrame(rows)


def span_summary(data_root: Path) -> pd.DataFrame:
    spans = pd.read_parquet(data_root / "manifests/canonical_span_table.parquet")
    if spans.empty:
        return pd.DataFrame(columns=["protein_id", "span_count", "span_tiers", "has_gold_high_span", "has_weak_span"])
    grouped = []
    for protein_id, sub in spans.groupby("protein_id"):
        tiers = sorted(str_set(sub["evidence_tier"]))
        grouped.append(
            {
                "protein_id": str(protein_id),
                "span_count": int(len(sub)),
                "span_tiers": ";".join(tiers),
                "has_gold_high_span": any(t in {"gold_causal", "high_curated"} for t in tiers),
                "has_weak_span": any(t == "weak_pseudo" for t in tiers),
            }
        )
    return pd.DataFrame(grouped)


def protein_tier(row: pd.Series) -> str:
    pool = str(row.get("source_pool", ""))
    negative_type = str(row.get("negative_type", ""))
    if bool(row.get("has_gold_high_span", False)):
        return "region_gold_high"
    if pool == "residue_supervised" or bool(row.get("has_weak_span", False)):
        return "region_weak"
    if pool == "bag_positive" or int(float(row.get("bag_label", row.get("llps_label", 0)) or 0)) == 1:
        return "bag_positive"
    if negative_type == "disordered":
        return "disordered_negative"
    if pool == "negative":
        return "structured_negative"
    return "unlabeled"


def build_proteins_manifest(sample_df: pd.DataFrame, excluded: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    spans = span_summary(data_root)
    df = sample_df.merge(spans, on="protein_id", how="left")
    for col in ("span_count", "has_gold_high_span", "has_weak_span"):
        if col not in df.columns:
            df[col] = 0 if col == "span_count" else False
    df["span_count"] = df["span_count"].fillna(0).astype(int)
    df["span_tiers"] = df.get("span_tiers", "").fillna("")
    df["has_gold_high_span"] = df["has_gold_high_span"].fillna(False).astype(bool)
    df["has_weak_span"] = df["has_weak_span"].fillna(False).astype(bool)
    df["tier"] = df.apply(protein_tier, axis=1)
    df["tier_priority"] = df["tier"].map(TIER_PRIORITY).fillna(99).astype(int)
    df["length"] = pd.to_numeric(df.get("seq_len"), errors="coerce").fillna(0).astype(int)
    df["length_bucket"] = df["length"].map(length_bucket)

    excluded_ids = set(excluded["protein_id"].astype(str)) if not excluded.empty else set()
    df["benchmark_excluded"] = df["protein_id"].astype(str).isin(excluded_ids)
    df = df.loc[~df["benchmark_excluded"]].copy()
    df["split"] = "train_all"

    df = df.sort_values(["protein_id", "tier_priority", "sample_weight"], ascending=[True, True, False])
    df = df.drop_duplicates("protein_id", keep="first").reset_index(drop=True)
    df["shard_id"] = -1
    df["row_index"] = -1
    df["offline_ready"] = False
    df["source_publication"] = ""

    columns = [
        "protein_id",
        "tier",
        "shard_id",
        "row_index",
        "length",
        "length_bucket",
        "cluster_id",
        "sample_weight",
        "bag_label",
        "has_region_label",
        "benchmark_excluded",
        "split",
        "sequence_hash",
        "sequence_sha256",
        "canonical_accession",
        "source_pool",
        "source_list",
        "source_publication",
        "span_count",
        "span_tiers",
        "has_gold_high_span",
        "has_weak_span",
        "region_label_path",
        "has_esm2",
        "has_biophys",
        "has_protenix",
        "has_starling",
        "has_merged_graph",
        "offline_ready",
    ]
    for col in columns:
        if col not in df.columns:
            if col.startswith("has_") or col == "benchmark_excluded" or col == "offline_ready":
                df[col] = False
            elif col in {"sample_weight", "bag_label"}:
                df[col] = 0.0
            else:
                df[col] = ""
    return df[columns]


def map_window_type(value: object) -> str:
    text = str(value)
    if text in {"positive_core", "left_boundary", "right_boundary"}:
        return text
    if text in {"same_protein_hard_negative", "full_span_context", "same_protein_context"}:
        return "same_protein_context"
    if text in {"negative_background", "external_negative"}:
        return "external_negative"
    return text


def build_windows_manifest(data_root: Path, proteins: pd.DataFrame) -> pd.DataFrame:
    windows = pd.read_parquet(data_root / "window_indexes/window_index.parquet")
    keep_ids = set(proteins["protein_id"].astype(str))
    windows = windows.loc[windows["protein_id"].astype(str).isin(keep_ids)].copy()
    lookup = proteins.set_index("protein_id")[["tier", "length_bucket"]]
    windows = windows.merge(lookup, left_on="protein_id", right_index=True, how="left", suffixes=("", "_protein"))
    windows["window_type_original"] = windows["window_type"].astype(str)
    windows["window_type"] = windows["window_type"].map(map_window_type)
    windows["split"] = "train_all"
    windows["shard_id"] = -1
    windows["row_index"] = -1
    columns = [
        "protein_id",
        "shard_id",
        "row_index",
        "window_start",
        "window_end",
        "window_type",
        "window_type_original",
        "tier",
        "sample_weight",
        "split",
        "length_bucket",
        "span_id",
        "boundary_type",
        "contains_positive",
        "positive_residue_count",
    ]
    for col in columns:
        if col not in windows.columns:
            windows[col] = ""
    return windows[columns].reset_index(drop=True)


def empty_shards_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "shard_id",
            "tier",
            "length_bucket",
            "path",
            "num_proteins",
            "num_windows",
            "bytes_total",
            "storage_mode",
            "offline_ready",
        ]
    )


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows_"
    view = df.head(max_rows).copy()
    headers = [str(c) for c in view.columns]

    def clean(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    benchmark_sets: dict[str, set[str]],
    excluded: pd.DataFrame,
    proteins: pd.DataFrame,
    windows: pd.DataFrame,
    old_homology_hits: pd.DataFrame,
    mmseqs_hits: dict[str, set[str]],
    publication_hits: dict[str, set[str]],
) -> None:
    reason_counts = (
        excluded["exclude_reasons"]
        .str.get_dummies(sep=";")
        .sum()
        .sort_values(ascending=False)
        .rename_axis("reason")
        .reset_index(name="excluded_proteins")
        if not excluded.empty
        else pd.DataFrame(columns=["reason", "excluded_proteins"])
    )
    tier_counts = proteins["tier"].value_counts().rename_axis("tier").reset_index(name="proteins")
    window_counts = windows["window_type"].value_counts().rename_axis("window_type").reset_index(name="windows")
    old_unique = int(old_homology_hits["protein_id"].nunique()) if not old_homology_hits.empty else 0
    lines = [
        "# Benchmark Exclusion Report",
        "",
        "This report was generated before offline feature materialization. Exclusion is applied before train/validation are merged into `train_all`.",
        "",
        "## Benchmark Evidence",
        "",
        f"- benchmark proteins/source ids: {len(benchmark_sets['ids'])}",
        f"- benchmark sequence hashes: {len(benchmark_sets['sequence_hashes'])}",
        f"- benchmark 30/40/50 cluster ids: {len(benchmark_sets['clusters'])}",
        f"- benchmark publication IDs parsed from region notes: {len(benchmark_sets['pmids'])}",
        f"- MMseqs30 cluster-overlap candidate proteins: {len(mmseqs_hits)}",
        f"- previous homology-hit candidate proteins: {old_unique}",
        f"- region source-publication overlap candidate proteins: {len(publication_hits)}",
        "",
        "## Exclusion Reasons",
        "",
        md_table(reason_counts),
        "",
        f"Total excluded training proteins: {len(excluded)}",
        "",
        "## Remaining Train-All Protein Tiers",
        "",
        md_table(tier_counts),
        "",
        "## Remaining Train-All Window Types",
        "",
        md_table(window_counts),
        "",
        "## Notes",
        "",
        "- Exact benchmark protein/sequence overlap is checked using IDs, sequence hashes, and existing exact-overlap flags.",
        "- 30% cluster exclusion uses `data/interim/server_final/mmseqs30_cluster.tsv` when benchmark IDs are present in that clustering.",
        "- Previous homology hits from `external_artifacts/stage2/dpr_stack_v1/audit/benchmark_homology_hits_stage2.csv` are treated as hard exclusion evidence.",
        "- Region source-publication overlap uses PMIDs parsed from benchmark region notes and `data/processed/model_region_spans.jsonl`.",
        "- Pseudo-label source and teacher-prediction source overlap do not have a benchmark-side teacher-source field in the current benchmark files; the audit records this limitation and does not fabricate exclusions from unavailable fields.",
        "- `proteins.parquet` and `windows.parquet` are pre-offline manifests. `shard_id` and `row_index` are placeholders until shard packing updates them.",
        "",
        "## Sample Exclusions",
        "",
        md_table(excluded[["protein_id", "source_pools", "exclude_reasons", "exclude_details"]], max_rows=30),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_root
    manifests_dir = out / "manifests"
    audit_dir = out / "audit"
    reports_dir = out / "reports"
    for directory in (manifests_dir, audit_dir, reports_dir, out / "offline", out / "logs", out / "checkpoints", out / "benchmark", out / "final"):
        directory.mkdir(parents=True, exist_ok=True)

    benchmark_sets, _ = read_benchmark_sets(args.benchmark_root)
    sample_df = read_sample_indexes(args.data_root)
    candidate_ids = set(sample_df["protein_id"].astype(str))
    mmseqs_hits = read_mmseqs_cluster_hits(args.mmseqs30_cluster, benchmark_sets["ids"], candidate_ids)
    old_homology_hits = read_old_homology_hits(args.old_homology_hits)
    publication_hits = read_region_publication_hits(args.region_span_jsonl, benchmark_sets["pmids"])
    publication_hits = {pid: pmids for pid, pmids in publication_hits.items() if pid in candidate_ids}

    excluded = build_exclusions(
        sample_df,
        benchmark_sets,
        mmseqs_hits=mmseqs_hits,
        old_homology_hits=old_homology_hits,
        publication_hits=publication_hits,
    )
    proteins = build_proteins_manifest(sample_df, excluded, args.data_root)
    windows = build_windows_manifest(args.data_root, proteins)
    shards = empty_shards_manifest()

    excluded.to_parquet(manifests_dir / "excluded_benchmark.parquet", index=False)
    excluded.to_csv(manifests_dir / "excluded_benchmark.csv", index=False)
    proteins.to_parquet(manifests_dir / "proteins.parquet", index=False)
    proteins.to_csv(manifests_dir / "proteins.csv", index=False)
    windows.to_parquet(manifests_dir / "windows.parquet", index=False)
    windows.to_csv(manifests_dir / "windows.csv", index=False)
    shards.to_parquet(manifests_dir / "shards.parquet", index=False)

    stats = {
        "excluded_proteins": int(len(excluded)),
        "remaining_proteins": int(len(proteins)),
        "remaining_windows": int(len(windows)),
        "tiers": proteins["tier"].value_counts().to_dict(),
        "window_types": windows["window_type"].value_counts().to_dict(),
        "mmseqs30_hits": int(len(mmseqs_hits)),
        "previous_homology_hit_proteins": int(old_homology_hits["protein_id"].nunique()) if not old_homology_hits.empty else 0,
        "publication_hit_proteins": int(len(publication_hits)),
    }
    (audit_dir / "benchmark_exclusion_summary.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    write_report(
        audit_dir / "benchmark_exclusion_report.md",
        benchmark_sets=benchmark_sets,
        excluded=excluded,
        proteins=proteins,
        windows=windows,
        old_homology_hits=old_homology_hits,
        mmseqs_hits=mmseqs_hits,
        publication_hits=publication_hits,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
