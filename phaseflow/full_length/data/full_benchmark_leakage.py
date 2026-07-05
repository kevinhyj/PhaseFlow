from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd


FULL_PPMC_BENCHMARK_SPLITS = {"train", "valid", "test_internal", "benchmark_holdout"}
LEGACY_PPMC_HELDOUT_SPLITS = {"valid", "test_internal", "benchmark_holdout"}
PPMC_RAW_REL = Path("data/benchmarks/protein_benchmark_ppmc/ppmc_ce_de_c_d_np_nd_raw.tsv")
PPMC_MANIFEST_REL = Path("data/benchmarks/protein_benchmark_ppmc/manifest.csv")
PPMC_SOURCE_RECORDS_REL = Path("data/benchmarks/protein_benchmark_ppmc/source_records.csv")
PHASEPRO_PROTEINS_REL = Path("data/benchmarks/dpr_benchmark_phasepro/proteins.csv")
PHASEPRO_SOURCE_RECORDS_REL = Path("data/benchmarks/dpr_benchmark_phasepro/source_records.csv")
MMSEQS40_CLUSTER_REL = Path("data/interim/server_final/mmseqs40_leakage_20260606_cluster.tsv")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def norm_accession(value: Any) -> str:
    text = norm_text(value).upper()
    if not text:
        return ""
    return text.split(".", 1)[0].split("-", 1)[0]


def clean_sequence(value: Any) -> str:
    return "".join(ch for ch in norm_text(value).upper() if ch.isalpha())


def sequence_md5(value: Any) -> str:
    sequence = clean_sequence(value)
    return hashlib.md5(sequence.encode("utf-8")).hexdigest() if sequence else ""


def sequence_sha256(value: Any) -> str:
    sequence = clean_sequence(value)
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest() if sequence else ""


def empty_key_sets() -> dict[str, set[str]]:
    return {"ids": set(), "sha256": set(), "md5": set()}


def merge_key_sets(*items: dict[str, set[str]]) -> dict[str, set[str]]:
    out = empty_key_sets()
    for item in items:
        for key in out:
            out[key].update(item.get(key, set()))
    return out


def full_ppmc_key_sets(root: Path | None = None) -> dict[str, set[str]]:
    root = (root or project_root()).resolve()
    keys = empty_key_sets()
    raw = root / PPMC_RAW_REL
    if raw.exists():
        frame = pd.read_csv(raw, sep="\t", dtype=str, keep_default_na=False)
        _collect_keys(frame, keys)
        if "mapped_protein_ids" in frame.columns:
            for value in frame["mapped_protein_ids"].astype(str):
                for part in value.split(";"):
                    acc = norm_accession(part)
                    if acc:
                        keys["ids"].add(acc)

    for rel in [PPMC_MANIFEST_REL, PPMC_SOURCE_RECORDS_REL]:
        path = root / rel
        if path.exists():
            _collect_keys(pd.read_csv(path, dtype=str, keep_default_na=False), keys)
    return _strip_empty(keys)


def ppmc_legacy_heldout_key_sets(root: Path | None = None) -> dict[str, set[str]]:
    root = (root or project_root()).resolve()
    path = root / PPMC_MANIFEST_REL
    keys = empty_key_sets()
    if path.exists():
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "split" in frame.columns:
            frame = frame[frame["split"].astype(str).isin(LEGACY_PPMC_HELDOUT_SPLITS)].copy()
        _collect_keys(frame, keys)
    return _strip_empty(keys)


def ppmc_final_eval_key_sets(root: Path | None = None) -> dict[str, set[str]]:
    return ppmc_legacy_heldout_key_sets(root)


def phasepro_key_sets(root: Path | None = None) -> dict[str, set[str]]:
    root = (root or project_root()).resolve()
    keys = empty_key_sets()
    for rel in [PHASEPRO_PROTEINS_REL, PHASEPRO_SOURCE_RECORDS_REL]:
        path = root / rel
        if path.exists():
            _collect_keys(pd.read_csv(path, dtype=str, keep_default_na=False), keys)
    return _strip_empty(keys)


def full_benchmark_key_sets(root: Path | None = None) -> dict[str, set[str]]:
    return merge_key_sets(full_ppmc_key_sets(root), phasepro_key_sets(root))


def overlap_flags(
    frame: pd.DataFrame,
    keys: dict[str, set[str]],
    *,
    prefix: str,
    cluster_path: Path | None = None,
) -> pd.DataFrame:
    row_ids = _row_id_sets(frame)
    row_sha256 = _row_hash_sets(frame, hash_kind="sha256")
    row_md5 = _row_hash_sets(frame, hash_kind="md5")
    key_ids = {norm_accession(value) for value in keys.get("ids", set()) if norm_accession(value)}
    key_sha256 = {norm_text(value).lower() for value in keys.get("sha256", set()) if norm_text(value)}
    key_md5 = {norm_text(value).lower() for value in keys.get("md5", set()) if norm_text(value)}

    direct = pd.Series([bool(ids & key_ids) for ids in row_ids], index=frame.index)
    hash_overlap = pd.Series(
        [bool(sha & key_sha256) or bool(md5 & key_md5) for sha, md5 in zip(row_sha256, row_md5, strict=False)],
        index=frame.index,
    )
    homolog = pd.Series(False, index=frame.index)
    if cluster_path and cluster_path.exists() and key_ids:
        cluster_map = read_cluster_map(cluster_path)
        benchmark_reps = {cluster_map.get(key, key) for key in key_ids}
        homolog = pd.Series(
            [any(cluster_map.get(row_id, row_id) in benchmark_reps for row_id in ids) for ids in row_ids],
            index=frame.index,
        )

    out = pd.DataFrame(index=frame.index)
    out[f"{prefix}_direct_overlap"] = direct.astype(bool)
    out[f"{prefix}_hash_overlap"] = hash_overlap.astype(bool)
    out[f"{prefix}_homolog_overlap"] = homolog.astype(bool)
    blocker = direct | hash_overlap
    out[f"{prefix}_benchmark_overlap"] = blocker.astype(bool)
    return out


def assert_no_full_benchmark_leakage(
    sample_index: str | Path,
    *,
    root: Path | None = None,
    report_dir: str | Path | None = None,
    context: str = "training",
) -> None:
    path = Path(sample_index)
    if not path.is_absolute():
        path = (root or project_root()) / path
    if not path.exists():
        return
    frame = _read_table(path)
    keys = full_benchmark_key_sets(root)
    cluster = (root or project_root()) / MMSEQS40_CLUSTER_REL
    flags = overlap_flags(frame, keys, prefix="full_benchmark", cluster_path=cluster)
    mask = flags["full_benchmark_benchmark_overlap"].astype(bool)
    if not bool(mask.any()):
        return

    sample_cols = [col for col in ["protein_id", "uniprot_id", "accession", "sequence_sha256", "sequence_hash", "sequence_md5", "source_dataset", "source"] if col in frame.columns]
    sample = pd.concat([frame.loc[mask, sample_cols].reset_index(drop=True), flags.loc[mask].reset_index(drop=True)], axis=1)
    report_path = None
    if report_dir is not None:
        out_dir = Path(report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"full_benchmark_leakage_blocker_{path.stem}.csv"
        sample.to_csv(report_path, index=False)
    detail = f"; report={report_path}" if report_path else ""
    raise RuntimeError(
        f"Full PPMC/PhasePro exact-duplicate leakage guard failed for {context} sample_index={path}: "
        f"overlap_rows={int(mask.sum())}, unique_proteins={int(frame.loc[mask, 'protein_id'].nunique()) if 'protein_id' in frame.columns else 'unknown'}"
        f"{detail}"
    )


def read_cluster_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep = norm_accession(parts[0])
            member = norm_accession(parts[1])
            if rep and member:
                mapping[member] = rep
    return mapping


def _collect_keys(frame: pd.DataFrame, keys: dict[str, set[str]]) -> None:
    for col in ["protein_id", "uniprot_id", "accession", "source_id", "source_record_id", "UniProt.Acc", "uniprot_accession_norm"]:
        if col in frame.columns:
            keys["ids"].update(acc for acc in frame[col].map(norm_accession).tolist() if acc)
    for col in ["sequence_hash", "sequence_sha256", "seq_hash"]:
        if col in frame.columns:
            keys["sha256"].update(norm_text(value).lower() for value in frame[col].tolist() if norm_text(value))
    for col in ["sequence_md5"]:
        if col in frame.columns:
            keys["md5"].update(norm_text(value).lower() for value in frame[col].tolist() if norm_text(value))
    for col in ["sequence", "Full.seq"]:
        if col in frame.columns:
            keys["sha256"].update(value for value in frame[col].map(sequence_sha256).tolist() if value)
            keys["md5"].update(value for value in frame[col].map(sequence_md5).tolist() if value)


def _row_id_sets(frame: pd.DataFrame) -> list[set[str]]:
    cols = [col for col in ["protein_id", "uniprot_id", "accession", "source_id"] if col in frame.columns]
    rows: list[set[str]] = []
    if cols:
        for _, row in frame[cols].iterrows():
            rows.append({acc for acc in (norm_accession(row[col]) for col in cols) if acc})
    else:
        rows = [set() for _ in range(len(frame))]
    return rows


def _row_hash_sets(frame: pd.DataFrame, *, hash_kind: str) -> list[set[str]]:
    rows = [set() for _ in range(len(frame))]
    if hash_kind == "sha256":
        cols = [col for col in ["sequence_sha256", "sequence_hash", "seq_hash"] if col in frame.columns]
        for col in cols:
            for idx, value in enumerate(frame[col].tolist()):
                text = norm_text(value).lower()
                if text:
                    rows[idx].add(text)
        if "sequence" in frame.columns:
            for idx, value in enumerate(frame["sequence"].map(sequence_sha256).tolist()):
                if value:
                    rows[idx].add(value)
    elif hash_kind == "md5":
        if "sequence_md5" in frame.columns:
            for idx, value in enumerate(frame["sequence_md5"].tolist()):
                text = norm_text(value).lower()
                if text:
                    rows[idx].add(text)
        if "sequence" in frame.columns:
            for idx, value in enumerate(frame["sequence"].map(sequence_md5).tolist()):
                if value:
                    rows[idx].add(value)
    else:
        raise ValueError(f"Unsupported hash kind: {hash_kind}")
    return rows


def _strip_empty(keys: dict[str, set[str]]) -> dict[str, set[str]]:
    return {key: {value for value in values if value} for key, values in keys.items()}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)
