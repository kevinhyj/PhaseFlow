from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import gzip
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd

from phaseflow.full_length.features.disorder import compute_disorder_features
from phaseflow.full_length.features.plm_embedder import clean_protein_sequence
from phaseflow.full_length.utils import load_yaml, write_json


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def build_teacher_pseudo_labels(
    config: dict[str, Any],
    *,
    manifest: str | Path | None = None,
    regions: str | Path | None = None,
    train_ids_file: str | Path | None = None,
    out_dir: str | Path | None = None,
    run_predictors: bool = True,
    selected_predictors: set[str] | None = None,
) -> dict[str, Any]:
    paths = config.get("paths", {})
    manifest_path = Path(manifest or paths["manifest"])
    regions_path = Path(regions or paths["regions"])
    train_ids_path = Path(train_ids_file or paths["train_ids_file"])
    output_dir = Path(out_dir or paths.get("out_dir", "data/pseudo_labels/round0_external"))
    input_dir = output_dir / "inputs"
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_frame = pd.read_csv(manifest_path)
    train_ids = _read_id_file(train_ids_path)
    train_frame = manifest_frame.loc[manifest_frame["protein_id"].astype(str).isin(train_ids)].copy()
    train_frame["sequence"] = train_frame["sequence"].map(lambda value: clean_protein_sequence(str(value)))
    train_frame = train_frame.loc[train_frame["sequence"].map(lambda seq: bool(seq) and set(seq).issubset(VALID_AA))]

    train_fasta = input_dir / "teacher_train.fasta"
    idr_fasta = input_dir / "teacher_idr_segments.fasta"
    metadata_csv = input_dir / "teacher_metadata.csv"
    _write_fasta(train_fasta, train_frame[["protein_id", "sequence"]].itertuples(index=False, name=None))
    idr_segments = _write_idr_segment_fasta(idr_fasta, train_frame)
    _write_metadata(metadata_csv, train_frame)

    context = {
        "manifest": str(manifest_path),
        "regions": str(regions_path),
        "train_ids_file": str(train_ids_path),
        "out_dir": str(output_dir),
        "input_dir": str(input_dir),
        "raw_dir": str(raw_dir),
        "train_fasta": str(train_fasta),
        "idr_fasta": str(idr_fasta),
        "metadata_csv": str(metadata_csv),
    }

    predictor_status: list[dict[str, Any]] = []
    if run_predictors:
        predictor_status = _run_predictors(config.get("predictors", {}), context, selected_predictors)

    collected_rows = _collect_scores(config.get("predictors", {}), context, manifest_frame, selected_predictors)
    score_rows = [row for row in collected_rows if str(row["protein_id"]) in train_ids]
    protein_rows = [row for row in score_rows if row["scope"] == "protein"]
    region_rows = [row for row in score_rows if row["scope"] == "region"]
    profile_rows = [row for row in score_rows if row["scope"] == "profile"]

    consensus_config = config.get("consensus", {})
    pseudo_regions = _consensus_regions(region_rows, consensus_config)
    protein_consensus = _consensus_proteins(protein_rows, consensus_config)
    manifest_with_teacher = _write_manifest_with_teacher(manifest_frame, protein_consensus, output_dir / "manifest_with_teacher.csv")
    teacher_scores_h5 = _write_teacher_profiles_h5(
        output_dir / "teacher_scores.h5",
        train_frame,
        region_rows,
        profile_rows,
        consensus_config,
    )

    scores_csv = output_dir / "teacher_scores.csv"
    regions_csv = output_dir / "teacher_region_candidates.csv"
    regions_jsonl = output_dir / "teacher_region_candidates.jsonl"
    protein_csv = output_dir / "teacher_protein_labels.csv"
    evidence_csv = output_dir / "teacher_evidence.csv"
    _write_score_csv(scores_csv, score_rows)
    _write_region_csv(regions_csv, pseudo_regions)
    _write_region_candidates_jsonl(regions_jsonl, pseudo_regions)
    _write_protein_csv(protein_csv, protein_consensus)
    _write_evidence_csv(evidence_csv, protein_rows, region_rows)

    report = {
        "manifest": str(manifest_path),
        "train_records": int(len(train_frame)),
        "idr_segments": int(idr_segments),
        "predictors": predictor_status,
        "teacher_score_rows_collected": int(len(collected_rows)),
        "teacher_score_rows": int(len(score_rows)),
        "protein_pseudo_labels": int(len(protein_consensus)),
        "pseudo_regions": int(len(pseudo_regions)),
        "outputs": {
            "manifest_with_teacher": str(manifest_with_teacher),
            "teacher_scores_h5": str(teacher_scores_h5),
            "teacher_scores": str(scores_csv),
            "teacher_region_candidates": str(regions_csv),
            "teacher_region_candidates_jsonl": str(regions_jsonl),
            "teacher_protein_labels": str(protein_csv),
            "teacher_evidence": str(evidence_csv),
        },
    }
    write_json(output_dir / "teacher_pseudo_label_report.json", report)
    _write_markdown_report(output_dir / "teacher_pseudo_label_report.md", report)
    return report


def _run_predictors(
    predictors: dict[str, Any],
    context: dict[str, str],
    selected: set[str] | None,
) -> list[dict[str, Any]]:
    status_by_name: dict[str, dict[str, Any]] = {}
    jobs: list[tuple[str, dict[str, Any], list[str], dict[str, str]]] = []
    ordered_names: list[str] = []
    for name, cfg in predictors.items():
        if selected is not None and name not in selected:
            continue
        ordered_names.append(name)
        if not bool(cfg.get("enabled", False)):
            status_by_name[name] = {"name": name, "status": "disabled"}
            continue
        command = cfg.get("command")
        if not command:
            status_by_name[name] = {"name": name, "status": "no_command_collect_only"}
            continue
        command = [_format_token(str(token), context) for token in command]
        for token in command:
            if token.startswith(str(context["raw_dir"])):
                Path(token).parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        for key, value in dict(cfg.get("env", {}) or {}).items():
            env[str(key)] = _format_token(str(value), context)
        jobs.append((name, cfg, command, env))

    if jobs:
        max_workers = int(os.environ.get("PHASEFLOW_TEACHER_PARALLEL", str(len(jobs))))
        max_workers = max(1, min(max_workers, len(jobs)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(_run_one_predictor, name, command, env): name
                for name, _, command, env in jobs
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                status_by_name[name] = future.result()
    return [status_by_name[name] for name in ordered_names if name in status_by_name]


def _run_one_predictor(name: str, command: list[str], env: dict[str, str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        return {
            "name": name,
            "status": "ok" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1000:],
            "stderr_tail": result.stderr[-1000:],
        }
    except FileNotFoundError as exc:
        return {"name": name, "status": "missing_executable", "error": str(exc)}


def _collect_scores(
    predictors: dict[str, Any],
    context: dict[str, str],
    manifest_frame: pd.DataFrame,
    selected: set[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sequence_by_id = {str(row["protein_id"]): clean_protein_sequence(str(row["sequence"])) for _, row in manifest_frame.iterrows()}
    for name, cfg in predictors.items():
        if selected is not None and name not in selected:
            continue
        if not bool(cfg.get("enabled", False)):
            continue
        output_pattern = _format_token(str(cfg.get("output", "")), context)
        paths = [Path(path) for path in glob.glob(output_pattern)] if any(char in output_pattern for char in "*?[]") else [Path(output_pattern)]
        existing = [path for path in paths if path.exists()]
        if not existing:
            continue
        parser_name = str(cfg.get("parser", ""))
        for path in existing:
            parsed = _parse_predictor_output(name, parser_name, path, cfg, sequence_by_id)
            rows.extend(parsed)
    return rows


def _parse_predictor_output(
    teacher: str,
    parser_name: str,
    path: Path,
    cfg: dict[str, Any],
    sequence_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    if parser_name == "deephase_tsv":
        return _parse_deephase(teacher, path, cfg)
    if parser_name == "pscore_text":
        return _parse_pscore(teacher, path, cfg)
    if parser_name == "psphunter_regions":
        return _parse_psphunter_regions(teacher, path, cfg)
    if parser_name == "phasemotif_csv":
        return _parse_phasemotif(teacher, path, cfg, sequence_by_id)
    if parser_name == "phaseflow_window_jsonl":
        return _parse_phaseflow_windows(teacher, path, cfg)
    if parser_name == "pspredictor_csv":
        return _parse_pspredictor(teacher, path, cfg)
    if parser_name == "molphase_summary_csv":
        return _parse_molphase(teacher, path, cfg)
    if parser_name == "fuzdrop_csv":
        return _parse_fuzdrop(teacher, path, cfg)
    if parser_name == "profile_h5":
        return _parse_profile_h5(teacher, path, cfg)
    if parser_name == "profile_jsonl":
        return _parse_profile_jsonl(teacher, path, cfg)
    return []


def _parse_deephase(teacher: str, path: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t")
    rows = []
    for _, row in frame.iterrows():
        protein_id = str(row.get("name", "")).split()[0]
        score = _float_or_none(row.get("deephase_score"))
        if protein_id and score is not None:
            rows.append(_score_row(teacher, "protein", protein_id, None, None, score, cfg, str(path)))
    return rows


def _parse_pscore(teacher: str, path: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        match = re.search(r"PScore:\s*([-+]?\d+(?:\.\d+)?)\s*>?(\S+)?", line)
        if not match:
            continue
        score = float(match.group(1))
        protein_id = (match.group(2) or "").strip()
        if protein_id:
            rows.append(_score_row(teacher, "protein", protein_id, None, None, score, cfg, str(path)))
    return rows


def _parse_pspredictor(teacher: str, path: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        protein_id = str(row.get("short_id", "")).split()[0]
        score = _float_or_none(row.get("pspredictor_score"))
        if protein_id and score is not None:
            rows.append(_score_row(teacher, "protein", protein_id, None, None, score, cfg, str(path)))
    return rows


def _parse_molphase(teacher: str, path: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        protein_id = str(row.get("seqname", "")).split()[0]
        score = _float_or_none(row.get("molphase_score"))
        if protein_id and score is not None:
            rows.append(_score_row(teacher, "protein", protein_id, None, None, score, cfg, str(path)))
    return rows


def _parse_fuzdrop(teacher: str, path: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    rows = []
    id_col = _first_existing_column(frame, ["protein_id", "id", "name", "short_id"])
    score_col = _first_existing_column(frame, ["pLLPS", "pDP", "fuzdrop_score", "score"])
    if id_col is None or score_col is None:
        return rows
    for _, row in frame.iterrows():
        score = _float_or_none(row.get(score_col))
        protein_id = str(row.get(id_col, "")).split()[0]
        if score is not None and protein_id:
            rows.append(_score_row(teacher, "protein", protein_id, None, None, score, cfg, str(path)))
    return rows


def _parse_phasemotif(
    teacher: str,
    path: Path,
    cfg: dict[str, Any],
    sequence_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        raw_name = str(row.get("IDR Name", "")).strip()
        score = _float_or_none(row.get("Predict Score"))
        if not raw_name or score is None:
            continue
        protein_id, start, end = _parse_segment_name(raw_name)
        if start is None or end is None:
            segment = clean_protein_sequence(str(row.get("IDR", "")))
            sequence = sequence_by_id.get(protein_id, "")
            found = sequence.find(segment) if sequence and segment else -1
            if found >= 0:
                start, end = found, found + len(segment) - 1
        if start is not None and end is not None:
            rows.append(_score_row(teacher, "region", protein_id, start, end, score, cfg, str(path)))
    return rows


def _parse_phaseflow_windows(teacher: str, path: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    window = int(cfg.get("window_size", 10))
    threshold = float(cfg.get("threshold", 0.5))
    min_len = int(cfg.get("min_region_len", cfg.get("profile_min_region_len", 8)))
    merge_gap = int(cfg.get("merge_gap", 5))
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        protein_id = str(payload.get("protein_id") or payload.get("entry") or payload.get("id") or payload.get("record_id") or "")
        scores = payload.get("scores")
        if scores is None:
            scores = payload.get("score") or []
        profile = np.asarray(scores, dtype=np.float32)
        payload_length = _int_or_none(payload.get("length"))
        if protein_id and profile.ndim == 1 and profile.size > 0 and payload_length == int(profile.size):
            rows.append(_profile_row(teacher, protein_id, profile, cfg, str(path)))
            mask = np.isfinite(profile) & (profile >= threshold)
            for start, end in _spans_from_mask(mask, min_len=min_len, merge_gap=merge_gap):
                score = float(np.nanmean(profile[start : end + 1]))
                rows.append(_score_row(teacher, "region", protein_id, start, end, score, cfg, str(path)))
            continue
        for start, raw_score in enumerate(scores):
            score = _float_or_none(raw_score)
            if protein_id and score is not None:
                rows.append(_score_row(teacher, "region", protein_id, start, start + window - 1, score, cfg, str(path)))
    return rows


def _parse_psphunter_regions(teacher: str, path: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_id = ""
    positions: list[int] = []
    scores: list[float] = []
    flags: list[int] = []

    def flush_table() -> None:
        if not current_id or not positions:
            return
        length = max(positions)
        profile = np.zeros(length, dtype=np.float32)
        mask = np.zeros(length, dtype=bool)
        for pos, score, flag in zip(positions, scores, flags):
            index = max(0, pos - 1)
            if flag > 0:
                profile[index] = float(1.0 - np.clip(score, 0.0, 1.0))
                mask[index] = True
        rows.append(_profile_row(teacher, current_id, profile, cfg, str(path)))
        for start, end in _spans_from_mask(mask, min_len=int(cfg.get("min_region_len", 1)), merge_gap=int(cfg.get("merge_gap", 0))):
            span_score = float(np.nanmean(profile[start : end + 1]))
            rows.append(_score_row(teacher, "region", current_id, start, end, span_score, cfg, str(path)))

    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#Sequecing ID:") or stripped.startswith("#Sequencing ID:"):
            flush_table()
            current_id = stripped.split(":", 1)[1].strip().split()[0]
            positions, scores, flags = [], [], []
            continue
        if stripped.startswith("#"):
            continue
        if stripped.lower().startswith("pos"):
            continue
        fields = re.split(r"\s+", stripped)
        if len(fields) >= 4 and fields[0].isdigit():
            score = _float_or_none(fields[2]) if fields[2] != "-" else 0.0
            flag = _float_or_none(fields[3])
            if score is not None and flag is not None:
                positions.append(int(fields[0]))
                scores.append(float(score))
                flags.append(int(flag))
            continue
        header = re.match(r">?([A-Za-z0-9_.|-]+)", stripped)
        if header and not re.search(r"\d+\s*[-,]\s*\d+", stripped):
            flush_table()
            current_id = header.group(1).split()[0]
            positions, scores, flags = [], [], []
    flush_table()
    return rows


def _parse_profile_h5(teacher: str, path: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset_name = str(cfg.get("dataset") or cfg.get("profile_dataset") or "score")
    threshold = float(cfg.get("threshold", 0.5))
    min_len = int(cfg.get("min_region_len", cfg.get("profile_min_region_len", 8)))
    merge_gap = int(cfg.get("merge_gap", 5))
    protein_score_attr = str(cfg.get("protein_score_attr", "protein_score"))
    with h5py.File(path, "r") as handle:
        for protein_id in handle:
            group = handle[protein_id]
            if dataset_name not in group:
                continue
            profile = np.asarray(group[dataset_name], dtype=np.float32)
            if profile.ndim != 1 or profile.size == 0:
                continue
            rows.append(_profile_row(teacher, str(protein_id), profile, cfg, str(path)))
            protein_score = _float_or_none(group.attrs.get(protein_score_attr))
            if protein_score is not None:
                rows.append(_score_row(teacher, "protein", str(protein_id), None, None, protein_score, cfg, str(path)))
            mask = np.isfinite(profile) & (profile >= threshold)
            for start, end in _spans_from_mask(mask, min_len=min_len, merge_gap=merge_gap):
                score = float(np.nanmean(profile[start : end + 1]))
                rows.append(_score_row(teacher, "region", str(protein_id), start, end, score, cfg, str(path)))
    return rows


def _parse_profile_jsonl(teacher: str, path: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    score_key = str(cfg.get("score_key", cfg.get("dataset", "score")))
    threshold = float(cfg.get("threshold", 0.5))
    min_len = int(cfg.get("min_region_len", cfg.get("profile_min_region_len", 8)))
    merge_gap = int(cfg.get("merge_gap", 5))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            protein_id = str(payload.get("protein_id") or payload.get("id") or "")
            profile_values = payload.get(score_key)
            if not protein_id or profile_values is None:
                continue
            profile = np.asarray(profile_values, dtype=np.float32)
            if profile.ndim != 1 or profile.size == 0:
                continue
            rows.append(_profile_row(teacher, protein_id, profile, cfg, str(path)))
            protein_score = _float_or_none(payload.get("protein_score"))
            if protein_score is not None:
                rows.append(_score_row(teacher, "protein", protein_id, None, None, protein_score, cfg, str(path)))
            mask = np.isfinite(profile) & (profile >= threshold)
            for start, end in _spans_from_mask(mask, min_len=min_len, merge_gap=merge_gap):
                score = float(np.nanmean(profile[start : end + 1]))
                rows.append(_score_row(teacher, "region", protein_id, start, end, score, cfg, str(path)))
    return rows


def _score_row(
    teacher: str,
    scope: str,
    protein_id: str,
    start: int | None,
    end: int | None,
    score: float,
    cfg: dict[str, Any],
    source_path: str,
) -> dict[str, Any]:
    threshold = float(cfg.get("threshold", 0.5))
    confidence = _confidence(score, threshold, str(cfg.get("direction", "high")), float(cfg.get("weight", 1.0)))
    return {
        "teacher": teacher,
        "scope": scope,
        "protein_id": protein_id,
        "start": start,
        "end": end,
        "score": float(score),
        "threshold": threshold,
        "confidence": confidence,
        "positive": bool(confidence >= 0.5 and _passes_threshold(score, threshold, str(cfg.get("direction", "high")))),
        "source_path": source_path,
    }


def _profile_row(
    teacher: str,
    protein_id: str,
    profile: np.ndarray,
    cfg: dict[str, Any],
    source_path: str,
) -> dict[str, Any]:
    profile = np.asarray(profile, dtype=np.float32)
    finite = np.isfinite(profile)
    score = float(np.nanmean(profile)) if np.any(finite) else float("nan")
    weight = float(cfg.get("weight", 1.0))
    return {
        "teacher": teacher,
        "scope": "profile",
        "protein_id": protein_id,
        "start": None,
        "end": None,
        "score": score,
        "threshold": float(cfg.get("threshold", 0.5)),
        "confidence": float(max(0.0, min(1.0, weight))),
        "positive": bool(score == score and _passes_threshold(score, float(cfg.get("threshold", 0.5)), str(cfg.get("direction", "high")))),
        "source_path": source_path,
        "profile": profile,
    }


def _consensus_regions(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    min_teachers = int(config.get("min_region_teachers", 1))
    min_confidence = float(config.get("min_region_confidence", 0.55))
    merge_gap = int(config.get("merge_gap", 5))
    min_len = int(config.get("min_region_len", 8))
    by_protein: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("positive") or row.get("start") is None or row.get("end") is None:
            continue
        by_protein.setdefault(str(row["protein_id"]), []).append(row)

    pseudo: list[dict[str, Any]] = []
    for protein_id, items in by_protein.items():
        items.sort(key=lambda row: (int(row["start"]), int(row["end"])))
        clusters: list[list[dict[str, Any]]] = []
        for row in items:
            if not clusters or int(row["start"]) > max(int(item["end"]) for item in clusters[-1]) + merge_gap + 1:
                clusters.append([row])
            else:
                clusters[-1].append(row)
        for cluster in clusters:
            teachers = sorted({str(row["teacher"]) for row in cluster})
            confidence = sum(float(row["confidence"]) for row in cluster) / max(len(cluster), 1)
            start = min(int(row["start"]) for row in cluster)
            end = max(int(row["end"]) for row in cluster)
            if len(teachers) < min_teachers or confidence < min_confidence or end - start + 1 < min_len:
                continue
            pseudo.append(
                {
                    "protein_id": protein_id,
                    "start": start,
                    "end": end,
                    "confidence": float(min(confidence, 1.0)),
                    "teachers": ";".join(teachers),
                    "n_teachers": len(teachers),
                    "region_type": "DPR_candidate",
                    "region_label": "candidate",
                    "evidence_level": "pseudo",
                    "source": "teacher_consensus",
                    "notes": f"teacher_consensus={';'.join(teachers)}",
                }
            )
    return pseudo


def _consensus_proteins(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    min_teachers = int(config.get("min_protein_teachers", 2))
    min_confidence = float(config.get("min_protein_confidence", 0.60))
    by_protein: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("positive"):
            by_protein.setdefault(str(row["protein_id"]), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for protein_id, items in by_protein.items():
        teachers = sorted({str(row["teacher"]) for row in items})
        confidence = sum(float(row["confidence"]) for row in items) / max(len(items), 1)
        if len(teachers) >= min_teachers and confidence >= min_confidence:
            out[protein_id] = {
                "protein_id": protein_id,
                "llps_label": 1,
                "confidence": float(min(confidence, 1.0)),
                "teachers": ";".join(teachers),
                "n_teachers": len(teachers),
            }
    return out


def _write_manifest_with_teacher(frame: pd.DataFrame, consensus: dict[str, dict[str, Any]], path: Path) -> Path:
    frame = frame.copy()
    frame["teacher_consensus_score"] = frame["protein_id"].astype(str).map(lambda pid: consensus.get(pid, {}).get("confidence", ""))
    frame["teacher_consensus_teachers"] = frame["protein_id"].astype(str).map(lambda pid: consensus.get(pid, {}).get("teachers", ""))
    if "llps_label" in frame.columns:
        for index, row in frame.iterrows():
            protein_id = str(row["protein_id"])
            if protein_id not in consensus:
                continue
            current = row.get("llps_label")
            if pd.isna(current) or int(current) == -100:
                frame.loc[index, "llps_label"] = 1
                frame.loc[index, "sample_weight"] = consensus[protein_id]["confidence"]
                frame.loc[index, "evidence_level"] = "pseudo"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_teacher_profiles_h5(
    path: Path,
    frame: pd.DataFrame,
    region_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    min_confidence = float(config.get("profile_min_confidence", 0.0))
    rows_by_protein: dict[str, list[dict[str, Any]]] = {}
    for row in region_rows:
        if row.get("start") is None or row.get("end") is None:
            continue
        if float(row.get("confidence", 0.0)) < min_confidence:
            continue
        rows_by_protein.setdefault(str(row["protein_id"]), []).append(row)
    profiles_by_protein: dict[str, list[dict[str, Any]]] = {}
    for row in profile_rows:
        if "profile" not in row:
            continue
        if float(row.get("confidence", 0.0)) < min_confidence:
            continue
        profiles_by_protein.setdefault(str(row["protein_id"]), []).append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["status"] = "ready"
        handle.attrs["policy"] = "soft_distillation_only_not_model_input"
        for _, row in frame.iterrows():
            protein_id = str(row["protein_id"])
            sequence = clean_protein_sequence(str(row["sequence"]))
            length = len(sequence)
            if length == 0:
                continue
            teacher_scores: dict[str, np.ndarray] = {}
            teacher_weights: dict[str, np.ndarray] = {}
            for item in rows_by_protein.get(protein_id, []):
                start = max(0, int(item["start"]))
                end = min(length - 1, int(item["end"]))
                if end < start:
                    continue
                teacher = str(item.get("teacher", "teacher"))
                confidence = float(item.get("confidence", 0.0))
                raw_score = float(item.get("score", confidence))
                value = float(np.clip(raw_score if 0.0 <= raw_score <= 1.0 else confidence, 0.0, 1.0))
                scores, weights = _teacher_arrays_for(teacher_scores, teacher_weights, teacher, length)
                current = weights[start : end + 1]
                replace = confidence >= current
                scores[start : end + 1][replace] = value
                weights[start : end + 1][replace] = confidence
            for item in profiles_by_protein.get(protein_id, []):
                teacher = str(item.get("teacher", "teacher"))
                profile = _fit_profile(np.asarray(item["profile"], dtype=np.float32), length)
                valid = np.isfinite(profile)
                if not np.any(valid):
                    continue
                scores, weights = _teacher_arrays_for(teacher_scores, teacher_weights, teacher, length)
                confidence = float(item.get("confidence", 1.0))
                replace = valid & (confidence >= weights)
                scores[replace] = np.clip(profile[replace], 0.0, 1.0)
                weights[replace] = confidence
            consensus, confidence, uncertainty = _teacher_consensus_from_arrays(teacher_scores, teacher_weights, length)
            group = handle.create_group(protein_id)
            for teacher, scores in sorted(teacher_scores.items()):
                group.create_dataset(_teacher_dataset_name(teacher), data=scores, compression="gzip")
            group.create_dataset("teacher_consensus", data=consensus, compression="gzip")
            group.create_dataset("teacher_confidence", data=confidence, compression="gzip")
            group.create_dataset("teacher_uncertainty", data=uncertainty, compression="gzip")
            group.attrs["n_teacher_regions"] = len(rows_by_protein.get(protein_id, []))
            group.attrs["teacher_names_json"] = json.dumps(
                sorted(set(teacher_scores))
            )
    return path


def _write_region_candidates_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    by_protein: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_protein.setdefault(str(row["protein_id"]), []).append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for protein_id in sorted(by_protein):
            handle.write(json.dumps({"protein_id": protein_id, "candidate_regions": by_protein[protein_id]}, sort_keys=True) + "\n")


def _write_score_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["teacher", "scope", "protein_id", "start", "end", "score", "threshold", "confidence", "positive", "source_path"]
    _write_dict_csv(path, rows, fieldnames)


def _write_region_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    converted = []
    for row in rows:
        item = dict(row)
        item["start_1based"] = int(row["start"]) + 1
        item["end_1based"] = int(row["end"]) + 1
        converted.append(item)
    fieldnames = [
        "protein_id",
        "start",
        "end",
        "start_1based",
        "end_1based",
        "confidence",
        "teachers",
        "n_teachers",
        "region_type",
        "region_label",
        "evidence_level",
        "source",
        "notes",
    ]
    _write_dict_csv(path, converted, fieldnames)


def _write_protein_csv(path: Path, rows_by_id: dict[str, dict[str, Any]]) -> None:
    _write_dict_csv(path, list(rows_by_id.values()), ["protein_id", "llps_label", "confidence", "teachers", "n_teachers"])


def _write_evidence_csv(path: Path, protein_rows: list[dict[str, Any]], region_rows: list[dict[str, Any]]) -> None:
    rows = []
    for row in protein_rows + region_rows:
        rows.append(
            {
                "protein_id": row["protein_id"],
                "label_scope": row["scope"],
                "teacher": row["teacher"],
                "start": row.get("start", ""),
                "end": row.get("end", ""),
                "score": row["score"],
                "confidence": row["confidence"],
                "positive": row["positive"],
                "source_path": row["source_path"],
            }
        )
    _write_dict_csv(path, rows, ["protein_id", "label_scope", "teacher", "start", "end", "score", "confidence", "positive", "source_path"])


def _write_dict_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Teacher Pseudo Label Report",
        "",
        f"- train records: {report['train_records']}",
        f"- IDR/candidate segments: {report['idr_segments']}",
        f"- teacher score rows collected: {report['teacher_score_rows_collected']}",
        f"- teacher score rows: {report['teacher_score_rows']}",
        f"- protein pseudo labels: {report['protein_pseudo_labels']}",
        f"- pseudo regions: {report['pseudo_regions']}",
        "",
        "## Outputs",
    ]
    for key, value in report["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def _write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for protein_id, sequence in records:
            handle.write(f">{protein_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def _write_idr_segment_fasta(path: Path, frame: pd.DataFrame) -> int:
    count = 0
    with path.open("w") as handle:
        for _, row in frame.iterrows():
            protein_id = str(row["protein_id"])
            sequence = clean_protein_sequence(str(row["sequence"]))
            spans = _candidate_spans(sequence)
            for start, end in spans:
                segment = sequence[start:end]
                if len(segment) < 8:
                    continue
                count += 1
                handle.write(f">{protein_id}|start={start + 1}|end={end}\n")
                handle.write(segment + "\n")
    return count


def _candidate_spans(sequence: str, threshold: float = 0.5, flank: int = 8, merge_gap: int = 8) -> list[tuple[int, int]]:
    disorder, _, _, _ = compute_disorder_features(sequence, mode="simple")
    score = disorder[:, :3].max(axis=1)
    mask = score >= threshold
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if bool(flag) and start is None:
            start = index
        elif not bool(flag) and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(sequence)))
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        start = max(0, start - flank)
        end = min(len(sequence), end + flank)
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [(start, end) for start, end in merged if end - start >= 8]


def _write_metadata(path: Path, frame: pd.DataFrame) -> None:
    out = pd.DataFrame(
        {
            "short_id": frame["protein_id"].astype(str),
            "mutation": "",
            "sequence": frame["sequence"].astype(str),
            "length": frame["sequence"].astype(str).map(len),
        }
    )
    out.to_csv(path, index=False)


def _read_id_file(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _format_token(token: str, context: dict[str, str]) -> str:
    return token.format(**context)


def _confidence(score: float, threshold: float, direction: str, weight: float) -> float:
    if 0.0 <= score <= 1.0 and 0.0 <= threshold <= 1.0:
        base = score if direction == "high" else 1.0 - score
    else:
        margin = score - threshold if direction == "high" else threshold - score
        base = 1.0 / (1.0 + math.exp(-margin))
    return float(max(0.0, min(1.0, base * max(weight, 0.0))))


def _passes_threshold(score: float, threshold: float, direction: str) -> bool:
    return score >= threshold if direction == "high" else score <= threshold


def _parse_segment_name(value: str) -> tuple[str, int | None, int | None]:
    protein_id = value.split("|")[0].split()[0]
    start_match = re.search(r"start=(\d+)", value)
    end_match = re.search(r"end=(\d+)", value)
    if start_match and end_match:
        return protein_id, int(start_match.group(1)) - 1, int(end_match.group(1)) - 1
    parts = value.split("|")
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return protein_id, int(parts[-2]) - 1, int(parts[-1]) - 1
    return protein_id, None, None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(str(value).strip())
    except ValueError:
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(float(str(value).strip()))
    except ValueError:
        return None


def _last_float(value: str) -> float | None:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", value)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _first_existing_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    lower = {name.lower(): name for name in frame.columns}
    for name in names:
        if name in frame.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _spans_from_mask(mask: np.ndarray, *, min_len: int = 1, merge_gap: int = 0) -> list[tuple[int, int]]:
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


def _fit_profile(profile: np.ndarray, length: int) -> np.ndarray:
    out = np.full(length, np.nan, dtype=np.float32)
    copy_len = min(length, int(profile.shape[0]))
    if copy_len > 0:
        out[:copy_len] = profile[:copy_len]
    return out


def _teacher_arrays_for(
    scores_by_teacher: dict[str, np.ndarray],
    weights_by_teacher: dict[str, np.ndarray],
    teacher: str,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if teacher not in scores_by_teacher:
        scores_by_teacher[teacher] = np.full(length, np.nan, dtype=np.float32)
        weights_by_teacher[teacher] = np.zeros(length, dtype=np.float32)
    return scores_by_teacher[teacher], weights_by_teacher[teacher]


def _teacher_dataset_name(teacher: str) -> str:
    mapping = {
        "pstp": "pstp_scan_score",
        "pstp_scan": "pstp_scan_score",
        "psphunter": "psphunter_key_score",
        "catgranule": "catgranule_score",
        "catgranule2": "catgranule_score",
        "phasemotif": "phasemotif_score",
        "fuzdrop": "fuzdrop_score",
    }
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", teacher.strip().lower()).strip("_")
    return mapping.get(safe, f"{safe}_score" if safe else "teacher_score")


def _teacher_consensus_from_arrays(
    scores_by_teacher: dict[str, np.ndarray],
    weights_by_teacher: dict[str, np.ndarray],
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    consensus = np.full(length, np.nan, dtype=np.float32)
    confidence = np.zeros(length, dtype=np.float32)
    uncertainty = np.ones(length, dtype=np.float32)
    if not scores_by_teacher:
        return consensus, confidence, uncertainty

    score_sum = np.zeros(length, dtype=np.float64)
    weight_sum = np.zeros(length, dtype=np.float64)
    total_possible = 0.0
    for teacher, scores in scores_by_teacher.items():
        weights = weights_by_teacher.get(teacher, np.zeros(length, dtype=np.float32)).astype(np.float64)
        valid = np.isfinite(scores) & (weights > 0)
        if np.any(valid):
            score_sum[valid] += scores[valid].astype(np.float64) * weights[valid]
            weight_sum[valid] += weights[valid]
            total_possible += float(np.nanmax(weights))

    valid = weight_sum > 0
    if not np.any(valid):
        return consensus, confidence, uncertainty
    mean = np.zeros(length, dtype=np.float64)
    mean[valid] = score_sum[valid] / weight_sum[valid]
    var_sum = np.zeros(length, dtype=np.float64)
    for teacher, scores in scores_by_teacher.items():
        weights = weights_by_teacher.get(teacher, np.zeros(length, dtype=np.float32)).astype(np.float64)
        valid_teacher = np.isfinite(scores) & (weights > 0)
        var_sum[valid_teacher] += weights[valid_teacher] * (scores[valid_teacher].astype(np.float64) - mean[valid_teacher]) ** 2
    variance = np.zeros(length, dtype=np.float64)
    variance[valid] = var_sum[valid] / weight_sum[valid]
    coverage = np.zeros(length, dtype=np.float64)
    coverage[valid] = np.clip(weight_sum[valid] / max(total_possible, 1.0e-6), 0.0, 1.0)
    agreement = np.ones(length, dtype=np.float64)
    agreement[valid] = np.clip(1.0 - 2.0 * np.sqrt(np.clip(variance[valid], 0.0, 0.25)), 0.0, 1.0)
    consensus[valid] = np.clip(mean[valid], 0.0, 1.0).astype(np.float32)
    confidence[valid] = (coverage[valid] * agreement[valid]).astype(np.float32)
    uncertainty[valid] = (1.0 - confidence[valid]).astype(np.float32)
    return consensus, confidence, uncertainty


def main() -> None:
    parser = argparse.ArgumentParser(description="Run teacher predictors and build train-only pseudo labels.")
    parser.add_argument("--config", default="configs/teacher_pseudo_labels.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--regions")
    parser.add_argument("--train-ids-file")
    parser.add_argument("--out-dir")
    parser.add_argument("--skip-predictors", action="store_true")
    parser.add_argument("--predictors", nargs="*", help="Optional subset of predictor keys from the config.")
    args = parser.parse_args()
    selected = set(args.predictors) if args.predictors else None
    report = build_teacher_pseudo_labels(
        load_yaml(args.config),
        manifest=args.manifest,
        regions=args.regions,
        train_ids_file=args.train_ids_file,
        out_dir=args.out_dir,
        run_predictors=not args.skip_predictors,
        selected_predictors=selected,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
