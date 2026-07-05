from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "artifacts/data/processed/evaluation/phasepro_official_v1"
OUT_ROOT = ROOT / "external_artifacts/pstp_official_benchmark_v1"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_official_package_is_exact_121_143() -> None:
    validation = load_json(DATA_ROOT / "validation_report.json")
    assert validation["protein_count"] == 121
    assert validation["unique_protein_count"] == 121
    assert validation["region_count"] == 143
    assert validation["raw_boundary_count"] == 144
    assert validation["invalid_region_count"] == 0
    assert validation["empty_positive_mask_count"] == 0
    assert validation["duplicate_region_count"] == 0
    assert validation["training_allowed"] is False
    assert validation["evaluation_only"] is True


def test_regions_are_in_bounds_and_not_clipped_silently() -> None:
    regions = pd.read_parquet(DATA_ROOT / "regions.parquet")
    assert ((regions["start_0based"] >= 0) & (regions["start_0based"] < regions["end_exclusive"])).all()
    assert (regions["end_exclusive"] <= regions["sequence_length"]).all()
    clipped = regions[regions["clipped_from_raw"].astype(bool)]
    assert len(clipped) == 3
    assert {"start_raw", "end_raw", "end_exclusive", "clipped_from_raw"}.issubset(clipped.columns)
    skipped = pd.read_csv(DATA_ROOT / "skipped_raw_regions.csv")
    assert len(skipped) == 1
    assert set(skipped["skip_reason"]) == {"pstp_start_after_sequence_end"}


def test_q09737_uses_official_span_policy() -> None:
    regions = pd.read_parquet(DATA_ROOT / "regions.parquet")
    q09737 = regions[regions["protein_id"].eq("Q09737")]
    assert len(q09737) == 1
    row = q09737.iloc[0]
    assert int(row["sequence_length"]) == 571
    assert int(row["start_raw"]) == 1
    assert int(row["end_raw"]) == 750
    assert int(row["start_0based"]) == 0
    assert int(row["end_exclusive"]) == 571
    assert int(row["pstp_notebook_start_0based"]) == 1
    assert int(row["pstp_notebook_end_exclusive"]) == 571


def test_no_local_sidecar_fallback_in_sources() -> None:
    proteins = pd.read_parquet(DATA_ROOT / "proteins.parquet")
    regions = pd.read_parquet(DATA_ROOT / "regions.parquet")
    source_manifest = pd.read_csv(DATA_ROOT / "source_manifest.csv")
    all_sources = "\n".join(
        list(proteins["source_file"].astype(str))
        + list(regions["source_file"].astype(str))
        + list(source_manifest["relative_path"].astype(str))
    )
    assert "PhaSePro_data_all.tsv" in all_sources
    assert "sidecar" not in all_sources.lower()
    assert "140" not in all_sources


def test_official_cached_p33_profiles_match_sequence_lengths() -> None:
    proteins = pd.read_parquet(DATA_ROOT / "proteins.parquet").set_index("protein_id")
    profile_npz = np.load(OUT_ROOT / "profiles/pstp/official_cached_best_p33_profiles.npz")
    assert set(profile_npz.files) == set(proteins.index)
    for protein_id in profile_npz.files:
        assert len(profile_npz[protein_id]) == int(proteins.loc[protein_id, "sequence_length"])
    manifest = pd.read_csv(OUT_ROOT / "profiles/pstp/profile_manifest.csv")
    best = manifest[manifest["model_name"].eq("official_cached_best")]
    assert len(best) == 121
    assert best["profile_length"].gt(0).all()


def test_native_cached_reproduction_passes_saved_notebook_output() -> None:
    status = load_json(OUT_ROOT / "official_native/native_reproduction_status.json")
    metrics = load_json(OUT_ROOT / "official_native/native_reproduction_metrics.json")
    assert status["status"] == "PASS"
    assert status["global_spearman_abs_diff_vs_saved_cell5"] == 0.0
    assert status["region_overlap_matches_saved_cell5"] is True
    native = metrics["notebook_truth_metrics"]
    assert native["total_residue_count"] == 86660
    assert native["positive_residue_count"] == 47129
    assert native["pstp_scanned_regions"] == 212
    assert native["pstp_mapped_regions"] == 123
    assert native["pstp_total_regions"] == 143
    assert native["per_protein_Spearman_valid_count"] == 85
    assert native["per_protein_Spearman_invalid_count"] == 36


def test_final_status_is_blocked_before_v5_or_training() -> None:
    run_info = load_json(OUT_ROOT / "manifests/run_info.json")
    assert run_info["final_status"] == "BLOCKED"
    assert run_info["training_started"] is False
    assert run_info["v5_evaluated"] is False
    assert run_info["full_12000_continued"] is False
    reasons = "\n".join(run_info["hard_stop_reasons"])
    assert "PhaSePro-included" in reasons
    assert "No official PhaSePro feature matrix cache" in reasons
