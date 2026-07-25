from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_full_length_public_entry_points_use_stable_names() -> None:
    expected = (
        "configs/full_length/llps.yaml",
        "configs/full_length/dpr.yaml",
        "scripts/full_length/train_llps.py",
        "scripts/full_length/train_dpr.py",
        "scripts/full_length/data/build_dataset.py",
        "scripts/full_length/data/build_region_targets.py",
        "scripts/full_length/benchmark/evaluate_benchmarks.py",
        "scripts/full_length/tables/generate_tables.py",
        "scripts/full_length/figures/plot_llps_benchmark.py",
        "scripts/full_length/figures/plot_llps_ablation.py",
        "scripts/full_length/figures/plot_dpr_benchmark.py",
        "scripts/full_length/figures/plot_dpr_ablation.py",
        "scripts/full_length/figures/plot_dpr_examples.py",
        "scripts/full_length/figures/plot_model_architecture.py",
        "scripts/full_length/figures/assemble_manuscript_assets.py",
    )
    for relative_path in expected:
        assert (ROOT / relative_path).is_file(), relative_path


def test_public_full_length_files_do_not_embed_machine_paths_or_release_labels() -> None:
    paths = (
        ROOT / "configs/full_length/llps.yaml",
        ROOT / "configs/full_length/dpr.yaml",
        ROOT / "scripts/full_length/train_llps.py",
        ROOT / "scripts/full_length/train_dpr.py",
        ROOT / "scripts/full_length/data/build_dataset.py",
        ROOT / "scripts/full_length/data/build_region_targets.py",
        *(ROOT / "scripts/full_length/figures").glob("*.py"),
    )
    forbidden = ("/data/mogoo7zn", "outputs/overall/final", "external_artifacts")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path}: {token}"


def test_retired_full_length_public_names_are_absent() -> None:
    retired = (
        "configs/full_length/final_llps.yaml",
        "configs/full_length/final_dpr.yaml",
        "scripts/full_length/training/run_dpr_v6.py",
        "scripts/full_length/data/build_final_region_targets.py",
        "scripts/full_length/data/build_server_final_dataset.py",
        "scripts/full_length/benchmark/final_overall_benchmark_from_profiles.py",
        "scripts/full_length/generate_paper_tables.py",
    )
    for relative_path in retired:
        assert not (ROOT / relative_path).exists(), relative_path
