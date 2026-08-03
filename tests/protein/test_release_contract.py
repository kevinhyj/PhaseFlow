"""Workflow-level regression checks for the protein release."""



# Source: test_release_contract.py

"""Release-level invariants for the compact protein reproduction workflow."""


import hashlib
import json
from pathlib import Path
import importlib

import yaml


ROOT = Path(__file__).resolve().parents[2]
PROTEIN_ROOT = ROOT / "phaseflow" / "protein"


def _load_config(name: str) -> dict:
    with (ROOT / "configs" / "protein" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_protein_source_has_only_named_core_modules() -> None:
    modules = {path.name for path in PROTEIN_ROOT.glob("*.py")}
    assert modules == {
        "__init__.py",
        "contracts.py",
        "data.py",
        "features.py",
        "model.py",
        "objectives.py",
        "postprocessing.py",
        "structure.py",
        "tokenizer.py",
    }
    assert not (PROTEIN_ROOT / "models.py").exists()


def test_protein_tokenizer_normalizes_and_encodes_unknown_residues() -> None:
    tokenizer_path = PROTEIN_ROOT / "tokenizer.py"
    assert tokenizer_path.is_file()

    from phaseflow.protein.tokenizer import ProteinTokenizer

    tokenizer = ProteinTokenizer()
    assert tokenizer.normalize("acdu-x") == "ACDX-X"
    assert tokenizer.encode("ACDX-X").tolist() == [1, 2, 3, 0, 0, 0]


def test_protein_postprocessing_module_has_domain_specific_name() -> None:
    assert (PROTEIN_ROOT / "postprocessing.py").is_file()
    assert not (PROTEIN_ROOT / "runtime.py").exists()


def test_protein_documentation_exposes_architecture_and_tokenizer() -> None:
    text = (ROOT / "docs" / "protein" / "README.md").read_text(encoding="utf-8")
    assert "architecture.md" in text
    assert "tokenizer.py" in text


def test_protein_dry_run_examples_include_all_required_roots() -> None:
    for relative_path in ("docs/protein/architecture.md", "scripts/protein/README.md"):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "--data-root" in text
        assert "--work-root" in text
        assert "--output-root" in text


def test_protein_artifact_placeholder_uses_the_canonical_name() -> None:
    assert (ROOT / "artifacts" / "data" / "protein" / "README.md").is_file()
    assert not (ROOT / "artifacts" / "data" / "_".join(("full", "length"))).exists()


def test_protein_package_excludes_workflow_and_cli_tools() -> None:
    forbidden_names = {
        "train_llps_main",
        "train_dpr_main",
        "refine_dpr_main",
        "evaluate_llps_main",
        "build_features_main",
        "build_features_sharded_main",
        "prepare_weak_dataset_main",
        "parse_args",
        "main",
    }
    for path in PROTEIN_ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import argparse" not in text, path
        assert 'if __name__ == "__main__"' not in text, path
        for name in forbidden_names:
            assert f"def {name}" not in text, f"{path} retains workflow entry point {name}"


def test_external_starling_process_runner_lives_in_scripts() -> None:
    core_features = (PROTEIN_ROOT / "features.py").read_text(encoding="utf-8")
    workflow_features = (ROOT / "scripts" / "protein" / "workflows" / "features.py").read_text(encoding="utf-8")

    assert "subprocess.run(" not in core_features
    assert "def run_starling_segment(" in workflow_features


def test_protein_module_descriptions_are_public_and_task_specific() -> None:
    for path in PROTEIN_ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        first_line = text.splitlines()[0].lower()
        assert "private" not in first_line, path
        assert "protein" in first_line, path
        assert "workflow-scoped private" not in text.lower(), path


def test_protein_implementation_has_no_private_duplicate_package() -> None:
    private_root = ROOT / "phaseflow" / "_protein_impl"
    assert not private_root.exists()


def test_training_contract_is_declared_in_public_configs() -> None:
    llps = _load_config("llps.yaml")
    dpr = _load_config("dpr.yaml")

    assert llps["training"]["max_epochs"] == 3
    assert llps["training"]["max_steps"] == 700
    assert llps["reproduction"]["selected_epoch"] == 2
    assert dpr["scheduler"]["total_updates"] == 4000
    assert dpr["refinement"]["stages"] == [
        "plan_d",
        "plan_c_strong",
        "plan_c_ranking",
    ]
    assert dpr["refinement"]["updates"] == [75, 50, 50]
    assert llps["reproduction"]["benchmark"]["ppmc"]["precision"] == "fp32"
    assert llps["reproduction"]["benchmark"]["ppmc"]["score_source"] == "region_global_llps_score"
    assert llps["reproduction"]["benchmark"]["ppmc"]["batch_size"] == 8
    assert dpr["reproduction"]["benchmark"]["phasepro"] == {
        "checkpoint_sha256": "7fb0091e6dd5a85bd3a6be7a0b606501700c4b8f28ff9b6e309267835a2fdff0",
        "checkpoint_variant": "raw",
        "precision": "bf16",
        "tolerance": 1.0e-6,
    }


# Source: test_protein_package_layout.py
def test_protein_workflow_has_one_flat_public_package() -> None:
    protein = importlib.import_module("phaseflow.protein")

    for name in (
        "PhaseFlowModel",
        "PhaseFlowDataset",
        "compute_biophys_node",
        "scores_to_regions",
    ):
        assert getattr(protein, name) is not None
    for name in ("train_llps_main", "train_dpr_main", "refine_dpr_main", "compile_llps_inputs"):
        assert not hasattr(protein, name)


def test_legacy_protein_subpackages_are_absent() -> None:
    for legacy_path in (
        "data",
        "features",
        "losses",
        "metrics",
        "models",
        "reproduction",
        "training",
        "teachers",
        "ablation",
    ):
        assert not (PROTEIN_ROOT / legacy_path).exists()


def test_protein_scripts_expose_one_flat_reproduction_surface() -> None:
    scripts = ROOT / "scripts" / "protein"
    expected_root_scripts = {"evaluate.py", "prepare.py", "run.py", "train.py"}

    assert {path.name for path in scripts.glob("*.py")} == expected_root_scripts
    assert {path.name for path in scripts.iterdir() if path.is_dir() and not path.name.startswith("__")} == {
        "analysis",
        "features",
        "inference",
        "workflows",
    }


def test_protein_workflow_files_use_domain_specific_names() -> None:
    workflows = ROOT / "scripts" / "protein" / "workflows"
    files = {path.name for path in workflows.glob("*.py")}

    assert "acceptance.py" in files
    assert "runtime_tools.py" not in files



# Source: test_package_boundaries.py


from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_protein_training_clis_delegate_to_workflows() -> None:
    text = (ROOT / "scripts/protein/run.py").read_text(encoding="utf-8")
    for command in ("train-llps", "train-dpr", "refine-dpr", "evaluate-llps", "evaluate-phasepro"):
        assert f'"{command}"' in text
    assert "from scripts.protein.workflows" in text


def test_protein_package_does_not_import_scripts() -> None:
    for path in (ROOT / "phaseflow/protein").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from scripts." not in text, path
        assert "import scripts." not in text, path


def test_protein_training_workflow_has_no_private_run_contract() -> None:
    assert not (ROOT / "phaseflow/protein/training.py").exists()
    module = ROOT / "scripts/protein/workflows/training.py"
    assert module.is_file()

    text = module.read_text(encoding="utf-8")
    for token in ("v8r1a", "remote_053"):
        assert token not in text



# Source: test_public_layout.py


from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_protein_public_entry_points_use_stable_names() -> None:
    expected = (
        "configs/protein/llps.yaml",
        "configs/protein/dpr.yaml",
        "scripts/protein/run.py",
        "scripts/protein/analysis/evaluate_benchmarks.py",
        "scripts/protein/inference/predict_protein_dpr.py",
        "scripts/protein/workflows/precompute_graph_cache.py",
        "artifacts/results/protein/scripts/tables/generate_tables.py",
        "artifacts/results/protein/scripts/tables/generate_tables_pdf.py",
        "artifacts/results/protein/scripts/figures/plot_llps_benchmark.py",
        "artifacts/results/protein/scripts/figures/plot_llps_embedding_ablation.py",
        "artifacts/results/protein/scripts/figures/plot_dpr_benchmark.py",
        "artifacts/results/protein/scripts/figures/plot_dpr_ablation_summary.py",
        "artifacts/results/protein/scripts/figures/plot_dpr_phaseflow_bridge_ablation.py",
        "artifacts/results/protein/scripts/figures/plot_dpr_stream_ablation.py",
        "artifacts/results/protein/scripts/figures/plot_phasepro_dpr_top12.py",
        "artifacts/results/protein/scripts/figures/plot_model_architecture.py",
        "artifacts/results/protein/scripts/figures/assemble_manuscript_assets.py",
    )
    for relative_path in expected:
        assert (ROOT / relative_path).is_file(), relative_path


def test_public_protein_files_do_not_embed_machine_paths_or_release_labels() -> None:
    paths = (
        ROOT / "configs/protein/llps.yaml",
        ROOT / "configs/protein/dpr.yaml",
        ROOT / "scripts/protein/run.py",
        *(ROOT / "artifacts/results/protein/scripts/figures").glob("*.py"),
    )
    forbidden = ("/data/mogoo7zn", "outputs/overall/final", "external_artifacts")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path}: {token}"


def test_retired_protein_public_names_are_absent() -> None:
    retired = (
        "configs/protein/final_llps.yaml",
        "configs/protein/final_dpr.yaml",
        "scripts/protein/training/run_dpr_v6.py",
        "scripts/protein/data/build_dataset.py",
        "scripts/protein/data/build_region_targets.py",
        "scripts/protein/benchmark/evaluate_benchmarks.py",
        "scripts/protein/generate_tables_pdf.py",
        "scripts/figures/protein",
        "scripts/paper/protein",
        "scripts/analysis",
        "scripts/inference",
    )
    for relative_path in retired:
        assert not (ROOT / relative_path).exists(), relative_path


def test_protein_domain_scripts_keep_importable_cli_contracts() -> None:
    import importlib
    import subprocess
    import sys

    for module_name in (
        "scripts.protein.analysis.dpr_plan",
        "scripts.protein.analysis.analyze_dpr_thresholds",
    ):
        assert importlib.import_module(module_name)

    scripts = (
        "scripts/protein/inference/predict_protein_dpr.py",
        "scripts/protein/analysis/analyze_dpr_thresholds.py",
        "scripts/protein/analysis/compare_dpr_phasepro.py",
        "scripts/protein/analysis/compare_dpr_threshold_policies.py",
        "scripts/protein/analysis/evaluate_benchmarks.py",
        "scripts/protein/analysis/select_dpr_plan.py",
        "scripts/protein/workflows/precompute_graph_cache.py",
    )
    for relative_path in scripts:
        result = subprocess.run(
            [sys.executable, str(ROOT / relative_path), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr



# Source: test_public_release_surface.py


from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ROOTS = (
    ROOT / "README.md",
    ROOT / "pyproject.toml",
    ROOT / "configs/protein",
    ROOT / "docs/protein",
    ROOT / "phaseflow/protein",
    ROOT / "scripts/protein",
    ROOT / "reproduction",
)
FORBIDDEN_TOKENS = (
    "CondenGT",
    "PhaseGT",
    "/data/mogoo7zn",
    "outputs/overall/final",
    "full" + "_length",
    "full" + "-length",
    "phasegt",
    "condengt",
)


def _public_text_files() -> list[Path]:
    paths: list[Path] = []
    for root in PUBLIC_ROOTS:
        if root.is_file():
            paths.append(root)
        else:
            paths.extend(path for path in root.rglob("*") if path.suffix in {".md", ".py", ".toml", ".yaml", ".yml", ".sh"})
    return sorted(paths)


def test_public_protein_release_surface_has_no_private_project_or_machine_names() -> None:
    for path in _public_text_files():
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            assert token not in text, f"{path.relative_to(ROOT)} contains {token!r}"



# Source: test_protein_artifact_paths.py

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_protein_configs_use_one_artifact_namespace() -> None:
    config_text = "\n".join(
        (ROOT / "configs" / "protein" / name).read_text(encoding="utf-8")
        for name in ("llps.yaml", "dpr.yaml")
    )

    assert "artifacts/derived/protein/" in config_text
    assert "artifacts/data/protein/" in config_text
    assert "artifacts/models/protein/" in config_text
    retired_namespace = "_".join(("full", "length"))
    assert "artifacts/derived/" + retired_namespace not in config_text
    assert "artifacts/data/" + retired_namespace not in config_text
    assert "artifacts/models/" + retired_namespace not in config_text


# Source: test_protein_result_artifacts.py


PROTEIN_RESULT_SOURCE_ARCHIVE_ROOT = "release-source"
PROTEIN_RESULT_FILES = {
    "artifacts/results/protein/ablation/dpr/dpr_finalchain_corrected_topline.csv": "ablation/dpr/reports/dpr_finalchain_corrected_topline.csv",
    "artifacts/results/protein/ablation/dpr/dpr_stream_ablation_summary.csv": "ablation/dpr/benchmarks/final_tables/dpr_stream_ablation_summary.csv",
    "artifacts/results/protein/ablation/llps/llps_ablation_summary_table.csv": "ablation/llps/nature_main/reports/llps_ablation_summary_table.csv",
    "artifacts/results/protein/ablation/llps/llps_embedding_ablation_bootstrap_ci.csv": "ablation/llps/nature_main/reports/llps_embedding_ablation_bootstrap_ci.csv",
    "artifacts/results/protein/benchmark/dpr/dpr_phasepro_region_metrics.csv": "benchmarks/dpr/dpr_phasepro_region_metrics.csv",
    "artifacts/results/protein/benchmark/dpr/dpr_phasepro_residue_metrics.csv": "benchmarks/dpr/dpr_phasepro_residue_metrics.csv",
    "artifacts/results/protein/benchmark/dpr/dpr_requested_table.csv": "benchmarks/dpr/dpr_requested_table.csv",
    "artifacts/results/protein/benchmark/dpr/final_overall_benchmark_summary.json": "benchmarks/dpr/final_overall_benchmark_summary.json",
    "artifacts/results/protein/benchmark/llps/llps_all_panel_metrics.csv": "benchmarks/llps/llps_all_panel_metrics.csv",
    "artifacts/results/protein/benchmark/llps/llps_requested_table.csv": "benchmarks/llps/llps_requested_table.csv",
    "artifacts/results/protein/figures/ablation/fig05_llps_key_metrics.png": "paper/protein/figures/ablation/fig05_llps_key_metrics.png",
    "artifacts/results/protein/figures/ablation/fig06_dpr_ablation_summary.png": "paper/protein/figures/ablation/fig06_dpr_ablation_summary.png",
    "artifacts/results/protein/figures/ablation/fig06_dpr_stream_ablation.png": "paper/protein/figures/ablation/fig06_dpr_stream_ablation.png",
    "artifacts/results/protein/figures/benchmark/fig01_dpr_benchmark.png": "paper/protein/figures/benchmark/fig01_dpr_benchmark.png",
    "artifacts/results/protein/figures/benchmark/fig01_llps_benchmark.png": "paper/protein/figures/benchmark/fig01_llps_benchmark.png",
    "artifacts/results/protein/figures/benchmark/fig02_phasepro_dpr_top12.png": "paper/protein/figures/benchmark/fig02_phasepro_dpr_top12.png",
}


def test_protein_result_artifacts_are_curated_and_integrity_checked() -> None:
    results_root = ROOT / "artifacts" / "results" / "protein"
    manifest_path = results_root / "manifest.json"
    readme_path = results_root / "README.md"

    assert readme_path.is_file()
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "phaseflow_protein_results_v1"
    assert manifest["source_archive_root"] == PROTEIN_RESULT_SOURCE_ARCHIVE_ROOT
    entries = manifest["files"]
    destinations = [entry["destination"] for entry in entries]
    assert destinations == sorted(destinations)
    assert {entry["destination"] for entry in entries} == set(PROTEIN_RESULT_FILES)
    assert {entry["destination"]: entry["source"] for entry in entries} == PROTEIN_RESULT_FILES
    assert all(not entry["source"].startswith("outputs/") for entry in entries)
    assert all(not Path(entry["source"]).is_absolute() for entry in entries)

    published_files = {
        (Path("artifacts/results/protein") / path.relative_to(results_root)).as_posix()
        for path in results_root.rglob("*")
            if path.is_file()
            and path.name not in {"README.md", "manifest.json"}
            and "scripts" not in path.relative_to(results_root).parts
    }
    assert published_files == set(PROTEIN_RESULT_FILES)

    forbidden_suffixes = {".pt", ".ckpt", ".h5", ".npz", ".log"}
    for path in results_root.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix not in forbidden_suffixes, path
        assert path.stat().st_size <= 25 * 1024 * 1024, path

    scripts_root = results_root / "scripts"
    assert all(path.name != "__pycache__" for path in scripts_root.rglob("*"))
    assert all(path.suffix != ".pyc" for path in scripts_root.rglob("*") if path.is_file())

    for entry in entries:
        result_path = ROOT / entry["destination"]
        payload = result_path.read_bytes()
        assert entry["bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()

    payload_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in results_root.rglob("*")
        if path.is_file() and path.suffix in {".csv", ".json"}
    )
    for forbidden in (
        "/data/mogoo7zn",
        ".pt",
        ".ckpt",
        ".npz",
        ".h5",
        "CondenGT",
        "PhaseGT",
        "condengt",
        "phasegt",
        "full" + "_length",
    ):
        assert forbidden not in payload_text

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--output runs/protein/idr_phaseflow_profiles.jsonl" in root_readme
    assert "--csv runs/protein/idr_phaseflow_profiles.csv" in root_readme
    assert "--output artifacts/results/protein/idr_phaseflow_profiles.jsonl" not in root_readme
    assert "--csv artifacts/results/protein/idr_phaseflow_profiles.csv" not in root_readme



# Source: test_rebuild_pipeline.py


from scripts.protein.workflows.release import RebuildPlan


def test_rebuild_plan_has_all_required_stages(tmp_path) -> None:
    plan = RebuildPlan.from_roots(
        data_root=tmp_path / "data",
        work_root=tmp_path / "work",
        output_root=tmp_path / "out",
    )

    assert plan.stage_names() == (
        "validate",
        "features",
        "llps-inputs",
        "llps",
        "llps-hidden",
        "dpr-inputs",
        "dpr",
        "refinement",
        "evaluate",
    )
    assert plan.as_dict()["paths"]["data_root"] == str(tmp_path / "data")
    assert plan.as_dict()["outputs"]["dpr_packed_sidecar"] == str(tmp_path / "work" / "dpr" / "packed")
    assert plan.as_dict()["contracts"]["dpr_packed_hidden_key"] == "phaseflow_llps_hidden"



# Source: test_release_layout.py


from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.protein.workflows.training import configure_schedule_paths, resolve_updates


ROOT = Path(__file__).resolve().parents[2]


def test_protein_release_uses_canonical_config_and_training_names() -> None:
    config_dir = ROOT / "configs" / "protein"
    script_dir = ROOT / "scripts" / "protein"

    assert (config_dir / "llps.yaml").is_file()
    assert (config_dir / "dpr.yaml").is_file()
    assert not (config_dir / "final_llps.yaml").exists()
    assert not (config_dir / "final_dpr.yaml").exists()
    assert (script_dir / "run.py").is_file()


def test_protein_configs_do_not_embed_local_artifact_paths() -> None:
    config_dir = ROOT / "configs" / "protein"
    text = "\n".join(
        (config_dir / name).read_text(encoding="utf-8")
        for name in ("llps.yaml", "dpr.yaml")
    )

    forbidden = ("/data/mogoo7zn/", "external_artifacts")
    for token in forbidden:
        assert token not in text


def test_public_protein_docs_do_not_expose_local_release_names() -> None:
    paths = (
        ROOT / "README.md",
        ROOT / "configs" / "protein" / "README.md",
        ROOT / "docs" / "protein" / "README.md",
        ROOT / "docs" / "protein" / "artifact_policy.md",
        ROOT / "scripts" / "protein" / "README.md",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "configs/protein/final_" not in text
    assert "docs/protein/final/" not in text
    assert "training/run_dpr_v6.py" not in text
    assert "Git LFS" not in text


def test_dpr_config_controls_default_update_count_and_runtime_schedule() -> None:
    config_path = ROOT / "configs" / "protein" / "dpr.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(updates=None, start_update=1, end_update=None)

    updates, end_update = resolve_updates(args, config)

    assert config["run"]["seed"] == 20260616
    assert config["input_representation"] == {
        "policy": "historical_cached_hidden",
        "packed_sidecar_key": "phaseflow_llps_hidden",
    }
    assert config["loss"]["objective"] == "mflat"
    assert config["optimizer"]["lr"] == 3.0e-4
    assert config["model"]["leaky_relu_slope"] == 0.01
    assert updates == config["scheduler"]["total_updates"] == 4000
    assert end_update == 4000
    frozen_schedule = "artifacts/data/protein/PhaseFlow-DPR/data/base_training_schedule.parquet"
    assert config["paths"]["schedule_current"] == frozen_schedule
    configure_schedule_paths(
        config,
        arm="dpr",
        updates=updates,
        schedule_seed=config["run"]["seed"],
        world_size=config["run"]["world_size"],
    )
    assert config["paths"]["schedule_current"] == frozen_schedule



# Source: test_release_paths.py


from scripts.protein.workflows.release import ReleasePaths


def test_release_paths_are_derived_only_from_explicit_roots(tmp_path) -> None:
    paths = ReleasePaths.from_roots(
        data_root=tmp_path / "data",
        work_root=tmp_path / "work",
        output_root=tmp_path / "out",
    )

    assert paths.llps_raw_root == tmp_path / "data" / "PhaseFlow-LLPS"
    assert paths.dpr_raw_root == tmp_path / "data" / "PhaseFlow-DPR"
    assert paths.llps_cache_root == tmp_path / "work" / "llps"
    assert paths.dpr_cache_root == tmp_path / "work" / "dpr"
    assert paths.dpr_packed_root == tmp_path / "work" / "dpr" / "packed"
    assert paths.run_root == tmp_path / "out"



# Source: test_figure_script_interfaces.py


import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PUBLICATION_RESULTS = ROOT / "artifacts/results/protein"
FIGURES = PUBLICATION_RESULTS / "scripts/figures"


def run(script: str, *arguments: str) -> None:
    result = subprocess.run([sys.executable, str(FIGURES / script), *arguments], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def assert_assets(directory: Path, stem: str) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        assert (directory / f"{stem}{suffix}").is_file()
    assert (directory / f"{stem}_plot_data.csv").is_file()


def test_independent_protein_figures_export_auditable_assets(tmp_path: Path) -> None:
    bootstrap = tmp_path / "llps_bootstrap.csv"
    arms = ("no_starling_reference", "no_disorder_no_starling", "no_physchem_no_starling", "no_protenix_no_starling", "no_esm2_no_starling", "all5_full_control", "no_pseudo_no_starling")
    records = []
    for index, arm in enumerate(arms):
        for metric, base in (("auprc", 0.72), ("auroc", 0.86), ("mcc_at_0.5", 0.52)):
            value = base - index * 0.01
            records.append({"arm_id": arm, "label": arm, "metric": metric, "value": value, "ci_low": value - 0.01, "ci_high": value + 0.01})
    pd.DataFrame(records).to_csv(bootstrap, index=False)
    output = tmp_path / "llps"
    run("plot_llps_embedding_ablation.py", "--bootstrap-input", str(bootstrap), "--output-dir", str(output))
    assert_assets(output, "llps_embedding_ablation")

    metrics = tmp_path / "dpr_metrics.csv"
    all_arms = ("s1111_full_retrain", "s1011_no_biophys", "s0111_no_esm2", "s1110_no_phaseflow_bridge", "s1001_esm2_phaseflow_bridge", "s0101_biophys_phaseflow_bridge", "s0011_biophys_esm2", "s0000_null_negative_control")
    data = []
    for index, arm in enumerate(all_arms):
        row = {"arm_id": arm, "phasepro_region_p257_segment_f1_iou_0_25": 0.59 - index * 0.01}
        for prefix, values in (("phasepro_official_all_scales_global_residue_AUPRC_scale-", (0.69, 0.70, 0.71, 0.70)), ("phasepro_official_all_scales_per_protein_Spearman_median_scale-", (0.49, 0.50, 0.51, 0.50))):
            for suffix, value in zip(("p33", "p129", "p257", "mean"), values, strict=True):
                row[prefix + suffix] = value - index * 0.005
        for suffix, value in zip(("p33", "p129", "p257", "mean"), (0.69, 0.70, 0.71, 0.70), strict=True):
            row[f"phasepro_{suffix}_global_residue_AUPRC"] = value - index * 0.005
        data.append(row)
    pd.DataFrame(data).to_csv(metrics, index=False)
    output = tmp_path / "dpr_summary"
    run("plot_dpr_ablation_summary.py", "--input", str(metrics), "--output-dir", str(output))
    assert_assets(output, "dpr_ablation_summary")
    for script, stem in (("plot_dpr_phaseflow_bridge_ablation.py", "dpr_phaseflow_bridge_ablation"), ("plot_dpr_stream_ablation.py", "dpr_stream_ablation")):
        output = tmp_path / stem
        run(script, "--input", str(metrics), "--output-dir", str(output))
        assert_assets(output, stem)

    benchmark = tmp_path / "benchmark.csv"
    pd.DataFrame({"model": ["PSTP", "PSPHunter", "catGRANULE", "PhaseFlow"], "auroc": [0.67, 0.51, 0.43, 0.67], "auprc": [0.71, 0.54, 0.48, 0.71], "iou_at_0_25": [0.54, 0.08, 0.46, 0.61], "precision": [0.60, 0.06, 0.38, 0.64], "f1": [0.63, 0.27, 0.60, 0.67], "recall": [0.50, 0.13, 0.57, 0.58], "ppmc_auprc": [0.67, 0.68, 0.63, 0.75], "mcc": [0.43, 0.40, 0.48, 0.55], "nd_auprc": [0.71, 0.73, 0.66, 0.77], "recall_at_fpr_5": [0.44, 0.44, 0.35, 0.54]}).to_csv(benchmark, index=False)
    for script, stem in (("plot_llps_benchmark.py", "llps_benchmark"), ("plot_dpr_benchmark.py", "dpr_benchmark")):
        output = tmp_path / stem
        run(script, "--input", str(benchmark), "--output-dir", str(output))
        assert_assets(output, stem)

    profiles, proteins, regions, per_protein = (tmp_path / name for name in ("profiles.npz", "proteins.csv", "regions.csv", "per_protein.csv"))
    np.savez(profiles, P00001=np.array([0.1, 0.7, 0.9, 0.2]))
    pd.DataFrame({"protein_id": ["P00001"], "gene_name": ["GENE1"]}).to_csv(proteins, index=False)
    pd.DataFrame({"protein_id": ["P00001"], "start_0based": [1], "end_exclusive": [3]}).to_csv(regions, index=False)
    pd.DataFrame({"protein_id": ["P00001"], "length": [4], "positive_count": [2], "auprc": [0.8], "auroc": [0.9], "spearman": [0.7]}).to_csv(per_protein, index=False)
    output = tmp_path / "top12"
    run("plot_phasepro_dpr_top12.py", "--profiles", str(profiles), "--proteins", str(proteins), "--regions", str(regions), "--per-protein", str(per_protein), "--output-dir", str(output))
    assert_assets(output, "phasepro_dpr_top12")


def test_released_result_scripts_render_from_adjacent_default_inputs(tmp_path: Path) -> None:
    for script, stem in (
        ("plot_llps_benchmark.py", "llps_benchmark"),
        ("plot_dpr_benchmark.py", "dpr_benchmark"),
        ("plot_llps_embedding_ablation.py", "llps_embedding_ablation"),
        ("plot_dpr_phaseflow_bridge_ablation.py", "dpr_phaseflow_bridge_ablation"),
        ("plot_dpr_stream_ablation.py", "dpr_stream_ablation"),
    ):
        output = tmp_path / stem
        run(script, "--output-dir", str(output))
        assert_assets(output, stem)



# Source: test_independent_figure_scripts.py


import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


CANONICAL_PROTEIN_FIGURES = {
    "plot_llps_embedding_ablation.py",
    "plot_dpr_ablation_summary.py",
    "plot_dpr_phaseflow_bridge_ablation.py",
    "plot_dpr_stream_ablation.py",
    "plot_llps_benchmark.py",
    "plot_dpr_benchmark.py",
    "plot_phasepro_dpr_top12.py",
}


def test_every_final_protein_figure_has_its_own_script() -> None:
    scripts = FIGURES
    assert {path.name for path in scripts.glob("plot_*.py")} >= CANONICAL_PROTEIN_FIGURES
    assert not (scripts / "plot_llps_ablation.py").exists()
    assert not (scripts / "plot_dpr_ablation.py").exists()
    assert not (scripts / "plot_dpr_examples.py").exists()


def test_every_final_protein_figure_exposes_a_command_line_interface() -> None:
    for name in sorted(CANONICAL_PROTEIN_FIGURES):
        result = subprocess.run(
            [sys.executable, str(FIGURES / name), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
