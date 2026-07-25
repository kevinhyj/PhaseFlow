from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def run_script(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_llps_ablation_exports_auditable_bootstrap_plot_data(tmp_path: Path) -> None:
    source = tmp_path / "llps_ablation.csv"
    pd.DataFrame(
        {
            "arm_id": ["no_starling_reference", "no_esm2_no_starling", "no_pseudo_no_starling"],
            "label": ["PhaseFlow", "w/o ESM2", "w/o pseudo"],
            "group": ["phaseflow", "wo", "pseudo"],
            "auprc_value": [0.72, 0.67, 0.61],
            "auprc_ci_low": [0.70, 0.65, 0.59],
            "auprc_ci_high": [0.74, 0.69, 0.63],
            "auroc_value": [0.88, 0.84, 0.80],
            "auroc_ci_low": [0.86, 0.82, 0.78],
            "auroc_ci_high": [0.90, 0.86, 0.82],
            "mcc_at_0.5_value": [0.55, 0.42, 0.31],
            "mcc_at_0.5_ci_low": [0.50, 0.37, 0.26],
            "mcc_at_0.5_ci_high": [0.60, 0.47, 0.36],
        }
    ).to_csv(source, index=False)
    output_dir = tmp_path / "figures"

    result = run_script(
        "scripts/full_length/figures/plot_llps_ablation.py",
        "--input",
        str(source),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    plot_data = pd.read_csv(output_dir / "llps_ablation_plot_data.csv")
    assert plot_data["arm_id"].tolist() == ["no_starling_reference", "no_esm2_no_starling", "no_pseudo_no_starling"]
    assert (output_dir / "llps_ablation.svg").is_file()


def test_dpr_ablation_exports_scale_ranges_from_public_summary(tmp_path: Path) -> None:
    source = tmp_path / "dpr_ablation.csv"
    pd.DataFrame(
        {
            "arm_id": ["s1111_full_retrain", "s1011_no_biophys", "s0111_no_esm2", "s1110_no_phaseflow_bridge"],
            "phasepro_official_all_scales_global_residue_AUPRC_scale-p33": [0.69, 0.65, 0.63, 0.60],
            "phasepro_official_all_scales_global_residue_AUPRC_scale-p129": [0.70, 0.66, 0.64, 0.61],
            "phasepro_official_all_scales_global_residue_AUPRC_scale-p257": [0.71, 0.67, 0.65, 0.62],
            "phasepro_official_all_scales_global_residue_AUPRC_scale-mean": [0.70, 0.66, 0.64, 0.61],
            "phasepro_region_p257_segment_f1_iou_0_25": [0.59, 0.56, 0.54, 0.50],
            "phasepro_official_all_scales_per_protein_Spearman_median_scale-p33": [0.49, 0.45, 0.43, 0.40],
            "phasepro_official_all_scales_per_protein_Spearman_median_scale-p129": [0.50, 0.46, 0.44, 0.41],
            "phasepro_official_all_scales_per_protein_Spearman_median_scale-p257": [0.51, 0.47, 0.45, 0.42],
            "phasepro_official_all_scales_per_protein_Spearman_median_scale-mean": [0.50, 0.46, 0.44, 0.41],
        }
    ).to_csv(source, index=False)
    output_dir = tmp_path / "figures"

    result = run_script(
        "scripts/full_length/figures/plot_dpr_ablation.py",
        "--input",
        str(source),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    plot_data = pd.read_csv(output_dir / "dpr_ablation_plot_data.csv")
    assert plot_data["arm_id"].tolist() == ["s1111_full_retrain", "s1011_no_biophys", "s0111_no_esm2", "s1110_no_phaseflow_bridge"]
    assert plot_data.loc[0, "auprc_scale_low"] == 0.69
    assert plot_data.loc[0, "auprc_scale_high"] == 0.71


def test_dpr_examples_accept_regions_and_export_selected_residue_data(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.npz"
    np.savez(profiles, P00001=np.array([0.1, 0.7, 0.9, 0.2]))
    proteins = tmp_path / "proteins.csv"
    pd.DataFrame({"protein_id": ["P00001"], "gene_name": ["GENE1"]}).to_csv(proteins, index=False)
    regions = tmp_path / "regions.csv"
    pd.DataFrame({"protein_id": ["P00001"], "start_0based": [1], "end_exclusive": [3]}).to_csv(regions, index=False)
    output_dir = tmp_path / "figures"

    result = run_script(
        "scripts/full_length/figures/plot_dpr_examples.py",
        "--profiles",
        str(profiles),
        "--proteins",
        str(proteins),
        "--regions",
        str(regions),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    plot_data = pd.read_csv(output_dir / "dpr_examples_plot_data.csv")
    assert plot_data[["residue_index_1based", "gold_dpr_label"]].values.tolist() == [[1, 0], [2, 1], [3, 1], [4, 0]]


def test_dpr_benchmark_accepts_a_single_metric(tmp_path: Path) -> None:
    source = tmp_path / "dpr_benchmark.csv"
    pd.DataFrame({"model": ["PhaseFlow", "baseline"], "auprc": [0.71, 0.54]}).to_csv(source, index=False)
    output_dir = tmp_path / "figures"

    result = run_script(
        "scripts/full_length/figures/plot_dpr_benchmark.py",
        "--input",
        str(source),
        "--output-dir",
        str(output_dir),
        "--metric",
        "auprc",
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "dpr_benchmark.svg").is_file()
