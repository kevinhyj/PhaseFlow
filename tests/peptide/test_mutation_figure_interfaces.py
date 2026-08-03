
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_mutation_scripts_have_portable_interfaces() -> None:
    scripts = (
        "plot_mutation_scatter.py",
        "plot_mutation_metrics.py",
        "plot_multi_mutation_dose.py",
    )
    forbidden = ("/data/", "outputs/overall/final")
    for name in scripts:
        path = ROOT / "scripts/peptide/figures/mutation" / name
        text = path.read_text(encoding="utf-8")
        assert "--output-dir" in text
        assert "--input" in text
        assert not any(token in text for token in forbidden)
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_multi_mutation_dose_figure_exports_publication_assets(tmp_path: Path) -> None:
    source = tmp_path / "mutation_scores.csv"
    pd.DataFrame(
        {
            "mutation": [
                "WT",
                "W334G",
                "W385G",
                "W412G",
                "W334G/W385G",
                "W334G/W412G",
                "W385G/W412G",
                "W334G/W385G/W412G",
            ],
            "mutation_count": [0, 1, 1, 1, 2, 2, 2, 3],
            "official_score": [0.712, 0.704, 0.706, 0.702, 0.696, 0.698, 0.694, 0.687],
        }
    ).to_csv(source, index=False)
    output_dir = tmp_path / "figures"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/peptide/figures/mutation/plot_multi_mutation_dose.py",
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    plot_data = pd.read_csv(output_dir / "multi_mutation_dose_plot_data.csv")
    assert plot_data["mutation_count"].tolist() == [0, 1, 2, 3]
    assert plot_data.loc[1, "n"] == 3
    for suffix in (".svg", ".pdf", ".png"):
        assert (output_dir / f"multi_mutation_dose{suffix}").is_file()


def test_mutation_metrics_figure_exports_ranked_plot_data(tmp_path: Path) -> None:
    source = tmp_path / "mutation_metrics.csv"
    pd.DataFrame(
        {
            "model": ["baseline", "PhaseFlow", "comparator"],
            "mean_auroc": [0.52, 0.61, 0.57],
            "mean_auprc": [0.41, 0.48, 0.45],
        }
    ).to_csv(source, index=False)
    output_dir = tmp_path / "figures"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/peptide/figures/mutation/plot_mutation_metrics.py",
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    plot_data = pd.read_csv(output_dir / "mutation_metrics_plot_data.csv")
    assert plot_data.loc[0, "auroc_model"] == "PhaseFlow"
    assert plot_data.loc[0, "auprc_model"] == "PhaseFlow"
    for suffix in (".svg", ".pdf", ".png"):
        assert (output_dir / f"mutation_metrics{suffix}").is_file()


def test_mutation_scatter_figure_exports_model_statistics(tmp_path: Path) -> None:
    source = tmp_path / "mutation_effects.csv"
    pd.DataFrame(
        {
            "experimental_effect": [-1.0, -0.3, 0.4, 1.0],
            "phaseflow_score": [-0.8, -0.1, 0.3, 0.9],
            "psphunter_effect": [0.7, 0.2, -0.2, -0.8],
            "phasemotif_effect": [0.6, 0.1, -0.1, -0.7],
            "pstp_scan_score_mean_effect": [-0.3, -0.1, 0.2, 0.4],
            "deephase_effect": [-0.5, -0.2, 0.2, 0.6],
            "pspredictor_effect": [0.4, 0.1, -0.2, -0.5],
        }
    ).to_csv(source, index=False)
    output_dir = tmp_path / "figures"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/peptide/figures/mutation/plot_mutation_scatter.py",
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    plot_data = pd.read_csv(output_dir / "mutation_scatter_plot_data.csv")
    assert plot_data["model"].tolist() == ["PhaseFlow", "PSPHunter", "PhaseMotif", "PSTP", "DeePhase", "PSPredictor"]
    assert plot_data.loc[0, "n"] == 4
    for suffix in (".svg", ".pdf", ".png"):
        assert (output_dir / f"mutation_scatter{suffix}").is_file()
