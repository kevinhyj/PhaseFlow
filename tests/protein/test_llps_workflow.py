"""Workflow-level regression checks for the protein release."""



# Source: test_historical_llps_compat.py


import torch

from scripts.protein.workflows.training import ExponentialMovingAverage
from scripts.protein.workflows.training import _restore_optimizer_state


def test_ema_accepts_a_historical_checkpoint_with_null_ema_state() -> None:
    model = torch.nn.Linear(3, 2)
    ema = ExponentialMovingAverage(model, decay=0.995, update_after_steps=50)
    original = ema.model_state_dict()

    ema.load_checkpoint_state(None, model)

    restored = ema.model_state_dict()
    assert restored.keys() == original.keys()
    assert all(torch.equal(restored[name], original[name]) for name in original)


def test_non_strict_resume_skips_an_incompatible_optimizer_state() -> None:
    source = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    source_optimizer = torch.optim.AdamW(
        [
            {"params": source[0].parameters(), "lr": 1e-4},
            {"params": source[1].parameters(), "lr": 1e-5},
        ]
    )
    target_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-4)

    restored = _restore_optimizer_state(
        target_optimizer,
        source_optimizer.state_dict(),
        strict_resume=False,
    )

    assert restored is False



# Source: test_historical_ppmc.py


import hashlib
import json
from pathlib import Path

import h5py
import pandas as pd
import pytest

from scripts.protein.workflows import evaluation as benchmark
from scripts.protein.workflows.release import (
    materialize_h5_locked_manifest,
    score_llps_panel,
    validate_h5_locked_manifest,
)
from scripts.protein.workflows.evaluation import _checkpoint_edge_attr_dim


def test_llps_inference_uses_the_checkpoint_graph_edge_dimension() -> None:
    assert _checkpoint_edge_attr_dim({"model": {"graph_transformer": {"edge_dim": 32}}}) == 32


def _write_feature(path, protein_id: str, sequence: str) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["protein_id"] = protein_id
        handle.attrs["sequence"] = sequence
        handle.attrs["length"] = len(sequence)


def test_materialize_h5_locked_manifest_replaces_drifted_sequence(tmp_path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _write_feature(feature_dir / "P1.h5", "P1", "ACDE")
    _write_feature(feature_dir / "P2.h5", "P2", "FGHIJ")
    source = tmp_path / "source.csv"
    pd.DataFrame(
        [
            {"protein_id": "P1", "sequence": "XXXX", "length": 4, "llps_label": 1},
            {"protein_id": "P2", "sequence": "FGHIJ", "length": 5, "llps_label": 0},
        ]
    ).to_csv(source, index=False)
    output = tmp_path / "locked.csv"

    report = materialize_h5_locked_manifest(source, feature_dir, output)
    locked = pd.read_csv(output, dtype={"protein_id": str})

    assert report["records"] == 2
    assert report["source_sequence_mismatches"] == 1
    assert locked.loc[locked["protein_id"].eq("P1"), "sequence"].item() == "ACDE"
    assert locked.loc[locked["protein_id"].eq("P1"), "length"].item() == 4
    assert locked.loc[locked["protein_id"].eq("P1"), "sequence_sha256"].item() == hashlib.sha256(
        b"ACDE"
    ).hexdigest()
    assert validate_h5_locked_manifest(output, feature_dir)["records"] == 2


def test_validator_rejects_a_manifest_that_does_not_match_h5_sequence(tmp_path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _write_feature(feature_dir / "P1.h5", "P1", "ACDE")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{"protein_id": "P1", "sequence": "XXXX", "length": 4, "llps_label": 1}]
    ).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="sequence mismatch"):
        validate_h5_locked_manifest(manifest, feature_dir)


def test_score_llps_panel_uses_historical_id_join_and_threshold() -> None:
    predictions = pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3", "P4"],
            "region_global_llps_score": [0.6, 0.7, 0.8, 0.1],
        }
    )
    panels = pd.DataFrame(
        {
            "panel_id": ["official", "official", "official", "official"],
            "protein_id": ["P1", "P2", "P3", "P4"],
            "llps_label": [1, 0, 1, 0],
        }
    )

    metrics = score_llps_panel(
        predictions,
        panels,
        panel_id="official",
        score_column="region_global_llps_score",
    )

    assert metrics["n"] == 4
    assert metrics["positive_n"] == 2
    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["auprc"] == pytest.approx(5.0 / 6.0)
    assert metrics["mcc_at_0.5"] == pytest.approx(1.0 / 3.0**0.5)
    assert metrics["f1_at_0.5"] == pytest.approx(0.8)


def test_llps_reference_metrics_reject_values_outside_the_published_tolerance() -> None:
    reference = {"auroc": 0.8742824045886142, "auprc": 0.7521831456622791}
    observed = {"auroc": 0.8742745986992265, "auprc": 0.7521890622737311}

    with pytest.raises(ValueError, match="outside the published tolerance"):
        benchmark._validate_llps_reference_metrics(observed, reference, tolerance=1.0e-6)

    benchmark._validate_llps_reference_metrics(observed, reference, tolerance=1.0e-5)


def test_llps_evaluator_infers_only_the_selected_frozen_panel(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"canonical-llps-checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = tmp_path / "llps.yaml"
    config.write_text(
        "reproduction:\n"
        "  benchmark:\n"
        "    ppmc:\n"
        f"      checkpoint_sha256: {checkpoint_hash}\n"
        "      precision: fp32\n"
        "      threshold: 0.5\n"
        "      panel_id: selected\n"
        "      metric_tolerance: 1.0e-5\n"
        "  reference_metrics:\n"
        "    ppmc:\n"
        "      auroc: 1.0\n"
        "      auprc: 1.0\n"
        "      mcc_at_0.5: 1.0\n"
        "      f1_at_0.5: 1.0\n",
        encoding="utf-8",
    )
    panel = tmp_path / "panels.csv"
    pd.DataFrame(
        [
            {"panel_id": "selected", "protein_id": "P2", "llps_label": 0},
            {"panel_id": "selected", "protein_id": "P1", "llps_label": 1},
            {"panel_id": "excluded", "protein_id": "X1", "llps_label": 1},
            {"panel_id": "excluded", "protein_id": "X2", "llps_label": 0},
        ]
    ).to_csv(panel, index=False)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _write_feature(feature_dir / "P1.h5", "P1", "ACDE")
    _write_feature(feature_dir / "P2.h5", "P2", "FGHIJ")
    observed_ids: list[str] = []
    observed_batch_sizes: list[int | None] = []

    def fake_inference(_checkpoint, _feature_dir, output_path, *, protein_ids, batch_size=None):
        observed_ids.extend(protein_ids)
        observed_batch_sizes.append(batch_size)
        scores = {
            "P1": {"protein": 0.1, "region": 0.9},
            "P2": {"protein": 0.9, "region": 0.1},
            "X1": {"protein": 0.2, "region": 0.8},
            "X2": {"protein": 0.8, "region": 0.2},
        }
        output_path.write_text(
            "".join(
                json.dumps(
                    {
                        "protein_id": protein_id,
                        "protein_llps_score": scores[protein_id]["protein"],
                        "region_global_llps_score": scores[protein_id]["region"],
                    }
                )
                + "\n"
                for protein_id in protein_ids
            ),
            encoding="utf-8",
        )

    from scripts.protein.workflows import evaluation as inference

    monkeypatch.setattr(inference, "run_inference", fake_inference)
    assert benchmark.evaluate_llps_main(
        [
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--feature-dir", str(feature_dir),
            "--panel", str(panel),
            "--output-root", str(tmp_path / "results"),
        ]
    ) == 0

    assert observed_ids == ["P1", "P2"]
    assert observed_batch_sizes == [8]
    predictions = pd.read_csv(tmp_path / "results" / "llps_predictions.csv")
    assert predictions["protein_id"].tolist() == ["P1", "P2"]
    summary = json.loads((tmp_path / "results" / "llps_summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["auroc"] == pytest.approx(1.0)
    assert summary["contract"]["metric_tolerance"] == pytest.approx(1.0e-5)
    assert summary["contract"]["reference_metrics"] == {
        "auroc": 1.0,
        "auprc": 1.0,
        "mcc_at_0.5": 1.0,
        "f1_at_0.5": 1.0,
    }


# Source: test_llps_input_compiler.py


import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from phaseflow.protein import PhaseFlowBatchPlanDataset
from phaseflow.protein import FeatureCacheWriter
from phaseflow.protein import FeatureCacheRecord
from scripts.protein.workflows.release import compile_llps_inputs


def test_llps_input_compiler_builds_a_batch_plan_dataset_from_public_tables_and_local_features(tmp_path: Path) -> None:
    release_root = tmp_path / "PhaseFlow-LLPS"
    plan_root = release_root / "data" / "training_plan"
    plan_root.mkdir(parents=True)
    sequence = "ACDE"
    sequence_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    pd.DataFrame(
        [{"protein_id": "P1", "sequence": sequence, "sequence_sha256": sequence_sha256, "sequence_length": len(sequence)}]
    ).to_parquet(release_root / "data" / "proteins.parquet", index=False)
    pd.DataFrame(
        [{"protein_id": "P1", "sequence_sha256": sequence_sha256, "dataset_index": 0, "llps_label": 1.0, "sample_weight": 1.0}]
    ).to_parquet(release_root / "data" / "training_units.parquet", index=False)
    pd.DataFrame(
        [
            {
                "epoch": 0,
                "global_step": 0,
                "local_rank": 0,
                "local_slot": 0,
                "dataset_index": 0,
                "plan_dataset_index": 0,
                "protein_id": "P1",
                "sequence_sha256": sequence_sha256,
                "length": len(sequence),
                "embedding_shard_id": "source",
                "pool_name": "P1_driver",
                "label_group": "positive",
                "tier": "positive",
                "source": "fixture",
            }
        ]
    ).to_parquet(plan_root / "batch_plan_epoch_000.parquet", index=False)

    feature_root = tmp_path / "features"
    _write_feature_cache(feature_root / "P1.h5", protein_id="P1", sequence=sequence)
    output_root = tmp_path / "derived"

    report = compile_llps_inputs(release_root=release_root, feature_root=feature_root, output_root=output_root)

    assert report == {"records": 1, "epochs": [0], "plans": 1}
    sample_index = output_root / "processed" / "tables" / "training_sample_index.parquet"
    training_config = yaml.safe_load((output_root / "llps.yaml").read_text(encoding="utf-8"))
    assert training_config["dataset"]["dataset_root"] == str(output_root / "processed")
    assert training_config["dataset"]["plan_dir"] == str(output_root / "training" / "plan")
    assert training_config["dataset"]["esm2_store_metadata"] is None
    dataset = PhaseFlowBatchPlanDataset(
        plan_dir=output_root / "training" / "plan",
        sample_index=sample_index,
        dataset_root=output_root / "processed",
        input_contract=output_root / "processed" / "configs" / "offline_input_contract.yaml",
        local_rank=0,
        rank=0,
        max_neighbors=96,
        edge_attr_dim=32,
    )

    batch = dataset[0]

    assert tuple(batch["plm"].shape) == (1, len(sequence), 1280)
    assert tuple(batch["edge_attr"].shape[-1:]) == (32,)
    assert batch["protein_ids"] == ["P1"]


def test_llps_input_compiler_cli_runs_from_a_source_checkout(tmp_path: Path) -> None:
    release_root, feature_root = _write_tiny_release_and_features(tmp_path)
    output_root = tmp_path / "derived"
    root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/protein/run.py",
            "compile-llps-inputs",
            "--release-root",
            str(release_root),
            "--feature-root",
            str(feature_root),
            "--output-root",
            str(output_root),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "processed" / "tables" / "training_sample_index.parquet").is_file()


def _write_feature_cache(path: Path, *, protein_id: str, sequence: str) -> None:
    length = len(sequence)
    record = FeatureCacheRecord(
        protein_id=protein_id,
        sequence=sequence,
        plm=np.ones((length, 1280), dtype=np.float32),
        physchem=np.ones((length, 90), dtype=np.float32),
        disorder=np.ones((length, 6), dtype=np.float32),
        protenix_embed=np.zeros((length, 512), dtype=np.float32),
        starling_embed=np.zeros((length, 512), dtype=np.float32),
        modality_mask=np.zeros((length, 5), dtype=np.float32),
        reliability=np.ones((length, 5), dtype=np.float32),
        edge_src=np.arange(length, dtype=np.int64),
        edge_dst=np.arange(length, dtype=np.int64),
        edge_type=np.zeros(length, dtype=np.int64),
        edge_attr=np.ones((length, 32), dtype=np.float32),
        y_llps=1.0,
    )
    FeatureCacheWriter.write_h5(path, record)


def _write_tiny_release_and_features(tmp_path: Path) -> tuple[Path, Path]:
    release_root = tmp_path / "PhaseFlow-LLPS"
    plan_root = release_root / "data" / "training_plan"
    plan_root.mkdir(parents=True)
    sequence = "ACDE"
    sequence_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    pd.DataFrame(
        [{"protein_id": "P1", "sequence": sequence, "sequence_sha256": sequence_sha256, "sequence_length": len(sequence)}]
    ).to_parquet(release_root / "data" / "proteins.parquet", index=False)
    pd.DataFrame(
        [{"protein_id": "P1", "sequence_sha256": sequence_sha256, "dataset_index": 0, "llps_label": 1.0, "sample_weight": 1.0}]
    ).to_parquet(release_root / "data" / "training_units.parquet", index=False)
    pd.DataFrame(
        [{"epoch": 0, "global_step": 0, "local_rank": 0, "local_slot": 0, "dataset_index": 0, "plan_dataset_index": 0, "protein_id": "P1", "sequence_sha256": sequence_sha256, "length": len(sequence), "embedding_shard_id": "source"}]
    ).to_parquet(plan_root / "batch_plan_epoch_000.parquet", index=False)
    feature_root = tmp_path / "features"
    _write_feature_cache(feature_root / "P1.h5", protein_id="P1", sequence=sequence)
    return release_root, feature_root



# Source: test_llps_package_validator.py

from pathlib import Path
import subprocess
import sys

import pandas as pd

from scripts.protein.workflows.release import build_manifest


def test_llps_manifest_records_validated_fixed_training_protocol(tmp_path: Path) -> None:
    package = tmp_path / "PhaseFlow-LLPS"
    data = package / "data"
    plan_dir = data / "training_plan"
    plan_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"protein_id": "p0", "sequence_sha256": "h0", "sequence": "ACDE", "sequence_length": 4}]
    ).to_parquet(data / "proteins.parquet", index=False)
    pd.DataFrame(
        [{"protein_id": "p0", "sequence_sha256": "h0", "llps_label": 1.0, "sample_weight": 1.0, "dataset_index": 3}]
    ).to_parquet(data / "training_units.parquet", index=False)
    pd.DataFrame(
        [
            {
                "epoch": 0,
                "global_step": 0,
                "local_rank": 0,
                "local_slot": 0,
                "dataset_index": 3,
                "plan_dataset_index": 0,
                "protein_id": "p0",
                "sequence_sha256": "h0",
            }
        ]
    ).to_parquet(plan_dir / "batch_plan_epoch_000.parquet", index=False)

    manifest = build_manifest("llps", package)

    assert manifest["training_protocol"] == {"records": 1, "epochs": [0]}


def test_llps_package_validator_cli_loads_the_local_package(tmp_path: Path) -> None:
    package = tmp_path / "PhaseFlow-LLPS"
    data = package / "data"
    plan_dir = data / "training_plan"
    plan_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"protein_id": "p0", "sequence_sha256": "h0", "sequence": "ACDE", "sequence_length": 4}]
    ).to_parquet(data / "proteins.parquet", index=False)
    pd.DataFrame(
        [{"protein_id": "p0", "sequence_sha256": "h0", "llps_label": 1.0, "sample_weight": 1.0, "dataset_index": 3}]
    ).to_parquet(data / "training_units.parquet", index=False)
    pd.DataFrame(
        [{"epoch": 0, "global_step": 0, "local_rank": 0, "local_slot": 0, "dataset_index": 3, "plan_dataset_index": 0, "protein_id": "p0", "sequence_sha256": "h0"}]
    ).to_parquet(plan_dir / "batch_plan_epoch_000.parquet", index=False)

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "scripts/protein/run.py", "validate-data", "--task", "llps", "--package-root", str(package), "--output", str(tmp_path / "manifest.json")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr



# Source: test_llps_training_protocol.py

import pandas as pd
import pytest

from scripts.protein.workflows import release


def test_llps_training_protocol_requires_exact_public_sample_identity(tmp_path) -> None:
    units = pd.DataFrame(
        [
            {"protein_id": "p0", "sequence_sha256": "h0", "dataset_index": 10},
            {"protein_id": "p1", "sequence_sha256": "h1", "dataset_index": 11},
        ]
    )
    plan = pd.DataFrame(
        [
            {
                "epoch": 0,
                "global_step": 0,
                "local_rank": 0,
                "local_slot": 0,
                "dataset_index": 10,
                "plan_dataset_index": 0,
                "protein_id": "p0",
                "sequence_sha256": "h0",
            }
        ]
    )

    report = release.validate_llps_training_protocol(units, [plan], world_size=1, batch_size=1)
    assert report.records == 1
    assert report.epochs == (0,)

    plan.loc[0, "sequence_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        release.validate_llps_training_protocol(units, [plan], world_size=1, batch_size=1)



# Source: test_imports.py

def test_imports() -> None:
    import phaseflow
    from phaseflow.protein import PhaseFlowModel

    assert phaseflow.__version__
    assert PhaseFlowModel is not None
