from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

fair_benchmark = pytest.importorskip("scripts.full_length.evaluation.run_pstp_fair_benchmark")
ALBATROSS_WEIGHTS = fair_benchmark.ALBATROSS_WEIGHTS
AA_ORDER = fair_benchmark.AA_ORDER
compute_fix_avgpool_scaler = fair_benchmark.compute_fix_avgpool_scaler


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "external_artifacts/pstp_official_benchmark_v1"
FEATURE_ROOT = OUT_ROOT / "reconstructed_features"


def test_nophasepro_weights_manifested_and_separated() -> None:
    manifest = pd.read_csv(OUT_ROOT / "manifests/nophasepro_weights.csv")
    assert len(manifest) == 60
    counts = manifest.groupby(["included_or_nophasepro", "family"]).size().to_dict()
    for variant in ("included", "nophasepro"):
        for family in ("SaPS", "PdPS", "Mix"):
            assert counts[(variant, family)] == 10
    included = manifest[manifest["included_or_nophasepro"].eq("included")]
    nophase = manifest[manifest["included_or_nophasepro"].eq("nophasepro")]
    assert included["relative_path"].str.contains("_nophasepro").sum() == 0
    assert nophase["relative_path"].str.contains("_nophasepro").all()
    assert nophase["SHA256"].is_unique


def test_pstp_weight_architecture_is_650_20_5_1() -> None:
    manifest = pd.read_csv(OUT_ROOT / "manifests/nophasepro_weights.csv")
    assert manifest["layer1_weight_shape"].eq("[20, 650]").all()
    assert manifest["layer2_weight_shape"].eq("[5, 20]").all()
    assert manifest["layer3_weight_shape"].eq("[1, 5]").all()


def test_albatross_hidden_contract_is_not_scalar_prediction() -> None:
    for name, path in ALBATROSS_WEIGHTS.items():
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert tuple(state["lstm.weight_ih_l0"].shape) == (220, 20), name
        assert tuple(state["fc.bias"].shape) == (1,), name
        hidden_per_direction = state["lstm.weight_ih_l0"].shape[0] // 4
        assert hidden_per_direction == 55
        assert hidden_per_direction * 2 == 110


def test_feature_channel_order_contract_constants() -> None:
    assert AA_ORDER == "ACDEFGHIKLMNPQRSTVWY"
    expected_scaler = [33 / 17, 33 / 18, 33 / 19, 33 / 20, 33 / 21]
    actual = compute_fix_avgpool_scaler(40, 33, 16)[:5]
    assert np.allclose(actual, expected_scaler)
    assert compute_fix_avgpool_scaler(10, 33, 16) == [33 / 10] * 10


def test_reconstructed_feature_manifest_when_present_is_650d() -> None:
    manifest_path = FEATURE_ROOT / "manifest.csv"
    if not manifest_path.exists():
        return
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 121
    assert manifest["sequence_sha256"].nunique() == 121
    assert manifest["merged_shape"].str.contains("650").all()
    validation = json.loads((FEATURE_ROOT / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["complete"] is True
    assert validation["evaluation_only"] is True
    assert validation["training_allowed"] is False
    assert validation["benchmark_labels_used_in_feature_generation"] is False
