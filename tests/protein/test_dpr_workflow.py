"""Workflow-level regression checks for the protein release."""



# Source: test_dpr.py


from collections import Counter

import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from phaseflow.protein import build_fixed_schedule, validate_schedule
from phaseflow.protein import (
    DPRV6Head,
    DPRV6LossConfig,
    bag_from_profiles,
    dpr_v6_loss,
    masked_avg_pool1d_same,
)
from phaseflow.protein import safe_spearman
from scripts.protein.workflows.training import reduce_rows


def test_masked_avg_pool_edges_short_and_padding() -> None:
    x = torch.arange(1, 6, dtype=torch.float32).view(1, 5, 1)
    mask = torch.tensor([[True, True, True, False, False]])
    out = masked_avg_pool1d_same(x, mask, kernel_size=5).squeeze(0).squeeze(-1)
    expected = torch.tensor([
        (1 + 2 + 3) / 3,
        (1 + 2 + 3) / 3,
        (1 + 2 + 3) / 3,
        0.0,
        0.0,
    ])
    assert torch.allclose(out, expected)


def test_bag_hard_is_mean_of_three_scale_maxima() -> None:
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    p33 = torch.tensor([[0.1, 0.7, 0.2, 0.0], [0.3, 0.4, 0.0, 0.0]])
    p129 = torch.tensor([[0.5, 0.2, 0.4, 0.0], [0.6, 0.2, 0.0, 0.0]])
    p257 = torch.tensor([[0.9, 0.1, 0.2, 0.0], [0.1, 0.8, 0.0, 0.0]])
    bag = bag_from_profiles([p33, p129, p257], mask)
    expected = torch.stack([torch.tensor([0.7, 0.4]), torch.tensor([0.5, 0.6]), torch.tensor([0.9, 0.8])]).mean(dim=0)
    assert torch.allclose(bag["bag_hard"], expected)


def test_tiny_head_no_bypass_and_gradients() -> None:
    torch.manual_seed(7)
    h = torch.randn(2, 8, 6, requires_grad=True)
    mask = torch.ones(2, 8, dtype=torch.bool)
    head = DPRV6Head(6, head_type="tiny")
    out = head(h, mask)
    recomputed = (out["p33"].max(dim=1).values + out["p129"].max(dim=1).values + out["p257"].max(dim=1).values) / 3.0
    assert torch.allclose(out["bag_hard"], recomputed)
    loss = out["bag_hard"].sum()
    loss.backward()
    assert h.grad is not None
    assert float(h.grad.abs().sum()) > 0.0


def test_constant_profile_loss_is_not_near_zero_and_mflat_active() -> None:
    mask = torch.ones(5, 10, dtype=torch.bool)
    z = torch.zeros(5, 10, requires_grad=True)
    p = torch.sigmoid(z)
    bag = bag_from_profiles([p, p, p], mask)
    out = {
        "z33": z,
        "p33": p,
        "bag_hard": bag["bag_hard"],
        "bag_topk": bag["bag_topk"],
        "seq_mask": mask,
    }
    batch = {
        "v3_tiers": ["S", "W", "M", "ND", "NP"],
        "seq_mask": mask,
        "residue_target": F.pad(torch.ones(5, 2), (0, 8)),
    }
    loss, parts = dpr_v6_loss(out, batch, cfg=DPRV6LossConfig(objective="mflat"))
    assert float(parts["L_bag_hard"].detach()) > 0.65
    assert float(parts["L_M_peak"].detach()) > 0.10
    assert float(loss.detach()) > 0.70
    loss.backward()
    assert z.grad is not None
    assert float(z.grad.abs().sum()) > 0.0


def test_strong_supervision_uses_safe_background_not_full_sequence_zero() -> None:
    z = torch.zeros(1, 80, requires_grad=True)
    p = torch.sigmoid(z)
    mask = torch.ones(1, 80, dtype=torch.bool)
    bag = bag_from_profiles([p, p, p], mask)
    target = torch.zeros(1, 80)
    target[:, 30:35] = 1.0
    out = {"z33": z, "p33": p, "bag_hard": bag["bag_hard"], "bag_topk": bag["bag_topk"], "seq_mask": mask}
    batch = {"v3_tiers": ["S"], "seq_mask": mask, "residue_target": target}
    loss, parts = dpr_v6_loss(out, batch, cfg=DPRV6LossConfig(objective="strong", bag=0.0, s_dice=0.0, s_rank=0.0))
    assert int(parts["active_S_bce"]) == 1
    loss.backward()
    grad = z.grad.detach().squeeze(0)
    ambiguous = torch.zeros(80, dtype=torch.bool)
    ambiguous[13:52] = True
    ambiguous[30:35] = False
    assert float(grad[ambiguous].abs().max()) == 0.0


def test_v6_schedule_two_step_composition_and_rotation() -> None:
    rows = []
    counts = {"S": 3, "W": 3, "M": 12, "ND": 4, "NP": 12}
    for tier, n in counts.items():
        for i in range(n):
            rows.append({"protein_id": f"{tier}_{i}", "sequence_sha256": f"hash_{tier}_{i}", "length": 100 + i, "v3_tier": tier})
    schedule = build_fixed_schedule(pd.DataFrame(rows), updates=100, seed=20260616)
    audit = validate_schedule(schedule, updates=100)
    assert audit["violation_count"] == 0
    even = Counter(schedule.loc[schedule["update"].eq(2), "v3_tier"])
    odd = Counter(schedule.loc[schedule["update"].eq(1), "v3_tier"])
    assert even == Counter({"S": 1, "W": 1, "M": 2, "ND": 1, "NP": 3})
    assert odd == Counter({"M": 4, "ND": 1, "NP": 3})
    assert all(len(set(schedule.loc[schedule["rank"].eq(rank), "v3_tier"])) > 1 for rank in range(8))


def test_reduce_rows_reports_tier_exposure() -> None:
    rows = [
        {"update": 1, "rank": 0, "tier": "S", "loss": 1.0},
        {"update": 1, "rank": 1, "tier": "NP", "loss": 3.0},
    ]
    reduced = reduce_rows(rows)
    assert reduced["tier_exposure"] == {"NP": 1, "S": 1}
    assert reduced["loss"] == 2.0


def test_fast_spearman_matches_scipy_with_ties() -> None:
    y = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.float32).numpy()
    score = torch.tensor([0.2, 0.9, 0.9, 0.1, 0.5, 0.5, 0.0, 0.7], dtype=torch.float32).numpy()
    expected = float(spearmanr(score, y).statistic)
    actual = safe_spearman(y, score)
    assert abs(actual - expected) < 1.0e-12



# Source: test_dpr_isolation.py


from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_canonical_dpr_files_do_not_reference_retired_head_resume_or_region_global_logits() -> None:
    paths = [
        ROOT / "phaseflow/protein/data.py",
        ROOT / "scripts/protein/workflows/training.py",
        ROOT / "scripts/protein/run.py",
        ROOT / "configs/protein/dpr.yaml",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = [
        "region_global_logits",
        "dpr_v5_state_dict",
        "load_dpr_v5",
    ]
    for token in forbidden:
        assert token not in text


def test_dpr_uses_separate_source_and_derived_artifact_paths() -> None:
    cfg = (ROOT / "configs/protein/dpr.yaml").read_text(encoding="utf-8")
    assert "artifacts/derived/protein/dpr" in cfg
    assert "artifacts/models/protein/llps" in cfg
    assert "external_artifacts" not in cfg
    assert ("outputs" + "/") not in cfg



# Source: test_dpr_offline_labels.py

import numpy as np

from phaseflow.protein import HalfOpenSpan, build_dpr_label_arrays


def test_half_open_span_end_boundary_is_end_minus_one() -> None:
    labels = build_dpr_label_arrays(
        length=32,
        spans=[HalfOpenSpan(start=10, end=20, confidence=1.0)],
        boundary_radius=0,
    )

    assert labels["residue_target"][9] == 0.0
    assert labels["residue_target"][10:20].tolist() == [1.0] * 10
    assert labels["residue_target"][20] == 0.0
    assert labels["residue_mask"][10:20].tolist() == [1.0] * 10
    assert labels["residue_mask"][20] == 0.0
    assert labels["start_target"][10] == 1.0
    assert labels["end_target"][19] == 1.0
    assert not np.isfinite(labels["end_target"][20])
    assert labels["boundary_weight"][10] == 1.0
    assert labels["boundary_weight"][19] == 1.0
    assert labels["boundary_weight"][20] == 0.0


def test_soft_boundary_does_not_shift_half_open_end() -> None:
    labels = build_dpr_label_arrays(
        length=32,
        spans=[HalfOpenSpan(start=10, end=20, confidence=0.75)],
        boundary_radius=2,
    )

    assert labels["end_target"][19] == 1.0
    assert labels["end_target"][20] < 1.0
    assert labels["end_target"][18] < 1.0
    assert labels["boundary_weight"][19] == 0.75
    assert labels["boundary_weight"][20] == 0.75



# Source: test_dpr_stage_checkpoint.py


import torch

from scripts.protein.workflows import training as dpr_stages


class _StageModel:
    def __init__(self) -> None:
        self.loaded_full_states: list[dict[str, torch.Tensor]] = []

    def load_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool) -> tuple[list[str], list[str]]:
        assert strict is True
        self.loaded_full_states.append(state)
        return [], []


def test_stage_checkpoint_restores_frozen_state_before_ema_head(monkeypatch, tmp_path) -> None:
    """A refinement stage must retain the previous frozen bridge, not re-randomize it."""

    checkpoint = tmp_path / "stage.pt"
    checkpoint.write_bytes(b"fixture")
    full_state = {
        "phase" + "gt.encoder.weight": torch.tensor([1.0]),
        "phaseflow_bridge.gate": torch.tensor([2.0]),
        "v6.projection.weight": torch.tensor([3.0]),
    }
    ema_state = {"projection.weight": torch.tensor([4.0])}
    monkeypatch.setattr(
        dpr_stages.torch,
        "load",
        lambda *args, **kwargs: {
            "format": "dpr_v6_checkpoint",
            "step": 75,
            "model_state_dict": full_state,
            "ema": {"shadow": ema_state},
        },
    )
    trainable_loaded: list[dict[str, torch.Tensor]] = []
    monkeypatch.setattr(
        dpr_stages,
        "load_trainable_dpr_state_dict",
        lambda model, state, *, strict: trainable_loaded.append(state),
    )
    model = _StageModel()

    summary = dpr_stages.load_dpr_stage_checkpoint(model, checkpoint, variant="ema")

    assert summary == {"format": "dpr_v6_checkpoint", "step": 75, "variant": "ema"}
    assert model.loaded_full_states == [
        {
            "llps_backbone.encoder.weight": torch.tensor([1.0]),
            "phaseflow_bridge.gate": torch.tensor([2.0]),
            "v6.projection.weight": torch.tensor([3.0]),
        }
    ]
    assert trainable_loaded == [ema_state]



# Source: test_dpr_surface.py

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_release_source_has_no_versioned_dpr_module_files() -> None:
    source_roots = (
        ROOT / "phaseflow" / "protein",
        ROOT / "scripts" / "protein",
        ROOT / "tests" / "protein",
    )

    assert not [path for root in source_roots for path in root.rglob("*dpr_v*.py")]



# Source: test_historical_dpr_input_contract.py


import pytest
import torch

from scripts.protein.workflows.training import require_dpr_input_representation
from scripts.protein.workflows.training import require_refine_input_representation


def test_historical_cached_hidden_requires_packed_hidden_tensor() -> None:
    cfg = {
        "input_representation": {
            "policy": "historical_cached_hidden",
            "packed_sidecar_key": "phaseflow_llps_hidden",
        }
    }

    with pytest.raises(RuntimeError, match="historical_cached_hidden requires packed phaseflow_llps_hidden"):
        require_dpr_input_representation(cfg, {"seq_mask": torch.ones(1, 3, dtype=torch.bool)})


def test_historical_cached_hidden_reports_primary_packed_key() -> None:
    cfg = {
        "input_representation": {
            "policy": "historical_cached_hidden",
            "packed_sidecar_key": "phaseflow_llps_hidden",
        }
    }
    batch = {
        "seq_mask": torch.ones(1, 3, dtype=torch.bool),
        "phaseflow_llps_hidden": torch.zeros(1, 3, 256),
    }

    assert require_dpr_input_representation(cfg, batch) == {
        "policy": "historical_cached_hidden",
        "resolved_key": "phaseflow_llps_hidden",
    }


def test_refinement_reuses_the_historical_cached_hidden_guard() -> None:
    cfg = {"input_representation": {"policy": "historical_cached_hidden"}}
    batch = {"seq_mask": torch.ones(1, 3, dtype=torch.bool)}

    with pytest.raises(RuntimeError, match="historical_cached_hidden requires packed phaseflow_llps_hidden"):
        require_refine_input_representation(cfg, batch)



# Source: test_packed_rebuild.py


import hashlib

import numpy as np
import torch

from phaseflow.protein import DPRV5BaseOnlySidecar, dpr_v5_collate
from phaseflow.protein import FeatureCacheWriter
from phaseflow.protein import zero_record
from phaseflow.protein import compute_biophys_node
from phaseflow.protein import compute_disorder_features
from phaseflow.protein import compute_physchem_features
from scripts.protein.workflows.release import (
    build_packed_sidecar,
    build_packed_sidecar_from_feature_cache,
    extract_llps_hidden,
    validate_packed_sidecar,
)


def _record(protein_id: str, sequence: str) -> dict[str, object]:
    length = len(sequence)
    return {
        "protein_id": protein_id,
        "sequence": sequence,
        "arrays": {
            "plm": np.zeros((length, 1280), dtype=np.float32),
            "biophys": np.zeros((length, 112), dtype=np.float32),
            "aa_ids": np.arange(length, dtype=np.int16),
            "modality_mask": np.zeros((length, 5), dtype=np.float32),
            "reliability": np.ones((length, 5), dtype=np.float32),
            "neighbors": np.zeros((length, 96), dtype=np.int64),
            "edge_attr": np.zeros((length, 96, 32), dtype=np.float32),
            "neighbor_mask": np.ones((length, 96), dtype=np.bool_),
            "neighbor_edge_type": np.zeros((length, 96), dtype=np.int64),
            "phaseflow_llps_hidden": np.ones((length, 256), dtype=np.float32),
            "residue_target": np.zeros(length, dtype=np.float32),
            "residue_mask": np.zeros(length, dtype=np.float32),
            "residue_weight": np.zeros(length, dtype=np.float32),
            "core_target": np.zeros(length, dtype=np.float32),
            "core_mask": np.zeros(length, dtype=np.float32),
            "start_target": np.zeros(length, dtype=np.float32),
            "end_target": np.zeros(length, dtype=np.float32),
            "boundary_weight": np.zeros(length, dtype=np.float32),
            "safe_background_mask": np.zeros(length, dtype=np.float32),
            "ignore_mask": np.ones(length, dtype=np.float32),
        },
    }


def test_packed_rebuild_writes_phaseflow_native_hidden_key_and_is_readable(tmp_path) -> None:
    records = [_record("P1", "ACD"), _record("P2", "EF")]

    report = build_packed_sidecar(records=records, output_root=tmp_path)

    assert report.hidden_key == "phaseflow_llps_hidden"
    assert report.records == 2
    assert report.residues == 5
    assert not (tmp_path / "shard_00000" / "phasegt_hidden.npy").exists()
    manifest = report.manifest
    assert manifest.loc[manifest["protein_id"].eq("P1"), "sequence_sha256"].item() == hashlib.sha256(
        b"ACD"
    ).hexdigest()
    assert manifest["array_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()

    sidecar = DPRV5BaseOnlySidecar(v2_data_root=tmp_path, packed_root=tmp_path)
    sample = sidecar.sample_from_tier_row(
        type("Tier", (), {"protein_id": "P1", "sequence_sha256": manifest.iloc[0].sequence_sha256, "v3_tier": "S"})()
    )
    assert tuple(sample["phaseflow_llps_hidden"].shape) == (3, 256)
    assert tuple(dpr_v5_collate([sample])["biophys"].shape) == (1, 3, 112)
    assert validate_packed_sidecar(tmp_path) == {"records": 2, "residues": 5}


def test_packed_rebuild_rejects_mismatched_record_hash(tmp_path) -> None:
    record = _record("P1", "ACD")
    record["sequence_sha256"] = "0" * 64

    try:
        build_packed_sidecar(records=[record], output_root=tmp_path)
    except ValueError as exc:
        assert "sequence_sha256" in str(exc)
    else:
        raise AssertionError("expected sequence identity validation to fail")


def test_packed_sidecar_validation_rejects_tampered_array(tmp_path) -> None:
    build_packed_sidecar(records=[_record("P1", "ACD")], output_root=tmp_path)
    hidden_path = tmp_path / "shard_00000" / "phaseflow_llps_hidden.npy"
    hidden_path.write_bytes(hidden_path.read_bytes() + b"tampered")

    try:
        validate_packed_sidecar(tmp_path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("expected sidecar hash validation to fail")


def test_packed_rebuild_rejects_arrays_that_cannot_feed_dpr(tmp_path) -> None:
    record = _record("P1", "ACD")
    record["arrays"]["biophys"] = np.zeros((3, 96), dtype=np.float32)

    try:
        build_packed_sidecar(records=[record], output_root=tmp_path)
    except ValueError as exc:
        assert "biophys" in str(exc)
    else:
        raise AssertionError("expected DPR shape validation to fail")


def test_feature_cache_rebuilds_dpr_runtime_arrays_and_llps_hidden(tmp_path) -> None:
    feature_dir = tmp_path / "features"
    record = zero_record(
        protein_id="P1",
        sequence="ACD",
        plm_dim=1280,
        phys_dim=90,
        disorder_dim=6,
        protenix_dim=512,
        starling_dim=512,
        edge_dim=32,
    )
    record.graph_neighbors = np.zeros((3, 96), dtype=np.int64)
    record.graph_edge_attr = np.zeros((3, 96, 32), dtype=np.float32)
    record.graph_neighbor_mask = np.ones((3, 96), dtype=np.bool_)
    record.physchem, _ = compute_physchem_features(record.sequence)
    record.disorder, _, _, _ = compute_disorder_features(record.sequence, mode="simple")
    FeatureCacheWriter.write_h5(feature_dir / "P1.h5", record)

    report = build_packed_sidecar_from_feature_cache(
        feature_dir=feature_dir,
        output_root=tmp_path / "packed",
        llps_hidden_provider=lambda batch: torch.full((1, 3, 256), 2.0),
    )

    sidecar = DPRV5BaseOnlySidecar(v2_data_root=tmp_path, packed_root=report.output_root)
    sample = sidecar.sample_from_tier_row(
        type("Tier", (), {"protein_id": "P1", "sequence_sha256": report.manifest.iloc[0].sequence_sha256, "v3_tier": "S"})()
    )
    expected_biophys, _ = compute_biophys_node("ACD")
    np.testing.assert_allclose(sample["biophys"].numpy(), expected_biophys.astype(np.float16))
    assert torch.all(sample["phaseflow_llps_hidden"] == 2.0)


def test_llps_hidden_extraction_uses_the_mean_of_the_two_residue_taps() -> None:
    outputs = {
        "llps_residue_repr": torch.ones((1, 3, 256)),
        "dpr_residue_repr": torch.full((1, 3, 256), 3.0),
    }

    hidden = extract_llps_hidden(outputs, seq_mask=torch.tensor([[True, True, False]]))

    assert tuple(hidden.shape) == (1, 3, 256)
    assert torch.all(hidden[:, :2] == 2.0)
    assert torch.all(hidden[:, 2] == 0.0)



# Source: test_packed_sidecar_cli.py


import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_packed_sidecar_builder_exposes_only_explicit_paths() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/protein/run.py", "build-dpr-sidecar", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    for option in ("--feature-dir", "--llps-checkpoint", "--output-root", "--device"):
        assert option in result.stdout



# Source: test_phasepro_cached_contract.py


import pandas as pd
import pytest

from scripts.protein.workflows.release import validate_release_sidecar_pairs
from scripts.protein.workflows.training import normalize_checkpoint_namespace


def test_phasepro_pair_validator_requires_exact_protein_sequence_and_length_pairs() -> None:
    release = pd.DataFrame(
        {
            "protein_id": ["P1", "P2"],
            "sequence_sha256": ["a", "b"],
            "sequence_length": [10, 20],
        }
    )
    sidecar = pd.DataFrame(
        {
            "protein_id": ["P1", "P2"],
            "sequence_sha256": ["a", "b"],
            "length": [10, 20],
        }
    )

    assert validate_release_sidecar_pairs(release, sidecar, expected_count=2)["matched_pairs"] == 2

    sidecar.loc[1, "length"] = 19
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_release_sidecar_pairs(release, sidecar, expected_count=2)


def test_legacy_checkpoint_prefix_is_migrated_for_evaluation_only() -> None:
    legacy_key = "phase" + "gt.encoder.weight"
    state = {legacy_key: "old", "projection.weight": "current"}

    migrated = normalize_checkpoint_namespace(state)

    assert migrated["llps_backbone.encoder.weight"] == "old"
    assert migrated["projection.weight"] == "current"
    assert legacy_key not in migrated



# Source: test_phasepro_eval_sidecar_guard.py

from pathlib import Path

import pytest

from phaseflow.protein import DPRV2HotpathSidecar
from phaseflow.protein import assert_no_eval_only_training_path


EVAL_SIDECAR = Path("artifacts/data/processed/evaluation_only/phasepro_pstp_v1")


def test_phasepro_eval_sidecar_is_forbidden_for_training_access() -> None:
    with pytest.raises(RuntimeError, match="Eval-only sidecar path is forbidden"):
        assert_no_eval_only_training_path(EVAL_SIDECAR / "packed" / "manifest.parquet")


def test_hotpath_reader_rejects_phasepro_eval_sidecar_by_default() -> None:
    with pytest.raises(RuntimeError, match="Eval-only sidecar path is forbidden"):
        DPRV2HotpathSidecar(
            data_root="artifacts/data/processed/stage2/dpr_v2",
            sidecar_root=EVAL_SIDECAR,
        )



# Source: test_pstp_fair_benchmark.py


import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

try:
    from scripts.external.protein import run_pstp_fair_benchmark as fair_benchmark
except ImportError:
    fair_benchmark = None

ALBATROSS_WEIGHTS = {} if fair_benchmark is None else fair_benchmark.ALBATROSS_WEIGHTS
AA_ORDER = "" if fair_benchmark is None else fair_benchmark.AA_ORDER
compute_fix_avgpool_scaler = None if fair_benchmark is None else fair_benchmark.compute_fix_avgpool_scaler


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "external_artifacts/pstp_official_benchmark_v1"
FEATURE_ROOT = OUT_ROOT / "reconstructed_features"

EXTERNAL_PSTP_REQUIRED = pytest.mark.skipif(
    fair_benchmark is None or not OUT_ROOT.exists(),
    reason="requires the optional complete protein benchmark artifacts",
)


@EXTERNAL_PSTP_REQUIRED
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


@EXTERNAL_PSTP_REQUIRED
def test_pstp_weight_architecture_is_650_20_5_1() -> None:
    manifest = pd.read_csv(OUT_ROOT / "manifests/nophasepro_weights.csv")
    assert manifest["layer1_weight_shape"].eq("[20, 650]").all()
    assert manifest["layer2_weight_shape"].eq("[5, 20]").all()
    assert manifest["layer3_weight_shape"].eq("[1, 5]").all()


@EXTERNAL_PSTP_REQUIRED
def test_albatross_hidden_contract_is_not_scalar_prediction() -> None:
    for name, path in ALBATROSS_WEIGHTS.items():
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert tuple(state["lstm.weight_ih_l0"].shape) == (220, 20), name
        assert tuple(state["fc.bias"].shape) == (1,), name
        hidden_per_direction = state["lstm.weight_ih_l0"].shape[0] // 4
        assert hidden_per_direction == 55
        assert hidden_per_direction * 2 == 110


@EXTERNAL_PSTP_REQUIRED
def test_feature_channel_order_contract_constants() -> None:
    assert AA_ORDER == "ACDEFGHIKLMNPQRSTVWY"
    expected_scaler = [33 / 17, 33 / 18, 33 / 19, 33 / 20, 33 / 21]
    actual = compute_fix_avgpool_scaler(40, 33, 16)[:5]
    assert np.allclose(actual, expected_scaler)
    assert compute_fix_avgpool_scaler(10, 33, 16) == [33 / 10] * 10


@EXTERNAL_PSTP_REQUIRED
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



# Source: test_pstp_official_benchmark.py


import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "artifacts/data/processed/evaluation/phasepro_official_v1"
OUT_ROOT = ROOT / "external_artifacts/pstp_official_benchmark_v1"

OFFICIAL_BENCHMARK_REQUIRED = pytest.mark.skipif(
    not DATA_ROOT.exists() or not OUT_ROOT.exists(),
    reason="requires the optional complete protein benchmark artifacts",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@OFFICIAL_BENCHMARK_REQUIRED
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


@OFFICIAL_BENCHMARK_REQUIRED
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


@OFFICIAL_BENCHMARK_REQUIRED
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


@OFFICIAL_BENCHMARK_REQUIRED
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


@OFFICIAL_BENCHMARK_REQUIRED
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


@OFFICIAL_BENCHMARK_REQUIRED
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


@OFFICIAL_BENCHMARK_REQUIRED
def test_final_status_is_blocked_before_v5_or_training() -> None:
    run_info = load_json(OUT_ROOT / "manifests/run_info.json")
    assert run_info["final_status"] == "BLOCKED"
    assert run_info["training_started"] is False
    assert run_info["v5_evaluated"] is False
    assert run_info["full_12000_continued"] is False
    reasons = "\n".join(run_info["hard_stop_reasons"])
    assert "PhaSePro-included" in reasons
    assert "No official PhaSePro feature matrix cache" in reasons
