from pathlib import Path

import pytest

from phaseflow.full_length.data.config import resolve_feature_dirs, resolve_phase_targets, validate_forbidden_data_paths
from phaseflow.full_length.train import _phase_train_ids, _validate_train_region_supervision


def test_feature_dir_fallbacks_are_opt_in() -> None:
    data_config = {
        "feature_dir": "data/features/merged_h5_teacher",
        "feature_dirs": [
            "data/features/merged_h5_teacher_phaseflow",
            "external_artifacts/phase_diagram_original_scale_h5",
        ],
    }

    assert resolve_feature_dirs(data_config) == ["data/features/merged_h5_teacher"]

    data_config["allow_feature_dir_fallbacks"] = True
    assert resolve_feature_dirs(data_config) == [
        "data/features/merged_h5_teacher_phaseflow",
        "external_artifacts/phase_diagram_original_scale_h5",
    ]


def test_phase_aux_ids_and_targets_are_opt_in(tmp_path: Path) -> None:
    ids_file = tmp_path / "phase_ids.txt"
    ids_file.write_text("phase_a\nphase_b\n")
    data_config = {
        "phase_train_ids_file": str(ids_file),
        "phase_targets": "data/processed/phaseflow/phase_targets.csv",
    }

    assert _phase_train_ids(data_config) == []
    assert resolve_phase_targets(data_config) is None

    data_config["allow_phase_aux_data"] = True
    assert _phase_train_ids(data_config) == ["phase_a", "phase_b"]
    assert resolve_phase_targets(data_config) == "data/processed/phaseflow/phase_targets.csv"


def test_forbidden_phaseflow_paths_fail_fast() -> None:
    data_config = {
        "forbid_phaseflow_data": True,
        "phase_targets": "data/processed/phaseflow/phase_targets.csv",
    }

    with pytest.raises(ValueError, match="forbidden"):
        validate_forbidden_data_paths(data_config, ["data/features/merged_h5_teacher"])


def test_feature_region_supervision_with_dpr_losses_fails_fast() -> None:
    config = {"training": {"loss_weights": {"region_gold": 0.1, "region": 0.1}}}
    with pytest.raises(ValueError, match="gold labels is forbidden"):
        _validate_train_region_supervision(config, "feature")


def test_pstp_region_supervision_requires_target_file_for_final_losses() -> None:
    config = {"data": {}, "training": {"loss_weights": {"final_region_teacher": 1.0}}}
    with pytest.raises(ValueError, match="requires data.region_targets"):
        _validate_train_region_supervision(config, "region_targets")


def test_pstp_region_supervision_rejects_mixed_dpr_teacher_losses() -> None:
    config = {
        "data": {"region_targets": "data/processed/pstp_scan_region_targets.h5"},
        "training": {"loss_weights": {"final_region_teacher": 1.0, "teacher_dpr": 0.1, "region": 0.1}},
    }
    with pytest.raises(ValueError, match="Pure PSTP-Scan DPR training"):
        _validate_train_region_supervision(config, "region_targets")
