from pathlib import Path

import pytest

from phaseflow.full_length.data.dpr_v2_hotpath import DPRV2HotpathSidecar
from phaseflow.full_length.data.runtime_guard import assert_no_eval_only_training_path


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
