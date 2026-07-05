from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v6_files_do_not_reference_v5_head_resume_or_region_global_logits() -> None:
    paths = [
        ROOT / "phaseflow/full_length/models/dpr_v6.py",
        ROOT / "phaseflow/full_length/data/dpr_v6.py",
        ROOT / "phaseflow/full_length/training/dpr_v6.py",
        ROOT / "scripts/full_length/training/run_dpr_v6.py",
        ROOT / "configs/full_length/final_dpr.yaml",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = [
        "region_global_logits",
        "dpr_v5_state_dict",
        "load_dpr_v5",
    ]
    for token in forbidden:
        assert token not in text


def test_v6_uses_own_namespace_paths() -> None:
    cfg = (ROOT / "configs/full_length/final_dpr.yaml").read_text(encoding="utf-8")
    assert "artifacts/model/checkpoints/update_000050.pt" in cfg
    assert "data/processed/stage2/dpr_v8r1a" in cfg
    assert "external_artifacts/overall/v6_v8r1a_region_finetune" in cfg
    assert ("outputs" + "/") not in cfg
