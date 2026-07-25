from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.full_length.train_dpr import configure_schedule_paths, resolve_updates


ROOT = Path(__file__).resolve().parents[2]


def test_full_length_release_uses_canonical_config_and_training_names() -> None:
    config_dir = ROOT / "configs" / "full_length"
    script_dir = ROOT / "scripts" / "full_length"

    assert (config_dir / "llps.yaml").is_file()
    assert (config_dir / "dpr.yaml").is_file()
    assert not (config_dir / "final_llps.yaml").exists()
    assert not (config_dir / "final_dpr.yaml").exists()
    assert (script_dir / "train_llps.py").is_file()
    assert (script_dir / "train_dpr.py").is_file()


def test_full_length_configs_do_not_embed_local_artifact_paths() -> None:
    config_dir = ROOT / "configs" / "full_length"
    text = "\n".join(
        (config_dir / name).read_text(encoding="utf-8")
        for name in ("llps.yaml", "dpr.yaml")
    )

    forbidden = ("/data/mogoo7zn/", "external_artifacts")
    for token in forbidden:
        assert token not in text


def test_public_full_length_docs_do_not_expose_local_release_names() -> None:
    paths = (
        ROOT / "README.md",
        ROOT / "configs" / "full_length" / "README.md",
        ROOT / "docs" / "full_length" / "README.md",
        ROOT / "docs" / "full_length" / "artifact_policy.md",
        ROOT / "scripts" / "full_length" / "README.md",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "configs/full_length/final_" not in text
    assert "docs/full_length/final/" not in text
    assert "training/run_dpr_v6.py" not in text
    assert "Git LFS" not in text


def test_dpr_config_controls_default_update_count_and_runtime_schedule() -> None:
    config_path = ROOT / "configs" / "full_length" / "dpr.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(updates=None, start_update=1, end_update=None)

    updates, end_update = resolve_updates(args, config)

    assert updates == config["scheduler"]["total_updates"] == 50
    assert end_update == 50
    configure_schedule_paths(
        config,
        arm="dpr",
        updates=updates,
        schedule_seed=config["run"]["seed"],
        world_size=config["run"]["world_size"],
    )
    assert config["paths"]["schedule_current"] == "runs/dpr/schedules/dpr/schedule_000050_seed202606188_world8.parquet"
    assert config["paths"]["schedule_audit"] == "runs/dpr/schedules/dpr/schedule_000050_seed202606188_world8_audit.json"
