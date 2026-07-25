from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_protenix_cache_setup_supports_help() -> None:
    result = subprocess.run(
        ["bash", "phaseflow/full_length/features/setup_protenix_cache.sh", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_af3_batch_runner_supports_help() -> None:
    result = subprocess.run(
        ["bash", "phaseflow/full_length/features/run_af3_batch.sh", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout
