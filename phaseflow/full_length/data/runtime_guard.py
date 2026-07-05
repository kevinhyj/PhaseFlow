from __future__ import annotations

import builtins
import os
from pathlib import Path
from typing import Any, Callable


FORBIDDEN_PATH_TOKENS = (
    ".h5",
    ".hdf5",
    "retired_",
    "features_v2",
    "graphs_v2",
    "tables_v2",
    "audit_v2",
)

EVAL_ONLY_TRAINING_PATH_TOKENS = (
    "data/processed/evaluation_only/phasepro_pstp_v1",
)

_ORIGINAL_OPEN: Callable[..., Any] | None = None
_PATCHED = False


def strict_offline_enabled() -> bool:
    return str(os.environ.get("PHASEFLOW_STRICT_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}


def assert_offline_path_allowed(path: str | Path, *, allow_legacy_h5: bool = False) -> None:
    text = str(path)
    normalized = text.replace("\\", "/").lower()
    if allow_legacy_h5:
        tokens = tuple(token for token in FORBIDDEN_PATH_TOKENS if token not in {".h5", ".hdf5"})
    else:
        tokens = FORBIDDEN_PATH_TOKENS
    offenders = [token for token in tokens if token in normalized]
    if offenders:
        raise RuntimeError(f"Forbidden strict-offline data path: {text} (matched {offenders})")


def assert_no_eval_only_training_path(path: str | Path) -> None:
    text = str(path)
    normalized = text.replace("\\", "/").lower()
    offenders = [token for token in EVAL_ONLY_TRAINING_PATH_TOKENS if token in normalized]
    if offenders:
        raise RuntimeError(f"Eval-only sidecar path is forbidden for training data access: {text} (matched {offenders})")


def assert_no_runtime_build(enabled: bool, name: str) -> None:
    if strict_offline_enabled() and enabled:
        raise RuntimeError(f"Runtime {name} is forbidden when PHASEFLOW_STRICT_OFFLINE=1")


def assert_no_forbidden_dataset_write(path: str | Path, mode: str = "r") -> None:
    write_mode = any(flag in mode for flag in ("w", "a", "x", "+"))
    if not write_mode:
        return
    normalized = str(path).replace("\\", "/").lower()
    forbidden_roots = (
        "data/processed/merged/features/",
        "data/processed/merged/graphs/",
    )
    if any(root in normalized for root in forbidden_roots):
        raise RuntimeError(f"Writing training inputs is forbidden in strict offline mode: {path}")


def install_strict_offline_guard() -> None:
    global _ORIGINAL_OPEN, _PATCHED
    if _PATCHED:
        return
    if not strict_offline_enabled():
        return
    _ORIGINAL_OPEN = builtins.open

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, Path)):
            assert_offline_path_allowed(file)
            assert_no_forbidden_dataset_write(file, mode)
        return _ORIGINAL_OPEN(file, mode, *args, **kwargs)  # type: ignore[misc]

    builtins.open = guarded_open
    _PATCHED = True
