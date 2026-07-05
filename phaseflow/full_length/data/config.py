from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_FORBIDDEN_DATA_TOKENS = ("phaseflow", "phase_diagram")


def resolve_feature_dirs(data_config: dict[str, Any]) -> list[str | Path]:
    if bool(data_config.get("allow_feature_dir_fallbacks", False)) and data_config.get("feature_dirs"):
        return list(data_config["feature_dirs"])
    if data_config.get("feature_dir"):
        return [data_config["feature_dir"]]
    if data_config.get("feature_dirs"):
        feature_dirs = list(data_config["feature_dirs"])
        if feature_dirs:
            return [feature_dirs[0]]
    raise ValueError("data.feature_dir is required; data.feature_dirs fallback is opt-in")


def phase_aux_data_enabled(data_config: dict[str, Any]) -> bool:
    return bool(data_config.get("allow_phase_aux_data", False))


def resolve_phase_targets(data_config: dict[str, Any]) -> str | Path | None:
    if not phase_aux_data_enabled(data_config):
        return None
    return data_config.get("phase_targets")


def validate_forbidden_data_paths(data_config: dict[str, Any], feature_dirs: list[str | Path]) -> None:
    if not bool(data_config.get("forbid_phaseflow_data", False)):
        return
    tokens = tuple(
        str(token).lower()
        for token in data_config.get("forbidden_data_path_tokens", DEFAULT_FORBIDDEN_DATA_TOKENS)
        if str(token).strip()
    )
    if not tokens:
        return
    candidate_paths: list[str] = [str(path) for path in feature_dirs]
    for key in ("phase_train_ids_file", "phase_targets"):
        value = data_config.get(key)
        if value:
            candidate_paths.append(str(value))
    offenders = sorted(
        path
        for path in candidate_paths
        if any(token in path.lower() for token in tokens)
    )
    if offenders:
        raise ValueError(
            "PhaseFlow/phase-diagram data is forbidden for this run, but these paths were configured: "
            + ", ".join(offenders)
        )
