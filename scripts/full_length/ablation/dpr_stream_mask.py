from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


PHASEFLOW_LLPS_DIRECT_STREAM = "phaseflow_llps_direct"
LEGACY_LLPS_PREFIX = "phase" + "gt"
LEGACY_LLPS_DIRECT_STREAM = LEGACY_LLPS_PREFIX + "_direct"
STREAM_KEYS = ("esm2", "biophys", PHASEFLOW_LLPS_DIRECT_STREAM, "phaseflow_bridge")


def load_stream_mask(path: str | Path | None, *, arm_id: str | None = None) -> dict[str, Any] | None:
    if path is None:
        return None
    mask_path = Path(path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    if mask_path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(mask_path.open("r", encoding="utf-8")))
        row = select_row(rows, arm_id=arm_id)
        return normalize_row(row, source=mask_path)
    payload = yaml.safe_load(mask_path.read_text(encoding="utf-8"))
    if is_single_mask(payload):
        return normalize_row(payload, source=mask_path)
    rows = matrix_rows(payload)
    row = select_row(rows, arm_id=arm_id)
    return normalize_row(row, source=mask_path)


def is_single_mask(payload: Any) -> bool:
    return isinstance(payload, dict) and ("streams" in payload or all(key in payload for key in STREAM_KEYS))


def matrix_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise TypeError("DPR ablation mask file must be a dict, matrix YAML, or CSV")
    rows: list[dict[str, Any]] = []
    rows.extend(dict(row) for row in payload.get("primary_arms", []) if isinstance(row, dict))
    extension = payload.get("full_factorial_extension", {})
    if isinstance(extension, dict):
        rows.extend(dict(row) for row in extension.get("additional_required_arms", []) if isinstance(row, dict))
    if not rows:
        raise ValueError("DPR ablation matrix did not contain any arm rows")
    return rows


def select_row(rows: list[dict[str, Any]], *, arm_id: str | None) -> dict[str, Any]:
    if arm_id is None:
        if len(rows) == 1:
            return rows[0]
        raise ValueError("arm_id is required when a mask file contains more than one arm")
    matches = [row for row in rows if str(row.get("id", row.get("arm_id", ""))) == str(arm_id)]
    if len(matches) != 1:
        raise KeyError(f"expected exactly one DPR ablation arm {arm_id!r}, found {len(matches)}")
    return matches[0]


def normalize_row(row: dict[str, Any], *, source: Path) -> dict[str, Any]:
    streams_raw = normalize_legacy_stream_names(dict(row.get("streams", row)))
    missing = [key for key in STREAM_KEYS if key not in streams_raw]
    if missing:
        raise KeyError(f"DPR stream mask is missing keys: {missing}")
    streams = {key: parse_bool(streams_raw[key]) for key in STREAM_KEYS}
    arm_id = str(row.get("arm_id", row.get("id", "")))
    bitmask = str(row.get("bitmask", "".join("1" if streams[key] else "0" for key in STREAM_KEYS)))
    out = {
        "schema": "dpr_stream_mask_v1",
        "source": str(source.resolve()),
        "arm_id": arm_id,
        "id": arm_id,
        "bitmask": bitmask,
        "streams": streams,
        "label": str(row.get("label", "")),
        "class": str(row.get("class", "")),
        "priority": str(row.get("priority", "")),
        "strict_no_phaseflow_llps_module": parse_bool(row.get("strict_no_phaseflow_llps_module", row.get("strict_no_" + LEGACY_LLPS_PREFIX + "_module", False))),
    }
    if "note" in row:
        out["note"] = str(row["note"])
    return out


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def normalize_legacy_stream_names(streams: dict[str, Any]) -> dict[str, Any]:
    if LEGACY_LLPS_DIRECT_STREAM in streams and PHASEFLOW_LLPS_DIRECT_STREAM not in streams:
        streams[PHASEFLOW_LLPS_DIRECT_STREAM] = streams.pop(LEGACY_LLPS_DIRECT_STREAM)
    return streams


def apply_stream_mask_to_model_config(model_cfg: dict[str, Any], mask: dict[str, Any] | None) -> dict[str, Any]:
    out = {"model": dict(model_cfg["model"])}
    if mask is not None:
        out["model"]["ablation_mask"] = mask
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
