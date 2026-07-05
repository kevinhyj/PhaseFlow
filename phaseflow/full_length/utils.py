from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        return yaml.safe_load(handle)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(sanitize_json(data), handle, indent=2, sort_keys=True, allow_nan=False)


def dumps_json(data: Any, **kwargs: Any) -> str:
    return json.dumps(sanitize_json(data), allow_nan=False, **kwargs)


def sanitize_json(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: sanitize_json(value) for key, value in data.items()}
    if isinstance(data, list):
        return [sanitize_json(value) for value in data]
    if isinstance(data, tuple):
        return [sanitize_json(value) for value in data]
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved
