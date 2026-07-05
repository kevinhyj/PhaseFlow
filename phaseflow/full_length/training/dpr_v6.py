from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, *, total_updates: int, warmup_updates: int, min_lr_ratio: float) -> None:
        self.optimizer = optimizer
        self.total_updates = int(total_updates)
        self.warmup_updates = int(warmup_updates)
        self.min_lr_ratio = float(min_lr_ratio)
        self.last_update = 0
        for group in self.optimizer.param_groups:
            group.setdefault("base_lr", float(group["lr"]))

    def scale(self, update: int) -> float:
        update = int(update)
        if self.warmup_updates > 0 and update <= self.warmup_updates:
            return max(1.0e-8, update / max(1.0, float(self.warmup_updates)))
        denom = max(1.0, float(self.total_updates - self.warmup_updates))
        progress = (update - self.warmup_updates) / denom
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def set_update(self, update: int) -> None:
        self.last_update = int(update)
        scale = self.scale(update)
        for group in self.optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * scale

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "dpr_v6_warmup_cosine_scheduler",
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "min_lr_ratio": self.min_lr_ratio,
            "last_update": self.last_update,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_update = int(state.get("last_update", 0))
        self.set_update(self.last_update)


class V6EMA:
    def __init__(self, model: torch.nn.Module, *, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().float().cpu().clone()
            for name, value in trainable_dpr_state_dict(model).items()
            if torch.is_floating_point(value)
        }

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            current = trainable_dpr_state_dict(model)
            for name, value in current.items():
                if name not in self.shadow or not torch.is_floating_point(value):
                    continue
                self.shadow[name].mul_(self.decay).add_(value.detach().float().cpu(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"format": "dpr_v6_ema", "decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.shadow = {name: value.detach().float().cpu().clone() for name, value in state["shadow"].items()}

    def apply_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup = {name: value.detach().cpu().clone() for name, value in trainable_dpr_state_dict(model).items()}
        state = trainable_dpr_state_dict(model)
        updated = {
            name: self.shadow[name].to(device=value.device, dtype=value.dtype) if name in self.shadow else value
            for name, value in state.items()
        }
        load_trainable_dpr_state_dict(model, updated, strict=True)
        return backup

    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        load_trainable_dpr_state_dict(model, backup, strict=True)


def trainable_dpr_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {name: value for name, value in model.v6.state_dict().items()}
    projectors = getattr(model, "v6_feature_projectors", None)
    if projectors is not None:
        for name, value in projectors.state_dict().items():
            state[f"v6_feature_projectors.{name}"] = value
    return state


def load_trainable_dpr_state_dict(model: torch.nn.Module, state: dict[str, torch.Tensor], *, strict: bool) -> None:
    v6_state = {name: value for name, value in state.items() if not name.startswith("v6_feature_projectors.")}
    model.v6.load_state_dict(v6_state, strict=strict)
    projectors = getattr(model, "v6_feature_projectors", None)
    if projectors is not None:
        projector_state = {
            name.removeprefix("v6_feature_projectors."): value
            for name, value in state.items()
            if name.startswith("v6_feature_projectors.")
        }
        projectors.load_state_dict(projector_state, strict=strict)


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def module_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    groups = {
        "grad_norm_v6": "v6.",
        "grad_norm_feature_projectors": "v6_feature_projectors.",
        "grad_norm_projection": "v6.projection.",
        "grad_norm_scanner": "v6.shared_scanner.",
    }
    out: dict[str, float] = {}
    for name, prefix in groups.items():
        total = 0.0
        for pname, parameter in model.named_parameters():
            if not pname.startswith(prefix) or parameter.grad is None:
                continue
            total += float(parameter.grad.detach().float().norm().item()) ** 2
        out[name] = math.sqrt(total)
    return out


def assert_frozen_grads(model: torch.nn.Module) -> None:
    bad: list[str] = []
    allowed_prefixes = ("v6.", "v6_feature_projectors.")
    for name, parameter in model.named_parameters():
        if name.startswith(allowed_prefixes):
            continue
        if parameter.requires_grad:
            bad.append(f"{name}:requires_grad")
        if parameter.grad is not None and float(parameter.grad.detach().abs().max().item()) != 0.0:
            bad.append(f"{name}:grad")
    if bad:
        raise RuntimeError(f"DPR v6 frozen contract violation: {bad[:20]}")


def profile_monitor(out: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, float]:
    seq_mask = batch["seq_mask"].bool().to(out["p33"].device)
    p33 = out["p33"].detach().float()
    vals = p33[seq_mask]
    if vals.numel() == 0:
        return {}
    quant = torch.quantile(vals, torch.tensor([0.10, 0.50, 0.75, 0.90, 0.99], device=vals.device))
    valid_frac = [(p33[i][seq_mask[i]] > 0.5).float().mean() for i in range(p33.shape[0])]
    top_bottom = []
    for i in range(p33.shape[0]):
        row = p33[i][seq_mask[i]]
        if row.numel() == 0:
            continue
        top_k = max(1, int(math.ceil(0.10 * int(row.numel()))))
        bottom_k = max(1, int(math.ceil(0.50 * int(row.numel()))))
        top_bottom.append(torch.topk(row, k=min(top_k, int(row.numel())), largest=True).values.mean() - torch.topk(row, k=min(bottom_k, int(row.numel())), largest=False).values.mean())
    return {
        "bag_hard": float(out["bag_hard"].detach().float().mean().item()),
        "bag_topk": float(out["bag_topk"].detach().float().mean().item()),
        "max_p33": float(out["hard_33"].detach().float().mean().item()),
        "max_p129": float(out["hard_129"].detach().float().mean().item()),
        "max_p257": float(out["hard_257"].detach().float().mean().item()),
        "top5_p33": float(out["topk_33"].detach().float().mean().item()),
        "top5_p129": float(out["topk_129"].detach().float().mean().item()),
        "top5_p257": float(out["topk_257"].detach().float().mean().item()),
        "p33_min": float(vals.min().item()),
        "p33_max": float(vals.max().item()),
        "p33_mean": float(vals.mean().item()),
        "p33_std": float(vals.std(unbiased=False).item()),
        "p33_p10": float(quant[0].item()),
        "p33_p50": float(quant[1].item()),
        "p33_p75": float(quant[2].item()),
        "p33_p90": float(quant[3].item()),
        "p33_p99": float(quant[4].item()),
        "p33_q90_q50": float((quant[3] - quant[1]).item()),
        "top10_bottom50": float(torch.stack(top_bottom).mean().item()) if top_bottom else 0.0,
        "pred_fraction_0p5": float(torch.stack(valid_frac).mean().item()) if valid_frac else 0.0,
    }


def gather_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    if not dist.is_initialized():
        return [row]
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, row)
    return [dict(item) for item in gathered]


def reduce_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"update": int(rows[0]["update"])}
    tiers = [str(row["tier"]) for row in rows]
    out["tier_exposure"] = dict(sorted(Counter(tiers).items()))
    numeric: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in {"update", "rank"}:
                numeric.setdefault(key, []).append(float(value))
    for key, values in numeric.items():
        out[key] = float(np.mean(values))
    return out


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()}
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


def rng_states_for_checkpoint(device: torch.device) -> list[dict[str, Any]]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    state = {
        "rank": int(rank),
        "torch_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_state": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if not dist.is_initialized():
        return [state]
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, state)
    return [dict(item) for item in gathered]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    ema: V6EMA,
    cfg: dict[str, Any],
    arm: str,
    step: int,
    elapsed: float,
    phaseflow_llps_raw: dict[str, Any],
    phaseflow_raw: dict[str, Any],
    sampler_state: dict[str, Any],
    rng_states_by_rank: list[dict[str, Any]] | None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "dpr_v6_checkpoint",
            "arm": str(arm),
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "dpr_v6_state_dict": model.v6.state_dict(),
            "dpr_v6_trainable_state_dict": trainable_dpr_state_dict(model),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sampler": sampler_state,
            "config": cfg,
            "elapsed_sec": float(elapsed),
            "phaseflow_llps_raw_metadata": phaseflow_llps_raw,
            "phaseflow_raw_metadata": phaseflow_raw,
            "rng_states_by_rank": rng_states_by_rank,
        },
        path,
    )


def forbidden_env() -> dict[str, str]:
    return {
        "PHASEFLOW_DISABLE_STARLING_READ": os.environ.get("PHASEFLOW_DISABLE_STARLING_READ", ""),
        "PHASEFLOW_DISABLE_PROTENIX_READ": os.environ.get("PHASEFLOW_DISABLE_PROTENIX_READ", ""),
        "PHASEFLOW_STRICT_OFFLINE": os.environ.get("PHASEFLOW_STRICT_OFFLINE", ""),
    }
