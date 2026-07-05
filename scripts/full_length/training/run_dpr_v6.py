#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseflow.full_length.data.dpr_v5_hotpath import DPRV5BaseOnlySidecar, edge_policy_summary  # noqa: E402
from phaseflow.full_length.data.dpr_v6 import (  # noqa: E402
    DPRV6ScheduleDataset,
    ExtraFeatureStore,
    build_fixed_schedule,
    load_v6_tier_manifest,
    read_table,
    sha256_file,
    write_schedule_artifacts,
)
from phaseflow.full_length.models.dpr_v6 import DPRV6LossConfig, dpr_v6_loss, load_dpr_v6_phasestack  # noqa: E402
from phaseflow.full_length.training.dpr_v6 import (  # noqa: E402
    V6EMA,
    WarmupCosineScheduler,
    append_csv,
    append_jsonl,
    assert_frozen_grads,
    forbidden_env,
    gather_rows,
    module_grad_norms,
    move_batch_to_device,
    profile_monitor,
    reduce_rows,
    rng_states_for_checkpoint,
    save_checkpoint,
    seed_all,
    unwrap,
)
from scripts.full_length.ablation.dpr_stream_mask import load_stream_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train isolated DPR v6 ablation arms.")
    parser.add_argument("--config", type=Path, required=True, help="Runtime DPR v6 training config.")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--start-update", type=int, default=1)
    parser.add_argument("--end-update", type=int, default=None)
    parser.add_argument("--make-schedule-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None, help="Override paths.output_root from the config.")
    parser.add_argument("--ablation-mask", type=Path, default=None, help="Single mask JSON/YAML, matrix YAML, or registry CSV.")
    parser.add_argument("--ablation-arm-id", default=None, help="Arm id to select when --ablation-mask contains multiple arms.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PHASEFLOW_DISABLE_STARLING_READ", "1")
    os.environ.setdefault("PHASEFLOW_DISABLE_PROTENIX_READ", "1")
    os.environ.setdefault("PHASEFLOW_STRICT_OFFLINE", "1")
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
    init_dist()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    cfg = resolve_config(args)
    seed = int(args.seed if args.seed is not None else cfg["run"]["seed"])
    seed_all(seed + rank)
    output_root = Path(cfg["paths"]["output_root"]).resolve()
    dirs = make_arm_dirs(output_root, str(args.arm))
    if rank == 0:
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        (dirs["configs"] / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        write_json(dirs["logs"] / "environment.json", {"host": socket.gethostname(), "world": world, "arm": args.arm, "forbidden_env": forbidden_env()})
        ensure_schedule(cfg, updates=int(args.updates))
    barrier()
    if args.make_schedule_only:
        finish_dist()
        return

    end_update = int(args.end_update if args.end_update is not None else args.updates)
    if int(world) != int(cfg["run"]["world_size"]):
        raise RuntimeError(f"DPR v6 requires world_size={cfg['run']['world_size']}, got {world}")
    schedule = pd.read_parquet(cfg["paths"]["schedule_current"])
    arm_cfg = cfg["arms"][str(args.arm)]
    model_cfg = resolved_model_config(cfg, arm_cfg)
    llps_checkpoint = cfg["paths"].get("phaseflow_llps_checkpoint", cfg["paths"].get("phase" + "gt_checkpoint"))
    model, phaseflow_llps_raw, phaseflow_raw = load_dpr_v6_phasestack(
        phaseflow_llps_checkpoint=llps_checkpoint,
        phaseflow_checkpoint=cfg["paths"].get("phaseflow_checkpoint"),
        config=model_cfg,
        device=device,
    )
    stream_mask_contract = model.stream_mask_summary() if hasattr(model, "stream_mask_summary") else {}
    if rank == 0:
        write_json(
            dirs["reports"] / "stream_mask_contract.json",
            {
                "format": "dpr_v6_stream_mask_contract",
                "arm": str(args.arm),
                "ablation_arm_id": cfg.get("ablation", {}).get("stream_mask", {}).get("arm_id", ""),
                "mask": cfg.get("ablation", {}).get("stream_mask"),
                "model_contract": stream_mask_contract,
            },
        )
    assert_trainable_contract(model)
    model.train(True)
    optimizer = make_optimizer(model, cfg, arm_cfg)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_updates=int(args.updates),
        warmup_updates=int(cfg["scheduler"]["warmup_updates"]),
        min_lr_ratio=float(cfg["scheduler"]["min_lr_ratio"]),
    )
    ema = V6EMA(model, decay=float(cfg["ema"]["decay"]))
    ddp: torch.nn.Module
    if dist.is_initialized():
        ddp = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    else:
        ddp = model

    sidecar = DPRV5BaseOnlySidecar(
        v2_data_root=cfg["paths"]["v2_data_root"],
        packed_root=cfg["paths"]["packed_root"],
        mmap=True,
    )
    extra = ExtraFeatureStore.from_config(arm_cfg.get("extra_features"))
    dataset = DPRV6ScheduleDataset(
        sidecar=sidecar,
        schedule=schedule,
        rank=rank,
        start_update=int(args.start_update),
        end_update=end_update,
        extra_features=extra,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": None,
        "num_workers": int(args.num_workers if args.num_workers is not None else cfg["run"]["num_workers"]),
        "pin_memory": True,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor if args.prefetch_factor is not None else cfg["run"]["prefetch_factor"])
    loader = DataLoader(dataset, **loader_kwargs)
    loss_cfg = DPRV6LossConfig(**cfg_for_loss(cfg, arm_cfg))
    save_updates = {int(x) for x in cfg["checkpoint"]["save_updates"]}

    raw_log = (dirs["logs"] / f"rank{rank:02d}_raw_metrics.jsonl").open("a", encoding="utf-8")
    exposures: Counter[str] = Counter()
    unique_by_tier: dict[str, set[str]] = {tier: set() for tier in ("S", "W", "M", "ND", "NP")}
    started = time.perf_counter()
    last_step = int(args.start_update) - 1
    try:
        for batch_cpu in loader:
            step = int(batch_cpu["update"])
            last_step = step
            scheduler.set_update(step)
            batch = move_batch_to_device(batch_cpu, device)
            tier = str(batch["v3_tiers"][0])
            pid = str(batch["protein_ids"][0])
            exposures[tier] += 1
            unique_by_tier.setdefault(tier, set()).add(pid)
            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = ddp(batch=batch, task="dpr", return_regions=False)["dpr"]
                loss, parts = dpr_v6_loss(out, batch, cfg=loss_cfg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"DPR v6 non-finite loss arm={args.arm} update={step} rank={rank}")
            edge_summary = edge_policy_summary(batch)
            if edge_summary["starling_edges_passed_to_model"] or edge_summary["protenix_edges_passed_to_model"]:
                raise RuntimeError(f"DPR v6 forbidden edge reached model: {edge_summary}")
            loss.backward()
            grad_parts = module_grad_norms(unwrap(ddp))
            assert_frozen_grads(unwrap(ddp))
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in unwrap(ddp).parameters() if p.requires_grad], float(cfg["optimizer"]["gradient_clip_norm"])).detach().item())
            optimizer.step()
            ema.update(unwrap(ddp))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_step = time.perf_counter() - t0
            row = {
                "arm": str(args.arm),
                "update": step,
                "rank": rank,
                "tier": tier,
                "protein_id": pid,
                "loss": float(loss.detach().item()),
                **parts_to_floats(parts),
                **profile_monitor(out, batch),
                **edge_summary,
                **grad_parts,
                "grad_norm": grad_norm,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "step_sec": elapsed_step,
                "batch_residues": int(batch["seq_mask"].sum().detach().item()),
            }
            raw_log.write(json.dumps(row, sort_keys=True) + "\n")
            raw_log.flush()
            gathered = gather_rows(row)
            save_now = step in save_updates or step == end_update
            rng_states = rng_states_for_checkpoint(device) if save_now else None
            if rank == 0:
                if step == int(args.start_update) or step % int(cfg["run"]["log_every"]) == 0:
                    reduced = reduce_rows(gathered)
                    append_jsonl(dirs["logs"] / "global_metrics.jsonl", {"update": step, "raw_rows": gathered, "reduced": reduced})
                    append_csv(dirs["logs"] / "global_metrics.csv", reduced)
                    print(json.dumps(reduced, sort_keys=True), flush=True)
                if save_now:
                    save_checkpoint(
                        dirs["checkpoints"] / f"update_{step:06d}.pt",
                        model=unwrap(ddp),
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ema=ema,
                        cfg=cfg,
                        arm=str(args.arm),
                        step=step,
                        elapsed=time.perf_counter() - started,
                        phaseflow_llps_raw=phaseflow_llps_raw,
                        phaseflow_raw=phaseflow_raw,
                        sampler_state={"format": "dpr_v6_fixed_schedule", "schedule_path": cfg["paths"]["schedule_current"], "schedule_sha256": sha256_file(cfg["paths"]["schedule_current"]), "next_update": step + 1},
                        rng_states_by_rank=rng_states,
                    )
            barrier()
    finally:
        raw_log.close()
    write_rank_exposure(dirs["logs"] / f"exposure_rank{rank:02d}.json", exposures, unique_by_tier)
    barrier()
    if rank == 0:
        summary = {
            "format": "dpr_v6_train_summary",
            "arm": str(args.arm),
            "status": "COMPLETE" if last_step >= end_update else "STOPPED",
            "last_update": int(last_step),
            "target_update": int(end_update),
            "elapsed_sec": float(time.perf_counter() - started),
            "checkpoint": str(dirs["checkpoints"] / f"update_{last_step:06d}.pt"),
            "checkpoint_sha256": sha256_file(dirs["checkpoints"] / f"update_{last_step:06d}.pt") if (dirs["checkpoints"] / f"update_{last_step:06d}.pt").exists() else "",
            "schedule_sha256": sha256_file(cfg["paths"]["schedule_current"]),
            "ablation_mask": cfg.get("ablation", {}).get("stream_mask"),
            "stream_mask_contract": stream_mask_contract,
            "exposure": merge_exposure_logs(dirs["logs"]),
        }
        write_json(dirs["reports"] / "train_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    barrier()
    finish_dist()


def init_dist() -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


def finish_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if str(args.arm) not in cfg["arms"]:
        raise KeyError(f"unknown DPR v6 arm {args.arm}; available={sorted(cfg['arms'])}")
    if args.output_root is not None:
        cfg["paths"]["output_root"] = str(args.output_root.resolve())
    mask = load_stream_mask(args.ablation_mask, arm_id=args.ablation_arm_id)
    if mask is not None:
        cfg.setdefault("ablation", {})["stream_mask"] = mask
    return cfg


def ensure_schedule(cfg: dict[str, Any], *, updates: int) -> None:
    schedule_path = Path(cfg["paths"]["schedule_current"])
    audit_path = Path(cfg["paths"]["schedule_audit"])
    if schedule_path.exists():
        return
    tier_manifest = load_v6_tier_manifest(
        cfg["paths"]["tier_manifest"],
        official_package_root=cfg["paths"]["official_phasepro_root"],
        require_no_benchmark_overlap=True,
    )
    schedule = build_fixed_schedule(
        tier_manifest,
        updates=int(updates),
        world_size=int(cfg["run"]["world_size"]),
        seed=int(cfg["run"]["seed"]),
    )
    write_schedule_artifacts(schedule, schedule_path=schedule_path, audit_path=audit_path)


def resolved_model_config(cfg: dict[str, Any], arm_cfg: dict[str, Any]) -> dict[str, Any]:
    model = dict(cfg["model"])
    model.update(dict(arm_cfg.get("model", {})))
    mask = cfg.get("ablation", {}).get("stream_mask")
    if mask is not None:
        model["ablation_mask"] = mask
    return {"model": model}


def cfg_for_loss(cfg: dict[str, Any], arm_cfg: dict[str, Any]) -> dict[str, Any]:
    loss = dict(cfg["loss"])
    loss.update(dict(arm_cfg.get("loss", {})))
    return loss


def make_optimizer(model: torch.nn.Module, cfg: dict[str, Any], arm_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("DPR v6 arm has no trainable parameters")
    opt_cfg = dict(cfg["optimizer"])
    opt_cfg.update(dict(arm_cfg.get("optimizer", {})))
    opt_type = str(opt_cfg.get("type", "Adam")).lower()
    lr = float(opt_cfg["lr"])
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    if opt_type == "adamw":
        return torch.optim.AdamW([{"params": params, "lr": lr, "base_lr": lr, "name": "v6_head"}], weight_decay=weight_decay)
    if opt_type == "adam":
        return torch.optim.Adam([{"params": params, "lr": lr, "base_lr": lr, "name": "v6_head"}], weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer type: {opt_type}")


def assert_trainable_contract(model: torch.nn.Module) -> None:
    allowed_prefixes = ("v6.", "v6_feature_projectors.")
    bad = [name for name, parameter in model.named_parameters() if parameter.requires_grad and not name.startswith(allowed_prefixes)]
    if bad:
        raise RuntimeError(f"DPR v6 non-DPR trainable parameters: {bad[:20]}")
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("DPR v6 model has no trainable parameters")


def parts_to_floats(parts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in parts.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().float().item()) if value.numel() == 1 else float(value.detach().float().mean().item())
        elif isinstance(value, (int, float)):
            out[key] = value
    return out


def make_arm_dirs(output_root: Path, arm: str) -> dict[str, Path]:
    return {
        "configs": output_root / "configs" / arm,
        "checkpoints": output_root / "checkpoints" / arm,
        "logs": output_root / "logs" / arm,
        "reports": output_root / "reports" / arm,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rank_exposure(path: Path, exposures: Counter[str], unique_by_tier: dict[str, set[str]]) -> None:
    payload = {
        "tier_exposure": dict(exposures),
        "unique_coverage": {tier: len(values) for tier, values in unique_by_tier.items()},
    }
    write_json(path, payload)


def merge_exposure_logs(log_dir: Path) -> dict[str, Any]:
    exposure: Counter[str] = Counter()
    unique: Counter[str] = Counter()
    for path in sorted(log_dir.glob("exposure_rank*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        exposure.update({str(k): int(v) for k, v in payload.get("tier_exposure", {}).items()})
        unique.update({str(k): int(v) for k, v in payload.get("unique_coverage", {}).items()})
    return {"tier_exposure": dict(exposure), "rank_unique_sum": dict(unique)}


if __name__ == "__main__":
    main()
