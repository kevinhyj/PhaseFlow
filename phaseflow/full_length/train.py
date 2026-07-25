from __future__ import annotations

import argparse
import csv
import json
import os
import math
import random
import shutil
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

from phaseflow.full_length.data.collator import PhaseFlowCollator
from phaseflow.full_length.data.config import (
    phase_aux_data_enabled,
    resolve_feature_dirs,
    resolve_phase_targets,
    validate_forbidden_data_paths,
)
from phaseflow.full_length.data.dataset import PhaseFlowDataset
from phaseflow.full_length.data.offline_dataset import PhaseFlowOfflineDataset
from phaseflow.full_length.data.batch_plan_dataset import PhaseFlowBatchPlanDataset
from phaseflow.full_length.data.packed_batches import PhaseFlowPackedBatchDataset, PackedBatchDataset
from phaseflow.full_length.data.runtime_guard import assert_no_runtime_build, install_strict_offline_guard, strict_offline_enabled
from phaseflow.full_length.data.schemas import IGNORE_INDEX
from phaseflow.full_length.data.splits import resolve_split_ids
from phaseflow.full_length.data.full_benchmark_leakage import assert_no_full_benchmark_leakage
from phaseflow.full_length.evaluate import evaluate_model
from phaseflow.full_length.features.build_features import build_feature_cache, build_feature_cache_from_manifest
from phaseflow.full_length.features.plm_embedder import ESM2Config, download_esm2_model
from phaseflow.full_length.features.run_esm2 import records_from_fasta, records_from_manifest, run_esm2_embeddings
from phaseflow.full_length.losses.multitask import compute_multitask_loss
from phaseflow.full_length.models.phaseflow import PhaseFlowModel
from phaseflow.full_length.utils import dumps_json, load_yaml, move_batch_to_device, resolve_device, set_seed, write_json
import torch.nn.functional as F


def _normalize_dataset_class(value: object) -> str:
    name = str(value)
    legacy = {
        "PhaseFlowDataset": "PhaseFlowDataset",
        "PhaseFlowOfflineDataset": "PhaseFlowOfflineDataset",
        "PhaseFlowBatchPlanDataset": "PhaseFlowBatchPlanDataset",
        "PhaseFlowPackedBatchDataset": "PhaseFlowPackedBatchDataset",
    }
    return legacy.get(name, name)


def train(config: dict, resume: str | Path | None = None) -> Path:
    install_strict_offline_guard()
    _validate_strict_offline_config(config)
    ddp = _init_distributed_from_env(config)
    set_seed(int(config.get("seed", 7)))
    _maybe_build_toy(config)
    _maybe_build_feature_pipeline(config)
    output_dir = Path(config.get("output_dir", "runs/phaseflow_train"))
    if ddp["is_rank0"]:
        output_dir.mkdir(parents=True, exist_ok=True)
    _distributed_barrier(ddp)
    device = ddp["device"] if ddp["enabled"] else resolve_device(str(config.get("device", "auto")))

    training_config = config.get("training", {})
    collator = PhaseFlowCollator(
        max_neighbors=int(training_config.get("max_neighbors", 96)),
        require_precomputed_graph=bool(training_config.get("require_precomputed_graph", False)),
    )
    num_workers = int(training_config.get("num_workers", 0))
    dataloader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(training_config.get("pin_memory", device.type == "cuda")),
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(training_config.get("persistent_workers", True))
        dataloader_kwargs["prefetch_factor"] = int(training_config.get("prefetch_factor", 2))
    batch_size = int(training_config.get("batch_size", 2))
    optimizer_batch_size = int(training_config.get("optimizer_batch_size", batch_size) or batch_size)
    if optimizer_batch_size <= 0 or optimizer_batch_size > batch_size:
        optimizer_batch_size = batch_size

    data_config = config.get("data", {})
    dataset_config = config.get("dataset", {}) or {}
    dataset_class = _normalize_dataset_class(
        dataset_config.get("type", data_config.get("dataset_class", config.get("dataset_class", "PhaseFlowDataset")))
    )
    _validate_full_benchmark_sample_index_guard(config, dataset_class)
    use_internal_validation = _use_internal_validation(config)
    train_sampler = None
    packed_batches_dir = training_config.get("packed_batches_dir")
    if dataset_class == "PhaseFlowPackedBatchDataset":
        train_dataset = PhaseFlowPackedBatchDataset(
            packed_dir=dataset_config["packed_dir"],
            batch_index=dataset_config.get("batch_index"),
            epoch_dirs=dataset_config.get("epoch_dirs"),
            epoch_index_files=dataset_config.get("epoch_index_files"),
            local_rank=int(ddp["local_rank"]),
            rank=int(ddp["rank"]),
            epoch_seed=dataset_config.get("epoch_seed"),
        )
        valid_dataset = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
            **dataloader_kwargs,
        )
        valid_loader = None
        train_sampler = train_dataset
        if ddp["enabled"] and int(ddp["world_size"]) != int(training_config.get("nproc_per_node", ddp["world_size"])):
            raise ValueError(
                f"Packed run expected nproc_per_node={training_config.get('nproc_per_node')}, "
                f"but torchrun WORLD_SIZE={ddp['world_size']}"
            )
    elif dataset_class == "PhaseFlowBatchPlanDataset":
        train_dataset = PhaseFlowBatchPlanDataset(
            plan_dir=dataset_config["plan_dir"],
            sample_index=dataset_config.get("sample_index", "data/processed/merged/tables/training_sample_index.parquet"),
            dataset_root=dataset_config.get("dataset_root", "data/processed/merged"),
            input_contract=dataset_config.get("input_contract"),
            esm2_store_metadata=dataset_config.get("esm2_store_metadata"),
            npz_mirror_manifest=dataset_config.get("npz_mirror_manifest"),
            region_supervision=str(dataset_config.get("region_supervision", "none")),
            local_rank=int(ddp["local_rank"]),
            rank=int(ddp["rank"]),
            max_neighbors=int(config.get("model", {}).get("max_neighbors", dataset_config.get("max_neighbors", 96))),
            edge_attr_dim=dataset_config.get("edge_attr_dim"),
            require_precomputed_graph=bool(dataset_config.get("require_precomputed_graph", False)),
            hot_cache_pools=dataset_config.get("hot_cache_pools"),
            hot_cache_max_samples=int(dataset_config.get("hot_cache_max_samples", 0) or 0),
        )
        train_sampler = train_dataset.make_sampler()
        valid_dataset = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=None,
            sampler=train_sampler,
            collate_fn=None,
            drop_last=False,
            **dataloader_kwargs,
        )
        valid_loader = None
        if ddp["enabled"] and int(ddp["world_size"]) != int(training_config.get("nproc_per_node", ddp["world_size"])):
            raise ValueError(
                f"Batch-plan run expected nproc_per_node={training_config.get('nproc_per_node')}, "
                f"but torchrun WORLD_SIZE={ddp['world_size']}"
            )
    elif packed_batches_dir:
        if ddp["enabled"]:
            raise ValueError("DDP training is only supported for map-style datasets, not packed batch datasets.")
        if dataset_class == "PhaseFlowOfflineDataset" or strict_offline_enabled():
            raise ValueError("Packed batch datasets are not part of the strict offline parquet/npz contract.")
        packed_dir = Path(str(packed_batches_dir))
        packed_num_workers = int(training_config.get("packed_num_workers", min(num_workers, 4)))
        packed_kwargs: dict[str, Any] = {
            "num_workers": packed_num_workers,
            "pin_memory": bool(training_config.get("pin_memory", device.type == "cuda")),
        }
        if packed_num_workers > 0:
            packed_kwargs["persistent_workers"] = bool(training_config.get("persistent_workers", True))
            packed_kwargs["prefetch_factor"] = int(training_config.get("packed_prefetch_factor", 2))
        train_dataset = PackedBatchDataset(packed_dir / "train")
        valid_dataset = PackedBatchDataset(packed_dir / "valid")
        train_loader = DataLoader(
            train_dataset,
            batch_size=None,
            shuffle=bool(training_config.get("shuffle_packed_batches", True)),
            **packed_kwargs,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=None,
            shuffle=False,
            **packed_kwargs,
        )
    elif dataset_class == "PhaseFlowOfflineDataset":
        train_dataset, valid_dataset = _make_offline_datasets(config)
        if not use_internal_validation:
            valid_dataset = None
        train_batch_sampler = None
        if ddp["enabled"]:
            train_batch_sampler = _offline_distributed_length_bucket_batch_sampler(
                train_dataset,
                batch_size=batch_size,
                training_config=training_config,
                shuffle=True,
                num_replicas=int(ddp["world_size"]),
                rank=int(ddp["rank"]),
            )
        else:
            train_batch_sampler = _offline_length_bucket_batch_sampler(
                train_dataset,
                batch_size=batch_size,
                training_config=training_config,
                shuffle=True,
            )
        valid_batch_sampler = _offline_length_bucket_batch_sampler(
            valid_dataset,
            batch_size=batch_size,
            training_config=training_config,
            shuffle=False,
            enabled=bool(training_config.get("length_bucketed_eval", training_config.get("length_bucketed_batches", False))),
        ) if valid_dataset is not None else None
        if ddp["enabled"] and train_batch_sampler is None:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=int(ddp["world_size"]),
                rank=int(ddp["rank"]),
                shuffle=True,
                drop_last=bool(training_config.get("drop_last", False)),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=train_sampler,
                collate_fn=collator,
                drop_last=bool(training_config.get("drop_last", False)),
                **dataloader_kwargs,
            )
        if train_batch_sampler is None:
            if not ddp["enabled"]:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collator,
                    drop_last=bool(training_config.get("drop_last", False)),
                    **dataloader_kwargs,
                )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                collate_fn=collator,
                **dataloader_kwargs,
            )
            train_sampler = train_batch_sampler
        if valid_dataset is None:
            valid_loader = None
        elif valid_batch_sampler is None:
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator,
                **dataloader_kwargs,
            )
        else:
            valid_loader = DataLoader(
                valid_dataset,
                batch_sampler=valid_batch_sampler,
                collate_fn=collator,
                **dataloader_kwargs,
            )
    else:
        if ddp["enabled"]:
            raise ValueError("DDP training is currently implemented for PhaseFlowOfflineDataset only.")
        train_ids = resolve_split_ids(config["data"], "train")
        valid_ids = resolve_split_ids(config["data"], "valid")
        phase_train_ids = _phase_train_ids(config["data"])
        train_dataset_ids = train_ids + phase_train_ids
        feature_dirs = resolve_feature_dirs(config["data"])
        phase_targets = resolve_phase_targets(config["data"])
        region_targets = config["data"].get("region_targets")
        validate_forbidden_data_paths(config["data"], feature_dirs)
        sampler = _phase_aux_sampler(
            base_ids=train_ids,
            phase_ids=phase_train_ids,
            training_config=training_config,
        )
        read_raw_edges = not bool(training_config.get("require_precomputed_graph", False))
        train_region_supervision = str(
            config["data"].get("train_region_supervision", config["data"].get("region_supervision", "feature"))
        )
        valid_region_supervision = str(config["data"].get("valid_region_supervision", "feature"))
        _validate_train_region_supervision(config, train_region_supervision)
        train_dataset = PhaseFlowDataset(
            feature_dirs,
            train_dataset_ids,
            phase_targets=phase_targets,
            region_targets=region_targets,
            region_supervision=train_region_supervision,
            read_raw_edges=read_raw_edges,
        )
        valid_dataset = PhaseFlowDataset(
            feature_dirs,
            valid_ids,
            phase_targets=phase_targets,
            region_targets=region_targets,
            region_supervision=valid_region_supervision,
            read_raw_edges=read_raw_edges,
        )
        train_batch_sampler = _length_bucket_batch_sampler(
            train_dataset,
            feature_dirs=feature_dirs,
            batch_size=batch_size,
            training_config=training_config,
            shuffle=True,
            enabled=sampler is None,
        )
        valid_batch_sampler = _length_bucket_batch_sampler(
            valid_dataset,
            feature_dirs=feature_dirs,
            batch_size=batch_size,
            training_config=training_config,
            shuffle=False,
            enabled=bool(training_config.get("length_bucketed_eval", training_config.get("length_bucketed_batches", False))),
        )
        if train_batch_sampler is None:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                collate_fn=collator,
                **dataloader_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                collate_fn=collator,
                **dataloader_kwargs,
            )
        if valid_batch_sampler is None:
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator,
                **dataloader_kwargs,
            )
        else:
            valid_loader = DataLoader(
                valid_dataset,
                batch_sampler=valid_batch_sampler,
                collate_fn=collator,
                **dataloader_kwargs,
            )

    model = PhaseFlowModel(config).to(device)
    resume_training_state = bool(training_config.get("resume_training_state", True))
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        strict_resume = bool(training_config.get("strict_resume", True))
        skipped_mismatched: list[dict[str, Any]] = []
        if not strict_resume:
            state_dict, skipped_mismatched = _compatible_state_dict(model, state_dict)
        incompatible = model.load_state_dict(state_dict, strict=strict_resume)
        if not strict_resume:
            print(
                dumps_json(
                    {
                        "resume_strict": False,
                        "missing_keys": list(incompatible.missing_keys),
                        "unexpected_keys": list(incompatible.unexpected_keys),
                        "skipped_mismatched_keys": skipped_mismatched,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not resume_training_state:
            print(
                dumps_json(
                    {
                        "event": "resume_weights_only",
                        "resume": str(resume),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    else:
        checkpoint = None
    _maybe_sync_llps_reference_dpr_head(model, training_config)
    _apply_freeze_config(model, training_config)
    if ddp["enabled"]:
        model = DistributedDataParallel(
            model,
            device_ids=[int(ddp["local_rank"])],
            output_device=int(ddp["local_rank"]),
            find_unused_parameters=bool(training_config.get("ddp_find_unused_parameters", False)),
        )
    else:
        if (
            str(data_config.get("dataset_class", config.get("dataset_class", ""))) == "PhaseFlowOfflineDataset"
            and bool(training_config.get("multi_gpu", False) or training_config.get("data_parallel", False))
        ):
            raise ValueError("DataParallel is disabled for strict offline training; launch multi-GPU runs with torchrun DDP.")
        model = _maybe_wrap_data_parallel(model, training_config, device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after applying training.freeze.")
    optimizer_param_groups = _build_optimizer_param_groups(model, training_config)
    optimizer = torch.optim.AdamW(
        optimizer_param_groups if optimizer_param_groups is not None else trainable_parameters,
        lr=float(training_config.get("lr", 1.0e-4)),
        weight_decay=float(training_config.get("weight_decay", 1.0e-4)),
    )
    amp_enabled = bool(training_config.get("amp", False)) and device.type == "cuda"
    amp_dtype = _amp_autocast_dtype(training_config)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and (amp_dtype is None or amp_dtype == torch.float16),
    )
    ema_config = training_config.get("ema", {}) or {}
    ema = _maybe_create_ema(model, ema_config)
    ema_eval = ema is not None and bool(ema_config.get("use_for_eval", True))
    ema_save = ema is not None and bool(ema_config.get("save_ema", True))
    if ema is not None and resume_training_state and isinstance(checkpoint, dict) and "ema" in checkpoint:
        ema.load_checkpoint_state(checkpoint["ema"], model)
    best_path = output_dir / "best.pt"
    best_train_loss_path = output_dir / "best_train_loss.pt"
    last_path = output_dir / "last.pt"
    best_score = -float("inf")
    best_smoothed_train_loss = float("inf")
    history: list[dict[str, float]] = []
    start_epoch = 1
    if resume_training_state and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        _move_optimizer_state_to_device(optimizer, device)
        if "scaler" in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])
        history = list(checkpoint.get("history", []))
        best_score = float(checkpoint.get("best_score", best_score))
        best_smoothed_train_loss = float(checkpoint.get("best_smoothed_train_loss", best_smoothed_train_loss))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        _maybe_seed_best_checkpoint_from_resume(Path(resume), best_path)
    max_epochs = int(training_config.get("max_epochs", 10))
    checkpoint_every_epochs = int(training_config.get("checkpoint_every_epochs", 0) or 0)
    early_stopping_config = training_config.get("early_stopping", {}) or {}
    early_stopping_enabled = bool(early_stopping_config.get("enabled", False)) and use_internal_validation
    early_stopping_patience = int(early_stopping_config.get("patience", 0) or 0)
    early_stopping_min_delta = float(early_stopping_config.get("min_delta", 0.0))
    epochs_without_improvement = 0
    log_every_batches = int(training_config.get("log_every_batches", 0) or 0)
    max_steps = int(training_config.get("max_steps", 0) or 0)
    timing_writer = _maybe_create_timing_writer(training_config, ddp)
    nan_debug_config = _nan_debug_config(config)
    nan_debug_writer = _maybe_create_nan_debug_writer(nan_debug_config, ddp)
    train_loss_smoothing = float(training_config.get("train_loss_smoothing", 0.9))
    total_train_batches = _total_optimizer_steps(
        sample_count=int(getattr(train_dataset, "sample_count", len(train_dataset))),
        loader_batch_size=batch_size,
        optimizer_batch_size=optimizer_batch_size,
        loader_batches=len(train_loader),
    )
    if start_epoch > max_epochs:
        if ddp["is_rank0"]:
            write_json(output_dir / "history.json", {"history": history})
        _cleanup_distributed(ddp)
        return best_path
    for epoch in range(start_epoch, max_epochs + 1):
        sampler_epoch = epoch - 1 if dataset_class in {"PhaseFlowPackedBatchDataset", "PhaseFlowBatchPlanDataset"} else epoch
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(sampler_epoch)
        if ddp["is_rank0"] and hasattr(train_sampler, "epoch_stats"):
            _write_sampler_epoch_stats(training_config.get("sampler_stats_csv"), train_sampler.epoch_stats(sampler_epoch))
        _reset_cuda_peak_memory(training_config, device)
        model.train()
        epoch_loss = 0.0
        batches = 0
        global_steps_this_run = 0
        dataloader_wait_sec = 0.0
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        loss_component_sums: dict[str, float] = {}
        tier_loss_sums: dict[str, float] = {}
        tier_loss_counts: dict[str, float] = {}
        samples_seen = 0.0
        residues_seen = 0.0
        epoch_start = time.perf_counter()
        last_batch_end = epoch_start
        timing_previous_end = epoch_start
        psutil.cpu_percent(None) if timing_writer is not None else None
        timing_disk_previous = psutil.disk_io_counters() if timing_writer is not None else None
        timing_disk_previous_time = time.perf_counter()
        for loader_batch in train_loader:
            batch_start = time.perf_counter()
            dataloader_wait_sec += batch_start - last_batch_end
            batch_load_sec = float(loader_batch.get("__packed_load_sec", 0.0) or 0.0) if isinstance(loader_batch, dict) else 0.0
            if bool(training_config.get("shuffle_microbatches", False)):
                loader_batch = _shuffle_batch_dimension(loader_batch)
            for batch in _iter_optimizer_microbatches(loader_batch, optimizer_batch_size):
                timing_data_wait_sec = batch_start - timing_previous_end
                timing_lengths = batch.get("lengths") if isinstance(batch, dict) else None
                if torch.is_tensor(timing_lengths):
                    timing_batch_size = int(timing_lengths.shape[0])
                    timing_real_residues = int(timing_lengths.long().sum().item())
                    timing_max_length = int(timing_lengths.max().item()) if timing_batch_size else 0
                    timing_padded_residues = int(timing_batch_size * timing_max_length)
                    timing_padding_ratio = (timing_padded_residues - timing_real_residues) / max(timing_padded_residues, 1)
                else:
                    timing_batch_size = 0
                    timing_real_residues = 0
                    timing_max_length = 0
                    timing_padded_residues = 0
                    timing_padding_ratio = 0.0
                _sync_cuda(device, enabled=timing_writer is not None)
                timing_h2d_start = time.perf_counter()
                batch = move_batch_to_device(batch, device)
                _sync_cuda(device, enabled=timing_writer is not None)
                timing_h2d_sec = time.perf_counter() - timing_h2d_start
                optimizer.zero_grad(set_to_none=True)
                autocast_kwargs: dict[str, Any] = {"enabled": amp_enabled}
                if amp_dtype is not None:
                    autocast_kwargs["dtype"] = amp_dtype
                _sync_cuda(device, enabled=timing_writer is not None)
                timing_forward_start = time.perf_counter()
                with torch.amp.autocast("cuda", **autocast_kwargs):
                    outputs = model(batch)
                _sync_cuda(device, enabled=timing_writer is not None)
                timing_forward_sec = time.perf_counter() - timing_forward_start
                timing_loss_start = time.perf_counter()
                loss_weights = _loss_weights_for_step(
                    training_config.get("loss_weights", {}),
                    training_config,
                    global_steps_this_run + 1,
                )
                loss_context = (
                    torch.amp.autocast("cuda", enabled=False)
                    if _force_fp32_loss(loss_weights, training_config) and device.type == "cuda"
                    else torch.amp.autocast("cuda", **autocast_kwargs)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with loss_context:
                    loss, loss_values = compute_multitask_loss(outputs, batch, loss_weights)
                _sync_cuda(device, enabled=timing_writer is not None)
                timing_loss_sec = time.perf_counter() - timing_loss_start
                batch_label_stats = _batch_label_logit_stats(outputs, batch) if timing_writer is not None else {}
                if not bool(torch.isfinite(loss).detach().cpu()):
                    _nan_debug_write_bad_batch(
                        nan_debug_config,
                        ddp,
                        epoch=epoch,
                        step=global_steps_this_run + 1,
                        local_batch_index=batches + 1,
                        batch=batch,
                        outputs=outputs,
                        loss_values=loss_values,
                        total_loss=loss,
                        amp_scale=scaler.get_scale(),
                        grad_norm=None,
                        issues=[
                            {
                                "name": "total_loss",
                                "reason": "nonfinite",
                                "value": float(loss.detach().cpu()),
                            }
                        ],
                    )
                    raise RuntimeError(f"Non-finite loss on rank {ddp['rank']}: {float(loss.detach().cpu())}")
                nan_debug_issues = _nan_debug_finite_issues(
                    nan_debug_config,
                    batch=batch,
                    outputs=outputs,
                    loss_values=loss_values,
                    total_loss=loss,
                    grad_norm=None,
                )
                if nan_debug_issues and bool(nan_debug_config.get("stop_on_nonfinite", True)):
                    _nan_debug_write_bad_batch(
                        nan_debug_config,
                        ddp,
                        epoch=epoch,
                        step=global_steps_this_run + 1,
                        local_batch_index=batches + 1,
                        batch=batch,
                        outputs=outputs,
                        loss_values=loss_values,
                        total_loss=loss,
                        amp_scale=scaler.get_scale(),
                        grad_norm=None,
                        issues=nan_debug_issues,
                    )
                    raise RuntimeError(
                        f"NaN diagnostic finite check failed on rank {ddp['rank']}: "
                        f"{json.dumps(nan_debug_issues[:5], sort_keys=True)}"
                    )
                for name, value in loss_values.items():
                    loss_component_sums[name] = loss_component_sums.get(name, 0.0) + float(value)
                _accumulate_tier_losses(outputs, batch, tier_loss_sums, tier_loss_counts)
                timing_backward_start = time.perf_counter()
                scaler.scale(loss).backward()
                _sync_cuda(device, enabled=timing_writer is not None)
                timing_backward_sec = time.perf_counter() - timing_backward_start
                timing_optimizer_start = time.perf_counter()
                scaler.unscale_(optimizer)
                grad_clip_norm = float(training_config.get("grad_clip_norm", 1.0))
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if not bool(torch.isfinite(grad_norm).detach().cpu()):
                    _nan_debug_write_bad_batch(
                        nan_debug_config,
                        ddp,
                        epoch=epoch,
                        step=global_steps_this_run + 1,
                        local_batch_index=batches + 1,
                        batch=batch,
                        outputs=outputs,
                        loss_values=loss_values,
                        total_loss=loss,
                        amp_scale=scaler.get_scale(),
                        grad_norm=grad_norm,
                        issues=[
                            {
                                "name": "grad_norm",
                                "reason": "nonfinite",
                                "value": float(grad_norm.detach().cpu()),
                            }
                        ],
                    )
                    raise RuntimeError(f"Non-finite gradient norm on rank {ddp['rank']}: {float(grad_norm.detach().cpu())}")
                grad_norm_value = float(grad_norm.detach().cpu())
                _nan_debug_write_row(
                    nan_debug_writer,
                    nan_debug_config,
                    ddp,
                    epoch=epoch,
                    step=global_steps_this_run + 1,
                    local_batch_index=batches + 1,
                    batch=batch,
                    outputs=outputs,
                    loss_values=loss_values,
                    total_loss=loss,
                    amp_scale=scaler.get_scale(),
                    grad_norm=grad_norm_value,
                    issues=[],
                )
                grad_norm_sum += grad_norm_value
                grad_norm_max = max(grad_norm_max, grad_norm_value)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                _sync_cuda(device, enabled=timing_writer is not None)
                timing_optimizer_sec = time.perf_counter() - timing_optimizer_start
                timing_ddp_sync_sec = _ddp_sync_probe(ddp) if timing_writer is not None else 0.0
                timing_step_end = time.perf_counter()
                epoch_loss += float(loss.detach().cpu())
                batches += 1
                global_steps_this_run += 1
                samples_seen += float(batch["y_llps"].shape[0])
                residues_seen += float(batch["seq_mask"].sum().detach().cpu())
                if timing_writer is not None:
                    timing_disk_now = psutil.disk_io_counters()
                    timing_disk_now_time = time.perf_counter()
                    disk_read_mb_sec = 0.0
                    if timing_disk_previous is not None and timing_disk_now is not None:
                        disk_read_mb_sec = max(float(timing_disk_now.read_bytes - timing_disk_previous.read_bytes), 0.0) / 1.0e6 / max(
                            timing_disk_now_time - timing_disk_previous_time,
                            1.0e-6,
                        )
                    timing_disk_previous = timing_disk_now
                    timing_disk_previous_time = timing_disk_now_time
                    gpu_util, gpu_mem = _gpu_sample(int(ddp["local_rank"])) if global_steps_this_run == 1 or global_steps_this_run % 10 == 0 else (float("nan"), float("nan"))
                    timing_writer.writerow(
                        {
                            "epoch": epoch,
                            "rank": int(ddp["rank"]),
                            "local_rank": int(ddp["local_rank"]),
                            "world_size": int(ddp["world_size"]),
                            "step": global_steps_this_run,
                            "total_step_sec": timing_step_end - timing_previous_end,
                            "data_wait_sec": timing_data_wait_sec,
                            "dataset_getitem_sec": batch_load_sec,
                            "batch_read_sec": batch_load_sec,
                            "graph_file_read_sec": 0.0,
                            "embedding_file_read_sec": 0.0,
                            "collate_sec": 0.0,
                            "host_to_device_sec": timing_h2d_sec,
                            "forward_sec": timing_forward_sec,
                            "loss_sec": timing_loss_sec,
                            "backward_sec": timing_backward_sec,
                            "optimizer_step_sec": timing_optimizer_sec,
                            "ddp_sync_sec": timing_ddp_sync_sec,
                            "batch_size": timing_batch_size,
                            "max_length": timing_max_length,
                            "real_residues": timing_real_residues,
                            "padded_residues": timing_padded_residues,
                            "padding_ratio": timing_padding_ratio,
                            "samples_per_sec_rank": timing_batch_size / max(timing_step_end - timing_previous_end, 1.0e-6),
                            "residues_per_sec_rank": timing_real_residues / max(timing_step_end - timing_previous_end, 1.0e-6),
                            "gpu_util_pct": gpu_util,
                            "gpu_mem_mb": gpu_mem,
                            "cuda_allocated_gb": _bytes_to_gb(torch.cuda.memory_allocated(device) if device.type == "cuda" else 0),
                            "cuda_reserved_gb": _bytes_to_gb(torch.cuda.memory_reserved(device) if device.type == "cuda" else 0),
                            "cuda_peak_allocated_gb": _bytes_to_gb(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
                            "cpu_util_pct": psutil.cpu_percent(None),
                            "disk_read_mb_sec": disk_read_mb_sec,
                            "loss_value": float(loss.detach().cpu()),
                            "isfinite_loss": int(bool(torch.isfinite(loss).detach().cpu())),
                            **_runtime_guard_counters(),
                            **batch_label_stats,
                            "protein_loss_pos": float(loss_values.get("protein_loss_pos", 0.0)),
                            "protein_loss_neg": float(loss_values.get("protein_loss_neg", 0.0)),
                            "protein_bce": float(loss_values.get("llps", 0.0)),
                            "nnpu_loss": float(loss_values.get("nnpu", 0.0)),
                        }
                    )
                timing_previous_end = timing_step_end
                if ddp["is_rank0"] and log_every_batches > 0 and batches % log_every_batches == 0:
                    elapsed = time.perf_counter() - epoch_start
                    progress = {
                        "event": "train_progress",
                        "epoch": epoch,
                        "batch": batches,
                        "batches": total_train_batches,
                        "avg_train_loss": epoch_loss / max(batches, 1),
                        "avg_grad_norm": grad_norm_sum / max(batches, 1),
                        "max_grad_norm": grad_norm_max,
                        "avg_dataloader_wait_sec": dataloader_wait_sec / max(batches, 1),
                        "samples_sec": (samples_seen * int(ddp.get("world_size", 1))) / max(elapsed, 1.0e-6),
                        "residues_sec": (residues_seen * int(ddp.get("world_size", 1))) / max(elapsed, 1.0e-6),
                        "lr": float(optimizer.param_groups[0].get("lr", 0.0)),
                        "elapsed_sec": round(elapsed, 2),
                        **_runtime_guard_counters(),
                    }
                    for name, value in sorted(loss_component_sums.items()):
                        progress[f"loss_component_{_metric_key(name)}"] = value / max(batches, 1)
                    for tier, value in sorted(tier_loss_sums.items()):
                        key = _metric_key(tier)
                        count = tier_loss_counts.get(tier, 0.0)
                        progress[f"tier_loss_{key}"] = value / max(count, 1.0)
                        progress[f"tier_count_{key}"] = count
                    print(
                        dumps_json(progress, sort_keys=True),
                        flush=True,
                    )
                if max_steps > 0 and global_steps_this_run >= max_steps:
                    break
            last_batch_end = time.perf_counter()
            if max_steps > 0 and global_steps_this_run >= max_steps:
                break
        train_loss = epoch_loss / max(batches, 1)
        train_loss = _distributed_mean_float(train_loss, ddp)
        global_batches = _distributed_sum_float(float(batches), ddp)
        global_samples_seen = _distributed_sum_float(samples_seen, ddp)
        global_residues_seen = _distributed_sum_float(residues_seen, ddp)
        global_loss_component_sums = _distributed_merge_number_dict(loss_component_sums, ddp)
        global_tier_loss_sums = _distributed_merge_number_dict(tier_loss_sums, ddp)
        global_tier_loss_counts = _distributed_merge_number_dict(tier_loss_counts, ddp)
        epoch_elapsed = time.perf_counter() - epoch_start
        if use_internal_validation and valid_loader is not None:
            if ema_eval and ema is not None:
                ema.store(model)
                ema.copy_to(model)
            try:
                metrics = evaluate_model(_module_for_eval(model), valid_loader, device, config.get("postprocess", {}))
            finally:
                if ema_eval and ema is not None:
                    ema.restore(model)
        else:
            metrics = {
                "checkpoint_policy": "full_train_no_internal_validation",
                "validation_used": 0.0,
            }
            if ema_eval and ema is not None:
                metrics["ema_updates"] = float(ema.num_updates)
        score = _score_for_checkpoint(metrics, target=str(training_config.get("checkpoint_target", "joint"))) if use_internal_validation else -train_loss
        row = {"epoch": float(epoch), "train_loss": train_loss, **metrics, **_cuda_memory_metrics(training_config, device)}
        row.update(
            {
                "avg_grad_norm": grad_norm_sum / max(batches, 1),
                "max_grad_norm": grad_norm_max,
                "dataloader_wait_sec": dataloader_wait_sec,
                "avg_dataloader_wait_sec": dataloader_wait_sec / max(batches, 1),
                "samples_seen": global_samples_seen,
                "residues_seen": global_residues_seen,
                "samples_sec": global_samples_seen / max(epoch_elapsed, 1.0e-6),
                "residues_sec": global_residues_seen / max(epoch_elapsed, 1.0e-6),
                "lr": float(optimizer.param_groups[0].get("lr", 0.0)),
                **_runtime_guard_counters(),
            }
        )
        for name, value in sorted(global_loss_component_sums.items()):
            row[f"loss_component_{_metric_key(name)}"] = value / max(global_batches, 1.0)
        for tier, value in sorted(global_tier_loss_sums.items()):
            key = _metric_key(tier)
            count = global_tier_loss_counts.get(tier, 0.0)
            row[f"tier_loss_{key}"] = value / max(count, 1.0)
            row[f"tier_count_{key}"] = count
        if ema is not None:
            row.update({"ema_decay": float(ema.decay), "ema_updates": float(ema.num_updates), "ema_eval": float(ema_eval)})
        previous_smoothed = history[-1].get("smoothed_train_loss", train_loss) if history else train_loss
        smoothed_train_loss = train_loss_smoothing * float(previous_smoothed) + (1.0 - train_loss_smoothing) * train_loss
        row["smoothed_train_loss"] = smoothed_train_loss
        history.append(row)
        if ddp["is_rank0"]:
            print(dumps_json(row, sort_keys=True), flush=True)
        improvement_delta = early_stopping_min_delta if early_stopping_enabled else 0.0
        improved = score > best_score + improvement_delta
        if ddp["is_rank0"] and improved and use_internal_validation:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": _best_checkpoint_model_state(model, ema, use_ema=ema_eval),
                    "config": config,
                    "metrics": metrics,
                    "epoch": epoch,
                    "history": history,
                    "best_score": best_score,
                    "best_smoothed_train_loss": best_smoothed_train_loss,
                    "ema": _checkpoint_ema_state(ema if ema_save else None),
                    "ema_eval": ema_eval,
                },
                best_path,
            )
        elif early_stopping_enabled and use_internal_validation:
            epochs_without_improvement += 1
        if ddp["is_rank0"] and smoothed_train_loss < best_smoothed_train_loss:
            best_smoothed_train_loss = smoothed_train_loss
            torch.save(
                {
                    "model": _best_checkpoint_model_state(model, ema, use_ema=ema_eval),
                    "config": config,
                    "metrics": metrics,
                    "epoch": epoch,
                    "history": history,
                    "best_score": best_score,
                    "best_smoothed_train_loss": best_smoothed_train_loss,
                    "ema": _checkpoint_ema_state(ema if ema_save else None),
                    "ema_eval": ema_eval,
                },
                best_train_loss_path,
            )
            shutil.copy2(best_train_loss_path, output_dir / "best_smoothed_train_loss.ckpt")
        if ddp["is_rank0"]:
            torch.save(
                {
                    "model": _checkpoint_state_dict(model),
                    "config": config,
                    "metrics": metrics,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "history": history,
                    "best_score": best_score,
                    "best_smoothed_train_loss": best_smoothed_train_loss,
                    "ema": _checkpoint_ema_state(ema if ema_save else None),
                    "ema_eval": ema_eval,
                },
                last_path,
            )
            shutil.copy2(last_path, output_dir / "last.ckpt")
            torch.save(
                {
                    "model": _best_checkpoint_model_state(model, ema, use_ema=ema_eval),
                    "config": config,
                    "metrics": metrics,
                    "epoch": epoch,
                    "history": history,
                    "best_score": best_score,
                    "best_smoothed_train_loss": best_smoothed_train_loss,
                    "ema": _checkpoint_ema_state(ema if ema_save else None),
                    "ema_eval": ema_eval,
                },
                output_dir / "model.pt",
            )
            shutil.copy2(output_dir / "model.pt", output_dir / "model.ckpt")
            if ema is not None and ema_save:
                torch.save(
                    {
                        "model": ema.model_state_dict(cpu=True),
                        "config": config,
                        "metrics": metrics,
                        "epoch": epoch,
                        "history": history,
                        "ema": _checkpoint_ema_state(ema),
                        "ema_eval": ema_eval,
                    },
                    output_dir / "ema.ckpt",
                )
            if checkpoint_every_epochs > 0 and epoch % checkpoint_every_epochs == 0:
                torch.save(
                    {
                        "model": _checkpoint_state_dict(model),
                        "config": config,
                        "metrics": metrics,
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "history": history,
                        "best_score": best_score,
                        "best_smoothed_train_loss": best_smoothed_train_loss,
                        "ema": _checkpoint_ema_state(ema if ema_save else None),
                        "ema_eval": ema_eval,
                    },
                    output_dir / f"epoch_{epoch:03d}.pt",
                )
            write_json(output_dir / "history.json", {"history": history})
            _write_history_csv(training_config.get("metrics_csv"), history)
            _write_tier_metrics_csv(training_config.get("tier_metrics_csv"), history)
            if bool(training_config.get("checkpoint_reload_check", False)):
                _checkpoint_reload_check(last_path, output_dir / "checkpoint_reload_check.json")
        if bool(training_config.get("empty_cache_each_epoch", False)) and device.type == "cuda":
            torch.cuda.empty_cache()
        if early_stopping_enabled and early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            if ddp["is_rank0"]:
                print(
                    dumps_json(
                        {
                            "early_stopped": True,
                            "epoch": epoch,
                            "best_score": best_score,
                            "patience": early_stopping_patience,
                            "min_delta": early_stopping_min_delta,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            break
        if max_steps > 0:
            break
    if ddp["is_rank0"]:
        write_json(output_dir / "history.json", {"history": history})
        _write_history_csv(training_config.get("metrics_csv"), history)
        _write_tier_metrics_csv(training_config.get("tier_metrics_csv"), history)
    _finalize_nan_debug_writer(nan_debug_writer, nan_debug_config, ddp)
    _finalize_timing_writer(timing_writer, training_config, ddp)
    _distributed_barrier(ddp)
    _cleanup_distributed(ddp)
    return best_path if use_internal_validation else best_train_loss_path


def _score_for_checkpoint(metrics: dict[str, float], target: str = "joint") -> float:
    target = str(target or "joint").strip().lower()
    if target in {"llps", "protein", "protein_llps"}:
        reward_names = ("prauc", "auc", "f1", "mcc")
        penalty_names = ("fpr", "ece", "FPR_on_ND", "FPR_on_NP")
        reward = _finite_metric_sum(metrics, reward_names, default=-float("inf"))
        penalty = _finite_metric_sum(metrics, penalty_names, default=0.0)
        return reward - 0.25 * penalty
    if target in {"dpr_region_first", "region_first", "paper_region", "dpr_paper"}:
        region_score = 3.0 * _finite_metric_sum(
            metrics,
            (
                "region_iou@0.3_precision",
                "region_iou@0.3_recall",
                "region_iou@0.5_precision",
                "region_iou@0.5_recall",
            ),
            default=0.0,
        )
        boundary_score = 2.0 * _finite_metric_sum(metrics, ("boundary_f1",), default=0.0)
        residue_score = 0.25 * _finite_metric_sum(metrics, ("residue_f1", "residue_dice"), default=0.0)
        score = region_score + boundary_score + residue_score
        return score if score > 0.0 else -float("inf")
    if target in {"dpr", "region", "residue"}:
        names = (
            "region_iou@0.5_precision",
            "region_iou@0.5_recall",
            "region_iou@0.3_precision",
            "region_iou@0.3_recall",
            "boundary_f1",
            "residue_f1",
            "residue_dice",
        )
        score = _finite_metric_sum(metrics, names, default=0.0)
        return score if score > 0.0 else -float("inf")
    names = (
        "prauc",
        "auc",
        "f1",
        "region_iou@0.5_precision",
        "region_iou@0.5_recall",
        "region_iou@0.3_precision",
        "region_iou@0.3_recall",
    )
    values = [float(metrics[name]) for name in names if name in metrics and metrics[name] == metrics[name]]
    if not values:
        return -float("inf")
    protein = values[:3]
    region = values[3:]
    return float(sum(protein) + 0.5 * sum(region))


def _finite_metric_sum(metrics: dict[str, float], names: tuple[str, ...], *, default: float) -> float:
    values = [float(metrics[name]) for name in names if name in metrics and metrics[name] == metrics[name]]
    return float(sum(values)) if values else float(default)


def _maybe_wrap_data_parallel(model: torch.nn.Module, training_config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    enabled = bool(training_config.get("multi_gpu", False) or training_config.get("data_parallel", False))
    if not enabled or device.type != "cuda" or torch.cuda.device_count() < 2:
        return model
    device_ids = _resolve_device_ids(training_config.get("device_ids"))
    if len(device_ids) < 2:
        return model
    print(f"Using torch.nn.DataParallel on CUDA device ids: {device_ids}", flush=True)
    return torch.nn.DataParallel(model, device_ids=device_ids)


def _init_distributed_from_env(config: dict[str, Any]) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    rank = int(os.environ.get("RANK", "0") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
    enabled = world_size > 1
    if not enabled:
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "is_rank0": True,
            "device": resolve_device(str(config.get("device", "auto"))),
        }
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA devices visible to every torchrun process.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        backend = str(config.get("training", {}).get("ddp_backend", "nccl"))
        try:
            dist.init_process_group(backend=backend, device_id=device)
        except TypeError:
            dist.init_process_group(backend=backend)
    return {
        "enabled": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_rank0": rank == 0,
        "device": device,
    }


def _distributed_barrier(ddp: dict[str, Any]) -> None:
    if bool(ddp.get("enabled")) and dist.is_available() and dist.is_initialized():
        try:
            dist.barrier(device_ids=[int(ddp["local_rank"])])
        except TypeError:
            dist.barrier()


def _cleanup_distributed(ddp: dict[str, Any]) -> None:
    if bool(ddp.get("enabled")) and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_mean_float(value: float, ddp: dict[str, Any]) -> float:
    if not bool(ddp.get("enabled")):
        return float(value)
    device = ddp["device"]
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= int(ddp["world_size"])
    return float(tensor.detach().cpu())


def _distributed_sum_float(value: float, ddp: dict[str, Any]) -> float:
    if not bool(ddp.get("enabled")):
        return float(value)
    tensor = torch.tensor(float(value), device=ddp["device"], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.detach().cpu())


def _distributed_merge_number_dict(values: dict[str, float], ddp: dict[str, Any]) -> dict[str, float]:
    if not bool(ddp.get("enabled")):
        return dict(values)
    gathered: list[Any] = [None for _ in range(int(ddp["world_size"]))]
    dist.all_gather_object(gathered, dict(values))
    merged: dict[str, float] = {}
    for item in gathered:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            merged[str(key)] = merged.get(str(key), 0.0) + float(value)
    return merged


TIMING_FIELDS = [
    "epoch",
    "rank",
    "local_rank",
    "world_size",
    "step",
    "total_step_sec",
    "data_wait_sec",
    "dataset_getitem_sec",
    "batch_read_sec",
    "graph_file_read_sec",
    "embedding_file_read_sec",
    "collate_sec",
    "host_to_device_sec",
    "forward_sec",
    "loss_sec",
    "backward_sec",
    "optimizer_step_sec",
    "ddp_sync_sec",
    "batch_size",
    "max_length",
    "real_residues",
    "padded_residues",
    "padding_ratio",
    "samples_per_sec_rank",
    "residues_per_sec_rank",
    "gpu_util_pct",
    "gpu_mem_mb",
    "cuda_allocated_gb",
    "cuda_reserved_gb",
    "cuda_peak_allocated_gb",
    "cpu_util_pct",
    "disk_read_mb_sec",
    "loss_value",
    "isfinite_loss",
    "runtime_guard_legacy_h5_read",
    "runtime_guard_blocked_path_read",
    "runtime_guard_embedding_build",
    "runtime_guard_graph_build",
    "runtime_guard_edge_merge",
    "runtime_guard_non_merged_graph_read",
    "batch_positive_count",
    "batch_negative_count",
    "batch_pu_count",
    "rank_local_positive_count",
    "positive_logit_mean",
    "positive_logit_std",
    "negative_logit_mean",
    "negative_logit_std",
    "pu_logit_mean",
    "pu_logit_std",
    "pred_score_mean",
    "pred_score_std",
    "protein_loss_pos",
    "protein_loss_neg",
    "protein_bce",
    "nnpu_loss",
]


def _maybe_create_timing_writer(training_config: dict[str, Any], ddp: dict[str, Any]) -> csv.DictWriter | None:
    path_value = training_config.get("perf_timing_csv")
    if not path_value or int(training_config.get("max_steps", 0) or 0) <= 0:
        return None
    path = Path(str(path_value))
    rank_path = _rank_timing_path(path, int(ddp["rank"]))
    rank_path.parent.mkdir(parents=True, exist_ok=True)
    handle = rank_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
    writer.writeheader()
    setattr(writer, "_phaseflow_handle", handle)
    setattr(writer, "_phaseflow_rank_path", rank_path)
    return writer


def _finalize_timing_writer(writer: csv.DictWriter | None, training_config: dict[str, Any], ddp: dict[str, Any]) -> None:
    if writer is None:
        return
    handle = getattr(writer, "_phaseflow_handle", None)
    if handle is not None:
        handle.flush()
        handle.close()
    _distributed_barrier(ddp)
    if not bool(ddp.get("is_rank0", False)):
        return
    final_path = Path(str(training_config.get("perf_timing_csv")))
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", newline="", encoding="utf-8") as out:
        csv_out = csv.DictWriter(out, fieldnames=TIMING_FIELDS)
        csv_out.writeheader()
        for rank in range(int(ddp.get("world_size", 1))):
            rank_path = _rank_timing_path(final_path, rank)
            if not rank_path.exists():
                continue
            with rank_path.open("r", newline="", encoding="utf-8") as handle_in:
                reader = csv.DictReader(handle_in)
                for row in reader:
                    csv_out.writerow(row)


def _rank_timing_path(path: Path, rank: int) -> Path:
    return path.with_name(f"{path.stem}.rank{int(rank)}{path.suffix}")


def _ddp_sync_probe(ddp: dict[str, Any]) -> float:
    if not bool(ddp.get("enabled")) or not dist.is_available() or not dist.is_initialized():
        return 0.0
    device = ddp["device"]
    tensor = torch.ones((), device=device)
    _sync_cuda(device, enabled=True)
    start = time.perf_counter()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    _sync_cuda(device, enabled=True)
    return time.perf_counter() - start


def _sync_cuda(device: torch.device, *, enabled: bool = True) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_sample(local_rank: int) -> tuple[float, float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={int(local_rank)}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            cwd=Path.cwd(),
            check=False,
            text=True,
            capture_output=True,
            timeout=2.0,
        )
    except Exception:
        return float("nan"), float("nan")
    if result.returncode != 0 or not result.stdout.strip():
        return float("nan"), float("nan")
    parts = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
    try:
        return float(parts[0]), float(parts[1])
    except (IndexError, ValueError):
        return float("nan"), float("nan")


def _batch_label_logit_stats(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, float]:
    logits = outputs.get("loss_llps_logits", outputs.get("llps_logits", outputs["llps_logits"])).detach().reshape(-1)
    labels = batch["y_llps"].detach().reshape(-1)
    pos = labels == 1.0
    neg = labels == 0.0
    pu = labels == float(IGNORE_INDEX)
    scores = torch.sigmoid(logits)

    def mean_std(mask: torch.Tensor, values: torch.Tensor) -> tuple[float, float]:
        if not bool(torch.any(mask).detach().cpu()):
            return float("nan"), float("nan")
        selected = values[mask].float()
        return float(selected.mean().detach().cpu()), float(selected.std(unbiased=False).detach().cpu())

    pos_mean, pos_std = mean_std(pos, logits)
    neg_mean, neg_std = mean_std(neg, logits)
    pu_mean, pu_std = mean_std(pu, logits)
    score_mean, score_std = mean_std(torch.ones_like(labels, dtype=torch.bool), scores)
    return {
        "batch_positive_count": int(pos.sum().detach().cpu()),
        "batch_negative_count": int(neg.sum().detach().cpu()),
        "batch_pu_count": int(pu.sum().detach().cpu()),
        "rank_local_positive_count": int(pos.sum().detach().cpu()),
        "positive_logit_mean": pos_mean,
        "positive_logit_std": pos_std,
        "negative_logit_mean": neg_mean,
        "negative_logit_std": neg_std,
        "pu_logit_mean": pu_mean,
        "pu_logit_std": pu_std,
        "pred_score_mean": score_mean,
        "pred_score_std": score_std,
    }


def _bytes_to_gb(value: int | float) -> float:
    return float(value) / (1024.0**3)


def _module_for_eval(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (torch.nn.DataParallel, DistributedDataParallel)):
        return model.module
    return model


def _use_internal_validation(config: dict[str, Any]) -> bool:
    training_config = config.get("training", {})
    if bool(training_config.get("full_train_no_internal_validation", False)):
        return False
    if "use_internal_validation" in training_config:
        return bool(training_config.get("use_internal_validation"))
    return True


def _accumulate_tier_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    sums: dict[str, float],
    counts: dict[str, float],
) -> None:
    protein_logits = outputs.get("loss_llps_logits", outputs["llps_logits"]).detach().reshape(-1)
    targets = batch["y_llps"].detach().reshape(-1)
    sample_weight = batch.get("sample_weight")
    if sample_weight is None:
        weights = torch.ones_like(targets, dtype=torch.float32)
    else:
        weights = sample_weight.detach().reshape(-1).float()
    valid = targets != float(IGNORE_INDEX)
    if not bool(torch.any(valid).detach().cpu()):
        return
    losses = F.binary_cross_entropy_with_logits(protein_logits[valid], targets[valid].float(), reduction="none")
    losses = losses * weights[valid]
    tiers = batch.get("label_quality") or batch.get("merged_label_tier") or []
    if not tiers:
        tiers = ["unknown" for _ in range(int(targets.numel()))]
    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1).detach().cpu().tolist()
    losses_cpu = losses.detach().cpu().tolist()
    for index, loss_value in zip(valid_indices, losses_cpu, strict=False):
        tier = str(tiers[int(index)]) if int(index) < len(tiers) else "unknown"
        sums[tier] = sums.get(tier, 0.0) + float(loss_value)
        counts[tier] = counts.get(tier, 0.0) + 1.0


def _metric_key(value: str) -> str:
    text = str(value).strip().lower()
    return "".join(character if character.isalnum() else "_" for character in text).strip("_") or "unknown"


def _write_history_csv(path_value: Any, history: list[dict[str, float]]) -> None:
    if not path_value or not history:
        return
    path = Path(str(path_value))
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in history for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _write_tier_metrics_csv(path_value: Any, history: list[dict[str, float]]) -> None:
    if not path_value or not history:
        return
    path = Path(str(path_value))
    path.parent.mkdir(parents=True, exist_ok=True)
    tiers: set[str] = set()
    for row in history:
        for key in row:
            if key.startswith("tier_loss_"):
                tiers.add(key[len("tier_loss_") :])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "tier", "train_loss", "count"])
        writer.writeheader()
        for row in history:
            for tier in sorted(tiers):
                loss_key = f"tier_loss_{tier}"
                count_key = f"tier_count_{tier}"
                if loss_key in row:
                    writer.writerow(
                        {
                            "epoch": row.get("epoch", ""),
                            "tier": tier,
                            "train_loss": row.get(loss_key, ""),
                            "count": row.get(count_key, ""),
                        }
                    )


def _runtime_guard_counters() -> dict[str, float]:
    return {
        "runtime_guard_legacy_h5_read": 0.0,
        "runtime_guard_blocked_path_read": 0.0,
        "runtime_guard_embedding_build": 0.0,
        "runtime_guard_graph_build": 0.0,
        "runtime_guard_edge_merge": 0.0,
        "runtime_guard_non_merged_graph_read": 0.0,
    }


LOSS_COMPONENT_KEYS = [
    "loss",
    "llps",
    "protein_loss_pos",
    "protein_loss_neg",
    "protein_loss_pos_count",
    "protein_loss_neg_count",
    "protein_loss_missing_class",
    "teacher_llps",
    "self_llps",
    "nnpu",
    "calibration",
    "dpr",
    "region_gold",
    "region_mil",
    "teacher_dpr",
    "teacher_distill",
    "self_dpr",
    "region",
    "coverage",
    "key",
    "smoothness",
    "negative_regularization",
    "phase_aux",
    "region_teacher",
    "region_key_teacher",
    "region_boundary",
    "region_contrastive",
    "top_negative_ranking",
    "hard_negative_focal",
    "weighted_focal_bce",
    "pairwise_rank",
    "driver_aux",
    "client_aux",
    "negtype_aux",
]


NAN_DEBUG_FIELDS = [
    "epoch",
    "global_step",
    "rank",
    "local_rank",
    "world_size",
    "local_batch_index",
    "protein_ids_json",
    "sample_ids_json",
    "sequence_lengths_json",
    "region_span_count",
    "region_span_count_json",
    "positive_residue_count",
    "negative_residue_count",
    "valid_residue_mask_sum",
    "proposal_mask_sum",
    "mil_window_count",
    "dpr_logit_min",
    "dpr_logit_max",
    "region_logit_min",
    "region_logit_max",
    "llps_logit_min",
    "llps_logit_max",
    "target_min",
    "target_max",
    "input_plm_min",
    "input_plm_max",
    "input_physchem_min",
    "input_physchem_max",
    "input_disorder_min",
    "input_disorder_max",
    "input_protenix_embed_min",
    "input_protenix_embed_max",
    "edge_attr_min",
    "edge_attr_max",
    "starling_abs_sum",
    "starling_reliability_sum",
    "total_loss",
    "amp_scale",
    "grad_norm",
    "finite_issue_count",
    "finite_issues_json",
    "loss_components_json",
    *[f"loss_component_{name}" for name in LOSS_COMPONENT_KEYS],
]


def _nan_debug_config(config: dict[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if isinstance(config.get("nan_debug"), dict):
        raw.update(config.get("nan_debug") or {})
    training_config = config.get("training", {}) or {}
    if isinstance(training_config.get("nan_debug"), dict):
        raw.update(training_config.get("nan_debug") or {})
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return {"enabled": False}
    output_dir = Path(str(config.get("output_dir", "runs/phaseflow_train")))
    run_root = output_dir.parent.parent if output_dir.parent.name == "checkpoints" else output_dir
    stage = str((config.get("metadata", {}) or {}).get("stage", output_dir.name))
    raw.setdefault("loss_components_csv", str(run_root / "metrics" / f"{stage}_nan_debug_loss_components.csv"))
    raw.setdefault("bad_batch_dir", str(run_root / "debug"))
    raw.setdefault("report_path", str(run_root / "reports" / f"{stage}_nan_debug_report.md"))
    raw.setdefault("stop_on_nonfinite", True)
    return raw


def _maybe_create_nan_debug_writer(config: dict[str, Any], ddp: dict[str, Any]) -> csv.DictWriter | None:
    if not bool(config.get("enabled", False)):
        return None
    path = Path(str(config.get("loss_components_csv")))
    rank_path = _rank_timing_path(path, int(ddp["rank"]))
    rank_path.parent.mkdir(parents=True, exist_ok=True)
    if bool(ddp.get("is_rank0", False)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=NAN_DEBUG_FIELDS).writeheader()
    handle = rank_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=NAN_DEBUG_FIELDS)
    writer.writeheader()
    setattr(writer, "_phaseflow_handle", handle)
    setattr(writer, "_phaseflow_rank_path", rank_path)
    return writer


def _finalize_nan_debug_writer(writer: csv.DictWriter | None, config: dict[str, Any], ddp: dict[str, Any]) -> None:
    if writer is None or not bool(config.get("enabled", False)):
        return
    handle = getattr(writer, "_phaseflow_handle", None)
    if handle is not None:
        handle.flush()
        handle.close()
    _distributed_barrier(ddp)
    if not bool(ddp.get("is_rank0", False)):
        return
    final_path = Path(str(config.get("loss_components_csv")))
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", newline="", encoding="utf-8") as out:
        csv_out = csv.DictWriter(out, fieldnames=NAN_DEBUG_FIELDS)
        csv_out.writeheader()
        for rank in range(int(ddp.get("world_size", 1))):
            rank_path = _rank_timing_path(final_path, rank)
            if not rank_path.exists():
                continue
            with rank_path.open("r", newline="", encoding="utf-8") as handle_in:
                reader = csv.DictReader(handle_in)
                for row in reader:
                    csv_out.writerow(row)


def _loss_weights_for_step(
    base_weights: dict[str, Any],
    training_config: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    weights = dict(base_weights or {})
    ramp = training_config.get("loss_ramp", {}) or {}
    if not bool(ramp.get("enabled", False)):
        return weights
    warmup_steps = max(int(ramp.get("warmup_steps", 0) or 0), 1)
    frac = min(max(float(step) / float(warmup_steps), 0.0), 1.0)
    schedule = ramp.get("schedule", {}) or {}
    for name, spec in schedule.items():
        if not isinstance(spec, dict):
            continue
        start = float(spec.get("start", weights.get(name, 0.0)))
        end = float(spec.get("end", weights.get(name, start)))
        weights[name] = start + (end - start) * frac
    return weights


def _force_fp32_loss(weights: dict[str, Any], training_config: dict[str, Any]) -> bool:
    return bool(
        weights.get("compute_region_loss_in_fp32", False)
        or weights.get("compute_loss_in_fp32", False)
        or training_config.get("compute_loss_in_fp32", False)
        or training_config.get("loss_compute_fp32", False)
    )


def _nan_debug_write_row(
    writer: csv.DictWriter | None,
    config: dict[str, Any],
    ddp: dict[str, Any],
    *,
    epoch: int,
    step: int,
    local_batch_index: int,
    batch: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    loss_values: dict[str, float],
    total_loss: torch.Tensor,
    amp_scale: float,
    grad_norm: float | torch.Tensor | None,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if writer is None or not bool(config.get("enabled", False)):
        return {}
    row = _nan_debug_row(
        ddp,
        epoch=epoch,
        step=step,
        local_batch_index=local_batch_index,
        batch=batch,
        outputs=outputs,
        loss_values=loss_values,
        total_loss=total_loss,
        amp_scale=amp_scale,
        grad_norm=grad_norm,
        issues=issues,
    )
    writer.writerow(row)
    handle = getattr(writer, "_phaseflow_handle", None)
    if handle is not None:
        handle.flush()
    return row


def _nan_debug_write_bad_batch(
    config: dict[str, Any],
    ddp: dict[str, Any],
    *,
    epoch: int,
    step: int,
    local_batch_index: int,
    batch: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    loss_values: dict[str, float],
    total_loss: torch.Tensor,
    amp_scale: float,
    grad_norm: float | torch.Tensor | None,
    issues: list[dict[str, Any]],
) -> None:
    if not bool(config.get("enabled", False)):
        return
    row = _nan_debug_row(
        ddp,
        epoch=epoch,
        step=step,
        local_batch_index=local_batch_index,
        batch=batch,
        outputs=outputs,
        loss_values=loss_values,
        total_loss=total_loss,
        amp_scale=amp_scale,
        grad_norm=grad_norm,
        issues=issues,
    )
    bad_dir = Path(str(config.get("bad_batch_dir")))
    bad_dir.mkdir(parents=True, exist_ok=True)
    rank = int(ddp.get("rank", 0))
    payload = {
        "event": "bad_batch_nonfinite",
        "epoch": int(epoch),
        "global_step": int(step),
        "rank": rank,
        "local_rank": int(ddp.get("local_rank", rank)),
        "local_batch_index": int(local_batch_index),
        "issues": issues,
        "row": row,
    }
    path = bad_dir / f"bad_batch_rank{rank}_step{int(step)}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest = bad_dir / f"bad_batch_rank{rank}.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = Path(str(config.get("report_path")))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage2 NaN Debug Report",
        "",
        "- status: stopped_on_nonfinite",
        f"- epoch: {int(epoch)}",
        f"- global_step: {int(step)}",
        f"- rank: {rank}",
        f"- local_batch_index: {int(local_batch_index)}",
        f"- bad_batch_json: `{path}`",
        "",
        "## Batch",
        "",
        f"- protein_ids: `{row.get('protein_ids_json', '[]')}`",
        f"- sequence_lengths: `{row.get('sequence_lengths_json', '[]')}`",
        f"- positive_residue_count: {row.get('positive_residue_count', 0)}",
        f"- negative_residue_count: {row.get('negative_residue_count', 0)}",
        f"- region_span_count: {row.get('region_span_count', 0)}",
        "",
        "## Issues",
        "",
    ]
    for issue in issues:
        lines.append(f"- {json.dumps(issue, sort_keys=True)}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _nan_debug_row(
    ddp: dict[str, Any],
    *,
    epoch: int,
    step: int,
    local_batch_index: int,
    batch: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    loss_values: dict[str, float],
    total_loss: torch.Tensor,
    amp_scale: float,
    grad_norm: float | torch.Tensor | None,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    protein_ids = [str(item) for item in batch.get("protein_ids", [])]
    sample_ids = [str(item) for item in batch.get("sample_ids", [])]
    lengths = _tensor_list(batch.get("lengths"))
    region_counts = [
        sum(1 for region in regions if str(region.get("type", "")).lower() != "key_region")
        for regions in batch.get("regions", [])
    ]
    seq_mask = batch.get("seq_mask")
    seq_valid = seq_mask.bool() if torch.is_tensor(seq_mask) else None
    region_weight = batch.get("region_teacher_weight")
    region_target = batch.get("region_teacher_target")
    valid_region = None
    if torch.is_tensor(region_weight):
        valid_region = region_weight.float().gt(0.0)
        if seq_valid is not None and seq_valid.shape == valid_region.shape:
            valid_region = valid_region & seq_valid
    positive = _count_mask(region_target, valid_region, predicate="positive")
    negative = _count_mask(region_target, valid_region, predicate="negative")
    row: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(step),
        "rank": int(ddp.get("rank", 0)),
        "local_rank": int(ddp.get("local_rank", 0)),
        "world_size": int(ddp.get("world_size", 1)),
        "local_batch_index": int(local_batch_index),
        "protein_ids_json": json.dumps(protein_ids),
        "sample_ids_json": json.dumps(sample_ids),
        "sequence_lengths_json": json.dumps(lengths),
        "region_span_count": int(sum(region_counts)),
        "region_span_count_json": json.dumps(region_counts),
        "positive_residue_count": positive,
        "negative_residue_count": negative,
        "valid_residue_mask_sum": int(seq_valid.sum().detach().cpu()) if torch.is_tensor(seq_valid) else 0,
        "proposal_mask_sum": int(valid_region.sum().detach().cpu()) if torch.is_tensor(valid_region) else 0,
        "mil_window_count": int((batch.get("region_bag_weight", torch.zeros(0, device=total_loss.device)) > 0).sum().detach().cpu())
        if torch.is_tensor(batch.get("region_bag_weight"))
        else 0,
        "total_loss": _float_scalar(total_loss),
        "amp_scale": float(amp_scale),
        "grad_norm": _float_scalar(grad_norm) if grad_norm is not None else "",
        "finite_issue_count": len(issues),
        "finite_issues_json": json.dumps(issues, sort_keys=True),
        "loss_components_json": json.dumps({key: _safe_float(value) for key, value in loss_values.items()}, sort_keys=True),
    }
    for prefix, tensor, mask in [
        ("dpr_logit", outputs.get("dpr_logits"), seq_valid),
        ("region_logit", outputs.get("region_logits"), None),
        ("llps_logit", outputs.get("loss_llps_logits", outputs.get("llps_logits", outputs.get("llps_logits"))), None),
        ("target", region_target, valid_region),
        ("input_plm", batch.get("plm"), seq_valid),
        ("input_physchem", batch.get("physchem"), seq_valid),
        ("input_disorder", batch.get("disorder"), seq_valid),
        ("input_protenix_embed", batch.get("protenix_embed"), seq_valid),
        ("edge_attr", batch.get("edge_attr"), None),
    ]:
        mn, mx = _tensor_min_max(tensor, mask)
        row[f"{prefix}_min"] = mn
        row[f"{prefix}_max"] = mx
    row["starling_abs_sum"] = _tensor_abs_sum(batch.get("starling_embed"))
    reliability = batch.get("reliability")
    row["starling_reliability_sum"] = (
        _tensor_abs_sum(reliability[..., 4]) if torch.is_tensor(reliability) and reliability.ndim >= 3 and reliability.shape[-1] > 4 else 0.0
    )
    for key in LOSS_COMPONENT_KEYS:
        row[f"loss_component_{key}"] = _safe_float(loss_values.get(key, ""))
    return row


def _nan_debug_finite_issues(
    config: dict[str, Any],
    *,
    batch: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    loss_values: dict[str, float],
    total_loss: torch.Tensor,
    grad_norm: float | torch.Tensor | None,
) -> list[dict[str, Any]]:
    if not bool(config.get("enabled", False)):
        return []
    issues: list[dict[str, Any]] = []
    seq_mask = batch.get("seq_mask")
    seq_valid = seq_mask.bool() if torch.is_tensor(seq_mask) else None
    if bool(config.get("finite_check_inputs", True)):
        for name in ("plm", "physchem", "disorder", "protenix_embed", "edge_attr"):
            issues.extend(_finite_tensor_issues(f"input.{name}", batch.get(name), mask=seq_valid if name != "edge_attr" else None))
    if bool(config.get("finite_check_logits", True)):
        for name, value in outputs.items():
            if torch.is_tensor(value) and any(token in name for token in ("logit", "start", "end")):
                issues.extend(_finite_tensor_issues(f"model.{name}", value, mask=seq_valid if value.ndim == 2 else None))
    region_valid = _target_valid_mask(batch.get("region_teacher_weight"), seq_valid)
    issues.extend(_finite_tensor_issues("targets.residue", batch.get("y_dpr"), mask=(batch.get("y_dpr") != IGNORE_INDEX) & seq_valid if torch.is_tensor(batch.get("y_dpr")) and seq_valid is not None else None))
    issues.extend(_finite_tensor_issues("targets.region_teacher_target", batch.get("region_teacher_target"), mask=region_valid))
    issues.extend(
        _finite_tensor_issues(
            "targets.region_boundary_target",
            batch.get("region_boundary_target"),
            mask=_target_valid_mask(batch.get("region_boundary_weight"), seq_valid),
        )
    )
    issues.extend(
        _finite_tensor_issues(
            "targets.region_contrast_target",
            batch.get("region_contrast_target"),
            mask=_target_valid_mask(batch.get("region_contrast_weight"), seq_valid),
        )
    )
    issues.extend(
        _finite_tensor_issues(
            "targets.region_key_target",
            batch.get("region_key_target"),
            mask=_target_valid_mask(batch.get("region_key_weight"), seq_valid),
        )
    )
    if bool(config.get("finite_check_loss_components", True)):
        issues.extend(_finite_value_issues("total_loss", _float_scalar(total_loss)))
        for name, value in loss_values.items():
            issues.extend(_finite_value_issues(f"loss_component.{name}", value))
    if grad_norm is not None and bool(config.get("finite_check_grads", True)):
        issues.extend(_finite_value_issues("grad_norm", _float_scalar(grad_norm)))
    return issues


def _finite_tensor_issues(name: str, value: Any, mask: torch.Tensor | None = None) -> list[dict[str, Any]]:
    if not torch.is_tensor(value):
        return []
    tensor = value.detach()
    if mask is not None and torch.is_tensor(mask):
        try:
            if tensor.shape[: mask.ndim] == mask.shape:
                tensor = tensor[mask.bool()]
        except Exception:
            pass
    if tensor.numel() == 0:
        return []
    finite = torch.isfinite(tensor.float())
    if bool(finite.all().detach().cpu()):
        return []
    return [
        {
            "name": name,
            "reason": "nonfinite_tensor",
            "numel": int(tensor.numel()),
            "nonfinite": int((~finite).sum().detach().cpu()),
        }
    ]


def _target_valid_mask(weight: Any, seq_valid: torch.Tensor | None) -> torch.Tensor | None:
    if not torch.is_tensor(weight):
        return None
    mask = weight.float().gt(0.0)
    if seq_valid is not None and torch.is_tensor(seq_valid) and seq_valid.shape == mask.shape:
        mask = mask & seq_valid
    return mask


def _finite_value_issues(name: str, value: Any) -> list[dict[str, Any]]:
    numeric = _safe_float(value)
    if isinstance(numeric, str) or math.isfinite(float(numeric)):
        return []
    return [{"name": name, "reason": "nonfinite_value", "value": numeric}]


def _tensor_list(value: Any) -> list[int]:
    if not torch.is_tensor(value):
        return []
    return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _count_mask(value: Any, mask: torch.Tensor | None, *, predicate: str) -> int:
    if not torch.is_tensor(value) or mask is None or not torch.is_tensor(mask):
        return 0
    selected = value.detach().float()[mask.bool()]
    if selected.numel() == 0:
        return 0
    if predicate == "positive":
        return int((selected >= 0.5).sum().detach().cpu())
    return int((selected < 0.5).sum().detach().cpu())


def _tensor_min_max(value: Any, mask: torch.Tensor | None = None) -> tuple[float, float]:
    if not torch.is_tensor(value):
        return float("nan"), float("nan")
    tensor = value.detach().float()
    if mask is not None and torch.is_tensor(mask):
        try:
            if tensor.shape[: mask.ndim] == mask.shape:
                tensor = tensor[mask.bool()]
        except Exception:
            pass
    if tensor.numel() == 0:
        return float("nan"), float("nan")
    finite = torch.isfinite(tensor)
    if not bool(finite.any().detach().cpu()):
        return float("nan"), float("nan")
    finite_values = tensor[finite]
    return float(finite_values.min().detach().cpu()), float(finite_values.max().detach().cpu())


def _tensor_abs_sum(value: Any) -> float:
    if not torch.is_tensor(value):
        return 0.0
    return float(torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).abs().sum().detach().cpu())


def _float_scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def _safe_float(value: Any) -> float | str:
    if isinstance(value, str) and value == "":
        return ""
    try:
        return _float_scalar(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return str(value)


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, *, decay: float, update_after_steps: int = 0) -> None:
        if decay < 0.0 or decay >= 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        self.decay = float(decay)
        self.update_after_steps = max(int(update_after_steps), 0)
        self.num_updates = 0
        self.shadow = _clone_state_dict(_checkpoint_state_dict(model), cpu=False)
        self.backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        state = _checkpoint_state_dict(model)
        warmup_copy = self.num_updates <= self.update_after_steps
        for name, value in state.items():
            if name not in self.shadow:
                self.shadow[name] = value.detach().clone()
                continue
            shadow_value = self.shadow[name]
            if torch.is_floating_point(value):
                if warmup_copy:
                    shadow_value.copy_(value.detach())
                else:
                    shadow_value.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow_value.copy_(value.detach())

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        self.backup = _clone_state_dict(_checkpoint_state_dict(model), cpu=False)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        state = _checkpoint_state_dict(model)
        for name, value in state.items():
            if name in self.shadow:
                value.copy_(self.shadow[name].to(device=value.device, dtype=value.dtype))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is None:
            return
        state = _checkpoint_state_dict(model)
        for name, value in state.items():
            if name in self.backup:
                value.copy_(self.backup[name].to(device=value.device, dtype=value.dtype))
        self.backup = None

    def model_state_dict(self, *, cpu: bool = True) -> dict[str, torch.Tensor]:
        return _clone_state_dict(self.shadow, cpu=cpu)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "shadow": self.model_state_dict(cpu=True),
            "decay": self.decay,
            "update_after_steps": self.update_after_steps,
            "num_updates": self.num_updates,
        }

    def load_checkpoint_state(self, checkpoint_state: dict[str, Any], model: torch.nn.Module) -> None:
        shadow = checkpoint_state.get("shadow", checkpoint_state)
        if not isinstance(shadow, dict):
            raise ValueError("EMA checkpoint state must be a state dict or contain a 'shadow' state dict.")
        current = _checkpoint_state_dict(model)
        loaded: dict[str, torch.Tensor] = {}
        for name, value in shadow.items():
            if not torch.is_tensor(value):
                continue
            target = current.get(name)
            if target is None:
                loaded[name] = value.detach().clone()
            else:
                loaded[name] = value.detach().to(device=target.device, dtype=target.dtype).clone()
        for name, value in current.items():
            if name not in loaded:
                loaded[name] = value.detach().clone()
        self.shadow = loaded
        self.num_updates = int(checkpoint_state.get("num_updates", self.num_updates))


def _maybe_create_ema(model: torch.nn.Module, ema_config: dict[str, Any]) -> ExponentialMovingAverage | None:
    if not bool(ema_config.get("enabled", False)):
        return None
    decay = float(ema_config.get("decay", 0.999))
    update_after_steps = int(ema_config.get("update_after_steps", 0) or 0)
    return ExponentialMovingAverage(model, decay=decay, update_after_steps=update_after_steps)


def _resolve_device_ids(raw: Any) -> list[int]:
    if raw is None:
        return list(range(torch.cuda.device_count()))
    if isinstance(raw, str):
        if not raw.strip():
            return list(range(torch.cuda.device_count()))
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple)):
        return [int(item) for item in raw]
    return [int(raw)]

def _phase_train_ids(data_config: dict[str, Any]) -> list[str]:
    if not phase_aux_data_enabled(data_config):
        return []
    path = data_config.get("phase_train_ids_file")
    if not path:
        return []
    ids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    max_samples = int(data_config.get("phase_max_train_samples", 0) or 0)
    if max_samples > 0:
        ids = ids[:max_samples]
    return ids


def _phase_aux_sampler(
    *,
    base_ids: list[str],
    phase_ids: list[str],
    training_config: dict[str, Any],
) -> WeightedRandomSampler | None:
    if not phase_ids:
        return None
    phase_fraction = float(training_config.get("phase_aux_fraction", 0.0) or 0.0)
    if phase_fraction <= 0.0:
        return None
    phase_fraction = min(max(phase_fraction, 1.0e-3), 0.95)
    base_count = max(len(base_ids), 1)
    phase_count = max(len(phase_ids), 1)
    base_weight = (1.0 - phase_fraction) / base_count
    phase_weight = phase_fraction / phase_count
    weights = torch.tensor(
        [base_weight for _ in base_ids] + [phase_weight for _ in phase_ids],
        dtype=torch.double,
    )
    batch_size = int(training_config.get("batch_size", 1))
    steps_per_epoch = int(training_config.get("steps_per_epoch", 0) or 0)
    if steps_per_epoch > 0:
        num_samples = steps_per_epoch * batch_size
    else:
        num_samples = int(math.ceil(base_count / max(1.0 - phase_fraction, 1.0e-3)))
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


class LengthBucketBatchSampler:
    def __init__(
        self,
        lengths: list[int],
        *,
        batch_size: int,
        bucket_size: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.bucket_size = max(int(bucket_size), self.batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        indices = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        buckets = [indices[start : start + self.bucket_size] for start in range(0, len(indices), self.bucket_size)]
        batches: list[list[int]] = []
        for bucket in buckets:
            if self.shuffle:
                rng.shuffle(bucket)
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start : start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)


class TokenBucketBatchSampler:
    def __init__(
        self,
        lengths: list[int],
        *,
        max_batch_size: int,
        max_padded_tokens: int,
        bucket_size: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_padded_tokens <= 0:
            raise ValueError("max_padded_tokens must be positive")
        self.lengths = [int(length) for length in lengths]
        self.max_batch_size = int(max_batch_size)
        self.max_padded_tokens = int(max_padded_tokens)
        self.bucket_size = max(int(bucket_size), self.max_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

    def __iter__(self):
        batches = self._batches_for_epoch(self._epoch)
        self._epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches_for_epoch(0, shuffle_batches=False))

    def _batches_for_epoch(self, epoch: int, *, shuffle_batches: bool = True) -> list[list[int]]:
        rng = random.Random(self.seed + epoch)
        indices = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        buckets = [indices[start : start + self.bucket_size] for start in range(0, len(indices), self.bucket_size)]
        batches: list[list[int]] = []
        for bucket in buckets:
            # Pack proteins of similar lengths together; shuffling final batches preserves epoch-level randomness.
            bucket = sorted(bucket, key=self.lengths.__getitem__)
            current: list[int] = []
            current_max = 0
            for index in bucket:
                length = max(int(self.lengths[index]), 1)
                next_max = max(current_max, length)
                would_overflow = bool(current) and (
                    len(current) >= self.max_batch_size
                    or next_max * (len(current) + 1) > self.max_padded_tokens
                )
                if would_overflow:
                    if len(current) == self.max_batch_size or not self.drop_last:
                        batches.append(current)
                    current = []
                    current_max = 0
                current.append(index)
                current_max = max(current_max, length)
            if current and (len(current) == self.max_batch_size or not self.drop_last):
                batches.append(current)
        if self.shuffle and shuffle_batches:
            rng.shuffle(batches)
        return batches


class DistributedLengthBucketBatchSampler:
    """DDP-safe length-bucketed batch sampler for variable-length map-style datasets."""

    def __init__(
        self,
        lengths: list[int],
        *,
        batch_size: int,
        bucket_boundaries: list[int] | None = None,
        bucket_size: int = 256,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        max_padded_tokens: int = 0,
        max_batch_size: int | None = None,
        min_batch_size: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.lengths = [max(int(length), 1) for length in lengths]
        self.batch_size = int(batch_size)
        self.bucket_boundaries = [int(value) for value in (bucket_boundaries or [])]
        self.bucket_size = max(int(bucket_size), self.batch_size * int(num_replicas))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.max_padded_tokens = int(max_padded_tokens or 0)
        self.max_batch_size = int(max_batch_size or batch_size)
        self.min_batch_size = max(int(min_batch_size or 1), 1)
        self._epoch = 0
        self._last_stats: dict[str, Any] = {}

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        batches, stats = self._rank_batches_for_epoch(self._epoch, shuffle_batches=True)
        self._last_stats = stats
        return iter(batches)

    def __len__(self) -> int:
        batches, _ = self._rank_batches_for_epoch(self._epoch, shuffle_batches=False)
        return len(batches)

    def epoch_stats(self, epoch: int | None = None) -> dict[str, Any]:
        if self._last_stats and (epoch is None or int(self._last_stats.get("epoch", -1)) == int(epoch)):
            return dict(self._last_stats)
        _, stats = self._rank_batches_for_epoch(self._epoch if epoch is None else int(epoch), shuffle_batches=False)
        return stats

    def _rank_batches_for_epoch(self, epoch: int, *, shuffle_batches: bool) -> tuple[list[list[int]], dict[str, Any]]:
        rng = random.Random(self.seed + int(epoch))
        global_batches = self._global_batches(rng)
        if self.shuffle and shuffle_batches:
            rng.shuffle(global_batches)
        rank_batches: list[list[int]] = []
        duplicate_count = 0
        for global_batch in global_batches:
            samples_per_rank = self._samples_per_rank(global_batch)
            if len(global_batch) < self.num_replicas:
                if self.drop_last:
                    continue
                target = self.num_replicas
                fill = self._pad_batch(global_batch, target)
                duplicate_count += len(fill)
                global_batch = global_batch + fill
                samples_per_rank = self._samples_per_rank(global_batch)
            global_batch = self._rank_balanced_order(global_batch, samples_per_rank)
            for replica in range(self.num_replicas):
                start = replica * samples_per_rank
                end = min(start + samples_per_rank, len(global_batch))
                if replica != self.rank:
                    continue
                batch = global_batch[start:end]
                if batch and (len(batch) == self.batch_size or not self.drop_last):
                    rank_batches.append(batch)
        stats = self._stats_for_batches(rank_batches)
        stats.update(
            {
                "epoch": int(epoch),
                "rank": self.rank,
                "world_size": self.num_replicas,
                "sampler": "length_bucketed_distributed",
                "batch_size_per_rank": self.batch_size,
                "global_batches": len(global_batches),
                "rank_batches": len(rank_batches),
                "duplicate_fill_samples_global": duplicate_count,
            }
        )
        return rank_batches, stats

    def _global_batches(self, rng: random.Random) -> list[list[int]]:
        grouped = self._bucketed_indices()
        global_batches: list[list[int]] = []
        global_batch_size = self.batch_size * self.num_replicas
        for bucket in grouped:
            if self.shuffle:
                rng.shuffle(bucket)
            if self.max_padded_tokens > 0:
                global_batches.extend(self._token_global_batches(bucket, global_batch_size))
            else:
                for start in range(0, len(bucket), global_batch_size):
                    batch = bucket[start : start + global_batch_size]
                    if len(batch) == global_batch_size or (batch and not self.drop_last):
                        global_batches.append(batch)
        return global_batches

    def _bucketed_indices(self) -> list[list[int]]:
        sorted_indices = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        if self.bucket_boundaries:
            buckets: list[list[int]] = [[] for _ in range(len(self.bucket_boundaries) + 1)]
            for index in sorted_indices:
                length = self.lengths[index]
                bucket_index = 0
                while bucket_index < len(self.bucket_boundaries) and length > self.bucket_boundaries[bucket_index]:
                    bucket_index += 1
                buckets[bucket_index].append(index)
            chunks: list[list[int]] = []
            for bucket in buckets:
                chunks.extend(
                    bucket[start : start + self.bucket_size]
                    for start in range(0, len(bucket), self.bucket_size)
                    if bucket[start : start + self.bucket_size]
                )
            return chunks
        return [sorted_indices[start : start + self.bucket_size] for start in range(0, len(sorted_indices), self.bucket_size)]

    def _token_global_batches(self, bucket: list[int], global_batch_size: int) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        per_rank_limit = max(self.max_padded_tokens, 1)
        global_token_limit = per_rank_limit * self.num_replicas
        max_global_batch_size = min(max(self.max_batch_size, self.batch_size), self.batch_size) * self.num_replicas
        if self.max_batch_size > self.batch_size:
            max_global_batch_size = self.max_batch_size * self.num_replicas
        for index in sorted(bucket, key=self.lengths.__getitem__):
            length = self.lengths[index]
            next_max = max(current_max, length)
            would_overflow = bool(current) and (
                len(current) >= min(max_global_batch_size, global_batch_size)
                or next_max * (len(current) + 1) > global_token_limit
            )
            if would_overflow:
                if len(current) >= self.min_batch_size * self.num_replicas or not self.drop_last:
                    batches.append(current)
                current = []
                current_max = 0
            current.append(index)
            current_max = max(current_max, length)
        if current and (len(current) >= self.min_batch_size * self.num_replicas or not self.drop_last):
            batches.append(current)
        return batches

    def _samples_per_rank(self, global_batch: list[int]) -> int:
        value = math.ceil(len(global_batch) / self.num_replicas)
        value = max(value, self.min_batch_size)
        cap = self.max_batch_size if self.max_padded_tokens > 0 else self.batch_size
        return min(value, cap)

    def _rank_balanced_order(self, global_batch: list[int], samples_per_rank: int) -> list[int]:
        if self.num_replicas <= 1 or len(global_batch) <= 1:
            return list(global_batch)
        chunks: list[list[int]] = [[] for _ in range(self.num_replicas)]
        sums = [0 for _ in range(self.num_replicas)]
        maxes = [0 for _ in range(self.num_replicas)]
        for index in sorted(global_batch, key=self.lengths.__getitem__, reverse=True):
            candidates = [rank for rank in range(self.num_replicas) if len(chunks[rank]) < samples_per_rank]
            if not candidates:
                break
            rank = min(candidates, key=lambda item: (maxes[item], sums[item], len(chunks[item])))
            chunks[rank].append(index)
            length = self.lengths[index]
            sums[rank] += length
            maxes[rank] = max(maxes[rank], length)
        return [index for chunk in chunks for index in chunk]

    def _pad_batch(self, batch: list[int], target: int) -> list[int]:
        if not batch:
            return []
        fill: list[int] = []
        cursor = 0
        while len(batch) + len(fill) < target:
            fill.append(batch[cursor % len(batch)])
            cursor += 1
        return fill

    def _stats_for_batches(self, batches: list[list[int]]) -> dict[str, float]:
        if not batches:
            return {
                "mean_batch_max_length": 0.0,
                "mean_real_residues": 0.0,
                "mean_padded_residues": 0.0,
                "padding_ratio": 0.0,
            }
        max_lengths = [max(self.lengths[index] for index in batch) for batch in batches]
        real_residues = [sum(self.lengths[index] for index in batch) for batch in batches]
        padded_residues = [max_len * len(batch) for max_len, batch in zip(max_lengths, batches, strict=False)]
        return {
            "mean_batch_max_length": sum(max_lengths) / len(max_lengths),
            "mean_real_residues": sum(real_residues) / len(real_residues),
            "mean_padded_residues": sum(padded_residues) / len(padded_residues),
            "padding_ratio": (sum(padded_residues) - sum(real_residues)) / max(sum(padded_residues), 1),
        }


def _length_bucket_batch_sampler(
    dataset: PhaseFlowDataset,
    *,
    feature_dirs: list[Path | str],
    batch_size: int,
    training_config: dict[str, Any],
    shuffle: bool,
    enabled: bool,
) -> LengthBucketBatchSampler | None:
    if not enabled or not bool(training_config.get("length_bucketed_batches", False)):
        return None
    bucket_size = int(training_config.get("length_bucket_size", 256))
    feature_paths = [Path(path) for path in feature_dirs]
    lengths = [_read_cached_length(feature_paths, protein_id) for protein_id in dataset.protein_ids]
    max_padded_tokens = int(training_config.get("max_padded_residues_per_batch", 0) or 0)
    if max_padded_tokens > 0:
        return TokenBucketBatchSampler(
            lengths,
            max_batch_size=int(training_config.get("max_batch_size", batch_size) or batch_size),
            max_padded_tokens=max_padded_tokens,
            bucket_size=bucket_size,
            shuffle=shuffle,
            seed=int(training_config.get("length_bucket_seed", 42)),
            drop_last=bool(training_config.get("drop_last", False)),
        )
    return LengthBucketBatchSampler(
        lengths,
        batch_size=batch_size,
        bucket_size=bucket_size,
        shuffle=shuffle,
        seed=int(training_config.get("length_bucket_seed", 42)),
        drop_last=bool(training_config.get("drop_last", False)),
    )


def _make_offline_datasets(config: dict) -> tuple[PhaseFlowOfflineDataset, PhaseFlowOfflineDataset]:
    data_config = config["data"]
    dataset_root = data_config.get("dataset_root", "data/processed/merged")
    input_contract = data_config.get("input_contract")
    sample_index = data_config.get("sample_index")
    train_ids = resolve_split_ids(data_config, "train") if _has_split_source(data_config, "train") else None
    valid_ids = resolve_split_ids(data_config, "valid") if _has_split_source(data_config, "valid") else None
    region_targets = data_config.get("region_targets")
    region_labels_dir = data_config.get("region_labels_dir")
    phase_targets = resolve_phase_targets(data_config)
    train_region_supervision = str(
        data_config.get("train_region_supervision", data_config.get("region_supervision", "none"))
    )
    valid_region_supervision = str(data_config.get("valid_region_supervision", "none"))
    _validate_train_region_supervision(config, train_region_supervision)
    kwargs = {
        "dataset_root": dataset_root,
        "sample_index": sample_index,
        "input_contract": input_contract,
        "phase_targets": phase_targets,
        "region_targets": region_targets,
        "region_labels_dir": region_labels_dir,
        "allow_legacy_h5": bool(data_config.get("allow_legacy_h5", False)),
    }
    train_dataset = PhaseFlowOfflineDataset(
        **kwargs,
        protein_ids=train_ids,
        split=None if train_ids is not None else "train",
        region_supervision=train_region_supervision,
    )
    valid_dataset = PhaseFlowOfflineDataset(
        **kwargs,
        protein_ids=valid_ids,
        split=None if valid_ids is not None else "valid",
        region_supervision=valid_region_supervision,
    )
    return train_dataset, valid_dataset


def _validate_full_benchmark_sample_index_guard(config: dict, dataset_class: str) -> None:
    data_config = config.get("data", {}) or {}
    dataset_config = config.get("dataset", {}) or {}
    candidates: list[Path] = []
    if dataset_class == "PhaseFlowBatchPlanDataset":
        dataset_root = Path(str(dataset_config.get("dataset_root", "data/processed/merged")))
        candidates.append(Path(str(dataset_config.get("sample_index", dataset_root / "tables/training_sample_index.parquet"))))
    elif dataset_class == "PhaseFlowOfflineDataset":
        dataset_root = Path(str(data_config.get("dataset_root", "data/processed/merged")))
        candidates.append(Path(str(data_config.get("sample_index", dataset_root / "tables/training_sample_index.parquet"))))
    elif dataset_class == "PhaseFlowPackedBatchDataset":
        sample_index = dataset_config.get("sample_index") or data_config.get("sample_index")
        if sample_index:
            candidates.append(Path(str(sample_index)))
        else:
            packed_dir_value = dataset_config.get("packed_dir") or training_config_packed_dir(config)
            if packed_dir_value:
                packed_dir = Path(str(packed_dir_value))
                if not packed_dir.is_absolute():
                    packed_dir = Path.cwd() / packed_dir
                source_manifest = packed_dir / "source_readonly_manifest.yaml"
                if source_manifest.exists():
                    manifest = load_yaml(source_manifest)
                    source_sample_index = (manifest.get("paths", {}) or {}).get("sample_index")
                    if source_sample_index:
                        candidates.append(Path(str(source_sample_index)))
                if not candidates:
                    raise ValueError(
                        "PhaseFlowPackedBatchDataset requires a source sample_index in dataset.sample_index "
                        "or packed_dir/source_readonly_manifest.yaml for full PPMC benchmark leakage guard."
                    )

    report_dir = Path(str(config.get("output_dir", "runs/phaseflow_train"))) / "audit"
    seen: set[str] = set()
    for path in candidates:
        resolved = path if path.is_absolute() else Path.cwd() / path
        key = str(resolved.resolve()) if resolved.exists() else str(resolved)
        if key in seen:
            continue
        seen.add(key)
        assert_no_full_benchmark_leakage(
            resolved,
            root=Path.cwd(),
            report_dir=report_dir,
            context=f"{dataset_class} training",
        )


def training_config_packed_dir(config: dict) -> str | Path | None:
    return (config.get("training", {}) or {}).get("packed_batches_dir")


def _has_split_source(data_config: dict[str, Any], split: str) -> bool:
    return bool(data_config.get(f"{split}_ids") or data_config.get(f"{split}_ids_file") or data_config.get("manifest"))


def _offline_length_bucket_batch_sampler(
    dataset: PhaseFlowOfflineDataset,
    *,
    batch_size: int,
    training_config: dict[str, Any],
    shuffle: bool,
    enabled: bool = True,
) -> LengthBucketBatchSampler | None:
    if not enabled or not bool(training_config.get("length_bucketed_batches", False)):
        return None
    lengths = [int(value) for value in dataset.frame["seq_len"].tolist()]
    bucket_size = int(training_config.get("length_bucket_size", 256))
    max_padded_tokens = int(training_config.get("max_padded_residues_per_batch", 0) or 0)
    if max_padded_tokens > 0:
        return TokenBucketBatchSampler(
            lengths,
            max_batch_size=int(training_config.get("max_batch_size", batch_size) or batch_size),
            max_padded_tokens=max_padded_tokens,
            bucket_size=bucket_size,
            shuffle=shuffle,
            seed=int(training_config.get("length_bucket_seed", 42)),
            drop_last=bool(training_config.get("drop_last", False)),
        )
    return LengthBucketBatchSampler(
        lengths,
        batch_size=batch_size,
        bucket_size=bucket_size,
        shuffle=shuffle,
        seed=int(training_config.get("length_bucket_seed", 42)),
        drop_last=bool(training_config.get("drop_last", False)),
    )


def _offline_distributed_length_bucket_batch_sampler(
    dataset: PhaseFlowOfflineDataset,
    *,
    batch_size: int,
    training_config: dict[str, Any],
    shuffle: bool,
    num_replicas: int,
    rank: int,
) -> DistributedLengthBucketBatchSampler | None:
    sampler_name = str(training_config.get("sampler", training_config.get("dataloader_sampler", ""))).strip().lower()
    enabled = bool(training_config.get("length_bucketed_batches", False)) or sampler_name in {
        "length_bucketed_distributed",
        "distributed_length_bucketed",
    }
    if not enabled:
        return None
    lengths = [int(value) for value in dataset.frame["seq_len"].tolist()]
    boundaries = training_config.get("bucket_boundaries", training_config.get("length_bucket_boundaries"))
    bucket_boundaries = _parse_int_list_config(boundaries)
    max_padded_tokens = int(training_config.get("max_padded_residues_per_batch", 0) or 0)
    seed = int(training_config.get("length_bucket_seed", training_config.get("seed", 42)) or 42)
    return DistributedLengthBucketBatchSampler(
        lengths,
        batch_size=batch_size,
        bucket_boundaries=bucket_boundaries,
        bucket_size=int(training_config.get("length_bucket_size", 256)),
        shuffle=shuffle,
        seed=seed,
        drop_last=bool(training_config.get("drop_last", False)),
        num_replicas=num_replicas,
        rank=rank,
        max_padded_tokens=max_padded_tokens if bool(training_config.get("dynamic_batching", False)) else 0,
        max_batch_size=int(training_config.get("max_samples_per_batch", training_config.get("max_batch_size", batch_size)) or batch_size),
        min_batch_size=int(training_config.get("min_samples_per_batch", 1) or 1),
    )


def _parse_int_list_config(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return None


def _amp_autocast_dtype(training_config: dict[str, Any]) -> torch.dtype | None:
    dtype = str(training_config.get("amp_dtype", training_config.get("precision", ""))).strip().lower()
    if dtype in {"", "amp", "auto", "true"}:
        return None
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return None
    raise ValueError(f"Unsupported amp dtype: {dtype}")


def _write_sampler_epoch_stats(path_value: Any, stats: dict[str, Any]) -> None:
    if not path_value or not stats:
        return
    path = Path(str(path_value))
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = sorted(stats.keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(stats)


def _read_cached_length(feature_dirs: list[Path], protein_id: str) -> int:
    import h5py

    for feature_dir in feature_dirs:
        path = feature_dir / f"{protein_id}.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as handle:
            if "length" in handle.attrs:
                return int(handle.attrs["length"])
            if "sequence" in handle.attrs:
                return len(str(handle.attrs["sequence"]))
            if "plm" in handle:
                return int(handle["plm"].shape[0])
        break
    raise FileNotFoundError(f"Missing feature cache for {protein_id}")


def _total_optimizer_steps(
    *,
    sample_count: int,
    loader_batch_size: int,
    optimizer_batch_size: int,
    loader_batches: int,
) -> int:
    if optimizer_batch_size <= 0 or optimizer_batch_size >= loader_batch_size:
        return loader_batches
    return math.ceil(sample_count / optimizer_batch_size)


def _validate_train_region_supervision(config: dict[str, Any], train_region_supervision: str) -> None:
    weights = config.get("training", {}).get("loss_weights", {})
    hard_region_weights = ("region_gold", "dpr", "key", "region", "coverage")
    active_hard = [name for name in hard_region_weights if float(weights.get(name, 0.0)) > 0.0]
    if train_region_supervision == "feature" and active_hard:
        raise ValueError(
            "DPR region training from feature-cache gold labels is forbidden. "
            "Use train_region_supervision='region_targets' with PSTP-Scan targets, "
            f"or train_region_supervision='none'. Active leakage-prone loss weights: {active_hard}"
        )
    region_supervision_weights = ("region_teacher", "region_boundary", "region_contrastive", "region_key_teacher")
    active_region_supervision = [name for name in region_supervision_weights if float(weights.get(name, 0.0)) > 0.0]
    data_config = config.get("data", {})
    has_region_target_source = bool(data_config.get("region_targets") or data_config.get("region_labels_dir"))
    if train_region_supervision == "region_targets" and active_region_supervision and not has_region_target_source:
        raise ValueError("Region-target training requires data.region_targets or data.region_labels_dir.")
    allow_aux_region_targets = bool(config.get("training", {}).get("allow_region_target_aux_losses", False))
    pure_pstp_forbidden = (
        "region_gold",
        "dpr",
        "key",
        "teacher_dpr",
        "teacher_distill",
        "self_dpr",
        "region",
        "coverage",
        "region_key_teacher",
    )
    allow_npz_span_losses = bool(data_config.get("region_labels_dir")) and bool(
        config.get("training", {}).get("allow_region_label_npz_span_losses", False)
    )
    if not allow_aux_region_targets:
        pure_pstp_forbidden = pure_pstp_forbidden + ("region_boundary", "region_contrastive")
    if allow_npz_span_losses:
        pure_pstp_forbidden = tuple(
            name for name in pure_pstp_forbidden if name not in {"region", "coverage", "region_boundary", "region_contrastive"}
        )
    active_forbidden = [name for name in pure_pstp_forbidden if float(weights.get(name, 0.0)) > 0.0]
    if train_region_supervision == "region_targets" and active_forbidden:
        raise ValueError(
            "Pure PSTP-Scan DPR training must not mix feature-cache DPR teachers, "
            "hard DPR labels, or unapproved pseudo-region span losses. "
            "Use only region_teacher, or set training.allow_region_target_aux_losses=true "
            "to enable boundary/contrastive losses from data.region_targets. "
            f"Active forbidden loss weights: {active_forbidden}"
        )


def _iter_optimizer_microbatches(batch: dict[str, Any], optimizer_batch_size: int):
    lengths = batch.get("lengths")
    if not torch.is_tensor(lengths):
        yield batch
        return
    batch_size = int(lengths.shape[0])
    if optimizer_batch_size <= 0 or optimizer_batch_size >= batch_size:
        yield batch
        return
    for start in range(0, batch_size, optimizer_batch_size):
        end = min(start + optimizer_batch_size, batch_size)
        yield _slice_batch(batch, start, end, batch_size)


def _shuffle_batch_dimension(batch: dict[str, Any]) -> dict[str, Any]:
    lengths = batch.get("lengths")
    if not torch.is_tensor(lengths):
        return batch
    batch_size = int(lengths.shape[0])
    if batch_size <= 1:
        return batch
    order = torch.randperm(batch_size)
    order_list = [int(index) for index in order.tolist()]
    shuffled: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            shuffled[key] = value.index_select(0, order)
        elif isinstance(value, list) and len(value) == batch_size:
            shuffled[key] = [value[index] for index in order_list]
        else:
            shuffled[key] = value
    return shuffled


def _slice_batch(batch: dict[str, Any], start: int, end: int, batch_size: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    seq_len = int(batch["seq_mask"].shape[1]) if torch.is_tensor(batch.get("seq_mask")) else 0
    micro_lengths = batch["lengths"][start:end] if torch.is_tensor(batch.get("lengths")) else None
    micro_seq_len = int(micro_lengths.max().item()) if torch.is_tensor(micro_lengths) and micro_lengths.numel() else seq_len
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            item = value[start:end]
            if key in SEQUENCE_ALIGNED_BATCH_KEYS and value.ndim >= 2 and seq_len > 0 and int(value.shape[1]) == seq_len:
                item = item[:, :micro_seq_len, ...]
            sliced[key] = item
        elif isinstance(value, list) and len(value) == batch_size:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


SEQUENCE_ALIGNED_BATCH_KEYS = {
    "seq_mask",
    "plm",
    "physchem",
    "disorder",
    "protenix_embed",
    "starling_embed",
    "modality_mask",
    "reliability",
    "y_dpr",
    "y_key",
    "y_weight",
    "teacher_dpr",
    "teacher_dpr_weight",
    "self_dpr",
    "self_dpr_weight",
    "candidate_prior",
    "candidate_prior_weight",
    "region_teacher_target",
    "region_teacher_weight",
    "region_key_target",
    "region_key_weight",
    "region_boundary_target",
    "region_boundary_weight",
    "region_contrast_target",
    "region_contrast_weight",
    "neighbors",
    "edge_attr",
    "neighbor_mask",
}


def _checkpoint_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, (torch.nn.DataParallel, DistributedDataParallel)):
        return model.module.state_dict()
    return model.state_dict()


def _clone_state_dict(state: dict[str, torch.Tensor], *, cpu: bool) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        tensor = value.detach()
        if cpu:
            tensor = tensor.cpu()
        cloned[name] = tensor.clone()
    return cloned


def _best_checkpoint_model_state(
    model: torch.nn.Module,
    ema: ExponentialMovingAverage | None,
    *,
    use_ema: bool,
) -> dict[str, torch.Tensor]:
    if ema is not None and use_ema:
        return ema.model_state_dict(cpu=True)
    return _checkpoint_state_dict(model)


def _checkpoint_ema_state(ema: ExponentialMovingAverage | None) -> dict[str, Any] | None:
    if ema is None:
        return None
    return ema.checkpoint_state()


def _compatible_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    current = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[dict[str, Any]] = []
    for key, value in state_dict.items():
        if key not in current:
            filtered[key] = value
            continue
        if current[key].shape == value.shape:
            filtered[key] = value
            continue
        skipped.append(
            {
                "key": key,
                "checkpoint_shape": list(value.shape),
                "model_shape": list(current[key].shape),
            }
        )
    return filtered, skipped


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _build_optimizer_param_groups(
    model: torch.nn.Module,
    training_config: dict[str, Any],
) -> list[dict[str, Any]] | None:
    group_config = training_config.get("parameter_groups", []) or []
    if not group_config:
        return None
    default_lr = float(training_config.get("lr", 1.0e-4))
    default_weight_decay = float(training_config.get("weight_decay", 1.0e-4))
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assigned: set[str] = set()
    groups: list[dict[str, Any]] = []
    for group in group_config:
        prefixes = tuple(str(prefix) for prefix in group.get("prefixes", []) if str(prefix))
        if not prefixes:
            continue
        params = []
        names = []
        for name, parameter in named_parameters:
            clean_name = name[7:] if name.startswith("module.") else name
            if name in assigned:
                continue
            if _name_matches_prefix(clean_name, prefixes):
                params.append(parameter)
                names.append(clean_name)
                assigned.add(name)
        if not params:
            continue
        payload: dict[str, Any] = {
            "params": params,
            "lr": float(group.get("lr", default_lr)),
            "weight_decay": float(group.get("weight_decay", default_weight_decay)),
        }
        if "name" in group:
            payload["name"] = str(group["name"])
        groups.append(payload)
    remaining = [parameter for name, parameter in named_parameters if name not in assigned]
    if remaining:
        groups.append({"params": remaining, "lr": default_lr, "weight_decay": default_weight_decay, "name": "default"})
    return groups or None


def _maybe_sync_llps_reference_dpr_head(model: torch.nn.Module, training_config: dict[str, Any]) -> None:
    freeze_config = training_config.get("freeze", {}) or {}
    if not bool(freeze_config.get("sync_llps_reference_dpr_head", False)):
        return
    module = model.module if isinstance(model, (torch.nn.DataParallel, DistributedDataParallel)) else model
    reference = getattr(module, "llps_reference_dpr_head", None)
    dpr_head = getattr(module, "dpr_head", None)
    if reference is None or dpr_head is None:
        return
    reference.load_state_dict(dpr_head.state_dict())
    for parameter in reference.parameters():
        parameter.requires_grad = False
    print(
        dumps_json(
            {
                "event": "synced_llps_reference_dpr_head",
                "parameters": sum(parameter.numel() for parameter in reference.parameters()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _apply_freeze_config(model: torch.nn.Module, training_config: dict[str, Any]) -> None:
    freeze_config = training_config.get("freeze", {}) or {}
    if not bool(freeze_config.get("enabled", False)):
        return
    module = model.module if isinstance(model, (torch.nn.DataParallel, DistributedDataParallel)) else model
    trainable_prefixes = tuple(str(prefix) for prefix in freeze_config.get("trainable_prefixes", []) if str(prefix))
    frozen_prefixes = tuple(str(prefix) for prefix in freeze_config.get("frozen_prefixes", []) if str(prefix))
    freeze_all_except_trainable = bool(freeze_config.get("freeze_all_except_trainable_prefixes", False))
    trainable_count = 0
    frozen_count = 0
    for name, parameter in module.named_parameters():
        trainable = parameter.requires_grad
        if freeze_all_except_trainable:
            trainable = _name_matches_prefix(name, trainable_prefixes)
        elif frozen_prefixes and _name_matches_prefix(name, frozen_prefixes):
            trainable = False
        if trainable_prefixes and _name_matches_prefix(name, trainable_prefixes):
            trainable = True
        parameter.requires_grad = trainable
        if trainable:
            trainable_count += parameter.numel()
        else:
            frozen_count += parameter.numel()
    print(
        dumps_json(
            {
                "event": "freeze_applied",
                "frozen_parameters": frozen_count,
                "trainable_parameters": trainable_count,
                "trainable_prefixes": list(trainable_prefixes),
                "freeze_all_except_trainable_prefixes": freeze_all_except_trainable,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _name_matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _active_cuda_device_ids(training_config: dict[str, Any], device: torch.device) -> list[int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return []
    if dist.is_available() and dist.is_initialized():
        return [device.index if device.index is not None else torch.cuda.current_device()]
    if bool(training_config.get("multi_gpu", False) or training_config.get("data_parallel", False)):
        return _resolve_device_ids(training_config.get("device_ids"))
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _reset_cuda_peak_memory(training_config: dict[str, Any], device: torch.device) -> None:
    for device_id in _active_cuda_device_ids(training_config, device):
        torch.cuda.reset_peak_memory_stats(device_id)


def _cuda_memory_metrics(training_config: dict[str, Any], device: torch.device) -> dict[str, float]:
    device_ids = _active_cuda_device_ids(training_config, device)
    if not device_ids:
        return {}
    peak_allocated = max(torch.cuda.max_memory_allocated(device_id) for device_id in device_ids)
    peak_reserved = max(torch.cuda.max_memory_reserved(device_id) for device_id in device_ids)
    current_reserved = max(torch.cuda.memory_reserved(device_id) for device_id in device_ids)
    divisor = 1024**3
    return {
        "cuda_peak_allocated_gb": float(peak_allocated / divisor),
        "cuda_peak_reserved_gb": float(peak_reserved / divisor),
        "cuda_reserved_gb": float(current_reserved / divisor),
    }


def _maybe_seed_best_checkpoint_from_resume(resume_path: Path, best_path: Path) -> None:
    if best_path.exists():
        return
    sibling_best = resume_path.with_name("best.pt")
    if sibling_best.exists() and sibling_best.resolve() != best_path.resolve():
        best_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sibling_best, best_path)


def _checkpoint_reload_check(checkpoint_path: Path, report_path: Path) -> None:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "passed": False,
        "elapsed_sec": 0.0,
        "has_model": False,
        "has_optimizer": False,
        "has_config": False,
        "error": "",
    }
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        result.update(
            {
                "has_model": isinstance(checkpoint, dict) and "model" in checkpoint,
                "has_optimizer": isinstance(checkpoint, dict) and "optimizer" in checkpoint,
                "has_config": isinstance(checkpoint, dict) and "config" in checkpoint,
            }
        )
        result["passed"] = bool(result["has_model"] and result["has_optimizer"] and result["has_config"])
    except Exception as exc:
        result["error"] = repr(exc)
    result["elapsed_sec"] = time.perf_counter() - started
    write_json(report_path, result)


def _maybe_build_toy(config: dict) -> None:
    assert_no_runtime_build(bool(config.get("auto_build_toy", False)), "toy feature build")
    if not bool(config.get("auto_build_toy", False)):
        return
    feature_dir = Path(config["data"]["feature_dir"])
    ids = list(config["data"].get("train_ids", [])) + list(config["data"].get("valid_ids", []))
    if ids and all((feature_dir / f"{protein_id}.h5").exists() for protein_id in ids):
        return
    build_feature_cache(
        fasta=config["data"]["toy_fasta"],
        protein_labels=config["data"]["toy_labels"],
        regions=config["data"]["toy_regions"],
        mil_bags=config["data"].get("mil_bags"),
        candidate_priors=config["data"].get("candidate_priors"),
        teacher_scores=config["data"].get("teacher_scores"),
        out_dir=feature_dir,
        mode="simple",
        overwrite=True,
    )


def _maybe_build_feature_pipeline(config: dict) -> None:
    feature_config = config.get("feature_generation", {})
    assert_no_runtime_build(bool(feature_config.get("enabled", False)), "feature generation")
    if not bool(feature_config.get("enabled", False)):
        return
    data_config = config["data"]
    training_config = config.get("training", {})
    feature_dir = Path(data_config["feature_dir"])
    ids = resolve_split_ids(data_config, "train") + resolve_split_ids(data_config, "valid")
    if ids and all((feature_dir / f"{protein_id}.h5").exists() for protein_id in ids):
        return

    manifest = feature_config.get("manifest") or data_config.get("manifest")
    fasta = feature_config.get("fasta") or data_config.get("fasta")
    if manifest is None and fasta is None:
        raise ValueError("feature_generation requires either data.manifest/feature_generation.manifest or data.fasta")

    plm_config = feature_config.get("esm2", {})
    mode = "esm2" if bool(plm_config.get("enabled", False)) else str(feature_config.get("mode", "simple"))
    esm2_dir = plm_config.get("embedding_dir")
    esm2_model_dir = plm_config.get("model_dir")
    if mode == "esm2":
        if bool(plm_config.get("download", False)):
            esm2_model_dir = str(download_esm2_model(plm_config.get("model_name", "facebook/esm2_t33_650M_UR50D"), esm2_model_dir))
        if esm2_dir is not None:
            _maybe_build_esm2_npz(manifest, fasta, esm2_dir, plm_config, esm2_model_dir)

    esm2_config = ESM2Config(
        model_name=plm_config.get("model_name", "facebook/esm2_t33_650M_UR50D"),
        model_dir=esm2_model_dir,
        device=str(plm_config.get("device", config.get("device", "auto"))),
        dtype=str(plm_config.get("dtype", "float32")),
        storage_dtype=str(plm_config.get("storage_dtype", "float32")),
        local_files_only=bool(plm_config.get("local_files_only", False)),
        chunk_size=plm_config.get("chunk_size"),
        overlap=int(plm_config.get("overlap", 128)),
    )
    if manifest is not None:
        build_feature_cache_from_manifest(
            manifest=manifest,
            out_dir=feature_dir,
            regions=feature_config.get("regions") or data_config.get("regions"),
            mil_bags=feature_config.get("mil_bags") or data_config.get("mil_bags"),
            candidate_priors=feature_config.get("candidate_priors") or data_config.get("candidate_priors"),
            teacher_scores=feature_config.get("teacher_scores") or data_config.get("teacher_scores"),
            mode=mode,
            esm2_dir=esm2_dir,
            esm2_config=esm2_config,
            structure_dir=feature_config.get("structure_dir") or feature_config.get("structure", {}).get("feature_dir"),
            af3_dir=feature_config.get("af3_dir") or feature_config.get("af3", {}).get("feature_dir"),
            starling_dir=feature_config.get("starling_dir") or feature_config.get("starling", {}).get("feature_dir"),
            starling_embedding_dir=feature_config.get("starling_embedding_dir")
            or feature_config.get("starling_embedding", {}).get("feature_dir"),
            starling_distance_dir=feature_config.get("starling_distance_dir")
            or feature_config.get("starling_distance", {}).get("feature_dir"),
            local_window=int(feature_config.get("local_window", config.get("model", {}).get("graph_transformer", {}).get("local_window", 16))),
            graph_max_neighbors=int(training_config.get("max_neighbors", config.get("model", {}).get("graph_transformer", {}).get("max_neighbors", 96))),
            graph_edge_dim=int(feature_config.get("graph_edge_dim", config.get("model", {}).get("graph_transformer", {}).get("edge_dim", 13))),
            starling_distance_topk=int(feature_config.get("starling_distance_topk", 48)),
            require_structure=bool(feature_config.get("require_structure", False)),
            require_starling=bool(feature_config.get("require_starling", False)),
            overwrite=bool(feature_config.get("overwrite", False)),
        )
    else:
        build_feature_cache(
            fasta=fasta,
            protein_labels=feature_config.get("protein_labels") or data_config.get("protein_labels"),
            regions=feature_config.get("regions") or data_config.get("regions"),
            mil_bags=feature_config.get("mil_bags") or data_config.get("mil_bags"),
            candidate_priors=feature_config.get("candidate_priors") or data_config.get("candidate_priors"),
            teacher_scores=feature_config.get("teacher_scores") or data_config.get("teacher_scores"),
            out_dir=feature_dir,
            mode=mode,
            esm2_dir=esm2_dir,
            esm2_config=esm2_config,
            structure_dir=feature_config.get("structure_dir") or feature_config.get("structure", {}).get("feature_dir"),
            af3_dir=feature_config.get("af3_dir") or feature_config.get("af3", {}).get("feature_dir"),
            starling_dir=feature_config.get("starling_dir") or feature_config.get("starling", {}).get("feature_dir"),
            starling_embedding_dir=feature_config.get("starling_embedding_dir")
            or feature_config.get("starling_embedding", {}).get("feature_dir"),
            starling_distance_dir=feature_config.get("starling_distance_dir")
            or feature_config.get("starling_distance", {}).get("feature_dir"),
            local_window=int(feature_config.get("local_window", config.get("model", {}).get("graph_transformer", {}).get("local_window", 16))),
            graph_max_neighbors=int(training_config.get("max_neighbors", config.get("model", {}).get("graph_transformer", {}).get("max_neighbors", 96))),
            graph_edge_dim=int(feature_config.get("graph_edge_dim", config.get("model", {}).get("graph_transformer", {}).get("edge_dim", 13))),
            starling_distance_topk=int(feature_config.get("starling_distance_topk", 48)),
            require_structure=bool(feature_config.get("require_structure", False)),
            require_starling=bool(feature_config.get("require_starling", False)),
            overwrite=bool(feature_config.get("overwrite", False)),
        )


def _maybe_build_esm2_npz(
    manifest: str | None,
    fasta: str | None,
    esm2_dir: str,
    plm_config: dict,
    model_dir: str | None,
) -> None:
    assert_no_runtime_build(True, "ESM2 embedding build")
    records = records_from_manifest(manifest) if manifest is not None else None
    if records is None:
        records = records_from_fasta(str(fasta))
    out_dir = Path(esm2_dir)
    overwrite = bool(plm_config.get("overwrite", False))
    missing = [protein_id for protein_id, _ in records if overwrite or not (out_dir / f"{protein_id}.npz").exists()]
    if not missing:
        return
    wanted = set(missing)
    selected = [(protein_id, sequence) for protein_id, sequence in records if protein_id in wanted]
    config = ESM2Config(
        model_name=plm_config.get("model_name", "facebook/esm2_t33_650M_UR50D"),
        model_dir=model_dir,
        device=str(plm_config.get("device", "auto")),
        dtype=str(plm_config.get("dtype", "float32")),
        storage_dtype=str(plm_config.get("storage_dtype", "float32")),
        local_files_only=bool(plm_config.get("local_files_only", False)),
        chunk_size=plm_config.get("chunk_size"),
        overlap=int(plm_config.get("overlap", 128)),
    )
    run_esm2_embeddings(selected, out_dir, config, overwrite=overwrite)


def _validate_strict_offline_config(config: dict) -> None:
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    dataset_config = config.get("dataset", {}) or {}
    dataset_class = str(dataset_config.get("type", data_config.get("dataset_class", config.get("dataset_class", ""))))
    if bool(data_config.get("allow_legacy_h5", False)):
        raise ValueError("allow_legacy_h5 must be false for production offline training.")
    strict_classes = {"PhaseFlowOfflineDataset", "PhaseFlowPackedBatchDataset", "PhaseFlowBatchPlanDataset"}
    if strict_offline_enabled() or dataset_class in strict_classes:
        if dataset_class not in strict_classes:
            raise ValueError(
                "Strict offline training requires PhaseFlowOfflineDataset, "
                "PhaseFlowPackedBatchDataset, or PhaseFlowBatchPlanDataset"
            )
        forbidden_flags = {
            "build_embeddings_at_runtime": data_config.get("build_embeddings_at_runtime", False),
            "build_graph_at_runtime": data_config.get("build_graph_at_runtime", False),
            "merge_edges_at_runtime": data_config.get("merge_edges_at_runtime", False),
            "feature_generation.enabled": config.get("feature_generation", {}).get("enabled", False),
            "auto_build_toy": config.get("auto_build_toy", False),
        }
        active = [name for name, value in forbidden_flags.items() if bool(value)]
        if active:
            raise ValueError(f"Runtime feature/graph building is forbidden for offline training: {active}")
        if str(data_config.get("graph_source", "")).strip() not in {"merged_sparse", "merged_sparse_multigraph"}:
            raise ValueError("Offline training graph_source must be merged_sparse.")
        if dataset_class == "PhaseFlowPackedBatchDataset" and not bool(dataset_config.get("source_dataset_readonly", False)):
            raise ValueError("Packed batch training must declare dataset.source_dataset_readonly=true.")
        if bool(training_config.get("require_precomputed_graph", False)):
            raise ValueError("Offline merged_sparse is an edge-list multigraph; do not require legacy precomputed graph cache.")
        loss_weights = training_config.get("loss_weights", {}) or {}
        train_region_supervision = str(data_config.get("train_region_supervision", data_config.get("region_supervision", "none"))).strip().lower()
        stage_name = str(config.get("metadata", {}).get("stage", "")).strip().lower()
        is_stage1_llps = stage_name.startswith("stage1") or "llps_base" in stage_name
        if is_stage1_llps and train_region_supervision in {"", "none", "false", "0"}:
            forbidden_region_losses = {
                "region_mil": loss_weights.get("region_mil", loss_weights.get("region_MIL", 0.0)),
                "negative_regularization": loss_weights.get("negative_regularization", 0.0),
                "region": loss_weights.get("region", 0.0),
                "coverage": loss_weights.get("coverage", 0.0),
                "region_gold": loss_weights.get("region_gold", loss_weights.get("dpr", 0.0)),
                "region_teacher": loss_weights.get("region_teacher", 0.0),
                "region_key_teacher": loss_weights.get("region_key_teacher", 0.0),
                "region_boundary": loss_weights.get("region_boundary", 0.0),
                "region_contrastive": loss_weights.get("region_contrastive", 0.0),
                "teacher_dpr": loss_weights.get("teacher_dpr", 0.0),
                "teacher_distill": loss_weights.get("teacher_distill", 0.0),
                "self_dpr": loss_weights.get("self_dpr", 0.0),
                "key": loss_weights.get("key", 0.0),
                "smoothness": loss_weights.get("smoothness", 0.0),
            }
            active_region_losses = {
                name: float(value)
                for name, value in forbidden_region_losses.items()
                if abs(float(value or 0.0)) > 0.0
            }
            if active_region_losses:
                raise ValueError(
                    "Stage 1 LLPS base with train_region_supervision=none forbids region/proposal/"
                    f"negative-regularization losses: {active_region_losses}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Optional checkpoint to initialize the model before training.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional short-run override for canary/profiling.")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = int(args.max_steps)
    best = train(config, resume=args.resume)
    rank = int(os.environ.get("RANK", "0") or "0")
    if rank == 0:
        print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
