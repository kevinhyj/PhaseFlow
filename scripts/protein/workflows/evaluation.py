"""Protein inference and fixed LLPS and DPR benchmark evaluation."""
from __future__ import annotations

# Source: evaluate.py


import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from phaseflow.protein.data import PhaseFlowCollator
from phaseflow.protein.contracts import resolve_feature_dirs, resolve_phase_targets, validate_forbidden_data_paths
from phaseflow.protein.data import PhaseFlowDataset
from phaseflow.protein.contracts import resolve_split_ids
from phaseflow.protein.objectives import key_topk_metrics
from phaseflow.protein.objectives import binary_classification_metrics
from phaseflow.protein.objectives import boundary_f1, region_metrics
from phaseflow.protein.objectives import residue_binary_metrics
from phaseflow.protein.model import PhaseFlowModel
from phaseflow.protein.postprocessing import combine_regions, decoder_regions, scores_to_regions
from phaseflow.protein.contracts import dumps_json, load_yaml, move_batch_to_device, resolve_device, write_json


@torch.no_grad()
def evaluate_model(
    model: PhaseFlowModel,
    loader: DataLoader,
    device: torch.device,
    postprocess_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    postprocess_config = postprocess_config or {}
    llps_labels: list[float] = []
    llps_scores: list[float] = []
    dpr_labels: list[np.ndarray] = []
    dpr_scores: list[np.ndarray] = []
    key_labels: list[np.ndarray] = []
    key_scores: list[np.ndarray] = []
    pred_regions: list[list[dict[str, float]]] = []
    true_regions: list[list[dict[str, object]]] = []
    negative_types: list[str] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(batch)
        llps_labels.extend(batch["y_llps"].detach().cpu().numpy().tolist())
        llps_output = outputs.get("llps_logits", outputs["llps_logits"])
        llps_scores.extend(torch.sigmoid(llps_output).detach().cpu().numpy().tolist())
        negative_types.extend(batch.get("negative_type", [""] * len(batch["protein_ids"])))
        lengths = batch["lengths"].detach().cpu().numpy()
        if "dpr_logits" in outputs and "key_logits" in outputs:
            dpr_prob = torch.sigmoid(outputs["dpr_logits"]).detach().cpu().numpy()
            key_prob = torch.sigmoid(outputs["key_logits"]).detach().cpu().numpy()
            dpr_label = batch["y_dpr"].detach().cpu().numpy()
            key_label = batch["y_key"].detach().cpu().numpy()
            region_logits = outputs.get("region_logits")
            region_start = outputs.get("region_start")
            region_end = outputs.get("region_end")
            if region_logits is not None:
                region_logits = region_logits.detach().cpu().numpy()
                region_start = region_start.detach().cpu().numpy()
                region_end = region_end.detach().cpu().numpy()
            for index, length in enumerate(lengths):
                dpr_labels.append(dpr_label[index, :length])
                dpr_scores.append(dpr_prob[index, :length])
                key_labels.append(key_label[index, :length])
                key_scores.append(key_prob[index, :length])
                post = scores_to_regions(
                    dpr_prob[index, :length],
                    threshold=float(postprocess_config.get("threshold", 0.5)),
                    smooth_window=int(postprocess_config.get("smooth_window", 5)),
                    merge_gap=int(postprocess_config.get("merge_gap", 5)),
                    min_region_len=int(postprocess_config.get("min_region_len", 6)),
                )
                if region_logits is not None and bool(postprocess_config.get("use_decoder_regions", False)):
                    dec = decoder_regions(
                        region_logits[index],
                        region_start[index],
                        region_end[index],
                        int(length),
                        score_threshold=float(postprocess_config.get("decoder_score_threshold", 0.5)),
                    )
                    pred_regions.append(combine_regions(dec, post))
                else:
                    pred_regions.append(post)
            true_regions.extend(batch["regions"])

    metrics = {}
    metrics.update(binary_classification_metrics(np.asarray(llps_labels), np.asarray(llps_scores)))
    if dpr_labels and dpr_scores:
        metrics.update(residue_binary_metrics(np.concatenate(dpr_labels), np.concatenate(dpr_scores)))
        metrics.update(key_topk_metrics(key_labels, key_scores, k=10))
        metrics.update(region_metrics(pred_regions, true_regions, iou_threshold=0.3))
        metrics.update(region_metrics(pred_regions, true_regions, iou_threshold=0.5))
        metrics.update(boundary_f1(pred_regions, true_regions))
    metrics.update(_hard_negative_fpr(np.asarray(llps_labels), np.asarray(llps_scores), negative_types))
    return metrics


def _hard_negative_fpr(labels: np.ndarray, scores: np.ndarray, negative_types: list[str], threshold: float = 0.5) -> dict[str, float]:
    result: dict[str, float] = {}
    normalized = [str(value).lower() for value in negative_types]
    groups = {
        "NP": ("structured", "np"),
        "ND": ("disordered", "nd"),
        "LCR": ("lcr",),
        "long_IDR": ("long_idr", "long-idr"),
    }
    preds = scores >= threshold
    labels = np.asarray(labels)
    for name, tokens in groups.items():
        mask = np.asarray([any(token in value for token in tokens) for value in normalized]) & (labels == 0)
        result[f"FPR_on_{name}"] = float(np.mean(preds[mask])) if np.any(mask) else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--out")
    args = parser.parse_args()
    config = load_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PhaseFlowModel(checkpoint.get("config", config))
    model.load_state_dict(checkpoint["model"])
    device = resolve_device(str(config.get("device", "auto")))
    model.to(device)
    ids = resolve_split_ids(config["data"], args.split)
    feature_dirs = resolve_feature_dirs(config["data"])
    validate_forbidden_data_paths(config["data"], feature_dirs)
    loader = DataLoader(
        PhaseFlowDataset(
            feature_dirs,
            ids,
            phase_targets=resolve_phase_targets(config["data"]),
            region_targets=config["data"].get("region_targets"),
        ),
        batch_size=int(config["training"].get("batch_size", 2)),
        collate_fn=PhaseFlowCollator(
            max_neighbors=int(config["training"].get("max_neighbors", 96)),
            require_precomputed_graph=bool(config["training"].get("require_precomputed_graph", False)),
        ),
    )
    metrics = evaluate_model(model, loader, device, config.get("postprocess", {}))
    if args.out:
        write_json(args.out, metrics)
    print(dumps_json(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
"""Protein workflow implementation."""

# Source: phaseflow_fusion.py


import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASEFLOW_ROOT = PROJECT_ROOT
DEFAULT_PHASEFLOW_CHECKPOINT = PROJECT_ROOT / "artifacts" / "models" / "peptide" / "best_model.pt"
DEFAULT_PHASEFLOW_PYTHON = Path(sys.executable)

# These bounds match the PhaseFlow website scan endpoint and the local benchmark wrapper.
# Lower PSSI means stronger phase-separation tendency, so pssi_to_score inverts it.
PSSI_MIN = -1.82773655
PSSI_MAX = 0.87820744


@dataclass(frozen=True)
class PhaseFlowFusionConfig:
    phaseflow_root: Path = DEFAULT_PHASEFLOW_ROOT
    checkpoint: Path = DEFAULT_PHASEFLOW_CHECKPOINT
    phaseflow_python: Path | None = None
    profile_jsonl: Path | None = None
    profile_out: Path | None = None
    device: str = "auto"
    batch_size: int = 512
    window_sizes: tuple[int, ...] = (10, 20)
    dpr_fusion_mode: str = "rank_blend"
    dpr_blend_alpha: float = 0.15
    phaseflow_low: float = 0.60
    phaseflow_high: float = 0.68
    phaseflow_rank_gate: float = 0.70
    lift: float = 0.70
    lift_span: float = 0.05
    llps_gate: float = 0.45
    llps_max_phaseflow: float = 1.00
    llps_boost_scale: float = 0.50
    profile_topk_ratio: float = 0.05
    min_sequence_len: int = 5


@dataclass(frozen=True)
class PhaseFlowFusionResult:
    dpr_scores: np.ndarray
    phaseflow_scores: np.ndarray
    phaseflow_rank: np.ndarray
    llps_probability: float
    phaseflow_llps_proxy: float
    changed_fraction: float
    lifted_residues: int
    suppressed_residues: int
    window_sizes: tuple[int, ...]


class PhaseFlowWindowScorer:
    def __init__(self, config: PhaseFlowFusionConfig) -> None:
        self.config = config
        root = Path(config.phaseflow_root)
        checkpoint = Path(config.checkpoint)
        if not root.exists():
            raise FileNotFoundError(f"Missing PhaseFlow root: {root}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing PhaseFlow checkpoint: {checkpoint}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from phaseflow import PhaseFlow
        from phaseflow.peptide.tokenizer import AminoAcidTokenizer

        self.device = resolve_torch_device(config.device)
        patch_phaseflow_cuda_attention(self.device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg = payload["config"]["model"]
        self.model = PhaseFlow(
            dim=cfg["dim"],
            depth=cfg["depth"],
            heads=cfg["heads"],
            dim_head=cfg["dim_head"],
            vocab_size=cfg["vocab_size"],
            phase_dim=cfg["phase_dim"],
            max_seq_len=cfg["max_seq_len"],
            dropout=0.0,
            use_set_encoder=cfg.get("use_set_encoder", False),
            diffusion_type=cfg.get("diffusion_type", "flow_matching"),
            num_timesteps=cfg.get("num_timesteps", 1000),
            beta_schedule=cfg.get("beta_schedule", "cosine"),
        )
        self.model.load_state_dict(payload["model_state_dict"], strict=False)
        self.model.to(self.device).eval()
        self.tokenizer = AminoAcidTokenizer()

    def score_sequence(self, sequence: str) -> tuple[np.ndarray, tuple[int, ...]]:
        sequence = sanitize_sequence(sequence)
        if len(sequence) < int(self.config.min_sequence_len):
            return np.zeros(len(sequence), dtype=np.float32), ()
        window_sizes = usable_window_sizes(sequence, self.config.window_sizes, self.config.min_sequence_len)
        if not window_sizes:
            return np.zeros(len(sequence), dtype=np.float32), ()
        profiles = []
        for window_size in window_sizes:
            windows, starts = extract_windows(sequence, window_size)
            raw_scores = self._predict_window_scores(windows)
            profiles.append(aggregate_window_scores(raw_scores, starts, len(sequence), window_size))
        phaseflow = np.mean(np.stack(profiles, axis=0), axis=0).astype(np.float32)
        return local_contrast_profile(phaseflow), window_sizes

    @torch.no_grad()
    def _predict_window_scores(self, windows: list[str]) -> np.ndarray:
        scores: list[np.ndarray] = []
        batch_size = max(1, int(self.config.batch_size))
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            tokens = self.tokenizer.batch_encode(batch, max_len=32).to(self.device)
            attn_mask = (tokens != self.tokenizer.PAD_ID).long().to(self.device)
            seq_lens = torch.tensor([len(seq) for seq in batch], device=self.device)
            pred = self.model.generate_phase(tokens, attn_mask, seq_lens, method="euler")
            mean_pssi = pred.detach().cpu().numpy().mean(axis=1)
            scores.append(pssi_to_score(mean_pssi))
        return np.concatenate(scores, axis=0).astype(np.float32) if scores else np.zeros(0, dtype=np.float32)

    @torch.no_grad()
    def _predict_raw_pssi_windows(self, windows: list[str]) -> np.ndarray:
        """预测每个窗口的原始 16 维 PSSI 向量（未转换）"""
        pssi_list: list[np.ndarray] = []
        batch_size = max(1, int(self.config.batch_size))
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            tokens = self.tokenizer.batch_encode(batch, max_len=32).to(self.device)
            attn_mask = (tokens != self.tokenizer.PAD_ID).long().to(self.device)
            seq_lens = torch.tensor([len(seq) for seq in batch], device=self.device)
            pred = self.model.generate_phase(tokens, attn_mask, seq_lens, method="euler")
            # pred shape: (batch, 16), 16 维 PSSI
            pssi_list.append(pred.detach().cpu().numpy())
        return np.concatenate(pssi_list, axis=0).astype(np.float32) if pssi_list else np.zeros((0, 16), dtype=np.float32)

    def score_sequence_global_pssi(self, sequence: str, window_size: int = 20) -> np.ndarray:
        """
        对整个序列生成一个 16 维 PSSI 向量（通过滑动窗口聚合）

        Args:
            sequence: 蛋白质序列
            window_size: 滑动窗口大小
        Returns:
            16 维 PSSI 向量
        """
        sequence = sanitize_sequence(sequence)
        if len(sequence) < int(self.config.min_sequence_len):
            return np.zeros(16, dtype=np.float32)

        windows, _ = extract_windows(sequence, window_size)
        if not windows:
            return np.zeros(16, dtype=np.float32)

        raw_pssi = self._predict_raw_pssi_windows(windows)
        if raw_pssi.size == 0:
            return np.zeros(16, dtype=np.float32)

        # 对所有窗口的 16 维 PSSI 取平均
        return np.mean(raw_pssi, axis=0).astype(np.float32)


def run_phaseflow_profile_subprocess(
    *,
    records: dict[str, str],
    config: PhaseFlowFusionConfig,
    out_path: Path,
) -> dict[str, tuple[np.ndarray, tuple[int, ...]]]:
    python_bin = config.phaseflow_python
    if python_bin is None:
        raise ValueError("phaseflow_python is required for subprocess profile generation.")
    python_path = Path(python_bin)
    if not python_path.exists():
        raise FileNotFoundError(f"Missing PhaseFlow python executable: {python_path}")
    script = PROJECT_ROOT / "benchmark_runtime" / "scripts" / "write_phaseflow_window_profiles.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing PhaseFlow profile script: {script}")

    out_path = Path(out_path)
    metadata_path = out_path.with_suffix(".input.csv")
    write_sequence_metadata_csv(metadata_path, records)
    command = [
        str(python_path),
        str(script),
        "--sequence-metadata",
        str(metadata_path),
        "--out",
        str(out_path),
        "--phaseflow-root",
        str(config.phaseflow_root),
        "--checkpoint",
        str(config.checkpoint),
        "--device",
        str(config.device),
        "--batch-size",
        str(config.batch_size),
        "--window-sizes",
        ",".join(str(size) for size in config.window_sizes),
    ]
    result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            "PhaseFlow subprocess failed with return code "
            f"{result.returncode}\nstdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
        )
    return load_phaseflow_profile_jsonl(out_path)


def write_sequence_metadata_csv(path: Path, records: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "protein_id", "sequence", "length"])
        writer.writeheader()
        for record_id in sorted(records):
            sequence = sanitize_sequence(records[record_id])
            writer.writerow(
                {
                    "record_id": record_id,
                    "protein_id": record_id,
                    "sequence": sequence,
                    "length": len(sequence),
                }
            )


def load_phaseflow_profile_jsonl(path: Path) -> dict[str, tuple[np.ndarray, tuple[int, ...]]]:
    profiles: dict[str, tuple[np.ndarray, tuple[int, ...]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            record_id = str(payload.get("record_id") or payload.get("protein_id") or payload.get("id") or "")
            scores = payload.get("score")
            if scores is None:
                scores = payload.get("scores")
            if not record_id or scores is None:
                continue
            window_sizes = payload.get("window_sizes", "")
            if isinstance(window_sizes, str) and window_sizes:
                parsed_windows = parse_window_sizes(window_sizes)
            elif isinstance(window_sizes, list | tuple):
                parsed_windows = parse_window_sizes(window_sizes)
            else:
                parsed_windows = ()
            profiles[record_id] = (np.asarray(scores, dtype=np.float32), parsed_windows)
    return profiles


def fuse_window_phaseflow_with_protein(
    *,
    protein_dpr: np.ndarray,
    protein_llps_probability: float,
    window_scores: np.ndarray,
    config: PhaseFlowFusionConfig,
    window_sizes: tuple[int, ...] = (),
) -> PhaseFlowFusionResult:
    base = np.clip(np.asarray(protein_dpr, dtype=np.float32), 0.0, 1.0)
    window = fit_profile(np.asarray(window_scores, dtype=np.float32), base.size)
    rank = percentile_rank(window)
    mode = str(config.dpr_fusion_mode).strip().lower()
    if mode == "rank_blend":
        fused = rank_blend(phaseflow_scores=base, phaseflow_rank=rank, alpha=float(config.dpr_blend_alpha))
    elif mode == "gated_lift":
        fused = gated_lift(
            phaseflow_scores=base,
            phaseflow_rank=rank,
            phaseflow_low=float(config.phaseflow_low),
            phaseflow_high=float(config.phaseflow_high),
            phaseflow_rank_gate=float(config.phaseflow_rank_gate),
            lift=float(config.lift),
            lift_span=float(config.lift_span),
        )
    else:
        raise ValueError(f"Unknown DPR PhaseFlow fusion mode: {config.dpr_fusion_mode}")
    phaseflow_proxy = phaseflow_global_score(window, topk_ratio=float(config.profile_topk_ratio))
    fused_llps = fuse_llps_probability(
        phaseflow_probability=float(protein_llps_probability),
        phaseflow_proxy=phaseflow_proxy,
        config=config,
    )
    changed = np.abs(fused - base) > 1.0e-8
    lifted = fused > base + 1.0e-8
    suppressed = fused < base - 1.0e-8
    return PhaseFlowFusionResult(
        dpr_scores=fused,
        phaseflow_scores=window,
        phaseflow_rank=rank,
        llps_probability=fused_llps,
        phaseflow_llps_proxy=phaseflow_proxy,
        changed_fraction=float(np.mean(changed)) if changed.size else 0.0,
        lifted_residues=int(np.sum(lifted)),
        suppressed_residues=int(np.sum(suppressed)),
        window_sizes=window_sizes,
    )


def fuse_phaseflow_with_phaseflow(
    *,
    phaseflow_dpr: np.ndarray,
    phaseflow_llps_probability: float,
    phaseflow_scores: np.ndarray,
    config: PhaseFlowFusionConfig,
    window_sizes: tuple[int, ...] = (),
) -> PhaseFlowFusionResult:
    """Compatibility wrapper for the pre-rename fusion API."""
    return fuse_window_phaseflow_with_protein(
        protein_dpr=phaseflow_dpr,
        protein_llps_probability=phaseflow_llps_probability,
        window_scores=phaseflow_scores,
        config=config,
        window_sizes=window_sizes,
    )


def rank_blend(*, phaseflow_scores: np.ndarray, phaseflow_rank: np.ndarray, alpha: float) -> np.ndarray:
    c = np.clip(np.asarray(phaseflow_scores, dtype=np.float32), 0.0, 1.0)
    p = fit_profile(np.asarray(phaseflow_rank, dtype=np.float32), c.size)
    a = float(np.clip(alpha, 0.0, 1.0))
    return np.clip((1.0 - a) * c + a * p, 0.0, 1.0).astype(np.float32)


def gated_lift(
    *,
    phaseflow_scores: np.ndarray,
    phaseflow_rank: np.ndarray,
    phaseflow_low: float,
    phaseflow_high: float,
    phaseflow_rank_gate: float,
    lift: float,
    lift_span: float,
) -> np.ndarray:
    c = np.clip(np.asarray(phaseflow_scores, dtype=np.float32), 0.0, 1.0)
    p = fit_profile(np.asarray(phaseflow_rank, dtype=np.float32), c.size)
    denom = max(1.0e-6, 1.0 - float(phaseflow_rank_gate))
    boost = np.maximum(0.0, (p - float(phaseflow_rank_gate)) / denom)
    mask = (c >= float(phaseflow_low)) & (c < float(phaseflow_high)) & (p >= float(phaseflow_rank_gate))
    out = c.copy()
    out[mask] = np.maximum(out[mask], float(lift) + float(lift_span) * boost[mask])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fuse_llps_probability(
    *,
    phaseflow_probability: float,
    phaseflow_proxy: float,
    config: PhaseFlowFusionConfig,
) -> float:
    base = float(np.clip(phaseflow_probability, 1.0e-6, 1.0 - 1.0e-6))
    proxy = float(np.clip(phaseflow_proxy, 0.0, 1.0))
    if base >= float(config.llps_max_phaseflow) or proxy < float(config.llps_gate):
        return base
    boost = float(config.llps_boost_scale) * max(0.0, proxy - float(config.llps_gate)) * (1.0 - base)
    return float(np.clip(base + boost, 1.0e-6, 1.0 - 1.0e-6))


def phaseflow_global_score(scores: np.ndarray, topk_ratio: float = 0.05) -> float:
    values = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if values.size == 0:
        return 0.0
    k = max(1, int(round(values.size * float(topk_ratio))))
    topk = np.partition(values, -min(k, values.size))[-min(k, values.size) :]
    return float(np.clip(0.7 * float(np.mean(topk)) + 0.3 * float(np.max(values)), 0.0, 1.0))


def pssi_to_score(mean_pssi: np.ndarray) -> np.ndarray:
    return np.clip((PSSI_MAX - np.asarray(mean_pssi, dtype=np.float32)) / (PSSI_MAX - PSSI_MIN), 0.0, 1.0).astype(np.float32)


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if values.size <= 1:
        return np.full_like(values, 0.5, dtype=np.float32)
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(np.float32)
    return np.clip(order / float(values.size - 1), 0.0, 1.0).astype(np.float32)


def aggregate_window_scores(raw_scores: np.ndarray, starts: list[int], length: int, window_size: int) -> np.ndarray:
    sums = np.zeros(length, dtype=np.float32)
    counts = np.zeros(length, dtype=np.float32)
    for score, start in zip(raw_scores, starts, strict=True):
        end = min(length, int(start) + int(window_size))
        sums[int(start) : end] += float(score)
        counts[int(start) : end] += 1.0
    out = np.zeros(length, dtype=np.float32)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    return out


def local_contrast_profile(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size <= 1:
        return np.clip(np.nan_to_num(scores, nan=0.0), 0.0, 1.0).astype(np.float32)
    finite = scores[np.isfinite(scores)]
    if finite.size <= 1:
        return np.nan_to_num(scores, nan=0.0).astype(np.float32)
    lo, hi = np.percentile(finite, [5, 95])
    if hi - lo < 1.0e-6:
        local = np.full_like(scores, 0.5, dtype=np.float32)
    else:
        local = np.clip((scores - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    global_score = np.clip(np.nan_to_num(scores, nan=0.0), 0.0, 1.0)
    return np.clip(0.55 * global_score + 0.45 * local, 0.0, 1.0).astype(np.float32)


def extract_windows(sequence: str, window_size: int) -> tuple[list[str], list[int]]:
    if len(sequence) <= window_size:
        return [sequence], [0]
    windows = []
    starts = []
    for start in range(0, len(sequence) - window_size + 1):
        windows.append(sequence[start : start + window_size])
        starts.append(start)
    return windows, starts


def parse_window_sizes(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
        sizes = [int(item) for item in raw]
    else:
        sizes = [int(item) for item in value]
    if not sizes:
        raise ValueError("At least one PhaseFlow window size is required.")
    for size in sizes:
        if size <= 0 or size > 32:
            raise ValueError(f"PhaseFlow window size must be in 1..32, got {size}")
    return tuple(sorted(set(sizes)))


def usable_window_sizes(sequence: str, window_sizes: Iterable[int], min_sequence_len: int = 5) -> tuple[int, ...]:
    length = len(sequence)
    sizes = [size for size in parse_window_sizes(window_sizes) if size <= length]
    if sizes:
        return tuple(sizes)
    if length >= int(min_sequence_len):
        return (min(length, 32),)
    return ()


def fit_profile(profile: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(int(length), dtype=np.float32)
    if length <= 0:
        return out
    values = np.nan_to_num(np.asarray(profile, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    copy_len = min(int(length), int(values.size))
    if copy_len:
        out[:copy_len] = values[:copy_len]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def sanitize_sequence(sequence: str) -> str:
    cleaned = str(sequence).upper().strip()
    cleaned = cleaned.replace("X", "G").replace("B", "N").replace("Z", "Q")
    return "".join(aa for aa in cleaned if aa in set("ACDEFGHIKLMNPQRSTVWY"))


def resolve_torch_device(device_name: str) -> torch.device:
    if str(device_name) == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_name))


def patch_phaseflow_cuda_attention(device: torch.device) -> None:
    if device.type != "cuda":
        return
    from phaseflow.peptide.transformer import Attention

    if getattr(Attention, "_phaseflow_sdpa_patch", False):
        return

    def sdpa_forward(
        self,
        x: torch.Tensor,
        rotary_emb: object,
        attention_mask: torch.Tensor | None = None,
        phase_start_idx: int | None = None,
        phase_end_idx: int | None = None,
        skip_phase_rope: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        k = self.k_proj(x).view(batch, seq_len, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        v = self.v_proj(x).view(batch, seq_len, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        if skip_phase_rope and phase_end_idx is not None:
            end = phase_end_idx
            q_seq = rotary_emb.rotate_queries_or_keys(q[:, :, end:, :])
            k_seq = rotary_emb.rotate_queries_or_keys(k[:, :, end:, :])
            q = torch.cat([q[:, :, :end, :], q_seq], dim=2).contiguous()
            k = torch.cat([k[:, :, :end, :], k_seq], dim=2).contiguous()
        elif skip_phase_rope and phase_start_idx is not None:
            start = phase_start_idx
            q_seq = rotary_emb.rotate_queries_or_keys(q[:, :, :start, :])
            k_seq = rotary_emb.rotate_queries_or_keys(k[:, :, :start, :])
            q = torch.cat([q_seq, q[:, :, start:, :]], dim=2).contiguous()
            k = torch.cat([k_seq, k[:, :, start:, :]], dim=2).contiguous()
        else:
            q = rotary_emb.rotate_queries_or_keys(q).contiguous()
            k = rotary_emb.rotate_queries_or_keys(k).contiguous()

        mask = self._build_attention_mask(batch, seq_len, x.device, phase_start_idx, phase_end_idx)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            mask = mask & attention_mask.bool()

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, self.heads * self.dim_head)
        return self.out_proj(out)

    Attention.forward = sdpa_forward
    Attention._phaseflow_sdpa_patch = True

# Source: infer.py


import argparse
import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

from phaseflow.protein.data import PhaseFlowCollator
from phaseflow.protein.data import PhaseFlowDataset
from phaseflow.protein.model import PhaseFlowModel
from scripts.protein.workflows.evaluation import (
    DEFAULT_PHASEFLOW_CHECKPOINT,
    DEFAULT_PHASEFLOW_PYTHON,
    DEFAULT_PHASEFLOW_ROOT,
    PhaseFlowFusionConfig,
    PhaseFlowWindowScorer,
    fuse_window_phaseflow_with_protein,
    load_phaseflow_profile_jsonl,
    parse_window_sizes,
    run_phaseflow_profile_subprocess,
)
from phaseflow.protein.postprocessing import combine_regions, decoder_regions, scores_to_regions
from phaseflow.protein.contracts import load_yaml, move_batch_to_device, resolve_device


@torch.no_grad()
def run_inference(
    checkpoint_path: str | Path,
    feature_dir: str | Path,
    out: str | Path,
    protein_ids: list[str] | None = None,
    phaseflow_config: PhaseFlowFusionConfig | None = None,
    batch_size: int | None = None,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = resolve_device(str(config.get("device", "auto")))
    model = PhaseFlowModel(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    edge_attr_dim = _checkpoint_edge_attr_dim(config)
    if protein_ids is None:
        protein_ids = sorted(path.stem for path in Path(feature_dir).glob("*.h5"))
    out_path = Path(out)
    phaseflow_profiles = None
    phaseflow_scorer = None
    if phaseflow_config is not None:
        if phaseflow_config.profile_jsonl is not None:
            phaseflow_profiles = load_phaseflow_profile_jsonl(Path(phaseflow_config.profile_jsonl))
        elif phaseflow_config.phaseflow_python is not None:
            profile_out = Path(phaseflow_config.profile_out or _phaseflow_profile_out_path(out_path))
            records = _read_sequences_from_feature_cache(feature_dir, protein_ids)
            phaseflow_profiles = run_phaseflow_profile_subprocess(
                records=records,
                config=phaseflow_config,
                out_path=profile_out,
            )
        else:
            try:
                phaseflow_scorer = PhaseFlowWindowScorer(phaseflow_config)
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "PhaseFlow direct import failed. Run with --phaseflow-python "
                    f"{DEFAULT_PHASEFLOW_PYTHON} or provide --phaseflow-profile-jsonl."
                ) from exc
    loader = DataLoader(
        PhaseFlowDataset(feature_dir, protein_ids),
        batch_size=int(batch_size or config.get("training", {}).get("batch_size", 2)),
        shuffle=False,
        collate_fn=PhaseFlowCollator(
            max_neighbors=int(config.get("training", {}).get("max_neighbors", 96)),
            edge_attr_dim=edge_attr_dim,
            require_precomputed_graph=bool(config.get("training", {}).get("require_precomputed_graph", False)),
        ),
    )
    post_config = config.get("postprocess", {})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            llps_output = outputs.get("llps_logits", outputs["llps_logits"])
            llps = torch.sigmoid(llps_output).detach().cpu().numpy()
            region_global_output = outputs.get("region_global_score")
            if region_global_output is None:
                region_global_output = torch.sigmoid(outputs.get("region_global_logits", llps_output))
            region_global = region_global_output.detach().cpu().numpy()
            dpr = torch.sigmoid(outputs["dpr_logits"]).detach().cpu().numpy()
            key = torch.sigmoid(outputs["key_logits"]).detach().cpu().numpy()
            modality_weights = outputs["modality_weights"].detach().cpu().numpy()
            region_logits = outputs["region_logits"].detach().cpu().numpy()
            region_start = outputs["region_start"].detach().cpu().numpy()
            region_end = outputs["region_end"].detach().cpu().numpy()
            lengths = batch["lengths"].detach().cpu().numpy()
            for index, protein_id in enumerate(batch["protein_ids"]):
                length = int(lengths[index])
                phaseflow_llps = float(llps[index])
                output_llps = phaseflow_llps
                phaseflow_dpr_scores = dpr[index, :length]
                dpr_scores = phaseflow_dpr_scores
                phaseflow_fusion = None
                if phaseflow_config is not None:
                    if phaseflow_profiles is not None:
                        if str(protein_id) not in phaseflow_profiles:
                            raise RuntimeError(f"Missing PhaseFlow profile for {protein_id}")
                        phaseflow_scores, used_windows = phaseflow_profiles[str(protein_id)]
                    elif phaseflow_scorer is not None:
                        phaseflow_scores, used_windows = phaseflow_scorer.score_sequence(batch["sequences"][index])
                    else:
                        raise RuntimeError("PhaseFlow fusion is enabled but no scorer or profile lookup is available.")
                    phaseflow_fusion = fuse_window_phaseflow_with_protein(
                        protein_dpr=phaseflow_dpr_scores,
                        protein_llps_probability=phaseflow_llps,
                        window_scores=phaseflow_scores,
                        config=phaseflow_config,
                        window_sizes=used_windows,
                    )
                    dpr_scores = phaseflow_fusion.dpr_scores
                    output_llps = phaseflow_fusion.llps_probability
                post = scores_to_regions(
                    dpr_scores,
                    threshold=float(post_config.get("threshold", 0.5)),
                    smooth_window=int(post_config.get("smooth_window", 5)),
                    merge_gap=int(post_config.get("merge_gap", 5)),
                    min_region_len=int(post_config.get("min_region_len", 6)),
                )
                if phaseflow_fusion is not None:
                    for region in post:
                        region["source"] = "phaseflow_fused_postprocess"
                dec = decoder_regions(
                    region_logits[index],
                    region_start[index],
                    region_end[index],
                    length,
                    score_threshold=float(post_config.get("decoder_score_threshold", 0.5)),
                )
                regions = combine_regions(dec, post)
                evidence = _evidence_for_sample(
                    modality_weights[index, :length],
                    batch["modality_mask"][index, :length].detach().cpu().numpy(),
                    batch["structure_metadata"][index],
                )
                if phaseflow_fusion is not None:
                    evidence["phaseflow_fusion"] = {
                        "enabled": True,
                        "method": phaseflow_config.dpr_fusion_mode,
                        "dpr_blend_alpha": float(phaseflow_config.dpr_blend_alpha),
                        "window_sizes": list(phaseflow_fusion.window_sizes),
                        "phaseflow_llps_proxy": float(phaseflow_fusion.phaseflow_llps_proxy),
                        "changed_fraction": float(phaseflow_fusion.changed_fraction),
                        "lifted_residues": int(phaseflow_fusion.lifted_residues),
                        "suppressed_residues": int(phaseflow_fusion.suppressed_residues),
                    }
                residue_scores = {
                    "DPR": [float(value) for value in dpr_scores],
                    "key_residue": [float(value) for value in key[index, :length]],
                }
                if phaseflow_fusion is not None:
                    residue_scores["phaseflow_DPR"] = [float(value) for value in phaseflow_dpr_scores]
                    residue_scores["phaseflow_DPR"] = [float(value) for value in phaseflow_fusion.phaseflow_scores]
                    residue_scores["phaseflow_rank"] = [float(value) for value in phaseflow_fusion.phaseflow_rank]
                row = {
                    "protein_id": protein_id,
                    "length": length,
                    "LLPS_probability": float(output_llps),
                    "protein_llps_score": float(output_llps),
                    "region_global_llps_score": float(region_global[index]),
                    "phaseflow_LLPS_probability": float(phaseflow_llps),
                    "coordinate_system": "1-based inclusive",
                    "residue_scores": residue_scores,
                    "dpr_regions": _public_regions(regions),
                    "DPR_regions": _public_regions(regions),
                    "evidence": evidence,
                }
                if phaseflow_fusion is not None:
                    row["phaseflow_LLPS_proxy"] = float(phaseflow_fusion.phaseflow_llps_proxy)
                    row["fusion_method"] = f"phaseflow_phaseflow_{phaseflow_config.dpr_fusion_mode}"
                handle.write(json.dumps(row) + "\n")


def _checkpoint_edge_attr_dim(config: dict) -> int:
    edge_dim = int(config.get("model", {}).get("graph_transformer", {}).get("edge_dim", 8))
    if edge_dim <= 0:
        raise ValueError(f"checkpoint graph edge_dim must be positive, got {edge_dim}")
    return edge_dim


def _public_regions(regions: list[dict[str, float]]) -> list[dict[str, float]]:
    converted: list[dict[str, float]] = []
    for region in regions:
        row = dict(region)
        row["start"] = int(row["start"]) + 1
        row["end"] = int(row["end"]) + 1
        converted.append(row)
    return converted


def _evidence_for_sample(modality_weights, modality_mask, structure_metadata: dict) -> dict[str, object]:
    names = ["plm", "physchem", "disorder", "protenix_embed", "starling_embed"]
    available = 1.0 - modality_mask.astype(float)
    weighted = modality_weights * available
    means = weighted.mean(axis=0) if weighted.size else [0.0] * len(names)
    order = sorted(range(len(names)), key=lambda index: float(means[index]), reverse=True)
    important = [names[index] for index in order if float(means[index]) > 0.05][:3]
    structure_provider = str(structure_metadata.get("structure_provider", "none"))
    return {
        "important_modalities": important,
        "modality_weights": {names[index]: float(means[index]) for index in range(len(names))},
        "structure_provider": structure_provider,
        "structure_success": str(structure_metadata.get("structure_success", "")),
        "structure_model": str(structure_metadata.get("model_name", "")),
    }


def _phaseflow_profile_out_path(out_path: Path) -> Path:
    if out_path.suffix:
        return out_path.with_suffix(".phaseflow_profiles.jsonl")
    return out_path.parent / f"{out_path.name}.phaseflow_profiles.jsonl"


def _read_sequences_from_feature_cache(feature_dir: str | Path, protein_ids: list[str]) -> dict[str, str]:
    directory = Path(feature_dir)
    records: dict[str, str] = {}
    for protein_id in protein_ids:
        path = directory / f"{protein_id}.h5"
        if not path.exists():
            raise FileNotFoundError(f"Missing feature cache for {protein_id}: {path}")
        with h5py.File(path, "r") as handle:
            value = handle.attrs.get("sequence", "")
            if isinstance(value, bytes):
                sequence = value.decode("utf-8")
            else:
                sequence = str(value)
        records[str(protein_id)] = sequence
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--protein-ids", nargs="*")
    parser.add_argument("--config", help="Accepted for CLI symmetry; checkpoint config is authoritative.")
    parser.add_argument("--phaseflow-fusion", action="store_true", help="Fuse PhaseFlow inference with PhaseFlow short-window evidence.")
    parser.add_argument("--phaseflow-root", type=Path, default=DEFAULT_PHASEFLOW_ROOT)
    parser.add_argument("--phaseflow-checkpoint", type=Path, default=DEFAULT_PHASEFLOW_CHECKPOINT)
    parser.add_argument(
        "--phaseflow-python",
        type=Path,
        default=DEFAULT_PHASEFLOW_PYTHON if DEFAULT_PHASEFLOW_PYTHON.exists() else None,
        help="Optional PhaseFlow environment Python. When set, profiles are generated in a subprocess.",
    )
    parser.add_argument("--phaseflow-profile-jsonl", type=Path, help="Reuse precomputed PhaseFlow profile JSONL.")
    parser.add_argument("--phaseflow-profile-out", type=Path, help="Where subprocess-generated PhaseFlow profiles should be written.")
    parser.add_argument("--phaseflow-device", default="auto")
    parser.add_argument("--phaseflow-batch-size", type=int, default=512)
    parser.add_argument("--phaseflow-window-sizes", default="10,20")
    parser.add_argument("--phaseflow-dpr-mode", default="rank_blend", choices=["rank_blend", "gated_lift"])
    parser.add_argument("--phaseflow-dpr-blend-alpha", type=float, default=0.15)
    parser.add_argument("--phaseflow-phaseflow-low", type=float, default=0.60)
    parser.add_argument("--phaseflow-phaseflow-high", type=float, default=0.68)
    parser.add_argument("--phaseflow-rank-gate", type=float, default=0.70)
    parser.add_argument("--phaseflow-lift", type=float, default=0.70)
    parser.add_argument("--phaseflow-lift-span", type=float, default=0.05)
    parser.add_argument("--phaseflow-llps-gate", type=float, default=0.45)
    parser.add_argument("--phaseflow-llps-max-phaseflow", type=float, default=1.00)
    parser.add_argument("--phaseflow-llps-boost-scale", type=float, default=0.50)
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError("Inference requires a trained checkpoint; no fallback prediction is provided.")
    if args.config:
        load_yaml(args.config)
    phaseflow_config = None
    if args.phaseflow_fusion:
        phaseflow_config = PhaseFlowFusionConfig(
            phaseflow_root=Path(args.phaseflow_root),
            checkpoint=Path(args.phaseflow_checkpoint),
            phaseflow_python=Path(args.phaseflow_python) if args.phaseflow_python else None,
            profile_jsonl=Path(args.phaseflow_profile_jsonl) if args.phaseflow_profile_jsonl else None,
            profile_out=Path(args.phaseflow_profile_out) if args.phaseflow_profile_out else None,
            device=str(args.phaseflow_device),
            batch_size=int(args.phaseflow_batch_size),
            window_sizes=parse_window_sizes(args.phaseflow_window_sizes),
            dpr_fusion_mode=str(args.phaseflow_dpr_mode),
            dpr_blend_alpha=float(args.phaseflow_dpr_blend_alpha),
            phaseflow_low=float(args.phaseflow_phaseflow_low),
            phaseflow_high=float(args.phaseflow_phaseflow_high),
            phaseflow_rank_gate=float(args.phaseflow_rank_gate),
            lift=float(args.phaseflow_lift),
            lift_span=float(args.phaseflow_lift_span),
            llps_gate=float(args.phaseflow_llps_gate),
            llps_max_phaseflow=float(args.phaseflow_llps_max_phaseflow),
            llps_boost_scale=float(args.phaseflow_llps_boost_scale),
        )
    run_inference(args.checkpoint, args.feature_dir, args.out, args.protein_ids, phaseflow_config)
    print(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
"""Evaluate a cached-hidden DPR checkpoint on the frozen official PhasePro set."""


import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseflow.protein.objectives import (  # noqa: E402
    build_truths,
    per_protein_metrics,
    threshold_free_metrics,
    to_jsonable,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_llps_reference_metrics(
    observed: dict[str, float], reference: dict[str, float], *, tolerance: float
) -> None:
    if tolerance < 0:
        raise ValueError("LLPS metric tolerance must be non-negative")
    for name, expected in reference.items():
        if name not in observed:
            raise ValueError(f"LLPS evaluator did not produce published metric: {name}")
        difference = abs(float(observed[name]) - float(expected))
        if difference > tolerance:
            raise ValueError(
                f"LLPS metric {name} is outside the published tolerance: "
                f"observed={observed[name]:.12g}, expected={expected:.12g}, "
                f"difference={difference:.12g}, tolerance={tolerance:.12g}"
            )


def evaluate_llps_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an LLPS checkpoint on an explicit PPMC panel.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panel-id")
    return parser.parse_args(argv)


def evaluate_llps_main(argv: list[str] | None = None) -> int:
    args = evaluate_llps_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    contract = config["reproduction"]["benchmark"]["ppmc"]
    observed_hash = _sha256_file(args.checkpoint)
    if observed_hash != str(contract["checkpoint_sha256"]):
        raise ValueError("LLPS checkpoint SHA256 does not match the published PPMC contract")
    panel_id = str(args.panel_id or contract["panel_id"])
    score_source = str(contract.get("score_source", "region_global_llps_score"))
    batch_size = int(contract.get("batch_size", 8))
    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_root / "llps_predictions.jsonl"
    from scripts.protein.workflows.evaluation import run_inference
    from scripts.protein.workflows.release import score_llps_panel

    panel = pd.read_csv(args.panel)
    required_columns = {"panel_id", "protein_id", "llps_label"}
    missing_columns = sorted(required_columns - set(panel.columns))
    if missing_columns:
        raise ValueError(f"LLPS panel is missing required columns: {missing_columns}")
    selected_panel = panel.loc[panel["panel_id"].astype(str).eq(panel_id)].copy()
    if selected_panel.empty:
        raise ValueError(f"unknown panel_id: {panel_id}")
    if selected_panel["protein_id"].astype(str).duplicated().any():
        raise ValueError(f"panel {panel_id} has duplicated protein_id values")
    protein_ids = _length_sorted_protein_ids(args.feature_dir, selected_panel["protein_id"].astype(str).tolist())
    run_inference(
        args.checkpoint,
        args.feature_dir,
        predictions_path,
        protein_ids=protein_ids,
        batch_size=batch_size,
    )
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line]
    predictions = pd.DataFrame(
        {
            "protein_id": [str(row["protein_id"]) for row in rows],
            "llps_score": [float(row[score_source]) for row in rows],
        }
    )
    predictions.to_csv(args.output_root / "llps_predictions.csv", index=False)
    metrics = score_llps_panel(predictions, selected_panel, panel_id=panel_id)
    reference_metrics = config["reproduction"].get("reference_metrics", {}).get("ppmc", {})
    published_reference_metrics = {
        str(name): float(value) for name, value in reference_metrics.items()
    }
    metric_tolerance = float(contract.get("metric_tolerance", 0.0))
    if reference_metrics:
        _validate_llps_reference_metrics(
            metrics,
            published_reference_metrics,
            tolerance=metric_tolerance,
        )
    summary = {
        "status": "PASS",
        "contract": {
            "checkpoint_sha256": observed_hash,
            "precision": str(contract["precision"]),
            "threshold": float(contract["threshold"]),
            "panel_id": panel_id,
            "score_source": score_source,
            "inference_order": "feature_length_then_protein_id",
            "batch_size": batch_size,
            "metric_tolerance": metric_tolerance,
            "reference_metrics": published_reference_metrics,
        },
        "metrics": metrics,
    }
    (args.output_root / "llps_summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return 0


def _length_sorted_protein_ids(feature_dir: Path, protein_ids: list[str]) -> list[str]:
    def feature_length(protein_id: str) -> tuple[int, str]:
        path = feature_dir / f"{protein_id}.h5"
        if not path.is_file():
            raise FileNotFoundError(f"missing LLPS feature cache for benchmark protein {protein_id}: {path}")
        with h5py.File(path, "r") as handle:
            length_value = handle.attrs.get("length")
            length = int(length_value) if length_value is not None else int(handle["plm"].shape[0])
        if length <= 0:
            raise ValueError(f"invalid LLPS feature length for benchmark protein {protein_id}: {length}")
        return length, protein_id

    return [protein_id for _, protein_id in sorted(feature_length(protein_id) for protein_id in protein_ids)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("PHASEFLOW_DISABLE_STARLING_READ", "1")
    os.environ.setdefault("PHASEFLOW_DISABLE_PROTENIX_READ", "1")
    os.environ.setdefault("PHASEFLOW_STRICT_OFFLINE", "1")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    from phaseflow.protein.data import DPRV5BaseOnlySidecar, dpr_v5_collate
    from phaseflow.protein.model import load_dpr_v6_phasestack
    from scripts.protein.workflows.release import validate_release_sidecar_pairs
    from scripts.protein.workflows.training import move_batch_to_device, normalize_checkpoint_namespace

    model, _, _ = load_dpr_v6_phasestack(
        phaseflow_llps_checkpoint=config["paths"]["phaseflow_llps_checkpoint"],
        phaseflow_checkpoint=config["paths"]["phaseflow_checkpoint"],
        config={"model": dict(config["model"])},
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = normalize_checkpoint_namespace(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    sidecar = DPRV5BaseOnlySidecar(
        v2_data_root=config["paths"]["v2_data_root"], packed_root=args.sidecar_root, mmap=True
    )
    proteins = pd.read_parquet(args.benchmark_root / "proteins.parquet")
    regions = pd.read_parquet(args.benchmark_root / "regions.parquet")
    pair_audit = validate_release_sidecar_pairs(proteins, sidecar.manifest, expected_count=121)
    profiles: dict[str, np.ndarray] = {}
    for item in sidecar.manifest.sort_values(["protein_id", "sequence_sha256"]).to_dict(orient="records"):
        row = type(
            "PhaseProRow", (), {"protein_id": str(item["protein_id"]), "sequence_sha256": str(item["sequence_sha256"]), "v3_tier": "S", "v3_pool": "phasepro_evaluation_only"}
        )()
        batch = move_batch_to_device(dpr_v5_collate([sidecar.sample_from_tier_row(row)]), device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch=batch, task="dpr")["dpr"]["p257"]
        mask = batch["seq_mask"][0].bool()
        profile = output[0, mask].detach().float().cpu().numpy().astype(np.float32, copy=False)
        if len(profile) != int(item["length"]) or not np.isfinite(profile).all():
            raise RuntimeError(f"invalid p257 profile for {row.protein_id}")
        profiles[row.protein_id] = profile
    if set(profiles) != set(proteins["protein_id"].astype(str)):
        raise RuntimeError("incomplete PhasePro profile coverage")
    truths = build_truths(proteins, regions)
    per_protein = per_protein_metrics(profiles, truths)
    metrics = threshold_free_metrics(profiles, truths, per_protein)
    args.output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_root / "raw_p257_profiles.npz", **profiles)
    per_protein.to_csv(args.output_root / "per_protein_p257.csv", index=False)
    summary = {
        "status": "PASS",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "llps_feature_policy": "historical_cached_hidden",
        "pair_audit": pair_audit,
        "threshold_free": metrics,
        "files": {"profiles": str((args.output_root / "raw_p257_profiles.npz").resolve())},
    }
    (args.output_root / "phasepro_summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
