from __future__ import annotations

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASEFLOW_ROOT = PROJECT_ROOT
DEFAULT_PHASEFLOW_CHECKPOINT = DEFAULT_PHASEFLOW_ROOT / "outputs_set" / "output_set_flow32_missing15" / "best_model.pt"
DEFAULT_PHASEFLOW_PYTHON = PROJECT_ROOT.parent / "conda_envs" / "phaseflow" / "bin" / "python"

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
        from phaseflow.tokenizer import AminoAcidTokenizer

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


def fuse_window_phaseflow_with_full_length(
    *,
    full_length_dpr: np.ndarray,
    full_length_llps_probability: float,
    window_scores: np.ndarray,
    config: PhaseFlowFusionConfig,
    window_sizes: tuple[int, ...] = (),
) -> PhaseFlowFusionResult:
    base = np.clip(np.asarray(full_length_dpr, dtype=np.float32), 0.0, 1.0)
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
        phaseflow_probability=float(full_length_llps_probability),
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
    return fuse_window_phaseflow_with_full_length(
        full_length_dpr=phaseflow_dpr,
        full_length_llps_probability=phaseflow_llps_probability,
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
    from phaseflow.transformer import Attention

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
