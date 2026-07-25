from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from phaseflow.full_length.models.phase_stack_dpr import (
    DEFAULT_PHASEFLOW_REPO,
    DEFAULT_PHASEFLOW_SITE_PACKAGES,
    LEGACY_LLPS_HIDDEN_DIM_KEY,
    LEGACY_LLPS_PREFIX,
    PHASEFLOW_LLPS_HIDDEN_DIM_KEY,
    _prepare_phaseflow_imports,
    _phaseflow_safe_sequence,
    _phaseflow_tokenizer,
    load_phaseflow_llps_checkpoint,
    load_phaseflow_checkpoint,
)
from phaseflow.full_length.models.phaseflow import PhaseFlowModel
from phaseflow.full_length.data.dpr_v6 import TIER_TO_BAG_LABEL, TIER_TO_WEIGHT


PHASEFLOW_LLPS_DIRECT_STREAM = "phaseflow_llps_direct"
PHASEFLOW_LLPS_HIDDEN_KEY = "phaseflow_llps_hidden"
PHASEFLOW_LLPS_CONFIG_KEY = "phaseflow_llps_config"
PHASEFLOW_LLPS_CHECKPOINT_KEY = "phaseflow_llps_checkpoint"
PHASEFLOW_LLPS_STREAM_KEY = "use_phaseflow_llps_stream"
LEGACY_LLPS_DIRECT_STREAM = LEGACY_LLPS_PREFIX + "_direct"
LEGACY_LLPS_HIDDEN_KEY = LEGACY_LLPS_PREFIX + "_hidden"
LEGACY_LLPS_LOCAL_KEY = LEGACY_LLPS_PREFIX + "_local"
LEGACY_LLPS_CENTERED_HIDDEN_KEY = LEGACY_LLPS_PREFIX + "_centered_hidden"
LEGACY_LLPS_CONFIG_KEY = LEGACY_LLPS_PREFIX + "_config"
LEGACY_LLPS_CHECKPOINT_KEY = LEGACY_LLPS_PREFIX + "_checkpoint"
LEGACY_LLPS_STREAM_KEY = "use_" + LEGACY_LLPS_PREFIX + "_stream"
STREAM_MASK_KEYS = ("esm2", "biophys", PHASEFLOW_LLPS_DIRECT_STREAM, "phaseflow_bridge")


def normalize_dpr_stream_mask(raw: Any | None) -> dict[str, Any]:
    default_streams = {key: True for key in STREAM_MASK_KEYS}
    if raw is None:
        return {"active": False, "streams": default_streams}
    if not isinstance(raw, dict):
        raise TypeError(f"DPR stream mask must be a dict, got {type(raw).__name__}")
    streams_raw = raw.get("streams", raw)
    if not isinstance(streams_raw, dict):
        raise TypeError("DPR stream mask 'streams' must be a dict")
    streams_raw = _normalize_legacy_stream_names(dict(streams_raw))
    missing = [key for key in STREAM_MASK_KEYS if key not in streams_raw]
    if missing:
        raise KeyError(f"DPR stream mask is missing required keys: {missing}")
    extra = [key for key in streams_raw if key not in STREAM_MASK_KEYS]
    if extra:
        raise KeyError(f"DPR stream mask has unknown stream keys: {extra}")
    streams = {key: bool(streams_raw[key]) for key in STREAM_MASK_KEYS}
    out = {key: value for key, value in raw.items() if key != "streams"}
    out["active"] = True
    out["streams"] = streams
    return out


def _normalize_legacy_stream_names(streams: dict[str, Any]) -> dict[str, Any]:
    if LEGACY_LLPS_DIRECT_STREAM in streams and PHASEFLOW_LLPS_DIRECT_STREAM not in streams:
        streams[PHASEFLOW_LLPS_DIRECT_STREAM] = streams.pop(LEGACY_LLPS_DIRECT_STREAM)
    return streams


def _fit_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    dim = int(x.shape[-1])
    target = int(target_dim)
    if dim == target:
        return x
    if dim > target:
        return x[..., :target]
    return F.pad(x, (0, target - dim))


class PhaseFlow32GlobalBridge(nn.Module):
    """Ordered 32-token bridge from full-length LLPS residue states into frozen PhaseFlow."""

    def __init__(
        self,
        *,
        llps_dim: int | None = None,
        phaseflow_dim: int,
        tokens: int = 32,
        heads: int = 8,
        adapter_layers: int = 2,
        dropout: float = 0.10,
        gate_init: float = 0.075,
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()
        if llps_dim is None:
            llps_dim = legacy_kwargs.pop(LEGACY_LLPS_PREFIX + "_dim", None)
        if legacy_kwargs:
            raise TypeError(f"unexpected bridge keyword argument(s): {sorted(legacy_kwargs)}")
        if llps_dim is None:
            raise ValueError("llps_dim is required")
        self.tokens = int(tokens)
        self.phaseflow_dim = int(phaseflow_dim)
        self.pool_proj = _bridge_projection(int(llps_dim), int(phaseflow_dim), layers=max(1, int(adapter_layers)), dropout=dropout)
        self.content_score = nn.Linear(int(phaseflow_dim), 1)
        self.log_width = nn.Parameter(torch.tensor(math.log(math.exp(0.05) - 1.0)))
        self.token_bias = nn.Parameter(torch.zeros(self.tokens, int(phaseflow_dim)))
        self.pre_adapters = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=int(phaseflow_dim),
                    nhead=int(heads),
                    dim_feedforward=int(phaseflow_dim) * 4,
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(max(0, int(adapter_layers) - 1))
            ]
        )
        self.query_proj = _bridge_projection(int(llps_dim), int(phaseflow_dim), layers=2, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(
            int(phaseflow_dim),
            int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(int(phaseflow_dim))
        self.out_ffn = nn.Sequential(
            nn.Linear(int(phaseflow_dim), int(phaseflow_dim) * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(phaseflow_dim) * 4, int(phaseflow_dim)),
        )
        self.gate_raw = nn.Parameter(torch.tensor(_gate_probability_to_raw(float(gate_init)), dtype=torch.float32))

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_raw)

    def forward(self, phaseflow: nn.Module, llps_hidden: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        seq_mask = seq_mask.bool()
        pooled_tokens = self.monotonic_pool(llps_hidden.float(), seq_mask)
        for layer in self.pre_adapters:
            pooled_tokens = layer(pooled_tokens)
        attention_mask = torch.ones(
            pooled_tokens.shape[0],
            pooled_tokens.shape[1],
            dtype=torch.float32,
            device=pooled_tokens.device,
        )
        phase_t = torch.zeros(pooled_tokens.shape[0], int(phaseflow.phase_dim), dtype=torch.float32, device=pooled_tokens.device)
        phase_mask = torch.ones_like(phase_t)
        time = torch.zeros(pooled_tokens.shape[0], dtype=torch.float32, device=pooled_tokens.device)
        phaseflow_tokens, _ = _phaseflow_forward_token_embeddings(
            phaseflow,
            token_emb=pooled_tokens,
            attention_mask=attention_mask,
            phase_t=phase_t,
            phase_mask=phase_mask,
            time=time,
        )
        query = self.query_proj(llps_hidden.float())
        attended, _ = self.cross_attn(query, phaseflow_tokens, phaseflow_tokens, need_weights=False)
        context = self.out_norm(attended + self.out_ffn(attended))
        return (self.gate().to(context.dtype) * context).masked_fill(~seq_mask.unsqueeze(-1), 0.0)

    def monotonic_pool(self, llps_hidden: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        x = self.pool_proj(llps_hidden)
        bsz, length, _ = x.shape
        valid_lengths = seq_mask.float().sum(dim=1).clamp_min(1.0)
        positions = (torch.arange(length, device=x.device, dtype=x.dtype).view(1, 1, length) + 0.5) / valid_lengths.view(bsz, 1, 1)
        centers = (torch.arange(self.tokens, device=x.device, dtype=x.dtype).view(1, self.tokens, 1) + 0.5) / float(self.tokens)
        width = F.softplus(self.log_width).to(dtype=x.dtype).clamp_min(1.0 / (4.0 * float(self.tokens)))
        distance_logits = -0.5 * ((positions - centers) / width).pow(2)
        content_logits = 0.25 * self.content_score(x).squeeze(-1).unsqueeze(1)
        logits = distance_logits + content_logits
        logits = logits.masked_fill(~seq_mask.unsqueeze(1), -1.0e4)
        weights = torch.softmax(logits, dim=-1).masked_fill(~seq_mask.unsqueeze(1), 0.0)
        tokens = torch.matmul(weights, x)
        return tokens + self.token_bias.unsqueeze(0).to(dtype=tokens.dtype)


def _bridge_projection(input_dim: int, output_dim: int, *, layers: int, dropout: float) -> nn.Sequential:
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    if int(layers) <= 1:
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim))
    hidden_dim = max(input_dim, output_dim)
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(float(dropout)),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


def _gate_probability_to_raw(value: float) -> float:
    value = float(value)
    if 0.0 < value < 1.0:
        return math.log(value / (1.0 - value))
    return value


def phaseflow_full_sequence_forward(
    model: nn.Module,
    sequences: list[str],
    *,
    shape: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tokenizer = _phaseflow_tokenizer()
    safe = [_phaseflow_safe_sequence(seq) for seq in sequences]
    token_lists = [tokenizer.build_input_sequence(seq, shape=shape) for seq in safe]
    token_len = max(len(tokens) for tokens in token_lists)
    input_ids = torch.full((len(sequences), token_len), int(tokenizer.PAD_ID), dtype=torch.long, device=device)
    attention_mask = torch.zeros(len(sequences), token_len, dtype=torch.float32, device=device)
    for i, tokens in enumerate(token_lists):
        input_ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        attention_mask[i, : len(tokens)] = 1.0
    phase_t = torch.zeros(len(sequences), int(model.phase_dim), dtype=torch.float32, device=device)
    phase_mask = torch.ones_like(phase_t)
    time = torch.zeros(len(sequences), dtype=torch.float32, device=device)
    token_hidden, logit = _phaseflow_forward_full_hidden(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        phase_t=phase_t,
        phase_mask=phase_mask,
        time=time,
    )
    max_residue_len = max(len(seq) for seq in safe) if safe else 1
    residue_hidden = torch.zeros(len(sequences), max_residue_len, int(token_hidden.shape[-1]), dtype=token_hidden.dtype, device=device)
    seq_mask = torch.zeros(len(sequences), max_residue_len, dtype=torch.bool, device=device)
    for i, seq in enumerate(safe):
        length = len(seq)
        residue_hidden[i, :length] = token_hidden[i, 1 : 1 + length]
        seq_mask[i, :length] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "residue_hidden": residue_hidden.masked_fill(~seq_mask.unsqueeze(-1), 0.0),
        "seq_mask": seq_mask,
        "logit": logit,
    }


def _phaseflow_forward_token_embeddings(
    model: nn.Module,
    *,
    token_emb: torch.Tensor,
    attention_mask: torch.Tensor,
    phase_t: torch.Tensor,
    phase_mask: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    phase_emb, phase_attn_mask = model.embed_phase(phase_t, phase_mask, time)
    x = torch.cat([token_emb, phase_emb.to(token_emb.dtype)], dim=1)
    extended_mask = torch.cat([attention_mask.float(), phase_attn_mask.float()], dim=1)
    phase_start_idx = int(token_emb.shape[1])
    for layer in model.transformer.layers:
        x = layer(
            x,
            model.transformer.rotary,
            extended_mask,
            phase_start_idx,
            None,
            bool(model.use_set_encoder),
        )
    hidden = model.transformer.final_norm(x)
    n_phase = int(phase_emb.shape[1])
    if bool(model.use_set_encoder):
        phase_hidden = hidden[:, -n_phase:, :]
        logit = model.velocity_per_pos(phase_hidden).squeeze(-1)
    else:
        logit = model.velocity_head(hidden[:, -1, :])
    return hidden[:, : token_emb.shape[1], :], logit


def _phaseflow_forward_full_hidden(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    phase_t: torch.Tensor,
    phase_mask: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_emb = model.embed_tokens(input_ids)
    phase_emb, phase_attn_mask = model.embed_phase(phase_t, phase_mask, time)
    x = torch.cat([token_emb, phase_emb.to(token_emb.dtype)], dim=1)
    extended_mask = torch.cat([attention_mask.float(), phase_attn_mask.float()], dim=1)
    phase_start_idx = int(input_ids.shape[1])
    for layer in model.transformer.layers:
        x = layer(
            x,
            model.transformer.rotary,
            extended_mask,
            phase_start_idx,
            None,
            bool(model.use_set_encoder),
        )
    hidden = model.transformer.final_norm(x)
    n_phase = int(phase_emb.shape[1])
    if bool(model.use_set_encoder):
        phase_hidden = hidden[:, -n_phase:, :]
        logit = model.velocity_per_pos(phase_hidden).squeeze(-1)
    else:
        logit = model.velocity_head(hidden[:, -1, :])
    return hidden[:, : input_ids.shape[1], :], logit


@dataclass(frozen=True)
class DPRV6LossConfig:
    objective: str = "max"
    bag: float = 1.0
    topk: float = 0.25
    s_bce: float = 1.0
    s_dice: float = 0.30
    s_rank: float = 0.50
    w_bce: float = 0.25
    w_rank: float = 0.15
    m_peak: float = 0.20
    anchor_s: float = 0.10
    anchor_w: float = 0.05
    boundary_radius: int = 17
    topk_fraction: float = 0.05
    ms_p33: float = 0.15
    ms_p129: float = 0.25
    ms_p257: float = 0.60


class TinyPSTPMLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 20, hidden2: int = 5, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden1)),
            nn.LeakyReLU(negative_slope=float(negative_slope)),
            nn.Linear(int(hidden1), int(hidden2)),
            nn.Linear(int(hidden2), 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float()).squeeze(-1)


class V6ResidueAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.10) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(int(input_dim))
        self.in_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.ffn = nn.Sequential(nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(int(hidden_dim), int(hidden_dim)))
        self.out_norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, h: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(self.input_norm(h.float()))
        y = self.ffn(x)
        return self.out_norm(x + y).masked_fill(~seq_mask.bool().unsqueeze(-1), 0.0)


class V6BigScanner(nn.Module):
    def __init__(self, dim: int = 256, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(float(dropout)),
            nn.Linear(128, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float()).squeeze(-1)


class DPRV6Head(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        head_type: str = "tiny",
        window_sizes: tuple[int, int, int] = (33, 129, 257),
        topk_fraction: float = 0.05,
        adapter_dim: int = 256,
        dropout: float = 0.10,
        leaky_relu_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.head_type = str(head_type)
        self.window_sizes = tuple(int(x) for x in window_sizes)
        self.topk_fraction = float(topk_fraction)
        if self.head_type == "tiny":
            self.projection = nn.Identity()
            self.scanner_input_dim = self.input_dim
            self.shared_scanner = TinyPSTPMLP(self.scanner_input_dim, negative_slope=leaky_relu_slope)
        elif self.head_type == "big":
            self.projection = V6ResidueAdapter(self.input_dim, int(adapter_dim), dropout=float(dropout))
            self.scanner_input_dim = int(adapter_dim)
            self.shared_scanner = V6BigScanner(self.scanner_input_dim, dropout=float(dropout))
        else:
            raise ValueError(f"unknown DPR v6 head_type: {head_type}")

    def forward(self, h: torch.Tensor, seq_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_mask = seq_mask.bool()
        if self.head_type == "big":
            features = self.projection(h, seq_mask)
        else:
            features = h.float().masked_fill(~seq_mask.unsqueeze(-1), 0.0)
        logits: dict[int, torch.Tensor] = {}
        probs: dict[int, torch.Tensor] = {}
        for window in self.window_sizes:
            pooled = masked_avg_pool1d_same(features, seq_mask, kernel_size=window)
            z = self.shared_scanner(pooled).masked_fill(~seq_mask, -20.0)
            p = torch.sigmoid(z.float()).masked_fill(~seq_mask, 0.0)
            logits[window] = z
            probs[window] = p
        bag = bag_from_profiles([probs[w] for w in self.window_sizes], seq_mask, topk_fraction=self.topk_fraction)
        return {
            "z33": logits[self.window_sizes[0]],
            "z129": logits[self.window_sizes[1]],
            "z257": logits[self.window_sizes[2]],
            "p33": probs[self.window_sizes[0]],
            "p129": probs[self.window_sizes[1]],
            "p257": probs[self.window_sizes[2]],
            "dpr_residue_prob": probs[self.window_sizes[0]],
            "residue_logits": logits[self.window_sizes[0]],
            "residue_probabilities": probs[self.window_sizes[0]],
            "residue_prob": probs[self.window_sizes[0]],
            "bag_hard": bag["bag_hard"],
            "bag_topk": bag["bag_topk"],
            "hard_33": bag["hard_0"],
            "hard_129": bag["hard_1"],
            "hard_257": bag["hard_2"],
            "topk_33": bag["topk_0"],
            "topk_129": bag["topk_1"],
            "topk_257": bag["topk_2"],
            "head_features": features,
            "seq_mask": seq_mask,
        }


class DPRV6PhaseStack(nn.Module):
    def __init__(self, *, llps_backbone: nn.Module | None, phaseflow: nn.Module | None, config: dict[str, Any]) -> None:
        super().__init__()
        self.llps_backbone = llps_backbone if llps_backbone is not None else nn.Identity()
        self.phaseflow = phaseflow if phaseflow is not None else nn.Identity()
        self.config = default_dpr_v6_config() | dict(config or {})
        model_cfg = dict(self.config.get("model", self.config))
        self.esm2_dim = int(model_cfg.get("esm2_dim", 1280))
        self.biophys_dim = int(model_cfg.get("biophys_dim", 112))
        self.phaseflow_llps_hidden_dim = int(model_cfg.get(PHASEFLOW_LLPS_HIDDEN_DIM_KEY, model_cfg.get(LEGACY_LLPS_HIDDEN_DIM_KEY, 256)))
        self.phaseflow_hidden_dim = int(model_cfg.get("phaseflow_hidden_dim", 256))
        self.pstp_esm8_dim = int(model_cfg.get("pstp_esm8_dim", 320))
        self.pstp_alb_dim = int(model_cfg.get("pstp_alb_dim", 330))
        self.pstp_650_dim = int(model_cfg.get("pstp_650_dim", 650))
        self.phaseflow_shape = str(model_cfg.get("phaseflow_shape", "4x4"))
        self.use_base_streams = bool(model_cfg.get("use_base_streams", True))
        self.use_phaseflow_llps_stream = bool(model_cfg.get(PHASEFLOW_LLPS_STREAM_KEY, model_cfg.get(LEGACY_LLPS_STREAM_KEY, True)))
        self.use_phaseflow_stream = bool(model_cfg.get("use_phaseflow_stream", True))
        self.use_pstp650 = bool(model_cfg.get("use_pstp650", False))
        self.use_pstp_esm8 = bool(model_cfg.get("use_pstp_esm8", False))
        self.use_pstp_alb = bool(model_cfg.get("use_pstp_alb", False))
        self.pstp_alb_project_dim = int(model_cfg.get("pstp_alb_project_dim", 0) or 0)
        self.stream_mask = normalize_dpr_stream_mask(model_cfg.get("ablation_mask", model_cfg.get("stream_mask")))
        self.v6_feature_projectors = nn.ModuleDict()
        if self.use_pstp_alb and self.pstp_alb_project_dim > 0:
            self.v6_feature_projectors["pstp_alb"] = nn.Sequential(
                nn.LayerNorm(self.pstp_alb_dim),
                nn.Linear(self.pstp_alb_dim, self.pstp_alb_project_dim),
                nn.GELU(),
            )
        if self.use_phaseflow_stream:
            if llps_backbone is None or phaseflow is None:
                raise ValueError("PhaseFlow stream requires frozen LLPS and peptide PhaseFlow modules for the bridge")
            bridge_kwargs = {
                "llps_dim": self.phaseflow_llps_hidden_dim,
                "phaseflow_dim": self.phaseflow_hidden_dim,
                "tokens": int(model_cfg.get("phaseflow_bridge_tokens", 32)),
                "heads": int(model_cfg.get("num_heads", 8)),
                "adapter_layers": int(model_cfg.get("phaseflow_bridge_adapter_layers", 2)),
                "dropout": float(model_cfg.get("dropout", 0.10)),
                "gate_init": float(model_cfg.get("phaseflow_bridge_gate_init", 0.075)),
            }
            self.phaseflow_bridge = PhaseFlow32GlobalBridge(
                **bridge_kwargs,
            )
        else:
            self.phaseflow_bridge = nn.Identity()
        input_dim = self.compute_input_dim()
        self.v6 = DPRV6Head(
            input_dim,
            head_type=str(model_cfg.get("head_type", "tiny")),
            window_sizes=tuple(int(x) for x in model_cfg.get("window_sizes", (33, 129, 257))),
            topk_fraction=float(model_cfg.get("topk_fraction", 0.05)),
            adapter_dim=int(model_cfg.get("adapter_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.10)),
            leaky_relu_slope=float(model_cfg.get("leaky_relu_slope", 0.01)),
        )
        self._freeze_original_models()

    def compute_input_dim(self) -> int:
        dim = 0
        if self.use_base_streams:
            dim += self.esm2_dim + self.biophys_dim
        if self.use_phaseflow_llps_stream:
            dim += self.phaseflow_llps_hidden_dim
        if self.use_phaseflow_stream:
            dim += self.phaseflow_hidden_dim
        if self.use_pstp650:
            dim += self.pstp_650_dim
        else:
            if self.use_pstp_esm8:
                dim += self.pstp_esm8_dim
            if self.use_pstp_alb:
                dim += self.pstp_alb_project_dim if self.pstp_alb_project_dim > 0 else self.pstp_alb_dim
        if dim <= 0:
            raise ValueError("DPR v6 input dimension is zero")
        return dim

    def stream_enabled(self, name: str) -> bool:
        if name not in STREAM_MASK_KEYS:
            raise KeyError(f"unknown DPR stream mask key: {name}")
        return bool(self.stream_mask["streams"][name])

    def zero_stream(self, seq_mask: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.zeros((*seq_mask.shape, int(dim)), dtype=torch.float32, device=seq_mask.device)

    def stream_slice_offsets(self) -> dict[str, dict[str, int]]:
        offsets: dict[str, dict[str, int]] = {}
        start = 0
        if self.use_base_streams:
            offsets["esm2"] = {"start": start, "end": start + self.esm2_dim, "dim": self.esm2_dim}
            start += self.esm2_dim
            offsets["biophys"] = {"start": start, "end": start + self.biophys_dim, "dim": self.biophys_dim}
            start += self.biophys_dim
        if self.use_phaseflow_llps_stream:
            offsets[PHASEFLOW_LLPS_DIRECT_STREAM] = {
                "start": start,
                "end": start + self.phaseflow_llps_hidden_dim,
                "dim": self.phaseflow_llps_hidden_dim,
            }
            start += self.phaseflow_llps_hidden_dim
        if self.use_phaseflow_stream:
            offsets["phaseflow_bridge"] = {"start": start, "end": start + self.phaseflow_hidden_dim, "dim": self.phaseflow_hidden_dim}
            start += self.phaseflow_hidden_dim
        if self.use_pstp650:
            offsets["pstp_650d"] = {"start": start, "end": start + self.pstp_650_dim, "dim": self.pstp_650_dim}
            start += self.pstp_650_dim
        else:
            if self.use_pstp_esm8:
                offsets["pstp_esm8"] = {"start": start, "end": start + self.pstp_esm8_dim, "dim": self.pstp_esm8_dim}
                start += self.pstp_esm8_dim
            if self.use_pstp_alb:
                dim = self.pstp_alb_project_dim if self.pstp_alb_project_dim > 0 else self.pstp_alb_dim
                offsets["pstp_alb"] = {"start": start, "end": start + dim, "dim": dim}
                start += dim
        return offsets

    def stream_mask_summary(self) -> dict[str, Any]:
        return {
            "active": bool(self.stream_mask.get("active", False)),
            "streams": dict(self.stream_mask["streams"]),
            "arm_id": self.stream_mask.get("arm_id", self.stream_mask.get("id", "")),
            "bitmask": self.stream_mask.get("bitmask", ""),
            "input_dim": int(self.v6.input_dim),
            "slices": self.stream_slice_offsets(),
        }

    def _freeze_original_models(self) -> None:
        self.llps_backbone.requires_grad_(False)
        self.phaseflow.requires_grad_(False)
        self.phaseflow_bridge.requires_grad_(False)
        self.llps_backbone.eval()
        self.phaseflow.eval()
        self.phaseflow_bridge.eval()
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("v6.") or name.startswith("v6_feature_projectors."))

    def train(self, mode: bool = True) -> "DPRV6PhaseStack":
        super().train(mode)
        self.llps_backbone.eval()
        self.phaseflow.eval()
        self.phaseflow_bridge.eval()
        return self

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def dpr_only_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.v6.state_dict().items()}

    def load_dpr_only_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> None:
        self.v6.load_state_dict(state, strict=strict)

    def phaseflow_llps_residue_hidden(self, batch: dict[str, Any]) -> torch.Tensor:
        cached = batch.get(
            PHASEFLOW_LLPS_HIDDEN_KEY,
            batch.get(LEGACY_LLPS_HIDDEN_KEY, batch.get(LEGACY_LLPS_LOCAL_KEY, batch.get(LEGACY_LLPS_CENTERED_HIDDEN_KEY))),
        )
        seq_mask = batch["seq_mask"].bool()
        if torch.is_tensor(cached):
            return _fit_last_dim(cached.float(), self.phaseflow_llps_hidden_dim).masked_fill(~seq_mask.unsqueeze(-1), 0.0)
        with torch.no_grad():
            out = self.llps_backbone(batch)
        names = ("llps_residue_repr", "dpr_residue_repr")
        layers = [out[name].detach().float() for name in names if name in out]
        if not layers:
            raise KeyError(f"PhaseFlow outputs did not include any of {names}")
        hidden = _fit_last_dim(torch.stack(layers, dim=0).mean(dim=0), self.phaseflow_llps_hidden_dim)
        return hidden.masked_fill(~seq_mask.unsqueeze(-1), 0.0)

    def extract_residue_aligned_h(self, batch: dict[str, Any]) -> torch.Tensor:
        seq_mask = batch["seq_mask"].bool()
        streams: list[torch.Tensor] = []
        if self.use_base_streams:
            if self.stream_enabled("esm2"):
                streams.append(_fit_last_dim(batch["plm"].float(), self.esm2_dim).masked_fill(~seq_mask.unsqueeze(-1), 0.0))
            else:
                streams.append(self.zero_stream(seq_mask, self.esm2_dim))
            if self.stream_enabled("biophys"):
                streams.append(_fit_last_dim(batch["biophys"].float(), self.biophys_dim).masked_fill(~seq_mask.unsqueeze(-1), 0.0))
            else:
                streams.append(self.zero_stream(seq_mask, self.biophys_dim))
        phaseflow_llps_hidden: torch.Tensor | None = None
        needs_phaseflow_llps_hidden = (self.use_phaseflow_llps_stream and self.stream_enabled(PHASEFLOW_LLPS_DIRECT_STREAM)) or (
            self.use_phaseflow_stream and self.stream_enabled("phaseflow_bridge")
        )
        if needs_phaseflow_llps_hidden:
            phaseflow_llps_hidden = self.phaseflow_llps_residue_hidden(batch)
        if self.use_phaseflow_llps_stream:
            if self.stream_enabled(PHASEFLOW_LLPS_DIRECT_STREAM):
                assert phaseflow_llps_hidden is not None
                streams.append(phaseflow_llps_hidden)
            else:
                streams.append(self.zero_stream(seq_mask, self.phaseflow_llps_hidden_dim))
        if self.use_phaseflow_stream:
            if self.stream_enabled("phaseflow_bridge"):
                if phaseflow_llps_hidden is None:
                    phaseflow_llps_hidden = self.phaseflow_llps_residue_hidden(batch)
                with torch.no_grad():
                    phaseflow = self.phaseflow_bridge(self.phaseflow, phaseflow_llps_hidden, seq_mask)
                streams.append(phaseflow.float().masked_fill(~seq_mask.unsqueeze(-1), 0.0))
            else:
                streams.append(self.zero_stream(seq_mask, self.phaseflow_hidden_dim))
        if self.use_pstp650:
            streams.append(_fit_last_dim(batch["pstp_650d"].float(), self.pstp_650_dim).masked_fill(~seq_mask.unsqueeze(-1), 0.0))
        else:
            if self.use_pstp_esm8:
                streams.append(_fit_last_dim(batch["pstp_esm8"].float(), self.pstp_esm8_dim).masked_fill(~seq_mask.unsqueeze(-1), 0.0))
            if self.use_pstp_alb:
                alb = _fit_last_dim(batch["pstp_alb"].float(), self.pstp_alb_dim).masked_fill(~seq_mask.unsqueeze(-1), 0.0)
                if "pstp_alb" in self.v6_feature_projectors:
                    alb = self.v6_feature_projectors["pstp_alb"](alb.float()).masked_fill(~seq_mask.unsqueeze(-1), 0.0)
                streams.append(alb)
        h = torch.cat(streams, dim=-1)
        return h.masked_fill(~seq_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        sequence: str | list[str] | None = None,
        *,
        task: str = "dpr",
        batch: dict[str, Any] | None = None,
        phaseflow_batch: dict[str, torch.Tensor] | None = None,
        expert_mode: str = "full",
        return_regions: bool = False,
    ) -> dict[str, Any]:
        del expert_mode, return_regions
        if task == "llps":
            return {"llps": self.forward_llps(batch=batch, sequence=sequence)}
        if task == "phaseflow":
            return {"phaseflow": self.forward_phaseflow(phaseflow_batch=phaseflow_batch, sequence=sequence)}
        if task == "dpr":
            if batch is None:
                raise ValueError("DPR v6 requires offline packed features")
            h = self.extract_residue_aligned_h(batch)
            out = self.v6(h, batch["seq_mask"].bool())
            out["residue_aligned_h"] = h.detach()
            return {"dpr": out}
        if task == "all":
            if batch is None:
                raise ValueError("task='all' requires an offline batch")
            return {
                "llps": self.forward_llps(batch=batch, sequence=sequence),
                "phaseflow": self.forward_phaseflow(phaseflow_batch=phaseflow_batch, sequence=sequence),
                "dpr": self.forward(task="dpr", batch=batch)["dpr"],
            }
        raise ValueError(f"unknown task: {task}")

    def forward_llps(self, *, batch: dict[str, Any] | None, sequence: str | list[str] | None) -> dict[str, torch.Tensor]:
        del sequence
        if batch is None:
            raise ValueError("DPR v6 LLPS preservation uses offline batches")
        with torch.no_grad():
            out = self.llps_backbone(batch)
        logit = out.get("llps_logits", out.get("llps_logits"))
        if logit is None:
            raise KeyError("PhaseFlow output does not include LLPS logits")
        return {"logit": logit, "probability": torch.sigmoid(logit.float()), **out}

    def forward_phaseflow(
        self,
        *,
        phaseflow_batch: dict[str, torch.Tensor] | None,
        sequence: str | list[str] | None,
    ) -> dict[str, torch.Tensor]:
        if phaseflow_batch is not None:
            with torch.no_grad():
                velocity = self.phaseflow.forward_flow(
                    input_ids=phaseflow_batch["input_ids"],
                    attention_mask=phaseflow_batch["attention_mask"],
                    phase_t=phaseflow_batch["phase_t"],
                    phase_mask=phaseflow_batch["phase_mask"],
                    time=phaseflow_batch["time"],
                    seq_len=phaseflow_batch["seq_len"],
                )
            return {"logit": velocity, "velocity": velocity, "probability": torch.sigmoid(velocity.float())}
        if sequence is None:
            raise ValueError("PhaseFlow preservation requires phaseflow_batch or explicit sequence")
        seqs = [sequence] if isinstance(sequence, str) else [str(x) for x in sequence]
        with torch.no_grad():
            out = phaseflow_full_sequence_forward(
                self.phaseflow,
                seqs,
                shape=self.phaseflow_shape,
                device=next(self.parameters()).device,
            )
        return {"logit": out["logit"], "velocity": out["logit"], "probability": torch.sigmoid(out["logit"].float())}


def masked_avg_pool1d_same(x: torch.Tensor, seq_mask: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    kernel = int(kernel_size)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError(f"DPR v6 requires positive odd window sizes, got {kernel_size}")
    seq_mask = seq_mask.bool()
    x = x.float().masked_fill(~seq_mask.unsqueeze(-1), 0.0)
    values = x.transpose(1, 2)
    weight = torch.ones(values.shape[1], 1, kernel, dtype=values.dtype, device=values.device)
    sums = F.conv1d(values, weight, padding=kernel // 2, groups=values.shape[1])
    counts = F.conv1d(
        seq_mask.float().unsqueeze(1),
        torch.ones(1, 1, kernel, dtype=values.dtype, device=values.device),
        padding=kernel // 2,
    )
    return (sums / counts.clamp_min(1.0)).transpose(1, 2).masked_fill(~seq_mask.unsqueeze(-1), 0.0)


def masked_max(values: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
    return values.float().masked_fill(~seq_mask.bool(), -1.0).max(dim=1).values.clamp_min(0.0)


def masked_top_fraction_mean(values: torch.Tensor, seq_mask: torch.Tensor, *, fraction: float) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for row, mask in zip(values.float(), seq_mask.bool(), strict=False):
        valid = row[mask]
        if valid.numel() == 0:
            rows.append(row.sum() * 0.0)
            continue
        k = max(1, int(math.ceil(float(fraction) * int(valid.numel()))))
        rows.append(torch.topk(valid, k=min(k, int(valid.numel()))).values.mean())
    return torch.stack(rows, dim=0)


def bag_from_profiles(
    profiles: list[torch.Tensor] | tuple[torch.Tensor, ...],
    seq_mask: torch.Tensor,
    *,
    topk_fraction: float = 0.05,
) -> dict[str, torch.Tensor]:
    hard_rows = [masked_max(profile.float(), seq_mask) for profile in profiles]
    topk_rows = [masked_top_fraction_mean(profile.float(), seq_mask, fraction=topk_fraction) for profile in profiles]
    hards = torch.stack(hard_rows, dim=0)
    topks = torch.stack(topk_rows, dim=0)
    out: dict[str, torch.Tensor] = {"bag_hard": hards.mean(dim=0), "bag_topk": topks.mean(dim=0)}
    for i, value in enumerate(hard_rows):
        out[f"hard_{i}"] = value
    for i, value in enumerate(topk_rows):
        out[f"topk_{i}"] = value
    return out


def weighted_bce_prob(prob: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    prob = prob.float().clamp(1.0e-6, 1.0 - 1.0e-6)
    loss = -(target.float() * torch.log(prob) + (1.0 - target.float()) * torch.log1p(-prob))
    w = weight.float().clamp_min(0.0)
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def dpr_v6_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    cfg: DPRV6LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    device = out["p33"].device
    tiers = [str(x) for x in batch["v3_tiers"]]
    tier_mask = {tier: torch.tensor([x == tier for x in tiers], dtype=torch.bool, device=device) for tier in ("S", "W", "M", "ND", "NP")}
    labels = torch.tensor([TIER_TO_BAG_LABEL[x] for x in tiers], dtype=torch.float32, device=device)
    weights = torch.tensor([TIER_TO_WEIGHT[x] for x in tiers], dtype=torch.float32, device=device)
    bag_hard = weighted_bce_prob(out["bag_hard"], labels, weights)
    bag_topk = weighted_bce_prob(out["bag_topk"], labels, weights)
    terms: dict[str, torch.Tensor] = {"L_bag_hard": bag_hard, "L_bag_topk": bag_topk, "L_bag": bag_hard}
    counts: dict[str, int] = {"active_bag": int(labels.numel())}
    weighted_terms: list[torch.Tensor] = [cfg.bag * bag_hard]
    objective = str(cfg.objective)
    if objective in {"topk", "v5loss", "v5loss_ms", "v5loss_p257", "rank_p257"}:
        weighted_terms.append(cfg.topk * bag_topk)
        terms["L_bag"] = bag_hard + cfg.topk * bag_topk
    if objective in {"strong", "v5loss"}:
        add_supervised_terms(out, batch, tier_mask, cfg, terms, counts, weighted_terms)
    elif objective in {"strong_ms", "v5loss_ms"}:
        add_multiscale_supervised_terms(out, batch, tier_mask, cfg, terms, counts, weighted_terms)
    elif objective in {"strong_p257", "v5loss_p257"}:
        add_supervised_terms_for_scale(
            out,
            batch,
            tier_mask,
            cfg,
            terms,
            counts,
            weighted_terms,
            prefix="p257",
            logits_key="z257",
            prob_key="p257",
            scale_weight=1.0,
        )
    elif objective == "rank_p257":
        add_rank_focus_terms_for_scale(
            out,
            batch,
            tier_mask,
            cfg,
            terms,
            counts,
            weighted_terms,
            prefix="p257",
            logits_key="z257",
            prob_key="p257",
        )
    if objective == "mflat" or objective == "v5loss":
        m_peak, m_count = m_peak_loss(out["p33"], batch["seq_mask"].bool().to(device), tier_mask["M"])
        if m_count > 0:
            terms["L_M_peak"] = m_peak
            counts["active_M_peak"] = m_count
            weighted_terms.append(cfg.m_peak * m_peak)
    elif objective == "v5loss_p257":
        m_peak, m_count = m_peak_loss(out["p257"], batch["seq_mask"].bool().to(device), tier_mask["M"])
        if m_count > 0:
            terms["L_p257_M_peak"] = m_peak
            counts["active_p257_M_peak"] = m_count
            weighted_terms.append(cfg.m_peak * m_peak)
    elif objective == "rank_p257":
        nd_suppress, nd_count = negative_top_suppression(out["p257"], batch["seq_mask"].bool().to(device), tier_mask["ND"] | tier_mask["NP"])
        if nd_count > 0:
            terms["L_p257_N_top_suppress"] = nd_suppress
            counts["active_p257_N_top_suppress"] = nd_count
            weighted_terms.append(0.35 * nd_suppress)
    elif objective == "v5loss_ms":
        for prefix, prob_key, scale_weight in [
            ("p33", "p33", float(cfg.ms_p33)),
            ("p129", "p129", float(cfg.ms_p129)),
            ("p257", "p257", float(cfg.ms_p257)),
        ]:
            if scale_weight <= 0.0:
                continue
            m_peak, m_count = m_peak_loss(out[prob_key], batch["seq_mask"].bool().to(device), tier_mask["M"])
            if m_count > 0:
                terms[f"L_{prefix}_M_peak"] = m_peak
                counts[f"active_{prefix}_M_peak"] = m_count
                weighted_terms.append(scale_weight * cfg.m_peak * m_peak)
    if objective == "anchor":
        s_anchor, s_count = positive_anchor_loss(out["z33"], batch, tier_mask["S"])
        w_anchor, w_count = positive_anchor_loss(out["z33"], batch, tier_mask["W"])
        if s_count > 0:
            terms["L_S_anchor"] = s_anchor
            counts["active_S_anchor"] = s_count
            weighted_terms.append(cfg.anchor_s * s_anchor)
        if w_count > 0:
            terms["L_W_anchor"] = w_anchor
            counts["active_W_anchor"] = w_count
            weighted_terms.append(cfg.anchor_w * w_anchor)
    zero = out["p33"].sum() * 0.0
    total = torch.stack([term if torch.is_tensor(term) else zero + float(term) for term in weighted_terms]).sum()
    terms["L_total"] = total
    for name in ("L_S_bce", "L_S_dice", "L_S_rank", "L_W_bce", "L_W_rank", "L_M_peak", "L_S_anchor", "L_W_anchor"):
        terms.setdefault(name, zero)
    for name in ("active_S_bce", "active_S_dice", "active_S_rank", "active_W_bce", "active_W_rank", "active_M_peak", "active_S_anchor", "active_W_anchor"):
        counts.setdefault(name, 0)
    return total, {**terms, **counts}


def add_supervised_terms(
    out: dict[str, torch.Tensor],
    batch: dict[str, Any],
    tier_mask: dict[str, torch.Tensor],
    cfg: DPRV6LossConfig,
    terms: dict[str, torch.Tensor],
    counts: dict[str, int],
    weighted_terms: list[torch.Tensor],
) -> None:
    s_bce, s_count = supervised_bce(out["z33"], batch, tier_mask["S"], radius=cfg.boundary_radius)
    if s_count > 0:
        terms["L_S_bce"] = s_bce
        counts["active_S_bce"] = s_count
        weighted_terms.append(cfg.s_bce * s_bce)
    s_dice, s_dice_count = supervised_dice(out["p33"], batch, tier_mask["S"])
    if s_dice_count > 0:
        terms["L_S_dice"] = s_dice
        counts["active_S_dice"] = s_dice_count
        weighted_terms.append(cfg.s_dice * s_dice)
    s_rank, s_rank_count = supervised_rank(out["p33"], batch, tier_mask["S"], margin=0.30, radius=cfg.boundary_radius)
    if s_rank_count > 0:
        terms["L_S_rank"] = s_rank
        counts["active_S_rank"] = s_rank_count
        weighted_terms.append(cfg.s_rank * s_rank)
    w_bce, w_count = supervised_bce(out["z33"], batch, tier_mask["W"], radius=cfg.boundary_radius)
    if w_count > 0:
        terms["L_W_bce"] = w_bce
        counts["active_W_bce"] = w_count
        weighted_terms.append(cfg.w_bce * w_bce)
    w_rank, w_rank_count = supervised_rank(out["p33"], batch, tier_mask["W"], margin=0.15, radius=cfg.boundary_radius)
    if w_rank_count > 0:
        terms["L_W_rank"] = w_rank
        counts["active_W_rank"] = w_rank_count
        weighted_terms.append(cfg.w_rank * w_rank)


def add_multiscale_supervised_terms(
    out: dict[str, torch.Tensor],
    batch: dict[str, Any],
    tier_mask: dict[str, torch.Tensor],
    cfg: DPRV6LossConfig,
    terms: dict[str, torch.Tensor],
    counts: dict[str, int],
    weighted_terms: list[torch.Tensor],
) -> None:
    for prefix, logits_key, prob_key, scale_weight in [
        ("p33", "z33", "p33", float(cfg.ms_p33)),
        ("p129", "z129", "p129", float(cfg.ms_p129)),
        ("p257", "z257", "p257", float(cfg.ms_p257)),
    ]:
        if scale_weight <= 0.0:
            continue
        add_supervised_terms_for_scale(
            out,
            batch,
            tier_mask,
            cfg,
            terms,
            counts,
            weighted_terms,
            prefix=prefix,
            logits_key=logits_key,
            prob_key=prob_key,
            scale_weight=scale_weight,
        )


def add_supervised_terms_for_scale(
    out: dict[str, torch.Tensor],
    batch: dict[str, Any],
    tier_mask: dict[str, torch.Tensor],
    cfg: DPRV6LossConfig,
    terms: dict[str, torch.Tensor],
    counts: dict[str, int],
    weighted_terms: list[torch.Tensor],
    *,
    prefix: str,
    logits_key: str,
    prob_key: str,
    scale_weight: float,
) -> None:
    scale = float(scale_weight)
    s_bce, s_count = supervised_bce(out[logits_key], batch, tier_mask["S"], radius=cfg.boundary_radius)
    if s_count > 0:
        terms[f"L_{prefix}_S_bce"] = s_bce
        counts[f"active_{prefix}_S_bce"] = s_count
        weighted_terms.append(scale * cfg.s_bce * s_bce)
    s_dice, s_dice_count = supervised_dice(out[prob_key], batch, tier_mask["S"])
    if s_dice_count > 0:
        terms[f"L_{prefix}_S_dice"] = s_dice
        counts[f"active_{prefix}_S_dice"] = s_dice_count
        weighted_terms.append(scale * cfg.s_dice * s_dice)
    s_rank, s_rank_count = supervised_rank(out[prob_key], batch, tier_mask["S"], margin=0.30, radius=cfg.boundary_radius)
    if s_rank_count > 0:
        terms[f"L_{prefix}_S_rank"] = s_rank
        counts[f"active_{prefix}_S_rank"] = s_rank_count
        weighted_terms.append(scale * cfg.s_rank * s_rank)
    w_bce, w_count = supervised_bce(out[logits_key], batch, tier_mask["W"], radius=cfg.boundary_radius)
    if w_count > 0:
        terms[f"L_{prefix}_W_bce"] = w_bce
        counts[f"active_{prefix}_W_bce"] = w_count
        weighted_terms.append(scale * cfg.w_bce * w_bce)
    w_rank, w_rank_count = supervised_rank(out[prob_key], batch, tier_mask["W"], margin=0.15, radius=cfg.boundary_radius)
    if w_rank_count > 0:
        terms[f"L_{prefix}_W_rank"] = w_rank
        counts[f"active_{prefix}_W_rank"] = w_rank_count
        weighted_terms.append(scale * cfg.w_rank * w_rank)


def add_rank_focus_terms_for_scale(
    out: dict[str, torch.Tensor],
    batch: dict[str, Any],
    tier_mask: dict[str, torch.Tensor],
    cfg: DPRV6LossConfig,
    terms: dict[str, torch.Tensor],
    counts: dict[str, int],
    weighted_terms: list[torch.Tensor],
    *,
    prefix: str,
    logits_key: str,
    prob_key: str,
) -> None:
    s_bce, s_count = supervised_bce(out[logits_key], batch, tier_mask["S"], radius=cfg.boundary_radius)
    if s_count > 0:
        terms[f"L_{prefix}_S_bce"] = s_bce
        counts[f"active_{prefix}_S_bce"] = s_count
        weighted_terms.append(0.35 * cfg.s_bce * s_bce)
    s_dice, s_dice_count = supervised_dice(out[prob_key], batch, tier_mask["S"])
    if s_dice_count > 0:
        terms[f"L_{prefix}_S_dice"] = s_dice
        counts[f"active_{prefix}_S_dice"] = s_dice_count
        weighted_terms.append(0.20 * cfg.s_dice * s_dice)
    s_pair, s_pair_count = supervised_pairwise_rank(out[prob_key], batch, tier_mask["S"], margin=0.18, radius=cfg.boundary_radius)
    if s_pair_count > 0:
        terms[f"L_{prefix}_S_pairrank"] = s_pair
        counts[f"active_{prefix}_S_pairrank"] = s_pair_count
        weighted_terms.append(1.50 * cfg.s_rank * s_pair)
    w_pair, w_pair_count = supervised_pairwise_rank(out[prob_key], batch, tier_mask["W"], margin=0.10, radius=cfg.boundary_radius)
    if w_pair_count > 0:
        terms[f"L_{prefix}_W_pairrank"] = w_pair
        counts[f"active_{prefix}_W_pairrank"] = w_pair_count
        weighted_terms.append(1.25 * cfg.w_rank * w_pair)


def supervised_bce(logits: torch.Tensor, batch: dict[str, Any], sample_mask: torch.Tensor, *, radius: int) -> tuple[torch.Tensor, int]:
    target = batch["residue_target"].float().to(logits.device)
    seq_mask = batch["seq_mask"].bool().to(logits.device)
    losses: list[torch.Tensor] = []
    active = 0
    for i in sample_mask.nonzero(as_tuple=False).flatten().tolist():
        pos = (target[i] > 0.5) & seq_mask[i]
        if not bool(pos.any()):
            continue
        bg = safe_background_from_positive(pos, seq_mask[i], radius=radius)
        if not bool(bg.any()):
            continue
        pos_loss = F.binary_cross_entropy_with_logits(logits[i][pos].float(), torch.ones_like(logits[i][pos].float()))
        bg_loss = F.binary_cross_entropy_with_logits(logits[i][bg].float(), torch.zeros_like(logits[i][bg].float()))
        losses.append(0.5 * pos_loss + 0.5 * bg_loss)
        active += 1
    if not losses:
        return logits.sum() * 0.0, 0
    return torch.stack(losses).mean(), active


def supervised_dice(prob: torch.Tensor, batch: dict[str, Any], sample_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    target = batch["residue_target"].float().to(prob.device)
    seq_mask = batch["seq_mask"].bool().to(prob.device)
    losses: list[torch.Tensor] = []
    active = 0
    for i in sample_mask.nonzero(as_tuple=False).flatten().tolist():
        pos = (target[i] > 0.5) & seq_mask[i]
        if not bool(pos.any()):
            continue
        p = prob[i][seq_mask[i]].float()
        y = pos[seq_mask[i]].float()
        losses.append(1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0))
        active += 1
    if not losses:
        return prob.sum() * 0.0, 0
    return torch.stack(losses).mean(), active


def supervised_rank(prob: torch.Tensor, batch: dict[str, Any], sample_mask: torch.Tensor, *, margin: float, radius: int) -> tuple[torch.Tensor, int]:
    target = batch["residue_target"].float().to(prob.device)
    seq_mask = batch["seq_mask"].bool().to(prob.device)
    losses: list[torch.Tensor] = []
    active = 0
    for i in sample_mask.nonzero(as_tuple=False).flatten().tolist():
        pos = (target[i] > 0.5) & seq_mask[i]
        if not bool(pos.any()):
            continue
        bg = safe_background_from_positive(pos, seq_mask[i], radius=radius)
        if not bool(bg.any()):
            continue
        bg_vals = prob[i][bg].float()
        k = max(1, int(math.ceil(0.20 * int(bg_vals.numel()))))
        hard_bg = torch.topk(bg_vals, k=min(k, int(bg_vals.numel()))).values.mean()
        losses.append(F.relu(prob.new_tensor(float(margin)) - (prob[i][pos].float().mean() - hard_bg)))
        active += 1
    if not losses:
        return prob.sum() * 0.0, 0
    return torch.stack(losses).mean(), active


def supervised_pairwise_rank(prob: torch.Tensor, batch: dict[str, Any], sample_mask: torch.Tensor, *, margin: float, radius: int) -> tuple[torch.Tensor, int]:
    target = batch["residue_target"].float().to(prob.device)
    seq_mask = batch["seq_mask"].bool().to(prob.device)
    losses: list[torch.Tensor] = []
    active = 0
    for i in sample_mask.nonzero(as_tuple=False).flatten().tolist():
        pos = (target[i] > 0.5) & seq_mask[i]
        if not bool(pos.any()):
            continue
        bg = safe_background_from_positive(pos, seq_mask[i], radius=radius)
        if not bool(bg.any()):
            continue
        pos_vals = limit_values(prob[i][pos].float(), max_items=256)
        bg_vals = prob[i][bg].float()
        k = max(1, int(math.ceil(0.25 * int(bg_vals.numel()))))
        hard_bg = torch.topk(bg_vals, k=min(k, int(bg_vals.numel()), 256), largest=True).values
        losses.append(F.softplus(prob.new_tensor(float(margin)) - (pos_vals[:, None] - hard_bg[None, :])).mean())
        active += 1
    if not losses:
        return prob.sum() * 0.0, 0
    return torch.stack(losses).mean(), active


def negative_top_suppression(prob: torch.Tensor, seq_mask: torch.Tensor, sample_mask: torch.Tensor, *, fraction: float = 0.08) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    active = 0
    for i in sample_mask.nonzero(as_tuple=False).flatten().tolist():
        vals = prob[i][seq_mask[i]].float()
        if vals.numel() == 0:
            continue
        k = max(1, int(math.ceil(float(fraction) * int(vals.numel()))))
        top_vals = torch.topk(vals, k=min(k, int(vals.numel()), 256), largest=True).values
        losses.append(top_vals.mean())
        active += 1
    if not losses:
        return prob.sum() * 0.0, 0
    return torch.stack(losses).mean(), active


def limit_values(values: torch.Tensor, *, max_items: int) -> torch.Tensor:
    if values.numel() <= int(max_items):
        return values
    idx = torch.linspace(0, int(values.numel()) - 1, steps=int(max_items), device=values.device).long()
    return values[idx]


def positive_anchor_loss(logits: torch.Tensor, batch: dict[str, Any], sample_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    target = batch["residue_target"].float().to(logits.device)
    seq_mask = batch["seq_mask"].bool().to(logits.device)
    losses: list[torch.Tensor] = []
    active = 0
    for i in sample_mask.nonzero(as_tuple=False).flatten().tolist():
        pos = (target[i] > 0.5) & seq_mask[i]
        if not bool(pos.any()):
            continue
        losses.append(F.binary_cross_entropy_with_logits(logits[i][pos].float(), torch.ones_like(logits[i][pos].float())))
        active += 1
    if not losses:
        return logits.sum() * 0.0, 0
    return torch.stack(losses).mean(), active


def m_peak_loss(prob: torch.Tensor, seq_mask: torch.Tensor, sample_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    active = 0
    for i in sample_mask.nonzero(as_tuple=False).flatten().tolist():
        vals = prob[i][seq_mask[i]].float()
        if vals.numel() == 0:
            continue
        top_k = max(1, int(math.ceil(0.10 * int(vals.numel()))))
        bottom_k = max(1, int(math.ceil(0.50 * int(vals.numel()))))
        top_mean = torch.topk(vals, k=min(top_k, int(vals.numel())), largest=True).values.mean()
        bottom_mean = torch.topk(vals, k=min(bottom_k, int(vals.numel())), largest=False).values.mean()
        losses.append(F.relu(prob.new_tensor(0.15) - (top_mean - bottom_mean)))
        active += 1
    if not losses:
        return prob.sum() * 0.0, 0
    return torch.stack(losses).mean(), active


def safe_background_from_positive(pos_mask: torch.Tensor, seq_mask: torch.Tensor, *, radius: int = 17) -> torch.Tensor:
    pos = pos_mask.bool() & seq_mask.bool()
    if not bool(pos.any()):
        return torch.zeros_like(seq_mask, dtype=torch.bool)
    x = pos.float().view(1, 1, -1)
    width = int(radius) * 2 + 1
    near = F.conv1d(x, torch.ones(1, 1, width, dtype=x.dtype, device=x.device), padding=int(radius)).view(-1) > 0
    return seq_mask.bool() & ~near


def default_dpr_v6_config() -> dict[str, Any]:
    return {
        "model": {
            "esm2_dim": 1280,
            "biophys_dim": 112,
            PHASEFLOW_LLPS_HIDDEN_DIM_KEY: 256,
            "phaseflow_hidden_dim": 256,
            "pstp_esm8_dim": 320,
            "pstp_alb_dim": 330,
            "pstp_650_dim": 650,
            "phaseflow_shape": "4x4",
            "phaseflow_bridge_tokens": 32,
            "phaseflow_bridge_adapter_layers": 2,
            "phaseflow_bridge_gate_init": 0.075,
            "num_heads": 8,
            "adapter_dim": 256,
            "dropout": 0.10,
            "window_sizes": [33, 129, 257],
            "topk_fraction": 0.05,
            "head_type": "tiny",
            "leaky_relu_slope": 0.01,
            "use_base_streams": True,
            PHASEFLOW_LLPS_STREAM_KEY: True,
            "use_phaseflow_stream": True,
            "use_pstp650": False,
            "use_pstp_esm8": False,
            "use_pstp_alb": False,
            "pstp_alb_project_dim": 0,
        }
    }


def load_dpr_v6_phasestack(
    *,
    phaseflow_llps_checkpoint: str | Path | None,
    phaseflow_checkpoint: str | Path | None,
    config: dict[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[DPRV6PhaseStack, dict[str, Any], dict[str, Any]]:
    model_cfg = dict(config.get("model", config))
    needs_phaseflow_llps = bool(model_cfg.get(PHASEFLOW_LLPS_STREAM_KEY, model_cfg.get(LEGACY_LLPS_STREAM_KEY, True)) or model_cfg.get("use_phaseflow_stream", True))
    needs_phaseflow = bool(model_cfg.get("use_phaseflow_stream", True))
    phaseflow_llps_raw: dict[str, Any] = {}
    phaseflow_raw: dict[str, Any] = {}
    llps_backbone = None
    phaseflow = None
    if needs_phaseflow_llps:
        if phaseflow_llps_checkpoint is None:
            raise ValueError("phaseflow_llps_checkpoint is required for this DPR v6 arm")
        llps_backbone, phaseflow_llps_raw = load_phaseflow_llps_checkpoint(phaseflow_llps_checkpoint, device=device)
    if needs_phaseflow:
        if phaseflow_checkpoint is None:
            raise ValueError("phaseflow_checkpoint is required for this DPR v6 arm")
        phaseflow, phaseflow_raw = load_phaseflow_checkpoint(
            phaseflow_checkpoint,
            repo_path=DEFAULT_PHASEFLOW_REPO,
            dependency_site_packages=DEFAULT_PHASEFLOW_SITE_PACKAGES,
            device=device,
        )
    model = DPRV6PhaseStack(llps_backbone=llps_backbone, phaseflow=phaseflow, config=config).to(device)
    model.eval()
    return model, phaseflow_llps_raw, phaseflow_raw


def load_dpr_v6_final_inference_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    phaseflow_repo: str | Path = DEFAULT_PHASEFLOW_REPO,
    phaseflow_dependency_site_packages: str | Path = DEFAULT_PHASEFLOW_SITE_PACKAGES,
) -> tuple[DPRV6PhaseStack, dict[str, Any]]:
    """Load a self-contained DPR v6 final inference checkpoint.

    The checkpoint is expected to contain final model weights plus the small
    constructor configs for the frozen PhaseFlow and PhaseFlow backbones.
    """
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    phaseflow_llps_config = payload.get(PHASEFLOW_LLPS_CONFIG_KEY, payload.get(LEGACY_LLPS_CONFIG_KEY))
    phaseflow_config = payload.get("phaseflow_config")
    if phaseflow_llps_config is None:
        raise KeyError("final inference checkpoint is missing phaseflow_llps_config")
    if phaseflow_config is None:
        raise KeyError("final inference checkpoint is missing phaseflow_config")

    llps_backbone = PhaseFlowModel(phaseflow_llps_config)
    _prepare_phaseflow_imports(
        repo_path=phaseflow_repo,
        dependency_site_packages=phaseflow_dependency_site_packages,
    )
    from phaseflow.model import PhaseFlow  # type: ignore

    phaseflow_model_config = dict(phaseflow_config.get("model", phaseflow_config))
    phaseflow = PhaseFlow(**phaseflow_model_config)
    model_cfg = payload.get("resolved_model_config")
    if model_cfg is None:
        cfg = payload["config"]
        arm = str(payload.get("arm", ""))
        model_cfg = {"model": dict(cfg.get("model", {}))}
        if arm and arm in cfg.get("arms", {}):
            model_cfg["model"].update(dict(cfg["arms"][arm].get("model", {})))

    model = DPRV6PhaseStack(llps_backbone=llps_backbone, phaseflow=phaseflow, config=model_cfg)
    model.load_state_dict(_upgrade_legacy_state_dict_keys(payload["model_state_dict"]), strict=True)
    model.to(device)
    model.eval()
    return model, payload


def _upgrade_legacy_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    old_prefix = LEGACY_LLPS_PREFIX + "."
    new_prefix = "llps_backbone."
    return {
        (new_prefix + key[len(old_prefix) :] if key.startswith(old_prefix) else key): value
        for key, value in state_dict.items()
    }
