from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LLPSProteinHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, *, use_dpr_pooling: bool = True) -> None:
        super().__init__()
        self.use_dpr_pooling = bool(use_dpr_pooling)
        self.pool = nn.Linear(d_model, 1)
        self.dpr_pool = nn.Linear(d_model, 1)
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor, dpr_logits: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.pool(x).squeeze(-1).masked_fill(~seq_mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        attention_pool = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        mean_pool = torch.sum(x * seq_mask.unsqueeze(-1), dim=1) / seq_mask.sum(dim=1, keepdim=True).clamp(min=1)
        if not self.use_dpr_pooling:
            high_dpr_pool = attention_pool
        elif dpr_logits is None:
            dpr_scores = self.dpr_pool(x).squeeze(-1)
            dpr_weights = torch.softmax(dpr_scores.masked_fill(~seq_mask, -1.0e4), dim=-1)
            high_dpr_pool = torch.sum(dpr_weights.unsqueeze(-1) * x, dim=1)
        else:
            dpr_scores = dpr_logits
            dpr_weights = torch.softmax(dpr_scores.masked_fill(~seq_mask, -1.0e4), dim=-1)
            high_dpr_pool = torch.sum(dpr_weights.unsqueeze(-1) * x, dim=1)
        protein = torch.cat([attention_pool, mean_pool, high_dpr_pool], dim=-1)
        return self.mlp(protein).squeeze(-1)


class DPRSummaryFusionHead(nn.Module):
    def __init__(
        self,
        summary_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(summary_dim + 1),
            nn.Linear(summary_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, llps_logits: torch.Tensor, dpr_summary: torch.Tensor) -> torch.Tensor:
        features = torch.cat([llps_logits.unsqueeze(-1), dpr_summary], dim=-1)
        residual = self.net(features).squeeze(-1)
        return llps_logits + self.residual_scale * residual


class PhaseDiagramHead(nn.Module):
    def __init__(self, d_model: int, phase_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.pool = nn.Linear(d_model, 1)
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, phase_dim),
        )

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor, dpr_logits: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.pool(x).squeeze(-1).masked_fill(~seq_mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        attention_pool = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        mean_pool = torch.sum(x * seq_mask.unsqueeze(-1), dim=1) / seq_mask.sum(dim=1, keepdim=True).clamp(min=1)
        if dpr_logits is None:
            dpr_weights = weights
        else:
            dpr_weights = torch.softmax(dpr_logits.masked_fill(~seq_mask, -1.0e4), dim=-1)
        high_dpr_pool = torch.sum(dpr_weights.unsqueeze(-1) * x, dim=1)
        return self.mlp(torch.cat([attention_pool, mean_pool, high_dpr_pool], dim=-1))


class ResidueHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class MultiScaleDPRHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        windows: list[int] | tuple[int, ...] = (33, 129, 257),
        topk_ratio: float = 0.05,
        max_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.windows = tuple(max(1, int(window)) for window in windows)
        self.topk_ratio = float(topk_ratio)
        self.max_weight = float(max_weight)
        self.residue_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.window_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )
                for _ in self.windows
            ]
        )

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = [self.residue_mlp(x).squeeze(-1)]
        transposed = (x * seq_mask.unsqueeze(-1)).transpose(1, 2)
        mask = seq_mask.float().unsqueeze(1)
        for window, mlp in zip(self.windows, self.window_mlps, strict=False):
            padding = window // 2
            pooled = F.avg_pool1d(transposed, kernel_size=window, stride=1, padding=padding, count_include_pad=False)
            denom = F.avg_pool1d(mask, kernel_size=window, stride=1, padding=padding, count_include_pad=False).clamp(min=1.0e-6)
            pooled = (pooled / denom).transpose(1, 2)
            if pooled.shape[1] != x.shape[1]:
                pooled = pooled[:, : x.shape[1], :]
            logits.append(mlp(pooled).squeeze(-1))
        dpr_logits = torch.stack(logits, dim=0).mean(dim=0).masked_fill(~seq_mask, -1.0e4)
        probs = torch.sigmoid(dpr_logits.float()).masked_fill(~seq_mask, 0.0)
        topk_values = []
        max_values = []
        for index in range(probs.shape[0]):
            length = int(seq_mask[index].sum().item())
            if length == 0:
                topk_values.append(probs[index].sum() * 0.0)
                max_values.append(probs[index].sum() * 0.0)
                continue
            k = max(1, int(round(length * self.topk_ratio)))
            values = probs[index, :length]
            topk_values.append(torch.topk(values, k=min(k, length)).values.mean())
            max_values.append(values.max())
        topk_mean = torch.stack(topk_values)
        max_score = torch.stack(max_values)
        region_global_score = ((1.0 - self.max_weight) * topk_mean + self.max_weight * max_score).float()
        region_global_score = region_global_score.clamp(min=1.0e-6, max=1.0 - 1.0e-6)
        region_global_logits = torch.logit(region_global_score, eps=1.0e-6)
        return {
            "dpr_logits": dpr_logits,
            "region_global_logits": region_global_logits,
            "region_global_score": region_global_score,
            "region_topk_score": topk_mean,
            "region_max_score": max_score,
        }


class DPRBranchAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        bottleneck_dim: int = 64,
        kernel_size: int = 9,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        bottleneck_dim = max(1, int(bottleneck_dim))
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.depthwise = nn.Conv1d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=bottleneck_dim,
        )
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, d_model)
        nn.init.normal_(self.up.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        mask = seq_mask.unsqueeze(-1)
        hidden = self.down(self.norm(x))
        hidden = hidden * mask
        local = self.depthwise(hidden.transpose(1, 2)).transpose(1, 2)
        if local.shape[1] != hidden.shape[1]:
            local = local[:, : hidden.shape[1], :]
        hidden = F.gelu(hidden + local) * mask
        residual = self.up(self.dropout(hidden)) * mask
        return x + self.residual_scale * residual


class GatedDPRScanResidual(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        windows: list[int] | tuple[int, ...] = (9, 17, 33, 65, 129),
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.windows = tuple(max(1, int(window)) for window in windows)
        self.residual_scale = float(residual_scale)
        branch_count = len(self.windows) + 1
        self.local_branch = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.window_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )
                for _ in self.windows
            ]
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, branch_count),
        )
        self.residual_norm = nn.LayerNorm(branch_count)
        self.residual_mixer = nn.Linear(branch_count, 1)
        nn.init.normal_(self.residual_mixer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.residual_mixer.bias)

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        branch_logits = [self.local_branch(x).squeeze(-1)]
        transposed = (x * seq_mask.unsqueeze(-1)).transpose(1, 2)
        mask = seq_mask.float().unsqueeze(1)
        for window, branch in zip(self.windows, self.window_branches, strict=False):
            padding = window // 2
            pooled = F.avg_pool1d(transposed, kernel_size=window, stride=1, padding=padding, count_include_pad=False)
            denom = F.avg_pool1d(mask, kernel_size=window, stride=1, padding=padding, count_include_pad=False).clamp(min=1.0e-6)
            pooled = (pooled / denom).transpose(1, 2)
            if pooled.shape[1] != x.shape[1]:
                pooled = pooled[:, : x.shape[1], :]
            branch_logits.append(branch(pooled).squeeze(-1))
        stacked = torch.stack(branch_logits, dim=-1)
        gates = torch.softmax(self.gate(x), dim=-1)
        gated = stacked * gates
        residual = self.residual_mixer(self.residual_norm(gated)).squeeze(-1)
        residual = (self.residual_scale * residual).masked_fill(~seq_mask, 0.0)
        return {
            "dpr_scan_residual_logits": residual,
            "dpr_scan_gate": gates,
            "dpr_scan_branch_logits": stacked.masked_fill(~seq_mask.unsqueeze(-1), 0.0),
        }


class DPRLocalizationBranch(nn.Module):
    """Independent DPR localization branch for frozen Stage1 representations."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        bottleneck_dim: int = 64,
        kernel_size: int = 9,
        residual_scale: float = 0.25,
        presence_topk_ratio: float = 0.05,
        windows: list[int] | tuple[int, ...] = (9, 17, 33, 64, 129, 257),
        aux_feature_dim: int = 106,
    ) -> None:
        super().__init__()
        self.presence_topk_ratio = float(presence_topk_ratio)
        self.windows = tuple(max(1, int(window)) for window in windows)
        self.aux_feature_dim = max(0, int(aux_feature_dim))
        self.adapter = DPRBranchAdapter(
            d_model=d_model,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            kernel_size=kernel_size,
            residual_scale=residual_scale,
        )
        self.aux_projection = (
            nn.Sequential(
                nn.LayerNorm(self.aux_feature_dim),
                nn.Linear(self.aux_feature_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            if self.aux_feature_dim > 0
            else None
        )
        if self.aux_projection is not None:
            last = self.aux_projection[-1]
            if isinstance(last, nn.Linear):
                nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)
                nn.init.zeros_(last.bias)
        self.multiscale_convs = nn.ModuleList()
        for window in self.windows:
            if window % 2 == 0:
                window += 1
            self.multiscale_convs.append(
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=window,
                    padding=window // 2,
                    groups=d_model,
                )
            )
        self.multiscale_mixer = nn.Sequential(
            nn.LayerNorm(d_model * (len(self.multiscale_convs) + 1)),
            nn.Linear(d_model * (len(self.multiscale_convs) + 1), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.idr_lcr_expert = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.folded_interaction_expert = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.mechanism_gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 2),
        )
        self.localization_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, d_model // 2), 1),
        )
        self.presence_head = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, max(8, d_model // 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, d_model // 16), 1),
        )

    def forward(
        self,
        frozen_repr: torch.Tensor,
        seq_mask: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.adapter(frozen_repr, seq_mask)
        if self.aux_projection is not None:
            aux = self._aux_features(batch, frozen_repr, seq_mask)
            x = x + self.aux_projection(aux) * seq_mask.unsqueeze(-1)
        x = self._multiscale_context(x, seq_mask)
        idr_repr = self.idr_lcr_expert(x)
        folded_repr = self.folded_interaction_expert(x)
        gate = torch.softmax(self.mechanism_gate(torch.cat([idr_repr, folded_repr], dim=-1)), dim=-1)
        x = (gate[..., :1] * idr_repr + gate[..., 1:] * folded_repr) * seq_mask.unsqueeze(-1)
        localization_logits = self.localization_head(x).squeeze(-1).masked_fill(~seq_mask, -1.0e4)
        boundary_logits = self.boundary_head(x).squeeze(-1).masked_fill(~seq_mask, -1.0e4)
        probs = torch.sigmoid(localization_logits.float()).masked_fill(~seq_mask, 0.0)
        lengths = seq_mask.float().sum(dim=1).clamp(min=1.0)
        mean_score = probs.sum(dim=1) / lengths
        topk_values = []
        max_values = []
        for index in range(probs.shape[0]):
            length = int(seq_mask[index].sum().item())
            if length <= 0:
                topk_values.append(probs[index].sum() * 0.0)
                max_values.append(probs[index].sum() * 0.0)
                continue
            k = max(1, int(round(length * self.presence_topk_ratio)))
            values = probs[index, :length]
            topk_values.append(torch.topk(values, k=min(k, length)).values.mean())
            max_values.append(values.max())
        presence_features = torch.stack([mean_score, torch.stack(topk_values), torch.stack(max_values)], dim=-1)
        presence_logit = self.presence_head(presence_features).squeeze(-1)
        return {
            "dpr_localization_repr": x,
            "dpr_localization_logits": localization_logits,
            "dpr_presence_logit": presence_logit,
            "dpr_boundary_logits": boundary_logits,
            "dpr_mechanism_gate": gate.masked_fill(~seq_mask.unsqueeze(-1), 0.0),
        }

    def _multiscale_context(self, x: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        masked = x * seq_mask.unsqueeze(-1)
        transposed = masked.transpose(1, 2)
        contexts = [masked]
        for conv in self.multiscale_convs:
            context = conv(transposed).transpose(1, 2)
            if context.shape[1] != x.shape[1]:
                context = context[:, : x.shape[1], :]
            contexts.append(context * seq_mask.unsqueeze(-1))
        mixed = self.multiscale_mixer(torch.cat(contexts, dim=-1))
        return (x + mixed) * seq_mask.unsqueeze(-1)

    def _aux_features(
        self,
        batch: dict[str, torch.Tensor] | None,
        frozen_repr: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        if batch is None or self.aux_feature_dim <= 0:
            return frozen_repr.new_zeros((*frozen_repr.shape[:2], self.aux_feature_dim))
        pieces = []
        for name in ("disorder", "physchem", "modality_mask", "reliability"):
            value = batch.get(name)
            if torch.is_tensor(value):
                pieces.append(value.to(device=frozen_repr.device, dtype=frozen_repr.dtype))
        if pieces:
            aux = torch.cat(pieces, dim=-1)
        else:
            aux = frozen_repr.new_zeros((*frozen_repr.shape[:2], 0))
        if aux.shape[-1] > self.aux_feature_dim:
            aux = aux[..., : self.aux_feature_dim]
        elif aux.shape[-1] < self.aux_feature_dim:
            pad = frozen_repr.new_zeros((*aux.shape[:2], self.aux_feature_dim - aux.shape[-1]))
            aux = torch.cat([aux, pad], dim=-1)
        return aux * seq_mask.unsqueeze(-1)
