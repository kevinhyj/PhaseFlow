"""Protein LLPS and DPR model architectures."""
from __future__ import annotations

# Source: models/adapters.py


import torch
from torch import nn


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityAdapters(nn.Module):
    names = ("plm", "physchem", "disorder", "protenix_embed", "starling_embed")
    aliases = {"star_node": "starling_embed"}

    def __init__(self, input_dims: dict[str, int], d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        input_dims = dict(input_dims)
        for alias, canonical in self.aliases.items():
            if canonical not in input_dims and alias in input_dims:
                input_dims[canonical] = input_dims[alias]
        self.adapters = nn.ModuleDict(
            {
                name: FeatureAdapter(int(input_dims[name]), d_model, dropout)
                for name in self.names
            }
        )
        self.modality_embedding = nn.Parameter(torch.zeros(len(self.names), d_model))
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = [
            self.adapters["plm"](batch["plm"]),
            self.adapters["physchem"](batch["physchem"]),
            self.adapters["disorder"](batch["disorder"]),
            self.adapters["protenix_embed"](batch["protenix_embed"]),
            self.adapters["starling_embed"](batch["starling_embed"]),
        ]
        for index, output in enumerate(outputs):
            outputs[index] = output + self.modality_embedding[index]
        return torch.stack(outputs, dim=2)

# Source: models/fusion.py


import torch
from torch import nn


class ReliabilityGatedFusion(nn.Module):
    def __init__(self, d_model: int, num_modalities: int = 5) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model + 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.num_modalities = num_modalities

    def forward(
        self,
        modality_repr: torch.Tensor,
        modality_mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # modality_mask uses 1 for missing and 0 for present.
        gate_input = torch.cat(
            [modality_repr, reliability.unsqueeze(-1), modality_mask.unsqueeze(-1)],
            dim=-1,
        )
        logits = self.gate(gate_input).squeeze(-1)
        logits = logits.masked_fill(modality_mask.bool(), -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        fused = torch.sum(weights.unsqueeze(-1) * modality_repr, dim=2)
        return fused, weights


class ConcatFusion(nn.Module):
    def __init__(self, d_model: int, num_modalities: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.proj = nn.Sequential(
            nn.Linear(num_modalities * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        modality_repr: torch.Tensor,
        modality_mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        present = (~modality_mask.bool()).float()
        weighted = modality_repr * present.unsqueeze(-1) * reliability.unsqueeze(-1).clamp(min=0.0, max=1.0)
        denom = present.sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights = present / denom
        fused = self.proj(weighted.flatten(start_dim=2))
        return fused, weights

# Source: models/heads.py


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

# Source: models/local_motif_encoder.py


import torch
from torch import nn


class LocalMotifBlock(nn.Module):
    def __init__(self, d_model: int, kernels: list[int], dilations: list[int], dropout: float) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for kernel in kernels:
            self.branches.append(
                nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2)
            )
        for dilation in dilations:
            kernel = 7
            padding = dilation * (kernel // 2)
            self.branches.append(
                nn.Conv1d(d_model, d_model, kernel_size=kernel, dilation=dilation, padding=padding)
            )
        self.proj = nn.Sequential(
            nn.Conv1d(len(self.branches) * d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        xt = x.transpose(1, 2)
        branches = [branch(xt) for branch in self.branches]
        out = self.proj(torch.cat(branches, dim=1)).transpose(1, 2)
        out = self.norm(residual + out)
        return out * seq_mask.unsqueeze(-1)


class LocalMotifEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        kernels: list[int] | None = None,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        kernels = kernels or [3, 5, 9]
        dilations = dilations or [2, 4]
        self.layers = nn.ModuleList(
            [LocalMotifBlock(d_model, kernels, dilations, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, seq_mask)
        return x

# Source: models/region_decoder.py


import torch
from torch import nn


class RegionQueryDecoder(nn.Module):
    def __init__(self, d_model: int, num_queries: int = 16, num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=max(1, min(8, d_model // 16)),
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.score = nn.Linear(d_model, 1)
        self.boundary = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))

    def forward(self, residue_repr: torch.Tensor, seq_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = residue_repr.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(
            tgt=queries,
            memory=residue_repr,
            memory_key_padding_mask=~seq_mask,
        )
        region_logits = self.score(decoded).squeeze(-1)
        boundaries = torch.sigmoid(self.boundary(decoded))
        start = torch.minimum(boundaries[..., 0], boundaries[..., 1])
        end = torch.maximum(boundaries[..., 0], boundaries[..., 1])
        width = boundaries[..., 2]
        return {"region_logits": region_logits, "region_start": start, "region_end": end, "region_width": width}

# Source: models/sparse_graph_transformer.py


import math

import torch
from torch import nn


class SparseGraphTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        edge_dim: int,
        ffn_dim: int,
        dropout: float,
        num_edge_types: int = 8,
        relative_position_bins: int = 32,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Linear(edge_dim, num_heads)
        self.edge_type_bias = nn.Embedding(num_edge_types, num_heads)
        self.relative_position_bias = nn.Embedding(relative_position_bins + 1, num_heads)
        self.relative_position_bins = relative_position_bins
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        edge_attr: torch.Tensor,
        neighbor_mask: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, k_neighbors = neighbors.shape
        clamped_neighbors = neighbors.clamp(min=0, max=max(length - 1, 0))
        neighbor_x = _gather_neighbor_features(x, clamped_neighbors)

        q = self.q_proj(x).view(batch_size, length, self.num_heads, self.head_dim)
        k = self.k_proj(neighbor_x).view(batch_size, length, k_neighbors, self.num_heads, self.head_dim)
        v = self.v_proj(neighbor_x).view(batch_size, length, k_neighbors, self.num_heads, self.head_dim)
        scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(edge_attr)
        rel_bins = torch.clamp((edge_attr[..., 0] * self.relative_position_bins).round().long(), 0, self.relative_position_bins)
        scores = scores + self.relative_position_bias(rel_bins)
        edge_type = _sparse_graph_transformer_edge_type_from_attr(edge_attr, self.edge_type_bias.num_embeddings)
        scores = scores + self.edge_type_bias(edge_type)
        valid_neighbor = neighbor_mask & torch.gather(seq_mask, 1, clamped_neighbors.reshape(batch_size, -1)).view(batch_size, length, k_neighbors)
        scores = scores.masked_fill(~valid_neighbor.unsqueeze(-1), -1.0e4)
        attn = torch.softmax(scores, dim=2)
        attn = attn.masked_fill(~valid_neighbor.unsqueeze(-1), 0.0)
        context = (attn.unsqueeze(-1) * v).sum(dim=2).reshape(batch_size, length, self.d_model)
        x = self.norm1(x + self.dropout(self.out_proj(context)))
        x = self.norm2(x + self.ffn(x))
        return x * seq_mask.unsqueeze(-1)


class SparseGraphTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        edge_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        num_edge_types: int = 8,
        relative_position_bins: int = 32,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SparseGraphTransformerLayer(
                    d_model,
                    num_heads,
                    edge_dim,
                    ffn_dim,
                    dropout,
                    num_edge_types=num_edge_types,
                    relative_position_bins=relative_position_bins,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        edge_attr: torch.Tensor,
        neighbor_mask: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, neighbors, edge_attr, neighbor_mask, seq_mask)
        return x


def _sparse_graph_transformer_edge_type_from_attr(edge_attr: torch.Tensor, num_edge_types: int) -> torch.Tensor:
    if edge_attr.shape[-1] <= 3:
        return torch.zeros(edge_attr.shape[:-1], dtype=torch.long, device=edge_attr.device)
    type_slice = edge_attr[..., 3 : 3 + num_edge_types]
    if type_slice.numel() == 0:
        return torch.zeros(edge_attr.shape[:-1], dtype=torch.long, device=edge_attr.device)
    return torch.clamp(torch.argmax(type_slice, dim=-1), 0, max(num_edge_types - 1, 0))


def _gather_neighbor_features(x: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    batch_size, length, k_neighbors = neighbors.shape
    batch_offsets = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1) * length
    flat_indices = (neighbors + batch_offsets).reshape(-1)
    flat_x = x.reshape(batch_size * length, x.shape[-1])
    return flat_x.index_select(0, flat_indices).view(batch_size, length, k_neighbors, x.shape[-1])

# Source: models/phaseflow.py


import copy

import torch
from torch import nn

from phaseflow.protein.model import ModalityAdapters
from phaseflow.protein.model import ConcatFusion, ReliabilityGatedFusion
from phaseflow.protein.model import (
    DPRSummaryFusionHead,
    DPRBranchAdapter,
    DPRLocalizationBranch,
    GatedDPRScanResidual,
    LLPSProteinHead,
    MultiScaleDPRHead,
    PhaseDiagramHead,
    ResidueHead,
)
from phaseflow.protein.model import LocalMotifEncoder
from phaseflow.protein.model import RegionQueryDecoder
from phaseflow.protein.model import SparseGraphTransformer
from phaseflow.protein.features import BIO_VEC_DIM, BIO_VEC_NAMES


class BioMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int] | tuple[int, ...], dropout: float) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(value) for value in hidden]
        layers: list[nn.Module] = [nn.LayerNorm(dims[0])]
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=False):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(float(dropout)), nn.LayerNorm(out_dim)])
        self.net = nn.Sequential(*layers)
        self.output_dim = dims[-1]

    def forward(self, bio_vec: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(bio_vec.float(), nan=0.0, posinf=10.0, neginf=-10.0))


class BioFusionResidualHead(nn.Module):
    def __init__(self, protein_dim: int, bio_dim: int, hidden_dim: int, dropout: float, residual_scale: float) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(protein_dim + bio_dim),
            nn.Linear(protein_dim + bio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, base_logits: torch.Tensor, protein_repr: torch.Tensor, bio_repr: torch.Tensor) -> torch.Tensor:
        residual = self.net(torch.cat([protein_repr, bio_repr], dim=-1)).squeeze(-1)
        return base_logits + self.residual_scale * residual


class ProteinAuxHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class PhaseFlowModel(nn.Module):
    modality_indices = {
        "plm": 0,
        "physchem": 1,
        "disorder": 2,
        "protenix_embed": 3,
        "protenix_embedding": 3,
        "protenix": 3,
        "star": 4,
        "starling": 4,
        "starling_embed": 4,
        "star_node": 4,
    }
    edge_type_indices = {
        "local": 0,
        "sequence": 0,
        "star": 2,
        "starling": 2,
        "physchem": 3,
        "candidate": 4,
        "candidate_segment": 4,
    }
    named_ablations = {
        "no_physchem": (("physchem",), ()),
        "no_disorder": (("disorder",), ()),
        "no_protenix": (("protenix",), ()),
        "no_starling": (("starling",), ("starling",)),
        "no_protenix_starling": (("protenix", "starling"), ("starling",)),
    }
    bio_vec_groups = {
        "plm": ("esm_mean", "esm_std"),
        "esm2": ("esm_mean", "esm_std"),
        "physchem": (
            "ncpr",
            "charge_kappa_proxy",
            "sticker_spacer_kappa_proxy",
            "frac_g",
            "frac_p",
            "frac_r",
            "frac_y",
            "frac_f",
            "frac_w",
            "frac_fyw",
            "rgg_density",
            "aromatic_cluster_density",
            "charge_blockiness",
            "hydropathy_mean",
            "rna_binding_proxy",
            "dna_binding_proxy",
            "ptm_density_proxy",
        ),
        "disorder": ("idr_fraction", "ordered_fraction", "prld_fraction", "low_complexity_fraction"),
        "protenix": ("protenix_available",),
        "protenix_embed": ("protenix_available",),
        "protenix_embedding": ("protenix_available",),
        "star": ("starling_mean_norm", "starling_std_norm", "starling_compaction_proxy"),
        "starling": ("starling_mean_norm", "starling_std_norm", "starling_compaction_proxy"),
        "starling_embed": ("starling_mean_norm", "starling_std_norm", "starling_compaction_proxy"),
        "star_node": ("starling_mean_norm", "starling_std_norm", "starling_compaction_proxy"),
    }

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config.get("model", config)
        self.model_type = str(model_config.get("model_type", "v2_region"))
        self.is_decoupled = self.model_type in {"v3", "v3_decoupled", "decoupled"}
        self.forward_mode = str(model_config.get("forward_mode", "full")).strip().lower()
        self.llps_only_forward = self.is_decoupled and self.forward_mode in {"llps_only", "protein_only"}
        self.ablation_name = str(model_config.get("ablation", {}).get("name", "full"))
        ablation_config = model_config.get("ablation", {})
        self.disabled_modality_indices = self._disabled_indices(
            ablation_config,
            key="disabled_modalities",
            lookup=self.modality_indices,
            named_index=0,
        )
        self.disabled_edge_types = self._disabled_indices(
            ablation_config,
            key="disabled_edge_types",
            lookup=self.edge_type_indices,
            named_index=1,
        )
        d_model = int(model_config.get("d_model", 128))
        dropout = float(model_config.get("dropout", 0.1))
        self.adapters = ModalityAdapters(model_config["input_dims"], d_model, dropout)
        if self.ablation_name == "concat_fusion":
            self.fusion = ConcatFusion(d_model=d_model, num_modalities=5, dropout=dropout)
        else:
            self.fusion = ReliabilityGatedFusion(d_model=d_model, num_modalities=5)
        if self.is_decoupled:
            self._init_decoupled_encoders(model_config, d_model, dropout)
        else:
            local_config = model_config.get("local_encoder", {})
            self.local_encoder = self._make_local_encoder(local_config, d_model, dropout)
            self.encoder, self.uses_graph = self._make_sequence_encoder(model_config, model_config, d_model, dropout)
        llps_head_config = model_config.get("llps_head", {})
        self.llps_head = LLPSProteinHead(
            d_model,
            dropout,
            use_dpr_pooling=bool(llps_head_config.get("use_dpr_pooling", not self.is_decoupled)),
        )
        bio_config = model_config.get("bio_mlp", {}) or {}
        self.bio_mlp_enabled = bool(bio_config.get("enabled", False))
        self.bio_vec_dim = int(bio_config.get("input_dim", BIO_VEC_DIM))
        self.disabled_bio_vec_indices = self._disabled_bio_vec_indices(ablation_config)
        self.bio_mlp = None
        self.bio_fusion_head = None
        self.driver_head = None
        self.client_head = None
        self.negtype_head = None
        if self.bio_mlp_enabled:
            bio_hidden = list(bio_config.get("hidden", [256, 256, 128]))
            bio_dropout = float(bio_config.get("dropout", dropout))
            self.bio_mlp = BioMLP(self.bio_vec_dim, bio_hidden, bio_dropout)
            fusion_dim = 3 * d_model + self.bio_mlp.output_dim
            aux_hidden = int(bio_config.get("aux_hidden", max(64, d_model // 2)))
            self.bio_fusion_head = BioFusionResidualHead(
                protein_dim=3 * d_model,
                bio_dim=self.bio_mlp.output_dim,
                hidden_dim=int(bio_config.get("fusion_hidden", d_model)),
                dropout=bio_dropout,
                residual_scale=float(bio_config.get("residual_scale", 1.0)),
            )
            self.driver_head = ProteinAuxHead(fusion_dim, aux_hidden, 1, bio_dropout)
            self.client_head = ProteinAuxHead(fusion_dim, aux_hidden, 1, bio_dropout)
            self.negtype_head = ProteinAuxHead(fusion_dim, aux_hidden, 2, bio_dropout)
        mil_config = model_config.get("region_mil_head", {})
        self.dpr_head = MultiScaleDPRHead(
            d_model=d_model,
            dropout=dropout,
            windows=list(mil_config.get("windows", [33, 129, 257])),
            topk_ratio=float(mil_config.get("topk_ratio", 0.05)),
            max_weight=float(mil_config.get("max_weight", 0.3)),
        )
        scan_config = model_config.get("dpr_scan_residual", {})
        self.dpr_scan_residual = (
            GatedDPRScanResidual(
                d_model=d_model,
                dropout=dropout,
                windows=list(scan_config.get("windows", [9, 17, 33, 65, 129])),
                residual_scale=float(scan_config.get("residual_scale", 0.5)),
            )
            if bool(scan_config.get("enabled", False))
            else None
        )
        adapter_config = model_config.get("dpr_adapter", {})
        self.dpr_adapter = (
            DPRBranchAdapter(
                d_model=d_model,
                dropout=dropout,
                bottleneck_dim=int(adapter_config.get("bottleneck_dim", max(16, d_model // 4))),
                kernel_size=int(adapter_config.get("kernel_size", 9)),
                residual_scale=float(adapter_config.get("residual_scale", 0.25)),
            )
            if bool(adapter_config.get("enabled", False))
            else None
        )
        reference_config = model_config.get("llps_reference_dpr_head", {})
        self.llps_reference_dpr_head = None
        if bool(reference_config.get("enabled", False)):
            self.llps_reference_dpr_head = copy.deepcopy(self.dpr_head)
        if self.llps_reference_dpr_head is not None:
            for parameter in self.llps_reference_dpr_head.parameters():
                parameter.requires_grad = False
        self.key_head = ResidueHead(d_model, dropout)
        self.llps_region_mix = float(model_config.get("llps_region_mix", 0.8))
        self.llps_logit_bias = float(model_config.get("llps_logit_bias", 0.0))
        self.llps_logit_temperature = max(float(model_config.get("llps_logit_temperature", 1.0)), 1.0e-6)
        summary_config = model_config.get("dpr_summary", {})
        self.dpr_summary_dim = 6
        self.dpr_summary_enabled = self.is_decoupled and bool(summary_config.get("enabled", True))
        self.dpr_summary_detach = bool(summary_config.get("detach", True))
        self.dpr_summary_threshold = float(summary_config.get("threshold", 0.5))
        self.dpr_summary_temperature = max(float(summary_config.get("temperature", 0.08)), 1.0e-6)
        self.dpr_summary_head = (
            DPRSummaryFusionHead(
                summary_dim=self.dpr_summary_dim,
                hidden_dim=int(summary_config.get("hidden_dim", max(16, d_model // 2))),
                dropout=dropout,
                residual_scale=float(summary_config.get("residual_scale", 0.5)),
            )
            if self.dpr_summary_enabled
            else None
        )
        phase_aux_config = model_config.get("phase_aux", {})
        self.phase_head = (
            PhaseDiagramHead(d_model, int(phase_aux_config.get("phase_dim", 16)), dropout)
            if bool(phase_aux_config.get("enabled", False))
            else None
        )
        region_config = model_config.get("region_decoder", {})
        self.region_decoder = RegionQueryDecoder(
            d_model=d_model,
            num_queries=int(region_config.get("num_queries", 16)),
            num_layers=int(region_config.get("num_layers", 1)),
            dropout=dropout,
        )
        independent_dpr_config = model_config.get("dpr_localization_branch", {}) or model_config.get(
            "independent_dpr_branch",
            {},
        )
        self.dpr_localization_detach_input = bool(independent_dpr_config.get("detach_input", True))
        self.dpr_localization_branch = (
            DPRLocalizationBranch(
                d_model=d_model,
                dropout=float(independent_dpr_config.get("dropout", dropout)),
                bottleneck_dim=int(independent_dpr_config.get("bottleneck_dim", max(16, d_model // 4))),
                kernel_size=int(independent_dpr_config.get("kernel_size", 9)),
                residual_scale=float(independent_dpr_config.get("residual_scale", 0.25)),
                presence_topk_ratio=float(independent_dpr_config.get("presence_topk_ratio", 0.05)),
                windows=list(independent_dpr_config.get("windows", [9, 17, 33, 64, 129, 257])),
                aux_feature_dim=int(independent_dpr_config.get("aux_feature_dim", 106)),
            )
            if bool(independent_dpr_config.get("enabled", False))
            else None
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.is_decoupled:
            return self._forward_decoupled(batch)
        seq_mask = batch["seq_mask"].bool()
        adapter_batch, modality_mask, reliability = self._prepare_modality_inputs(batch)
        modality_repr = self.adapters(adapter_batch)
        modality_mask, reliability = self._apply_ablation(modality_mask, reliability)
        x, weights = self.fusion(modality_repr, modality_mask, reliability)
        x = self.local_encoder(x, seq_mask)
        if self.uses_graph:
            neighbor_mask = self._apply_edge_ablation(batch["edge_attr"], batch["neighbor_mask"])
            x = self.encoder(
                x=x,
                neighbors=batch["neighbors"],
                edge_attr=batch["edge_attr"],
                neighbor_mask=neighbor_mask,
                seq_mask=seq_mask,
            )
        else:
            x = self.encoder(x, src_key_padding_mask=~seq_mask)
            x = x * seq_mask.unsqueeze(-1)
        dpr_x = self._apply_dpr_adapter(x, seq_mask)
        region = self.region_decoder(dpr_x, seq_mask)
        dpr = self.dpr_head(dpr_x, seq_mask)
        dpr = self._apply_scan_residual(dpr, dpr_x, seq_mask)
        dpr_logits = dpr["dpr_logits"]
        llps_reference = self.llps_reference_dpr_head(x, seq_mask) if self.llps_reference_dpr_head is not None else dpr
        llps_reference_logits = llps_reference["dpr_logits"]
        llps_logits = self.llps_head(x, seq_mask, dpr_logits=llps_reference_logits)
        llps_logits, aux_outputs = self._apply_bio_mlp(
            batch=batch,
            base_logits=llps_logits,
            protein_x=x,
            seq_mask=seq_mask,
            dpr_logits=llps_reference_logits,
        )
        llps_probability = (
            self.llps_region_mix * torch.sigmoid(llps_logits.float())
            + (1.0 - self.llps_region_mix) * torch.sigmoid(llps_reference["region_global_logits"].float())
        ).clamp(min=1.0e-6, max=1.0 - 1.0e-6)
        outputs = {
            "residue_repr": dpr_x,
            "llps_residue_repr": x,
            "dpr_residue_repr": dpr_x,
            "raw_llps_logits": llps_logits,
            "llps_logits": self._calibrated_llps_logits(torch.logit(llps_probability, eps=1.0e-6)),
            "loss_llps_logits": self._calibrated_llps_logits(torch.logit(llps_probability, eps=1.0e-6)),
            "dpr_logits": dpr_logits,
            "residue_dpr_logits": dpr_logits,
            "key_logits": self.key_head(dpr_x),
            "modality_weights": weights,
            **aux_outputs,
            **dpr,
            **region,
        }
        outputs.update(self._independent_dpr_outputs(x, seq_mask, batch=batch))
        if self.phase_head is not None:
            outputs["phase_values"] = self.phase_head(dpr_x, seq_mask, dpr_logits=dpr_logits)
        return outputs

    def _forward_decoupled(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        seq_mask = batch["seq_mask"].bool()
        adapter_batch, modality_mask, reliability = self._prepare_modality_inputs(batch)
        modality_repr = self.adapters(adapter_batch)
        modality_mask, reliability = self._apply_ablation(modality_mask, reliability)
        x, weights = self.fusion(modality_repr, modality_mask, reliability)
        shared = self.shared_local_encoder(x, seq_mask)
        if self.uses_graph:
            neighbor_mask = self._apply_edge_ablation(batch["edge_attr"], batch["neighbor_mask"])
            shared = self.shared_encoder(
                x=shared,
                neighbors=batch["neighbors"],
                edge_attr=batch["edge_attr"],
                neighbor_mask=neighbor_mask,
                seq_mask=seq_mask,
            )
        else:
            shared = self.shared_encoder(shared, src_key_padding_mask=~seq_mask)
            shared = shared * seq_mask.unsqueeze(-1)

        llps_x = self.llps_branch_norm(shared)
        llps_x = self.llps_local_encoder(llps_x, seq_mask)
        if self.uses_graph:
            neighbor_mask = self._apply_edge_ablation(batch["edge_attr"], batch["neighbor_mask"])
            llps_x = self.llps_encoder(
                x=llps_x,
                neighbors=batch["neighbors"],
                edge_attr=batch["edge_attr"],
                neighbor_mask=neighbor_mask,
                seq_mask=seq_mask,
            )
        else:
            llps_x = self.llps_encoder(llps_x, src_key_padding_mask=~seq_mask) * seq_mask.unsqueeze(-1)

        raw_llps_logits = self.llps_head(llps_x, seq_mask, dpr_logits=None)
        raw_llps_logits, aux_outputs = self._apply_bio_mlp(
            batch=batch,
            base_logits=raw_llps_logits,
            protein_x=llps_x,
            seq_mask=seq_mask,
            dpr_logits=None,
        )
        if self.llps_only_forward:
            llps_logits = self._calibrated_llps_logits(raw_llps_logits)
            return {
                "residue_repr": llps_x,
                "shared_residue_repr": shared,
                "llps_residue_repr": llps_x,
                "raw_llps_logits": raw_llps_logits,
                "llps_logits": llps_logits,
                "loss_llps_logits": llps_logits,
                "modality_weights": weights,
                **aux_outputs,
            }

        dpr_x = self.dpr_branch_norm(shared)
        dpr_x = self.dpr_local_encoder(dpr_x, seq_mask)
        if self.uses_graph:
            neighbor_mask = self._apply_edge_ablation(batch["edge_attr"], batch["neighbor_mask"])
            dpr_x = self.dpr_encoder(
                x=dpr_x,
                neighbors=batch["neighbors"],
                edge_attr=batch["edge_attr"],
                neighbor_mask=neighbor_mask,
                seq_mask=seq_mask,
            )
        else:
            dpr_x = self.dpr_encoder(dpr_x, src_key_padding_mask=~seq_mask) * seq_mask.unsqueeze(-1)

        dpr_x = self._apply_dpr_adapter(dpr_x, seq_mask)
        region = self.region_decoder(dpr_x, seq_mask)
        dpr = self.dpr_head(dpr_x, seq_mask)
        dpr = self._apply_scan_residual(dpr, dpr_x, seq_mask)
        dpr_logits = dpr["dpr_logits"]
        dpr_summary = self._dpr_summary_features(dpr, dpr_logits, seq_mask)
        if self.dpr_summary_head is not None:
            summary_for_llps = dpr_summary.detach() if self.dpr_summary_detach else dpr_summary
            llps_logits = self.dpr_summary_head(raw_llps_logits, summary_for_llps)
        else:
            llps_logits = raw_llps_logits
        llps_logits = self._calibrated_llps_logits(llps_logits)
        outputs = {
            "residue_repr": dpr_x,
            "shared_residue_repr": shared,
            "llps_residue_repr": llps_x,
            "dpr_residue_repr": dpr_x,
            "raw_llps_logits": raw_llps_logits,
            "llps_logits": llps_logits,
            "loss_llps_logits": llps_logits,
            "dpr_logits": dpr_logits,
            "residue_dpr_logits": dpr_logits,
            "key_logits": self.key_head(dpr_x),
            "dpr_summary_features": dpr_summary,
            "dpr_summary_detached": torch.full_like(raw_llps_logits, float(self.dpr_summary_detach)),
            "modality_weights": weights,
            **aux_outputs,
            **dpr,
            **region,
        }
        outputs.update(self._independent_dpr_outputs(shared, seq_mask, batch=batch))
        if self.phase_head is not None:
            outputs["phase_values"] = self.phase_head(dpr_x, seq_mask, dpr_logits=dpr_logits)
        return outputs

    def _calibrated_llps_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.llps_logit_bias == 0.0 and self.llps_logit_temperature == 1.0:
            return logits
        return logits / float(self.llps_logit_temperature) + float(self.llps_logit_bias)

    def _apply_bio_mlp(
        self,
        *,
        batch: dict[str, torch.Tensor],
        base_logits: torch.Tensor,
        protein_x: torch.Tensor,
        seq_mask: torch.Tensor,
        dpr_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.bio_mlp_enabled or self.bio_mlp is None or self.bio_fusion_head is None:
            return base_logits, {}
        bio_vec = batch.get("bio_vec")
        if bio_vec is None:
            bio_vec = torch.zeros(base_logits.shape[0], self.bio_vec_dim, dtype=protein_x.dtype, device=protein_x.device)
        bio_vec = bio_vec.to(device=protein_x.device, dtype=protein_x.dtype)
        if bio_vec.shape[-1] != self.bio_vec_dim:
            if bio_vec.shape[-1] > self.bio_vec_dim:
                bio_vec = bio_vec[:, : self.bio_vec_dim]
            else:
                pad = torch.zeros(
                    bio_vec.shape[0],
                    self.bio_vec_dim - bio_vec.shape[-1],
                    dtype=bio_vec.dtype,
                    device=bio_vec.device,
                )
                bio_vec = torch.cat([bio_vec, pad], dim=-1)
        if self.disabled_bio_vec_indices:
            bio_vec = bio_vec.clone()
            bio_vec[:, list(self.disabled_bio_vec_indices)] = 0.0
        bio_repr = self.bio_mlp(bio_vec)
        protein_repr = self._llps_protein_repr(protein_x, seq_mask, dpr_logits)
        final_logits = self.bio_fusion_head(base_logits, protein_repr, bio_repr)
        aux_input = torch.cat([protein_repr, bio_repr], dim=-1)
        aux: dict[str, torch.Tensor] = {
            "bio_vec": bio_vec,
            "bio_repr": bio_repr,
            "bio_llps_logits": final_logits,
        }
        if self.driver_head is not None:
            aux["driver_logits"] = self.driver_head(aux_input)
        if self.client_head is not None:
            aux["client_logits"] = self.client_head(aux_input)
        if self.negtype_head is not None:
            aux["negtype_logits"] = self.negtype_head(aux_input)
        return final_logits, aux

    def _llps_protein_repr(
        self,
        x: torch.Tensor,
        seq_mask: torch.Tensor,
        dpr_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = self.llps_head.pool(x).squeeze(-1).masked_fill(~seq_mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        attention_pool = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        mean_pool = torch.sum(x * seq_mask.unsqueeze(-1), dim=1) / seq_mask.sum(dim=1, keepdim=True).clamp(min=1)
        if dpr_logits is None:
            dpr_scores = self.llps_head.dpr_pool(x).squeeze(-1)
        else:
            dpr_scores = dpr_logits
        dpr_weights = torch.softmax(dpr_scores.masked_fill(~seq_mask, -1.0e4), dim=-1)
        high_dpr_pool = torch.sum(dpr_weights.unsqueeze(-1) * x, dim=1)
        return torch.cat([attention_pool, mean_pool, high_dpr_pool], dim=-1)

    def _apply_dpr_adapter(self, x: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        if self.dpr_adapter is None:
            return x
        return self.dpr_adapter(x, seq_mask)

    def _apply_scan_residual(
        self,
        dpr: dict[str, torch.Tensor],
        x: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.dpr_scan_residual is None:
            return dpr
        scan = self.dpr_scan_residual(x, seq_mask)
        base_logits = dpr["dpr_logits"]
        dpr_logits = (base_logits + scan["dpr_scan_residual_logits"]).masked_fill(~seq_mask, -1.0e4)
        global_scores = self._dpr_global_scores(
            dpr_logits,
            seq_mask,
            topk_ratio=float(getattr(self.dpr_head, "topk_ratio", 0.05)),
            max_weight=float(getattr(self.dpr_head, "max_weight", 0.3)),
        )
        return {
            **dpr,
            "base_dpr_logits": base_logits,
            "dpr_logits": dpr_logits,
            "region_global_logits": global_scores["region_global_logits"],
            "region_global_score": global_scores["region_global_score"],
            "region_topk_score": global_scores["region_topk_score"],
            "region_max_score": global_scores["region_max_score"],
            **scan,
        }

    def _independent_dpr_outputs(
        self,
        x: torch.Tensor,
        seq_mask: torch.Tensor,
        *,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.dpr_localization_branch is None:
            return {}
        dpr_input = x.detach() if self.dpr_localization_detach_input else x
        return self.dpr_localization_branch(dpr_input, seq_mask, batch=batch)

    @staticmethod
    def _dpr_global_scores(
        dpr_logits: torch.Tensor,
        seq_mask: torch.Tensor,
        *,
        topk_ratio: float,
        max_weight: float,
    ) -> dict[str, torch.Tensor]:
        probs = torch.sigmoid(dpr_logits.float()).masked_fill(~seq_mask, 0.0)
        topk_values = []
        max_values = []
        for index in range(probs.shape[0]):
            length = int(seq_mask[index].sum().item())
            if length == 0:
                topk_values.append(probs[index].sum() * 0.0)
                max_values.append(probs[index].sum() * 0.0)
                continue
            k = max(1, int(round(length * topk_ratio)))
            values = probs[index, :length]
            topk_values.append(torch.topk(values, k=min(k, length)).values.mean())
            max_values.append(values.max())
        topk_mean = torch.stack(topk_values)
        max_score = torch.stack(max_values)
        region_global_score = ((1.0 - max_weight) * topk_mean + max_weight * max_score).float().clamp(
            min=1.0e-6,
            max=1.0 - 1.0e-6,
        )
        return {
            "region_global_logits": torch.logit(region_global_score, eps=1.0e-6),
            "region_global_score": region_global_score,
            "region_topk_score": topk_mean,
            "region_max_score": max_score,
        }

    def _init_decoupled_encoders(self, model_config: dict, d_model: int, dropout: float) -> None:
        local_config = dict(model_config.get("local_encoder", {}))
        graph_config = dict(model_config.get("graph_transformer", {}))
        decoupled_config = dict(model_config.get("decoupled", {}))
        total_local_layers = int(local_config.get("num_layers", 2))
        total_graph_layers = int(graph_config.get("num_layers", 2))

        shared_local_layers = int(decoupled_config.get("shared_local_layers", min(max(total_local_layers, 0), 1)))
        branch_local_layers = int(
            decoupled_config.get("branch_local_layers", max(total_local_layers - shared_local_layers, 1))
        )
        shared_graph_layers = int(decoupled_config.get("shared_graph_layers", max(total_graph_layers // 2, 1)))
        branch_graph_layers = int(
            decoupled_config.get("branch_graph_layers", max(total_graph_layers - shared_graph_layers, 1))
        )

        self.shared_local_encoder = self._make_local_encoder(
            _merged_encoder_config(local_config, model_config.get("shared_local_encoder", {}), shared_local_layers),
            d_model,
            dropout,
        )
        self.llps_local_encoder = self._make_local_encoder(
            _merged_encoder_config(local_config, model_config.get("llps_local_encoder", {}), branch_local_layers),
            d_model,
            dropout,
        )
        self.dpr_local_encoder = self._make_local_encoder(
            _merged_encoder_config(local_config, model_config.get("dpr_local_encoder", {}), branch_local_layers),
            d_model,
            dropout,
        )

        shared_sequence_config = dict(model_config)
        shared_sequence_config["graph_transformer"] = _merged_encoder_config(
            graph_config,
            model_config.get("shared_graph_transformer", {}),
            shared_graph_layers,
        )
        self.shared_encoder, self.uses_graph = self._make_sequence_encoder(
            model_config,
            shared_sequence_config,
            d_model,
            dropout,
        )

        llps_sequence_config = dict(model_config)
        llps_sequence_config["graph_transformer"] = _merged_encoder_config(
            graph_config,
            model_config.get("llps_graph_transformer", {}),
            branch_graph_layers,
        )
        dpr_sequence_config = dict(model_config)
        dpr_sequence_config["graph_transformer"] = _merged_encoder_config(
            graph_config,
            model_config.get("dpr_graph_transformer", {}),
            branch_graph_layers,
        )
        self.llps_encoder, llps_uses_graph = self._make_sequence_encoder(
            model_config,
            llps_sequence_config,
            d_model,
            dropout,
        )
        self.dpr_encoder, dpr_uses_graph = self._make_sequence_encoder(
            model_config,
            dpr_sequence_config,
            d_model,
            dropout,
        )
        if llps_uses_graph != self.uses_graph or dpr_uses_graph != self.uses_graph:
            raise ValueError("Decoupled encoders must all use the same graph/non-graph mode.")
        self.llps_branch_norm = nn.LayerNorm(d_model)
        self.dpr_branch_norm = nn.LayerNorm(d_model)

    def _make_local_encoder(self, local_config: dict, d_model: int, dropout: float) -> LocalMotifEncoder:
        return LocalMotifEncoder(
            d_model=d_model,
            num_layers=int(local_config.get("num_layers", 2)),
            kernels=list(local_config.get("kernels", [3, 5, 9])),
            dilations=list(local_config.get("dilations", [2, 4])),
            dropout=dropout,
        )

    def _make_sequence_encoder(
        self,
        root_model_config: dict,
        encoder_model_config: dict,
        d_model: int,
        dropout: float,
    ) -> tuple[nn.Module, bool]:
        if self.model_type == "v0":
            trans_config = encoder_model_config.get("transformer", root_model_config.get("transformer", {}))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(trans_config.get("num_heads", 4)),
                dim_feedforward=int(trans_config.get("ffn_dim", 4 * d_model)),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=int(trans_config.get("num_layers", 2))), False

        graph_config = encoder_model_config.get("graph_transformer", root_model_config.get("graph_transformer", {}))
        return (
            SparseGraphTransformer(
                d_model=d_model,
                num_layers=int(graph_config.get("num_layers", 2)),
                num_heads=int(graph_config.get("num_heads", 4)),
                edge_dim=int(graph_config.get("edge_dim", 8)),
                ffn_dim=int(graph_config.get("ffn_dim", 4 * d_model)),
                dropout=dropout,
                num_edge_types=int(graph_config.get("num_edge_types", 8)),
                relative_position_bins=int(graph_config.get("relative_position_bins", 32)),
            ),
            True,
        )

    def _dpr_summary_features(
        self,
        dpr: dict[str, torch.Tensor],
        dpr_logits: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(dpr_logits).masked_fill(~seq_mask, 0.0)
        lengths = seq_mask.float().sum(dim=1).clamp(min=1.0)
        mean_score = probs.sum(dim=1) / lengths
        high_fraction = torch.sigmoid((probs - self.dpr_summary_threshold) / self.dpr_summary_temperature)
        high_fraction = high_fraction.masked_fill(~seq_mask, 0.0).sum(dim=1) / lengths
        uncertainty = (probs * (1.0 - probs)).sum(dim=1) / lengths
        return torch.stack(
            [
                dpr["region_global_score"],
                dpr["region_topk_score"],
                dpr["region_max_score"],
                mean_score,
                high_fraction,
                uncertainty,
            ],
            dim=-1,
        )

    def _prepare_modality_inputs(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        modality_mask = batch["modality_mask"]
        reliability = batch["reliability"]
        esm2_available_mask = batch.get("esm2_available_mask")
        if esm2_available_mask is None:
            return batch, modality_mask, reliability

        esm2_available = esm2_available_mask.to(dtype=batch["plm"].dtype).clamp(min=0.0, max=1.0)
        adapter_batch = dict(batch)
        adapter_batch["plm"] = batch["plm"] * esm2_available.unsqueeze(-1)

        modality_mask = modality_mask.clone()
        reliability = reliability.clone()
        plm_missing = 1.0 - esm2_available.to(dtype=modality_mask.dtype)
        modality_mask[..., self.modality_indices["plm"]] = torch.maximum(
            modality_mask[..., self.modality_indices["plm"]],
            plm_missing,
        )
        reliability[..., self.modality_indices["plm"]] = (
            reliability[..., self.modality_indices["plm"]] * esm2_available.to(dtype=reliability.dtype)
        )
        return adapter_batch, modality_mask, reliability

    def _apply_ablation(self, modality_mask: torch.Tensor, reliability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.disabled_modality_indices:
            return modality_mask, reliability
        modality_mask = modality_mask.clone()
        reliability = reliability.clone()
        for modality_index in self.disabled_modality_indices:
            modality_mask[..., modality_index] = 1.0
            reliability[..., modality_index] = 0.0
        return modality_mask, reliability

    def _apply_edge_ablation(self, edge_attr: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        if not self.disabled_edge_types or edge_attr.shape[-1] <= 3:
            return neighbor_mask
        edge_types = _edge_type_from_attr(edge_attr)
        disabled = torch.zeros_like(neighbor_mask, dtype=torch.bool)
        for edge_type in self.disabled_edge_types:
            disabled |= edge_types == edge_type
        return neighbor_mask & ~disabled

    def _disabled_indices(
        self,
        ablation_config: dict,
        key: str,
        lookup: dict[str, int],
        named_index: int,
    ) -> tuple[int, ...]:
        names: list[str] = []
        if self.ablation_name in self.named_ablations:
            names.extend(self.named_ablations[self.ablation_name][named_index])
        raw = ablation_config.get(key, ())
        if isinstance(raw, str):
            raw = [raw]
        names.extend(str(name) for name in raw)
        indices = []
        for name in names:
            normalized = name.strip().lower()
            if not normalized:
                continue
            if normalized not in lookup:
                raise ValueError(f"Unknown ablation {key} entry: {name}")
            value = lookup[normalized]
            if isinstance(value, tuple):
                indices.extend(value)
            else:
                indices.append(value)
        return tuple(sorted(set(indices)))

    def _disabled_bio_vec_indices(self, ablation_config: dict) -> tuple[int, ...]:
        if not bool(ablation_config.get("zero_disabled_bio_vec", True)):
            return ()
        names: list[str] = []
        if self.ablation_name in self.named_ablations:
            names.extend(self.named_ablations[self.ablation_name][0])
        raw = ablation_config.get("disabled_modalities", ())
        if isinstance(raw, str):
            raw = [raw]
        names.extend(str(name) for name in raw)
        raw_groups = ablation_config.get("disabled_bio_vec_groups", ())
        if isinstance(raw_groups, str):
            raw_groups = [raw_groups]
        names.extend(str(name) for name in raw_groups)
        raw_features = ablation_config.get("disabled_bio_vec_features", ())
        if isinstance(raw_features, str):
            raw_features = [raw_features]
        feature_names = [str(name).strip() for name in raw_features if str(name).strip()]
        for name in names:
            normalized = str(name).strip().lower()
            if normalized in self.bio_vec_groups:
                feature_names.extend(self.bio_vec_groups[normalized])
        indices: list[int] = []
        for feature_name in feature_names:
            if feature_name in BIO_VEC_NAMES:
                indices.append(BIO_VEC_NAMES.index(feature_name))
        return tuple(sorted(set(index for index in indices if index < self.bio_vec_dim)))


def _edge_type_from_attr(edge_attr: torch.Tensor) -> torch.Tensor:
    type_slice = edge_attr[..., 3:11]
    if type_slice.numel() == 0:
        return torch.zeros(edge_attr.shape[:-1], dtype=torch.long, device=edge_attr.device)
    return torch.argmax(type_slice, dim=-1)


def _merged_encoder_config(base: dict, override: dict | None, num_layers: int) -> dict:
    config = dict(base)
    config.update(dict(override or {}))
    config["num_layers"] = int(config.get("num_layers", num_layers))
    if override is None or "num_layers" not in override:
        config["num_layers"] = int(num_layers)
    return config
"""Protein workflow implementation."""

# Source: models/phase_stack_dpr.py


import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

from phaseflow.protein.model import PhaseFlowModel
from phaseflow.protein.model import SparseGraphTransformer


DEFAULT_PHASEFLOW_REPO = Path("external/phaseflow-peptide")
DEFAULT_PHASEFLOW_SITE_PACKAGES = Path("external/phaseflow-peptide/site-packages")
PROTENIX_EDGE_TYPE = 20
PHASEFLOW_LLPS_LAYER_KEY = "phaseflow_llps"
PHASEFLOW_LLPS_HIDDEN_LAYERS_KEY = "phaseflow_llps_hidden_layers"
PHASEFLOW_LLPS_HIDDEN_DIM_KEY = "phaseflow_llps_hidden_dim"
LEGACY_LLPS_PREFIX = "phase" + "gt"
LEGACY_LLPS_HIDDEN_LAYERS_KEY = LEGACY_LLPS_PREFIX + "_hidden_layers"
LEGACY_LLPS_HIDDEN_DIM_KEY = LEGACY_LLPS_PREFIX + "_hidden_dim"


class ScalarLayerMix(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        self.logits = nn.Parameter(torch.zeros(int(num_layers)))
        self.gamma = nn.Parameter(torch.ones(()))

    def forward(self, layers: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(layers) != int(self.logits.numel()):
            raise ValueError(f"Expected {self.logits.numel()} layers, got {len(layers)}")
        stacked = torch.stack([layer.float() for layer in layers], dim=0)
        weights = torch.softmax(self.logits.float(), dim=0).view(-1, 1, 1, 1)
        return self.gamma.float() * torch.sum(weights * stacked, dim=0)


class PhaseStackDPR(nn.Module):
    """Frozen PhaseFlow + frozen PhaseFlow with a trainable DPR graph head."""

    def __init__(
        self,
        *,
        llps_backbone: nn.Module | None = None,
        phaseflow: nn.Module,
        dpr_config: dict[str, Any] | None = None,
        llps_hidden_names: tuple[str, ...] = ("llps_residue_repr", "dpr_residue_repr"),
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()
        if llps_backbone is None:
            llps_backbone = legacy_kwargs.pop(LEGACY_LLPS_PREFIX, None)
        if legacy_kwargs:
            raise TypeError(f"unexpected keyword argument(s): {sorted(legacy_kwargs)}")
        if llps_backbone is None:
            raise ValueError("llps_backbone is required")
        self.llps_backbone = llps_backbone
        self.phaseflow = phaseflow
        self.dpr_config = _default_dpr_config() | dict(dpr_config or {})
        self.llps_hidden_names = tuple(llps_hidden_names)
        self.phaseflow_window_size = int(self.dpr_config.get("phaseflow_window_size", 20))
        self.phaseflow_stride = int(self.dpr_config.get("phaseflow_stride", 1))
        self.phaseflow_shape = str(self.dpr_config.get("phaseflow_shape", "4x4"))
        self.disable_protenix_for_dpr = bool(self.dpr_config.get("disable_protenix_for_dpr", True))
        self.disable_protenix_edges_for_dpr = bool(self.dpr_config.get("disable_protenix_edges_for_dpr", True))
        self.phaseflow_max_windows_per_sequence = int(self.dpr_config.get("phaseflow_max_windows_per_sequence", 0))
        self.phaseflow_window_batch_size = int(self.dpr_config.get("phaseflow_window_batch_size", 128))

        d_model = int(self.dpr_config["d_model"])
        llps_dim = int(self.dpr_config.get(PHASEFLOW_LLPS_HIDDEN_DIM_KEY, self.dpr_config.get(LEGACY_LLPS_HIDDEN_DIM_KEY, d_model)))
        phaseflow_dim = int(self.dpr_config.get("phaseflow_hidden_dim", getattr(phaseflow, "dim", d_model)))
        edge_dim = int(self.dpr_config.get("edge_dim", 32))

        self.layer_mix = nn.ModuleDict(
            {
                PHASEFLOW_LLPS_LAYER_KEY: ScalarLayerMix(len(self.llps_hidden_names)),
                "phaseflow": ScalarLayerMix(int(self.dpr_config.get("phaseflow_num_mixed_layers", 4))),
            }
        )
        self.dpr_stem = nn.Sequential(
            nn.Linear(llps_dim + phaseflow_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.dpr_blocks = SparseGraphTransformer(
            d_model=d_model,
            num_layers=int(self.dpr_config["num_graph_transformer_layers"]),
            num_heads=int(self.dpr_config["num_heads"]),
            edge_dim=edge_dim,
            ffn_dim=int(self.dpr_config["ffn_dim"]),
            dropout=float(self.dpr_config["dropout"]),
            num_edge_types=int(self.dpr_config.get("num_edge_types", 40)),
            relative_position_bins=int(self.dpr_config.get("relative_position_bins", 32)),
        )
        self.dpr_residue_head = _mlp_head(d_model, 1, float(self.dpr_config["dropout"]))
        self.dpr_boundary_head = _mlp_head(d_model, 2, float(self.dpr_config["dropout"]))

        self._freeze_backbones()

    def _freeze_backbones(self) -> None:
        self.llps_backbone.requires_grad_(False)
        self.phaseflow.requires_grad_(False)
        self.llps_backbone.eval()
        self.phaseflow.eval()

    def train(self, mode: bool = True) -> "PhaseStackDPR":
        super().train(mode)
        self.llps_backbone.eval()
        self.phaseflow.eval()
        self.layer_mix.train(mode)
        self.dpr_stem.train(mode)
        self.dpr_blocks.train(mode)
        self.dpr_residue_head.train(mode)
        self.dpr_boundary_head.train(mode)
        return self

    def forward_llps(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.llps_backbone(batch)

    def forward_phaseflow(self, phaseflow_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            velocity = self.phaseflow.forward_flow(
                input_ids=phaseflow_batch["input_ids"],
                attention_mask=phaseflow_batch["attention_mask"],
                phase_t=phaseflow_batch["phase_t"],
                phase_mask=phaseflow_batch["phase_mask"],
                time=phaseflow_batch["time"],
                seq_len=phaseflow_batch["seq_len"],
            )
        return {"velocity": velocity}

    def forward_dpr(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        seq_mask = batch["seq_mask"].bool()
        llps_layers, pf_layers = self.extract_frozen_dpr_layers(batch)
        llps_hidden = torch.nan_to_num(self.layer_mix[PHASEFLOW_LLPS_LAYER_KEY](llps_layers), nan=0.0, posinf=0.0, neginf=0.0)
        pf_hidden = torch.nan_to_num(self.layer_mix["phaseflow"](pf_layers), nan=0.0, posinf=0.0, neginf=0.0)
        x = self.dpr_stem(torch.cat([llps_hidden, pf_hidden], dim=-1))
        neighbor_mask = batch["neighbor_mask"].bool()
        if self.disable_protenix_edges_for_dpr and "neighbor_edge_type" in batch:
            neighbor_mask = neighbor_mask & (batch["neighbor_edge_type"].to(neighbor_mask.device) != PROTENIX_EDGE_TYPE)
        x = self.dpr_blocks(
            x=x,
            neighbors=batch["neighbors"].long(),
            edge_attr=batch["edge_attr"].float(),
            neighbor_mask=neighbor_mask,
            seq_mask=seq_mask,
        )
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        residue_logits = self.dpr_residue_head(x).squeeze(-1).masked_fill(~seq_mask, 0.0)
        boundary_logits = self.dpr_boundary_head(x).masked_fill(~seq_mask.unsqueeze(-1), 0.0)
        out = {
            "residue_logits": residue_logits,
            "residue_prob": torch.sigmoid(residue_logits.float()),
            "start_logits": boundary_logits[..., 0],
            "end_logits": boundary_logits[..., 1],
            "boundary_logits": boundary_logits,
            "dpr_hidden": x,
            "seq_mask": seq_mask,
        }
        if bool(batch.get("return_regions", False)):
            out["regions"] = self.decode_regions(residue_logits, boundary_logits, seq_mask)
        return out

    def extract_frozen_dpr_layers(self, batch: dict[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return DPR hidden taps without touching trainable DPR modules."""

        cached_llps = batch.get(PHASEFLOW_LLPS_HIDDEN_LAYERS_KEY, batch.get(LEGACY_LLPS_HIDDEN_LAYERS_KEY))
        if torch.is_tensor(cached_llps):
            llps_layers = [cached_llps[:, layer_index].detach() for layer_index in range(int(cached_llps.shape[1]))]
        else:
            dpr_llps_batch = self._llps_batch_for_dpr(batch)
            with torch.no_grad():
                llps_outputs = self.llps_backbone(dpr_llps_batch)
                llps_layers = [llps_outputs[name].detach() for name in self.llps_hidden_names if name in llps_outputs]
                if not llps_layers:
                    raise KeyError(f"PhaseFlow outputs did not include any of {self.llps_hidden_names}")
                while len(llps_layers) < len(self.llps_hidden_names):
                    llps_layers.append(llps_layers[-1])

        cached_pf = batch.get("phaseflow_hidden_layers")
        if torch.is_tensor(cached_pf):
            pf_layers = [cached_pf[:, layer_index].detach() for layer_index in range(int(cached_pf.shape[1]))]
        else:
            with torch.no_grad():
                pf_layers = [layer.detach() for layer in self.phaseflow_residue_hidden(batch)]
        return llps_layers, pf_layers

    def forward(
        self,
        batch: dict[str, Any] | None = None,
        *,
        task: str,
        phaseflow_batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        if task == "llps":
            if batch is None:
                raise ValueError("batch is required for task='llps'")
            return {"llps": self.forward_llps(batch)}
        if task == "phaseflow":
            if phaseflow_batch is None:
                raise ValueError("phaseflow_batch is required for task='phaseflow'")
            return {"phaseflow": self.forward_phaseflow(phaseflow_batch)}
        if task == "dpr":
            if batch is None:
                raise ValueError("batch is required for task='dpr'")
            return {"dpr": self.forward_dpr(batch)}
        raise ValueError(f"Unknown task: {task}")

    def phaseflow_residue_hidden(self, batch: dict[str, Any]) -> list[torch.Tensor]:
        tokenizer = _phaseflow_tokenizer()
        sequences = [str(seq) for seq in batch["sequences"]]
        lengths = batch["lengths"].detach().cpu().tolist()
        device = next(self.phaseflow.parameters()).device
        num_layers = int(self.layer_mix["phaseflow"].logits.numel())
        hidden_dim = int(getattr(self.phaseflow, "dim", self.dpr_config.get("phaseflow_hidden_dim", 256)))
        max_len = int(batch["seq_mask"].shape[1])
        accum = [torch.zeros(len(sequences), max_len, hidden_dim, device=device) for _ in range(num_layers)]
        counts = torch.zeros(len(sequences), max_len, 1, device=device)
        windows: list[tuple[int, int, str, int]] = []
        for batch_index, (sequence, length) in enumerate(zip(sequences, lengths, strict=False)):
            for start, end in _phaseflow_windows(int(length), self.phaseflow_window_size, self.phaseflow_stride):
                windows.append((batch_index, start, sequence[start:end], end - start))
                if self.phaseflow_max_windows_per_sequence > 0:
                    seen = sum(1 for item in windows if item[0] == batch_index)
                    if seen >= self.phaseflow_max_windows_per_sequence:
                        break
        if not windows:
            return accum
        for offset in range(0, len(windows), self.phaseflow_window_batch_size):
            chunk = windows[offset : offset + self.phaseflow_window_batch_size]
            encoded = []
            window_lengths = []
            for _, _, subseq, window_length in chunk:
                tokens = tokenizer.build_input_sequence(_phaseflow_safe_sequence(subseq), shape=self.phaseflow_shape)
                encoded.append(tokenizer.pad_sequence(tokens, int(self.phaseflow.max_seq_len)))
                window_lengths.append(window_length)
            input_ids = torch.tensor(encoded, dtype=torch.long, device=device)
            attention_mask = (input_ids != int(tokenizer.PAD_ID)).float()
            phase_t = torch.zeros(len(chunk), int(self.phaseflow.phase_dim), dtype=torch.float32, device=device)
            phase_mask = torch.ones_like(phase_t)
            time = torch.zeros(len(chunk), dtype=torch.float32, device=device)
            with torch.no_grad():
                layer_outputs = _phaseflow_forward_flow_layers(
                    self.phaseflow,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    phase_t=phase_t,
                    phase_mask=phase_mask,
                    time=time,
                )
            selected_layers = layer_outputs[-num_layers:]
            for local_index, (batch_index, start, _subseq, window_length) in enumerate(chunk):
                end = min(start + window_length, max_len)
                residue_slice = slice(1, 1 + (end - start))
                for layer_index, layer_hidden in enumerate(selected_layers):
                    accum[layer_index][batch_index, start:end] += layer_hidden[local_index, residue_slice].detach()
                counts[batch_index, start:end] += 1.0
        counts = counts.clamp_min(1.0)
        return [layer / counts for layer in accum]

    def _llps_batch_for_dpr(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not self.disable_protenix_for_dpr:
            return batch
        out = dict(batch)
        if "protenix_embed" in batch:
            out["protenix_embed"] = torch.zeros_like(batch["protenix_embed"])
        if "modality_mask" in batch:
            mask = batch["modality_mask"].clone()
            mask[..., 3] = 1.0
            out["modality_mask"] = mask
        if "reliability" in batch:
            reliability = batch["reliability"].clone()
            reliability[..., 3] = 0.0
            out["reliability"] = reliability
        return out

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    @staticmethod
    def decode_regions(
        residue_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        seq_mask: torch.Tensor,
        *,
        residue_threshold: float = 0.5,
        min_length: int = 3,
    ) -> list[list[dict[str, float | int]]]:
        residue_prob = torch.sigmoid(residue_logits.float()).detach().cpu()
        start_prob = torch.sigmoid(boundary_logits[..., 0].float()).detach().cpu()
        end_prob = torch.sigmoid(boundary_logits[..., 1].float()).detach().cpu()
        mask = seq_mask.detach().cpu().bool()
        all_regions: list[list[dict[str, float | int]]] = []
        for b in range(residue_prob.shape[0]):
            valid_len = int(mask[b].sum().item())
            regions: list[dict[str, float | int]] = []
            in_region = False
            start = 0
            for i in range(valid_len):
                active = bool(residue_prob[b, i].item() >= residue_threshold)
                if active and not in_region:
                    start = i
                    in_region = True
                if in_region and (not active or i == valid_len - 1):
                    end = i if active and i == valid_len - 1 else i - 1
                    if end - start + 1 >= min_length:
                        score = float(residue_prob[b, start : end + 1].mean().item())
                        regions.append(
                            {
                                "start": int(start),
                                "end": int(end),
                                "score": score,
                                "start_score": float(start_prob[b, start].item()),
                                "end_score": float(end_prob[b, end].item()),
                            }
                        )
                    in_region = False
            all_regions.append(regions)
        return all_regions


def load_phaseflow_llps_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> tuple[PhaseFlowModel, dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model = PhaseFlowModel(checkpoint["config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, checkpoint


def load_phaseflow_checkpoint(
    checkpoint_path: str | Path,
    *,
    repo_path: str | Path = DEFAULT_PHASEFLOW_REPO,
    dependency_site_packages: str | Path = DEFAULT_PHASEFLOW_SITE_PACKAGES,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    _prepare_phaseflow_imports(repo_path=repo_path, dependency_site_packages=dependency_site_packages)
    from phaseflow.peptide.model import PhaseFlow  # type: ignore

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model_config = dict(checkpoint["config"]["model"])
    model = PhaseFlow(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, checkpoint


def _prepare_phaseflow_imports(*, repo_path: str | Path, dependency_site_packages: str | Path) -> None:
    dep = str(Path(dependency_site_packages))
    repo = str(Path(repo_path))
    if dep not in sys.path:
        sys.path.append(dep)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    if "ot" not in sys.modules:
        ot_module = types.ModuleType("ot")

        def _unavailable_emd(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("ot.emd is unavailable in the PhaseStackDPR inference environment")

        ot_module.emd = _unavailable_emd  # type: ignore[attr-defined]
        sys.modules["ot"] = ot_module


def _phaseflow_tokenizer() -> Any:
    from phaseflow.peptide.tokenizer import AminoAcidTokenizer  # type: ignore

    return AminoAcidTokenizer()


def _phaseflow_forward_flow_layers(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    phase_t: torch.Tensor,
    phase_mask: torch.Tensor,
    time: torch.Tensor,
) -> list[torch.Tensor]:
    token_emb = model.embed_tokens(input_ids)
    phase_emb, phase_attn_mask = model.embed_phase(phase_t, phase_mask, time)
    x = torch.cat([token_emb, phase_emb], dim=1)
    extended_mask = torch.cat([attention_mask, phase_attn_mask], dim=1)
    phase_start_idx = int(model.max_seq_len)
    layers: list[torch.Tensor] = []
    for layer in model.transformer.layers:
        x = layer(
            x,
            model.transformer.rotary,
            extended_mask,
            phase_start_idx,
            None,
            bool(model.use_set_encoder),
        )
        layers.append(x[:, : input_ids.shape[1], :])
    x = model.transformer.final_norm(x)
    layers.append(x[:, : input_ids.shape[1], :])
    return layers


def _phaseflow_windows(length: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    length = int(length)
    window_size = max(int(window_size), 1)
    stride = max(int(stride), 1)
    if length <= window_size:
        return [(0, length)]
    starts = list(range(0, max(length - window_size + 1, 1), stride))
    last_start = length - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, min(start + window_size, length)) for start in starts]


def _phaseflow_safe_sequence(sequence: str) -> str:
    replacements = {"X": "G", "B": "N", "Z": "Q", "U": "C", "O": "K"}
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    out = []
    for aa in str(sequence).upper():
        aa = replacements.get(aa, aa)
        out.append(aa if aa in allowed else "G")
    return "".join(out)


def _mlp_head(d_model: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, 128),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(128, out_dim),
    )


def _default_dpr_config() -> dict[str, Any]:
    return {
        "d_model": 256,
        "num_graph_transformer_layers": 4,
        "num_heads": 8,
        "ffn_dim": 1024,
        "dropout": 0.10,
        "pre_layer_norm": True,
        "edge_dim": 32,
        "num_edge_types": 40,
        "relative_position_bins": 32,
        PHASEFLOW_LLPS_HIDDEN_DIM_KEY: 256,
        "phaseflow_hidden_dim": 256,
        "phaseflow_num_mixed_layers": 4,
        "phaseflow_window_size": 20,
        "phaseflow_stride": 1,
        "phaseflow_shape": "4x4",
        "disable_protenix_for_dpr": True,
        "disable_protenix_edges_for_dpr": True,
        "phaseflow_max_windows_per_sequence": 0,
        "phaseflow_window_batch_size": 128,
    }

# Source: models/dpr.py


import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from phaseflow.protein.model import (
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
from phaseflow.protein.model import PhaseFlowModel
from phaseflow.protein.data import TIER_TO_BAG_LABEL, TIER_TO_WEIGHT


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
    """Ordered 32-token bridge from protein LLPS residue states into frozen PhaseFlow."""

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
    from phaseflow.peptide.model import PhaseFlow  # type: ignore

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
