from __future__ import annotations

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
