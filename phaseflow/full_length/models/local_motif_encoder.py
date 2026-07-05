from __future__ import annotations

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
