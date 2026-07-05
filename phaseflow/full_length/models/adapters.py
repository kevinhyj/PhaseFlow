from __future__ import annotations

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
