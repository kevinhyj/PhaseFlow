from __future__ import annotations

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
