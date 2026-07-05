from __future__ import annotations

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
        edge_type = _edge_type_from_attr(edge_attr, self.edge_type_bias.num_embeddings)
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


def _edge_type_from_attr(edge_attr: torch.Tensor, num_edge_types: int) -> torch.Tensor:
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
