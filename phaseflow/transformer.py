"""
Transformer backbone for PhaseFlow.
Adapted from transfusion-pytorch with modifications for phase diagram prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch,) containing positions or times

        Returns:
            Embeddings of shape (batch, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# RotaryEmbedding, apply_rotary_emb 现在从 rotary_embedding_torch 导入


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.weight


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(x * W1) * (x * W3)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    """Multi-head attention with rotary embeddings."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        causal: bool = True
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.causal = causal

        inner_dim = heads * dim_head

        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
        phase_start_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq, dim)
            rotary_emb: Rotary embedding instance
            attention_mask: Attention mask (batch, seq) or (batch, seq, seq)
            phase_start_idx: Index where phase tokens start (for bidirectional attention)

        Returns:
            Output tensor (batch, seq, dim)
        """
        batch, seq_len, _ = x.shape

        # Project to queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # Apply rotary embeddings using the library
        # rotary_embedding_torch expects shape (batch, heads, seq, dim)
        q = apply_rotary_emb(rotary_emb, q)
        k = apply_rotary_emb(rotary_emb, k)

        # Compute attention scores
        scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # Build attention mask
        mask = self._build_attention_mask(batch, seq_len, x.device, phase_start_idx)

        if attention_mask is not None:
            # Expand attention mask to (batch, 1, 1, seq)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            mask = mask & attention_mask.bool()

        # Apply mask
        scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Compute output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)

        return out

    def _build_attention_mask(
        self,
        batch: int,
        seq_len: int,
        device: torch.device,
        phase_start_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Build attention mask with causal + bidirectional for phase tokens.

        The mask allows:
        - Causal attention for sequence tokens (can only attend to previous)
        - Bidirectional attention within phase diagram tokens
        - Phase tokens can attend to all sequence tokens
        """
        if self.causal:
            # Start with causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

            # If phase_start_idx is specified, allow bidirectional within phase region
            if phase_start_idx is not None and phase_start_idx < seq_len:
                # Phase tokens can attend to each other bidirectionally
                mask[phase_start_idx:, phase_start_idx:] = True
        else:
            mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)

        # Expand for batch dimension
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
        mask = mask.expand(batch, 1, -1, -1)

        return mask


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = True
    ):
        super().__init__()

        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            causal=causal
        )

        self.ff_norm = RMSNorm(dim)
        self.ff = FeedForward(
            dim=dim,
            hidden_dim=dim * ff_mult,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
        phase_start_idx: Optional[int] = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(
            self.attn_norm(x),
            rotary_emb,
            attention_mask,
            phase_start_idx
        )

        # Pre-norm feed-forward with residual
        x = x + self.ff(self.ff_norm(x))

        return x


class Transformer(nn.Module):
    """Full transformer model."""

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        ff_mult: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 128,
        causal: bool = True
    ):
        """
        Args:
            dim: Model dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            dim_head: Dimension per head
            ff_mult: Feed-forward hidden dimension multiplier
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            causal: Whether to use causal attention
        """
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.max_seq_len = max_seq_len

        # Rotary embeddings
        self.rotary = RotaryEmbedding(dim_head)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                causal=causal
            )
            for _ in range(depth)
        ])

        # Final normalization
        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        phase_start_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input embeddings (batch, seq, dim)
            attention_mask: Attention mask (batch, seq)
            phase_start_idx: Index where phase tokens start

        Returns:
            Output embeddings (batch, seq, dim)
        """
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, self.rotary, attention_mask, phase_start_idx)

        # Final normalization
        x = self.final_norm(x)

        return x
