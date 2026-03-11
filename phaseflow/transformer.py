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
        phase_end_idx: Optional[int] = None,
        skip_phase_rope: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq, dim)
            rotary_emb: Rotary embedding instance
            attention_mask: Attention mask (batch, seq) or (batch, seq, seq)
            phase_start_idx: Index where phase tokens start (for bidirectional attention)
            phase_end_idx: Index where phase tokens end (for LM direction layout)
            skip_phase_rope: If True, only apply RoPE to seq tokens, skip phase tokens

        Returns:
            Output tensor (batch, seq, dim)
        """
        batch, seq_len, _ = x.shape

        # Project to queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (b, n, h*d) -> (b, h, n, d)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # Apply RoPE selectively: skip phase tokens when using set encoder
        if skip_phase_rope and phase_end_idx is not None:
            # LM direction: [phase(0..E-1), seq(E..end)]
            # Only rotate seq tokens; seq RoPE positions start from 0
            E = phase_end_idx
            q_seq = rotary_emb.rotate_queries_or_keys(q[:, :, E:, :])
            k_seq = rotary_emb.rotate_queries_or_keys(k[:, :, E:, :])
            q = torch.cat([q[:, :, :E, :], q_seq], dim=2)
            k = torch.cat([k[:, :, :E, :], k_seq], dim=2)
        elif skip_phase_rope and phase_start_idx is not None:
            # Flow direction: [seq(0..S-1), phase(S..end)]
            # Only rotate seq tokens
            S = phase_start_idx
            q_seq = rotary_emb.rotate_queries_or_keys(q[:, :, :S, :])
            k_seq = rotary_emb.rotate_queries_or_keys(k[:, :, :S, :])
            q = torch.cat([q_seq, q[:, :, S:, :]], dim=2)
            k = torch.cat([k_seq, k[:, :, S:, :]], dim=2)
        else:
            # Legacy: apply RoPE to all positions
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)

        # Compute attention scores
        scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # Build attention mask
        mask = self._build_attention_mask(batch, seq_len, x.device, phase_start_idx, phase_end_idx)

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
        phase_start_idx: Optional[int] = None,
        phase_end_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Build attention mask with causal + bidirectional for phase tokens.

        Three modes:
        1. phase_start_idx=None, phase_end_idx=None: Pure causal (legacy LM).
        2. phase_start_idx=S, phase_end_idx=None: Flow direction.
           [seq(0..S-1) causal] [phase(S..end) bidirectional + can attend all seq].
        3. phase_start_idx=0, phase_end_idx=E: LM direction with set encoder.
           [phase(0..E-1) bidirectional] [seq(E..end) causal + can attend all phase].
        """
        if self.causal:
            # Start with causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

            if phase_start_idx is not None and phase_end_idx is not None:
                # Mode 3: LM direction — [phase(0..E-1) bidir] [seq(E..end) causal+attend phase]
                E = phase_end_idx
                mask[:E, :E] = True       # phase tokens attend to each other bidirectionally
                mask[E:, :E] = True        # seq tokens can attend to all phase tokens

            elif phase_start_idx is not None and phase_end_idx is None:
                # Mode 2: Flow direction — [seq(0..S-1)] [phase(S..end) bidir]
                if phase_start_idx < seq_len:
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
        phase_end_idx: Optional[int] = None,
        skip_phase_rope: bool = False,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(
            self.attn_norm(x),
            rotary_emb,
            attention_mask,
            phase_start_idx,
            phase_end_idx,
            skip_phase_rope,
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

        # Rotary embeddings (from rotary_embedding_torch)
        # dim_head: dimension per head
        # theta: base for computing rotary frequencies (default 10000)
        # use_xpos: use extended rotary positional embeddings (default True for better extrapolation)
        self.rotary = RotaryEmbedding(
            dim=dim_head,
            theta=10000.0,
            use_xpos=False,  # Disable xpos for simplicity
        )

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
        phase_end_idx: Optional[int] = None,
        skip_phase_rope: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input embeddings (batch, seq, dim)
            attention_mask: Attention mask (batch, seq)
            phase_start_idx: Index where phase tokens start
            phase_end_idx: Index where phase tokens end (for LM direction with set encoder)
            skip_phase_rope: If True, only apply RoPE to seq tokens, skip phase tokens

        Returns:
            Output embeddings (batch, seq, dim)
        """
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, self.rotary, attention_mask, phase_start_idx, phase_end_idx, skip_phase_rope)

        # Final normalization
        x = self.final_norm(x)

        return x
