"""
PhaseFlow: Main model combining Transformer backbone with Flow Matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from functools import partial
from einops import rearrange, repeat
from torchdiffeq import odeint

from .transformer import Transformer, SinusoidalPosEmb
from .tokenizer import AminoAcidTokenizer


class PhaseCNNEncoder(nn.Module):
    """Legacy phase diagram encoder. Linear(16, dim), single token output.

    Missing values are filled with 0. Mask is unused.
    """

    def __init__(self, embed_dim: int = 256, phase_dim: int = 16):
        super().__init__()
        self.phase_dim = phase_dim
        self.proj = nn.Linear(phase_dim, embed_dim)

    def forward(self, phase: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phase: (batch, 16) Phase values, missing values filled with 0
            mask: (batch, 16) Mask (unused, kept for API compatibility)

        Returns:
            (batch, 1, embed_dim) Phase embeddings
        """
        out = self.proj(phase)  # (batch, dim)
        return out.unsqueeze(1)  # (batch, 1, dim)


class SetPhaseEncoder(nn.Module):
    """Set-based phase diagram encoder.

    Each valid grid position becomes an independent token:
        token_i = MLP(sinusoidal(pssi_i)) + pos_emb[i]
    Missing positions are zeroed out and excluded via attention mask.
    """

    def __init__(self, dim: int = 256, phase_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.phase_dim = phase_dim

        # Sinusoidal encoding for continuous PSSI values (scalar → high-dim)
        self.value_sinusoidal = SinusoidalPosEmb(dim)

        # MLP to make sinusoidal features learnable
        self.value_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

        # 16 independent learnable position embeddings (no row/col decomposition)
        self.pos_emb = nn.Embedding(phase_dim, dim)

    def forward(
        self,
        phase: torch.Tensor,
        phase_mask: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            phase: (batch, 16) PSSI values, missing filled with 0
            phase_mask: (batch, 16) mask, 1=valid, 0=missing
            time_emb: (batch, dim) optional pre-computed time embedding

        Returns:
            tokens: (batch, 16, dim) phase token embeddings (invalid zeroed)
            attn_mask: (batch, 16) attention mask (1=valid, 0=pad)
        """
        batch = phase.shape[0]
        device = phase.device

        # Encode each position's value: scalar → sinusoidal → MLP
        flat_values = phase.reshape(-1)                          # (B*16,)
        flat_sin = self.value_sinusoidal(flat_values)            # (B*16, dim)
        flat_val = self.value_mlp(flat_sin)                      # (B*16, dim)
        val_emb = flat_val.reshape(batch, self.phase_dim, self.dim)  # (B, 16, dim)

        # Add position embeddings
        pos_ids = torch.arange(self.phase_dim, device=device)
        tokens = val_emb + self.pos_emb(pos_ids).unsqueeze(0)   # (B, 16, dim)

        # Add time embedding if provided (broadcast to all phase tokens)
        if time_emb is not None:
            tokens = tokens + time_emb.unsqueeze(1)              # (B, 16, dim)

        # Zero out invalid positions
        tokens = tokens * phase_mask.unsqueeze(-1)               # (B, 16, dim)

        return tokens, phase_mask


class PhaseFlow(nn.Module):
    """
    Transfusion-based model for bidirectional prediction between
    amino acid sequences and phase diagrams using Flow Matching.

    Architecture:
        [sos] [amino tokens...] [meta] [shape] [som] [phase tokens...] [eom] [eos]

    Supports:
        - Forward: sequence → phase diagram (flow matching)
        - Backward: phase diagram → sequence (language modeling)
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        vocab_size: int = 64,
        phase_dim: int = 16,  # 4x4 grid
        max_seq_len: int = 32,  # 序列5-20，加特殊token后约25，留余量
        dropout: float = 0.0,
        time_embed_dim: Optional[int] = None,
        use_set_encoder: bool = False,
    ):
        """
        Args:
            dim: Model dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            dim_head: Dimension per head
            vocab_size: Size of amino acid vocabulary
            phase_dim: Dimension of phase diagram (16 for 4x4)
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            time_embed_dim: Time embedding dimension (default: dim)
            use_set_encoder: Use set-based multi-token phase encoder
        """
        super().__init__()

        self.dim = dim
        self.phase_dim = phase_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.time_embed_dim = time_embed_dim or dim
        self.use_set_encoder = use_set_encoder
        self.phase_slots = phase_dim  # 16

        # Token embedding for amino acids
        self.token_embed = nn.Embedding(vocab_size, dim)

        # Phase diagram encoder
        if use_set_encoder:
            self.phase_encoder = SetPhaseEncoder(dim=dim, phase_dim=phase_dim)
            # Per-position velocity output: each phase hidden → MLP → 1D
            self.velocity_per_pos = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, 1),
            )
        else:
            self.phase_encoder = PhaseCNNEncoder(embed_dim=dim, phase_dim=phase_dim)

        # Time embedding for flow matching (added to phase tokens only)
        self.time_encoder = SinusoidalPosEmb(self.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

        # Transformer backbone
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            max_seq_len=max_seq_len,
            causal=True  # Causal with bidirectional for phase tokens
        )

        # Output head for language modeling (predicting next token)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Velocity head for flow matching (predicting dx/dt)
        self.velocity_head = nn.Linear(dim, phase_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.token_embed.weight, std=0.02)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs.

        Args:
            input_ids: (batch, seq) token IDs

        Returns:
            (batch, seq, dim) embeddings
        """
        return self.token_embed(input_ids)

    def embed_phase(
        self,
        phase: torch.Tensor,
        phase_mask: torch.Tensor,
        time: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed phase diagram with optional time conditioning.

        Args:
            phase: (batch, phase_dim) Phase diagram values, missing filled with 0
            phase_mask: (batch, phase_dim) Mask, 1 for valid, 0 for missing
            time: (batch,) Time values in [0, 1] for flow matching

        Returns:
            phase_emb: (batch, N, dim) Phase embeddings (N=16 for set, N=1 for legacy)
            phase_attn_mask: (batch, N) Attention mask for phase tokens
        """
        if self.use_set_encoder:
            # Compute time embedding once
            time_emb = None
            if time is not None:
                time_emb = self.time_mlp(self.time_encoder(time))  # (batch, dim)
            phase_emb, phase_attn_mask = self.phase_encoder(phase, phase_mask, time_emb)
            return phase_emb, phase_attn_mask
        else:
            # Legacy single-token path
            phase_emb = self.phase_encoder(phase, phase_mask)  # (batch, 1, dim)
            if time is not None:
                time_emb = self.time_encoder(time)
                time_emb = self.time_mlp(time_emb)
                phase_emb = phase_emb + time_emb.unsqueeze(1)
            return phase_emb, torch.ones(phase.shape[0], 1, device=phase.device)

    def forward_flow(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phase_t: torch.Tensor,
        phase_mask: torch.Tensor,
        time: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for flow matching (sequence → phase).

        Predicts the velocity field v(x_t, t) for flow matching.

        Layout:
            Legacy:  [seq_tokens(B, seq, dim)] [phase_token(B, 1, dim)]
            Set:     [seq_tokens(B, seq, dim)] [phase_tokens(B, 16, dim)]

        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: (batch, seq) attention mask
            phase_t: (batch, phase_dim) noisy phase at time t
            phase_mask: (batch, phase_dim) mask (1=valid, 0=missing)
            time: (batch,) diffusion time in [0, 1]
            seq_len: (batch,) original sequence lengths

        Returns:
            (batch, phase_dim) predicted velocity
        """
        batch = phase_t.shape[0]

        token_emb = self.embed_tokens(input_ids)  # (batch, seq, dim)

        # Embed noisy phase with time conditioning
        phase_emb, phase_attn_mask = self.embed_phase(phase_t, phase_mask, time)
        # phase_emb: (batch, N, dim)  N=16 for set, N=1 for legacy
        # phase_attn_mask: (batch, N)  1=valid, 0=missing

        n_phase = phase_emb.shape[1]

        # Concatenate: [seq_tokens, phase_tokens]
        x = torch.cat([token_emb, phase_emb], dim=1)

        # Build extended attention mask: [seq_mask, phase_attn_mask]
        extended_mask = torch.cat([attention_mask, phase_attn_mask], dim=1)

        # Phase tokens start after sequence tokens
        phase_start_idx = self.max_seq_len

        # Forward through transformer
        hidden = self.transformer(
            x, extended_mask, phase_start_idx,
            skip_phase_rope=self.use_set_encoder,
        )

        # Extract phase hidden states and predict velocity
        if self.use_set_encoder:
            # Per-position: each phase hidden independently outputs 1D velocity
            phase_hidden = hidden[:, -n_phase:, :]  # (batch, 16, dim)
            velocity = self.velocity_per_pos(phase_hidden).squeeze(-1)  # (batch, 16)
        else:
            # Legacy path: single phase token at last position
            phase_hidden = hidden[:, -1, :]  # (batch, dim)
            velocity = self.velocity_head(phase_hidden)  # (batch, phase_dim)

        return velocity

    def forward_lm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phase: torch.Tensor,
        phase_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for language modeling (phase → sequence).

        Predicts next token probabilities.

        Layout:
            Legacy:  [phase(B,1,dim)] [seq_tokens(B,seq,dim)]  — pure causal
            Set:     [phase(B,16,dim)] [seq_tokens(B,seq,dim)] — phase bidir, seq causal+attend phase

        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: (batch, seq) attention mask
            phase: (batch, phase_dim) phase diagram (clean)
            phase_mask: (batch, phase_dim) mask (1=valid, 0=missing)

        Returns:
            (batch, seq, vocab_size) logits
        """
        batch = phase.shape[0]

        # Embed phase (no time conditioning for LM)
        phase_emb, phase_attn_mask = self.embed_phase(phase, phase_mask, time=None)
        # phase_emb: (batch, N, dim)  N=16 for set, N=1 for legacy

        n_phase = phase_emb.shape[1]

        token_emb = self.embed_tokens(input_ids)  # (batch, seq, dim)

        # Concatenate: [phase_tokens, seq_tokens]
        x = torch.cat([phase_emb, token_emb], dim=1)

        # Build extended attention mask: [phase_attn_mask, seq_mask]
        extended_mask = torch.cat([phase_attn_mask, attention_mask], dim=1)

        if self.use_set_encoder:
            # Set path: phase tokens are bidirectional, seq tokens are causal + attend phase
            hidden = self.transformer(
                x, extended_mask,
                phase_start_idx=0, phase_end_idx=n_phase,
                skip_phase_rope=True,
            )
        else:
            # Legacy path: pure causal (single phase token at front)
            hidden = self.transformer(x, extended_mask, phase_start_idx=None)

        # Get token predictions (skip phase tokens)
        token_hidden = hidden[:, n_phase:, :]  # (batch, seq, dim)

        # Predict next tokens
        logits = self.lm_head(token_hidden)  # (batch, seq, vocab_size)

        return logits

    def compute_flow_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phase: torch.Tensor,
        phase_mask: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute flow matching loss.

        Uses CondOT path: x_t = (1-t) * x_0 + t * x_1
        Velocity target: v = x_1 - x_0

        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: (batch, seq) attention mask
            phase: (batch, phase_dim) target phase diagram
            phase_mask: (batch, phase_dim) mask (1=valid, 0=missing)
            seq_len: (batch,) sequence lengths

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch = phase.shape[0]
        device = phase.device

        # Sample time uniformly
        t = torch.rand(batch, device=device)

        # Sample noise x_0
        x_0 = torch.randn_like(phase)

        # Compute x_t = (1-t) * x_0 + t * x_1
        t_expand = t.unsqueeze(-1)  # (batch, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * phase

        # Target velocity: v = x_1 - x_0
        v_target = phase - x_0

        # Predict velocity (pass phase_mask to handle missing values)
        v_pred = self.forward_flow(input_ids, attention_mask, x_t, phase_mask, t, seq_len)

        # Compute MSE loss with masking
        # Only compute loss on valid (non-missing) values
        diff = (v_pred - v_target) ** 2  # (batch, phase_dim)
        masked_diff = diff * phase_mask  # Zero out missing values

        # Quadratic weighting: w = (n / 16)^2
        # Complete (n=16): w=1.0, Half (n=8): w=0.25, Single (n=1): w~0.004
        valid_count = phase_mask.sum(dim=-1).clamp(min=1)  # (batch,)
        weight = (valid_count / 16.0) ** 2

        # Loss per sample: mean over valid positions, then apply weight
        loss_per_sample = (masked_diff.sum(dim=-1) / valid_count) * weight

        # Mean over all samples
        loss = loss_per_sample.mean()

        metrics = {
            'flow_loss': loss.detach(),
            'valid_ratio': phase_mask.mean().detach(),
            'avg_weight': weight.mean().detach(),  # Track average weight for monitoring
        }

        return loss, metrics

    def compute_lm_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phase: torch.Tensor,
        phase_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute language modeling loss.

        Args:
            input_ids: (batch, seq) input token IDs
            attention_mask: (batch, seq) attention mask
            phase: (batch, phase_dim) conditioning phase diagram
            phase_mask: (batch, phase_dim) mask (1=valid, 0=missing)
            labels: (batch, seq) target token IDs (-100 for ignored)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Get logits
        logits = self.forward_lm(input_ids, attention_mask, phase, phase_mask)

        # Compute cross-entropy loss
        # logits: (batch, seq, vocab) -> (batch * seq, vocab)
        # labels: (batch, seq) -> (batch * seq,)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction='mean'
        )

        # Compute perplexity
        with torch.no_grad():
            valid_mask = labels != -100
            valid_logits = logits[valid_mask]
            valid_labels = labels[valid_mask]
            if len(valid_labels) > 0:
                nll = F.cross_entropy(valid_logits, valid_labels, reduction='mean')
                perplexity = torch.exp(nll)
            else:
                perplexity = torch.tensor(float('inf'), device=logits.device)

        metrics = {
            'lm_loss': loss.detach(),
            'perplexity': perplexity.detach(),
        }

        return loss, metrics

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phase: torch.Tensor,
        phase_mask: torch.Tensor,
        seq_len: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        flow_weight: float = 1.0,
        lm_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Combined forward pass for both tasks.

        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: (batch, seq) attention mask
            phase: (batch, phase_dim) phase diagram
            phase_mask: (batch, phase_dim) mask (1=valid, 0=missing)
            seq_len: (batch,) sequence lengths
            labels: (batch, seq) LM labels (optional)
            flow_weight: Weight for flow matching loss
            lm_weight: Weight for language modeling loss

        Returns:
            Dict with 'loss' and various metrics
        """
        # Flow matching loss (seq → phase)
        flow_loss, flow_metrics = self.compute_flow_loss(
            input_ids, attention_mask, phase, phase_mask, seq_len
        )

        # Language modeling loss (phase → seq) if labels provided
        if labels is not None:
            lm_loss, lm_metrics = self.compute_lm_loss(
                input_ids, attention_mask, phase, phase_mask, labels
            )
        else:
            lm_loss = torch.tensor(0.0, device=phase.device)
            lm_metrics = {'lm_loss': lm_loss, 'perplexity': torch.tensor(0.0)}

        # Combined loss
        total_loss = flow_weight * flow_loss + lm_weight * lm_loss

        return {
            'loss': total_loss,
            'flow_loss': flow_metrics['flow_loss'],
            'lm_loss': lm_metrics['lm_loss'],
            'perplexity': lm_metrics['perplexity'],
            'valid_ratio': flow_metrics['valid_ratio'],
        }

    @torch.no_grad()
    def generate_phase(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_len: torch.Tensor,
        method: str = 'dopri5',
        atol: float = 1e-5,
        rtol: float = 1e-5,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Generate phase diagram from sequence using ODE integration.

        Uses torchdiffeq for high-quality ODE solving.

        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: (batch, seq) attention mask
            seq_len: (batch,) sequence lengths
            method: ODE solver method ('dopri5', 'euler', 'midpoint', etc.)
            atol: Absolute tolerance for adaptive methods
            rtol: Relative tolerance for adaptive methods
            return_trajectory: Whether to return full trajectory

        Returns:
            (batch, phase_dim) generated phase diagram
            or (batch, T, phase_dim) if return_trajectory (T depends on solver)
        """
        batch = input_ids.shape[0]
        device = input_ids.device

        # Initialize with noise (t=0)
        x_init = torch.randn(batch, self.phase_dim, device=device)

        # Create all-ones mask for phase generation (predicting all values)
        phase_mask = torch.ones(batch, self.phase_dim, device=device)

        # Define ODE function: dx/dt = velocity(x, t)
        def ode_func(t, x):
            """
            Velocity function for the ODE.
            Args:
                t: scalar time value
                x: (batch, phase_dim) current state
            Returns:
                v: (batch, phase_dim) velocity
            """
            # t is a scalar, broadcast to batch
            t_batch = torch.full((batch,), t.item() if t.dim() == 0 else t, device=device)
            # Predict velocity field
            v = self.forward_flow(input_ids, attention_mask, x, phase_mask, t_batch, seq_len)
            return v

        # Time grid from t=0 to t=1
        if return_trajectory:
            # More time points for visualization
            t_span = torch.linspace(0.0, 1.0, 50, device=device)
        else:
            # Just start and end
            t_span = torch.tensor([0.0, 1.0], device=device)

        # Solve ODE using torchdiffeq
        trajectory = odeint(
            ode_func,
            x_init,
            t_span,
            method=method,
            atol=atol,
            rtol=rtol,
        )

        if return_trajectory:
            # trajectory shape: (T, batch, phase_dim) -> (batch, T, phase_dim)
            return trajectory.permute(1, 0, 2)
        else:
            # Return final state only
            return trajectory[-1]

    @torch.no_grad()
    def generate_sequence(
        self,
        phase: torch.Tensor,
        tokenizer: AminoAcidTokenizer,
        max_len: int = 25,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Generate amino acid sequence from phase diagram.

        Args:
            phase: (batch, phase_dim) phase diagram
            tokenizer: Tokenizer for decoding
            max_len: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
            top_p: Top-p (nucleus) sampling (optional)

        Returns:
            Tuple of (token_ids, decoded_sequences)
        """
        batch = phase.shape[0]
        device = phase.device

        # Start with SOS token
        tokens = torch.full((batch, 1), tokenizer.SOS_ID, dtype=torch.long, device=device)

        for _ in range(max_len):
            # Create attention mask
            attention_mask = torch.ones_like(tokens)

            # Get logits (all-ones mask: generating full phase diagram)
            phase_mask = torch.ones(phase.shape[0], phase.shape[1], device=phase.device)
            logits = self.forward_lm(tokens, attention_mask, phase, phase_mask)

            # Get next token logits (last position)
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

            # Check for EOS
            if (next_token == tokenizer.EOS_ID).all():
                break

            # Additional stop conditions for special tokens (safety net)
            # These should not appear in generated sequences after EOS fix
            if (next_token == tokenizer.META_ID).all():
                break
            if (next_token == tokenizer.SOM_ID).all():
                break
            if (next_token == tokenizer.PAD_ID).all():
                break

        # Decode sequences
        decoded = []
        for i in range(batch):
            seq_tokens = tokens[i].tolist()
            decoded.append(tokenizer.decode_sequence(seq_tokens))

        return tokens, decoded
