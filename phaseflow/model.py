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
        """
        super().__init__()

        self.dim = dim
        self.phase_dim = phase_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.time_embed_dim = time_embed_dim or dim

        # Token embedding for amino acids
        self.token_embed = nn.Embedding(vocab_size, dim)

        # Phase diagram projection (continuous → dim)
        self.phase_in_proj = nn.Linear(phase_dim, dim)
        self.phase_out_proj = nn.Linear(dim, phase_dim)

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
        # Initialize embeddings
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
        time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Embed phase diagram with optional time conditioning.

        Args:
            phase: (batch, phase_dim) phase diagram values
            time: (batch,) time values in [0, 1]

        Returns:
            (batch, 1, dim) phase embeddings
        """
        # Project phase to model dimension
        phase_emb = self.phase_in_proj(phase)  # (batch, dim)

        # Add time embedding if provided (for flow matching)
        if time is not None:
            time_emb = self.time_encoder(time)  # (batch, time_dim)
            time_emb = self.time_mlp(time_emb)  # (batch, dim)
            phase_emb = phase_emb + time_emb

        # Add sequence dimension
        phase_emb = phase_emb.unsqueeze(1)  # (batch, 1, dim)

        return phase_emb

    def forward_flow(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phase_t: torch.Tensor,
        time: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for flow matching (sequence → phase).

        Predicts the velocity field v(x_t, t) for flow matching.

        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: (batch, seq) attention mask
            phase_t: (batch, phase_dim) noisy phase at time t
            time: (batch,) diffusion time in [0, 1]
            seq_len: (batch,) original sequence lengths

        Returns:
            (batch, phase_dim) predicted velocity
        """
        batch = input_ids.shape[0]

        # Embed tokens
        token_emb = self.embed_tokens(input_ids)  # (batch, seq, dim)

        # Embed noisy phase with time conditioning
        phase_emb = self.embed_phase(phase_t, time)  # (batch, 1, dim)

        # Concatenate: [tokens..., phase]
        # Find where to insert phase token (after last real token)
        # For simplicity, append at the end
        x = torch.cat([token_emb, phase_emb], dim=1)  # (batch, seq+1, dim)

        # Extend attention mask for phase token
        phase_mask = torch.ones(batch, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([attention_mask, phase_mask], dim=1)

        # Phase token index for bidirectional attention
        phase_start_idx = self.max_seq_len  # Phase is after all sequence tokens

        # Forward through transformer
        hidden = self.transformer(x, extended_mask, phase_start_idx)

        # Extract phase token output (last position)
        phase_hidden = hidden[:, -1, :]  # (batch, dim)

        # Predict velocity
        velocity = self.velocity_head(phase_hidden)  # (batch, phase_dim)

        return velocity

    def forward_lm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for language modeling (phase → sequence).

        Predicts next token probabilities.

        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: (batch, seq) attention mask
            phase: (batch, phase_dim) phase diagram (clean)

        Returns:
            (batch, seq, vocab_size) logits
        """
        batch = input_ids.shape[0]

        # Embed phase (no time conditioning for LM)
        phase_emb = self.embed_phase(phase, time=None)  # (batch, 1, dim)

        # Embed tokens
        token_emb = self.embed_tokens(input_ids)  # (batch, seq, dim)

        # Concatenate: [phase, tokens...]
        # Phase conditions the sequence generation
        x = torch.cat([phase_emb, token_emb], dim=1)  # (batch, 1+seq, dim)

        # Extend attention mask
        phase_mask = torch.ones(batch, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([phase_mask, attention_mask], dim=1)

        # Forward through transformer (no bidirectional needed for LM)
        hidden = self.transformer(x, extended_mask, phase_start_idx=None)

        # Get token predictions (skip phase token)
        token_hidden = hidden[:, 1:, :]  # (batch, seq, dim)

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

        # Predict velocity
        v_pred = self.forward_flow(input_ids, attention_mask, x_t, t, seq_len)

        # Compute MSE loss with masking
        # Only compute loss on valid (non-missing) values
        diff = (v_pred - v_target) ** 2  # (batch, phase_dim)
        masked_diff = diff * phase_mask  # Zero out missing values

        # Mean over valid values only
        valid_count = phase_mask.sum(dim=-1).clamp(min=1)  # (batch,)
        loss_per_sample = masked_diff.sum(dim=-1) / valid_count
        loss = loss_per_sample.mean()

        metrics = {
            'flow_loss': loss.detach(),
            'valid_ratio': phase_mask.mean().detach(),
        }

        return loss, metrics

    def compute_lm_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phase: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute language modeling loss.

        Args:
            input_ids: (batch, seq) input token IDs
            attention_mask: (batch, seq) attention mask
            phase: (batch, phase_dim) conditioning phase diagram
            labels: (batch, seq) target token IDs (-100 for ignored)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Get logits
        logits = self.forward_lm(input_ids, attention_mask, phase)

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
                input_ids, attention_mask, phase, labels
            )
        else:
            lm_loss = torch.tensor(0.0, device=input_ids.device)
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
            v = self.forward_flow(input_ids, attention_mask, x, t_batch, seq_len)
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

            # Get logits
            logits = self.forward_lm(tokens, attention_mask, phase)

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

        # Decode sequences
        decoded = []
        for i in range(batch):
            seq_tokens = tokens[i].tolist()
            decoded.append(tokenizer.decode_sequence(seq_tokens))

        return tokens, decoded
