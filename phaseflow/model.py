"""
PhaseFlow: Main model combining Transformer backbone with Flow Matching / DDPM.
"""

import math
import numpy as np
import ot
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
        diffusion_type: str = "flow_matching",  # "flow_matching" | "ddpm"
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",  # "cosine" | "linear"
        use_ot_coupling: bool = False,  # OT minibatch coupling for flow matching
        use_quadratic_weighting: bool = True,  # Quadratic reliability weighting for missing data
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
            diffusion_type: "flow_matching" or "ddpm"
            num_timesteps: Number of DDPM diffusion steps (default: 1000)
            beta_schedule: "cosine" or "linear" beta schedule for DDPM
            use_ot_coupling: Use minibatch OT to pair noise x_0 with target x_1,
                             reducing trajectory crossings and variance collapse.
            use_quadratic_weighting: Use (n/16)^2 weighting for samples with missing values.
                                     If False, all samples weighted equally (uniform).
        """
        super().__init__()

        self.dim = dim
        self.phase_dim = phase_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.time_embed_dim = time_embed_dim or dim
        self.use_set_encoder = use_set_encoder
        self.phase_slots = phase_dim  # 16
        self.diffusion_type = diffusion_type
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.use_ot_coupling = use_ot_coupling
        self.use_quadratic_weighting = use_quadratic_weighting

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

        # Initialize DDPM noise schedule if needed
        if self.diffusion_type == "ddpm":
            self._init_noise_schedule()

        # Initialize weights
        self._init_weights()

    def _init_noise_schedule(self):
        """Initialize DDPM noise schedule buffers (betas, alphas, etc.)."""
        T = self.num_timesteps

        if self.beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, T)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
        elif self.beta_schedule == "cosine":
            # Improved DDPM cosine schedule (Nichol & Dhariwal, 2021), s=0.008
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float64)
            f_t = torch.cos((steps / T + s) / (1 + s) * (math.pi / 2)) ** 2
            # alphas_cumprod directly from cosine formula (T values, index 0~T-1)
            alphas_cumprod = (f_t[1:] / f_t[0]).float()
            # Derive betas and alphas from alphas_cumprod
            alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
            betas = (1 - alphas_cumprod / alphas_cumprod_prev).clamp(max=0.999)
            alphas = 1.0 - betas
        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")

        # Clamp alphas_cumprod to prevent division by zero at tail
        alphas_cumprod = alphas_cumprod.clamp(min=1e-6)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

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
            Tuple of (mse_loss, metrics_dict)
        """
        batch = phase.shape[0]
        device = phase.device

        # Sample time uniformly
        t = torch.rand(batch, device=device)

        # Sample noise x_0
        x_0 = torch.randn_like(phase)

        # Minibatch OT coupling: re-pair x_0 with x_1 to reduce trajectory crossings
        if self.use_ot_coupling:
            with torch.no_grad():
                # Only use valid (non-missing) dimensions for cost computation.
                # For missing positions, substitute x_0 values so their contribution
                # to the distance is zero (x_0[i] - x_0[i] = 0).
                phase_for_cost = phase.clone()
                phase_for_cost[phase_mask == 0] = x_0[phase_mask == 0]
                cost = torch.cdist(x_0, phase_for_cost, p=2).pow(2).cpu().numpy()  # (B, B)
                a = np.ones(batch) / batch
                b = np.ones(batch) / batch
                # Use exact OT (no regularization) to get a meaningful transport plan
                T = ot.emd(a, b, cost)                         # (B, B)
                indices = torch.from_numpy(T).argmax(dim=0).to(device)  # (B,)
                x_0 = x_0[indices]

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
        if self.use_quadratic_weighting:
            weight = (valid_count / 16.0) ** 2
        else:
            weight = torch.ones_like(valid_count)

        # Loss per sample: mean over valid positions, then apply weight
        loss_per_sample = (masked_diff.sum(dim=-1) / valid_count) * weight

        # Mean over all samples
        mse_loss = loss_per_sample.mean()

        metrics = {
            'flow_loss': mse_loss.detach(),
            'valid_ratio': phase_mask.mean().detach(),
            'avg_weight': weight.mean().detach(),
        }

        return mse_loss, metrics

    def compute_ddpm_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phase: torch.Tensor,
        phase_mask: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute DDPM denoising loss.

        Forward: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        Target: eps (noise)

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

        # Sample random integer timesteps
        t_int = torch.randint(0, self.num_timesteps, (batch,), device=device)

        # Normalize to [0, 1] for SinusoidalPosEmb (reuse time encoder)
        t = t_int.float() / self.num_timesteps

        # Sample noise
        eps = torch.randn_like(phase)

        # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
        sqrt_ab = self.sqrt_alphas_cumprod[t_int].unsqueeze(-1)       # (batch, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t_int].unsqueeze(-1)  # (batch, 1)
        x_t = sqrt_ab * phase + sqrt_one_minus_ab * eps

        # Predict noise using the same forward_flow backbone
        eps_pred = self.forward_flow(input_ids, attention_mask, x_t, phase_mask, t, seq_len)

        # MSE loss with masking (same weighting scheme as flow matching)
        diff = (eps_pred - eps) ** 2
        masked_diff = diff * phase_mask

        valid_count = phase_mask.sum(dim=-1).clamp(min=1)
        if self.use_quadratic_weighting:
            weight = (valid_count / 16.0) ** 2
        else:
            weight = torch.ones_like(valid_count)

        loss_per_sample = (masked_diff.sum(dim=-1) / valid_count) * weight
        loss = loss_per_sample.mean()

        metrics = {
            'flow_loss': loss.detach(),  # reuse key to avoid changing logging
            'valid_ratio': phase_mask.mean().detach(),
            'avg_weight': weight.mean().detach(),
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
        # Diffusion loss (seq → phase): flow matching or DDPM
        if self.diffusion_type == "ddpm":
            flow_loss, flow_metrics = self.compute_ddpm_loss(
                input_ids, attention_mask, phase, phase_mask, seq_len
            )
        else:
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
        **kwargs,
    ) -> torch.Tensor:
        """Generate phase diagram from sequence.

        Dispatches to ODE (flow matching) or DDPM/DDIM sampling based on diffusion_type.

        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: (batch, seq) attention mask
            seq_len: (batch,) sequence lengths
            method: ODE solver method (flow matching only)
            atol: Absolute tolerance (flow matching only)
            rtol: Relative tolerance (flow matching only)
            return_trajectory: Whether to return full trajectory (flow matching only)
            **kwargs: DDPM-specific args: num_steps, use_ddim, eta

        Returns:
            (batch, phase_dim) generated phase diagram
        """
        if self.diffusion_type == "ddpm":
            return self.generate_phase_ddpm(
                input_ids, attention_mask, seq_len, **kwargs
            )
        return self._generate_phase_flow(
            input_ids, attention_mask, seq_len,
            method=method, atol=atol, rtol=rtol,
            return_trajectory=return_trajectory,
        )

    @torch.no_grad()
    def _generate_phase_flow(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_len: torch.Tensor,
        method: str = 'dopri5',
        atol: float = 1e-5,
        rtol: float = 1e-5,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Generate phase diagram using Flow Matching ODE integration."""
        batch = input_ids.shape[0]
        device = input_ids.device

        # Initialize with noise (t=0)
        x_init = torch.randn(batch, self.phase_dim, device=device)

        # Create all-ones mask for phase generation (predicting all values)
        phase_mask = torch.ones(batch, self.phase_dim, device=device)

        # Define ODE function: dx/dt = velocity(x, t)
        def ode_func(t, x):
            t_batch = torch.full((batch,), t.item() if t.dim() == 0 else t, device=device)
            v = self.forward_flow(input_ids, attention_mask, x, phase_mask, t_batch, seq_len)
            return v

        # Time grid from t=0 to t=1
        if return_trajectory:
            t_span = torch.linspace(0.0, 1.0, 50, device=device)
        else:
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
            return trajectory.permute(1, 0, 2)
        else:
            return trajectory[-1]

    @torch.no_grad()
    def generate_phase_ddpm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_len: torch.Tensor,
        num_steps: Optional[int] = None,
        use_ddim: bool = False,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate phase diagram using DDPM or DDIM sampling.

        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: (batch, seq) attention mask
            seq_len: (batch,) sequence lengths
            num_steps: Number of sampling steps (None = full T steps for DDPM)
            use_ddim: Use DDIM deterministic sampling (faster)
            eta: DDIM stochasticity (0=deterministic, 1=DDPM)

        Returns:
            (batch, phase_dim) generated phase diagram
        """
        batch = input_ids.shape[0]
        device = input_ids.device
        T = self.num_timesteps

        # Start from pure noise
        x = torch.randn(batch, self.phase_dim, device=device)
        phase_mask = torch.ones(batch, self.phase_dim, device=device)

        if use_ddim:
            # DDIM: sub-sampled timestep sequence for fast sampling
            steps = num_steps or 50
            # Uniformly spaced timesteps from T-1 down to 0
            timesteps = torch.linspace(T - 1, 0, steps + 1, device=device).long()

            for i in range(steps):
                t_cur = timesteps[i]
                t_next = timesteps[i + 1]

                t_batch = torch.full((batch,), t_cur.item(), device=device)
                t_norm = t_batch / T  # normalize to [0,1] for time encoder

                # Predict noise
                eps_pred = self.forward_flow(
                    input_ids, attention_mask, x, phase_mask, t_norm, seq_len
                )

                # DDIM update
                alpha_bar_t = self.alphas_cumprod[t_cur]
                alpha_bar_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

                # Predict x_0 (clamp to prevent numerical explosion at high noise levels)
                x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t).clamp(min=1e-3)
                x0_pred = x0_pred.clamp(-5, 5)

                # Direction pointing to x_t
                sigma = eta * torch.sqrt(((1 - alpha_bar_next) / (1 - alpha_bar_t).clamp(min=1e-8)) * (1 - alpha_bar_t / alpha_bar_next.clamp(min=1e-8)))
                dir_xt = torch.sqrt((1 - alpha_bar_next - sigma ** 2).clamp(min=0)) * eps_pred

                # DDIM step
                x = torch.sqrt(alpha_bar_next) * x0_pred + dir_xt
                if eta > 0 and t_next > 0:
                    x = x + sigma * torch.randn_like(x)

        else:
            # Full DDPM reverse process: T -> 0
            steps = num_steps or T
            # Use all timesteps or sub-sample
            if steps < T:
                timestep_seq = torch.linspace(T - 1, 0, steps, device=device).long()
            else:
                timestep_seq = torch.arange(T - 1, -1, -1, device=device)

            for t_cur in timestep_seq:
                t_batch = torch.full((batch,), t_cur.item(), device=device)
                t_norm = t_batch / T

                eps_pred = self.forward_flow(
                    input_ids, attention_mask, x, phase_mask, t_norm, seq_len
                )

                alpha_t = self.alphas[t_cur]
                alpha_bar_t = self.alphas_cumprod[t_cur]
                beta_t = self.betas[t_cur]

                # DDPM reverse step: x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_pred) + sigma * z
                x = self.sqrt_recip_alphas[t_cur] * (x - beta_t / self.sqrt_one_minus_alphas_cumprod[t_cur].clamp(min=1e-4) * eps_pred)

                if t_cur > 0:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(self.posterior_variance[t_cur]) * noise

        return x

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

    @torch.no_grad()
    def compute_sequence_log_likelihood(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phase: torch.Tensor,
        phase_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sequence log-likelihood P(seq | phase).

        Args:
            input_ids: (batch, seq) token IDs [SOS, AA1, ..., AAn, EOS, PAD, ...]
            attention_mask: (batch, seq) attention mask
            phase: (batch, phase_dim) conditioning phase diagram
            phase_mask: (batch, phase_dim) mask (1=valid, 0=missing)

        Returns:
            (batch,) average per-token log-likelihood for each sequence
        """
        logits = self.forward_lm(input_ids, attention_mask, phase, phase_mask)

        # Shift: logits[i] predicts input_ids[i+1]
        shift_logits = logits[:, :-1, :]   # (batch, seq-1, vocab)
        shift_labels = input_ids[:, 1:]     # (batch, seq-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_ll = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (batch, seq-1)

        # Mask: only count real tokens (not PAD, not positions after EOS)
        valid = (shift_labels != 20) & (shift_labels != -100)  # PAD_ID=20
        valid_count = valid.sum(dim=-1).clamp(min=1)

        per_seq_ll = (token_ll * valid).sum(dim=-1) / valid_count  # (batch,)
        return per_seq_ll
