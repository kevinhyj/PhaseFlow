"""
Utility functions for PhaseFlow.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy import stats


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create a logger.

    Args:
        name: Logger name
        log_file: Optional path to log file

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffix.

    Args:
        n: Number to format

    Returns:
        Formatted string
    """
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    scheduler: Optional[Any] = None,
    **kwargs
):
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        **kwargs: Additional items to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    checkpoint.update(kwargs)

    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to load to

    Returns:
        Checkpoint dictionary with epoch, loss, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def create_labels_for_lm(
    input_ids: torch.Tensor,
    pad_token_id: int = 20,
    som_token_id: int = 24,
    eos_token_id: int = 22
) -> torch.Tensor:
    """Create labels for language modeling by shifting input_ids.

    The target for position i is the token at position i+1.
    Padding tokens and positions after EOS are ignored (-100).

    Args:
        input_ids: (batch, seq) input token IDs
        pad_token_id: Padding token ID
        som_token_id: Start of modality token ID
        eos_token_id: End of sequence token ID

    Returns:
        (batch, seq) labels tensor
    """
    # Shift: labels[i] = input_ids[i+1]
    labels = input_ids[:, 1:].clone()

    # Pad the last position
    batch_size = input_ids.shape[0]
    padding = torch.full((batch_size, 1), -100, dtype=labels.dtype, device=labels.device)
    labels = torch.cat([labels, padding], dim=1)

    # Mask padding tokens
    labels[input_ids == pad_token_id] = -100

    return labels


def compute_metrics(
    pred_phase: torch.Tensor,
    target_phase: torch.Tensor,
    phase_mask: torch.Tensor
) -> Dict[str, float]:
    """Compute evaluation metrics for phase diagram prediction.

    Args:
        pred_phase: (batch, 16) predicted phase values
        target_phase: (batch, 16) target phase values
        phase_mask: (batch, 16) mask (1=valid, 0=missing)

    Returns:
        Dictionary of metrics
    """
    # Mask invalid values
    valid = phase_mask.bool()

    if not valid.any():
        return {
            'mse': float('nan'),
            'mae': float('nan'),
            'rmse': float('nan'),
            'pearson': float('nan'),
            'spearman': float('nan'),
        }

    pred_valid = pred_phase[valid]
    target_valid = target_phase[valid]

    # Convert to numpy for correlation calculation
    pred_np = pred_valid.detach().cpu().numpy()
    target_np = target_valid.detach().cpu().numpy()

    # MSE
    mse = ((pred_valid - target_valid) ** 2).mean().item()

    # MAE
    mae = (pred_valid - target_valid).abs().mean().item()

    # RMSE
    rmse = np.sqrt(mse)

    # Pearson correlation
    if len(pred_np) > 1:
        pearson_r, _ = stats.pearsonr(pred_np, target_np)
    else:
        pearson_r = float('nan')

    # Spearman correlation
    if len(pred_np) > 1:
        spearman_r, _ = stats.spearmanr(pred_np, target_np)
    else:
        spearman_r = float('nan')

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'pearson': pearson_r,
        'spearman': spearman_r,
    }


def visualize_phase_diagram(
    phase: np.ndarray,
    title: str = "Phase Diagram",
    save_path: Optional[str] = None
):
    """Visualize a 4x4 phase diagram as a heatmap.

    Args:
        phase: (16,) or (4, 4) phase values
        title: Plot title
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    # Reshape to 4x4 if needed
    if phase.shape == (16,):
        phase = phase.reshape(4, 4)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(phase, cmap='RdBu_r', vmin=-1, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("PSSI Value", rotation=-90, va="bottom")

    # Add labels
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f'G{i+1}' for i in range(4)])
    ax.set_yticklabels([f'G{i+1}' for i in range(4)])

    # Add values as text
    for i in range(4):
        for j in range(4):
            val = phase[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.2f}',
                              ha="center", va="center", color="black", fontsize=8)

    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0
):
    """Create a schedule with linear warmup and cosine annealing.

    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial lr

    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
