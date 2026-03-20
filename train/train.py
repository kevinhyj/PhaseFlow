"""
Training script for PhaseFlow.
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server

from phaseflow import PhaseFlow, AminoAcidTokenizer, PhaseDataset
from phaseflow.data import create_dataloader, collate_fn
from phaseflow.utils import (
    set_seed,
    load_config,
    save_config,
    get_logger,
    count_parameters,
    format_number,
    AverageMeter,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    create_labels_for_lm,
    compute_metrics,
    get_cosine_schedule_with_warmup,
)


def plot_training_curves(history: dict, save_dir: Path):
    """Plot and save training curves.

    Args:
        history: Dictionary containing training history
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Loss curves (train vs val)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'loss.png', dpi=150)
    plt.close()

    # 2. Flow and LM Loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_flow_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_flow_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Flow Loss', fontsize=12)
    axes[0].set_title('Flow Matching Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_lm_loss'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_lm_loss'], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('LM Loss', fontsize=12)
    axes[1].set_title('Language Model Loss', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'flow_lm_loss.png', dpi=150)
    plt.close()

    # 3. Perplexity
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['perplexity'], 'g-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Validation Perplexity', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'perplexity.png', dpi=150)
    plt.close()

    # 4. Correlation metrics (Pearson & Spearman)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['pearson'], 'b-', label='Pearson', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, history['spearman'], 'r-', label='Spearman', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Correlation Metrics', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.0)
    plt.tight_layout()
    plt.savefig(save_dir / 'correlation.png', dpi=150)
    plt.close()

    # 5. Phase prediction metrics (MSE, MAE, RMSE)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['mse'], 'b-', label='MSE', linewidth=2)
    ax.plot(epochs, history['mae'], 'g-', label='MAE', linewidth=2)
    ax.plot(epochs, history['rmse'], 'r-', label='RMSE', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Phase Prediction Metrics', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'phase_metrics.png', dpi=150)
    plt.close()

    # 6. Summary plot (all key metrics in one figure)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=12)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Perplexity
    axes[0, 1].plot(epochs, history['perplexity'], 'g-', linewidth=2)
    axes[0, 1].set_title('Perplexity', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    # Correlation
    axes[0, 2].plot(epochs, history['pearson'], 'b-', label='Pearson', linewidth=2)
    axes[0, 2].plot(epochs, history['spearman'], 'r-', label='Spearman', linewidth=2)
    axes[0, 2].set_title('Correlation', fontsize=12)
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    # Flow Loss
    axes[1, 0].plot(epochs, history['train_flow_loss'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_flow_loss'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_title('Flow Loss', fontsize=12)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # LM Loss
    axes[1, 1].plot(epochs, history['train_lm_loss'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, history['val_lm_loss'], 'r-', label='Val', linewidth=2)
    axes[1, 1].set_title('LM Loss', fontsize=12)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    # Phase Metrics
    axes[1, 2].plot(epochs, history['mae'], 'g-', label='MAE', linewidth=2)
    axes[1, 2].plot(epochs, history['rmse'], 'r-', label='RMSE', linewidth=2)
    axes[1, 2].set_title('Phase Metrics', fontsize=12)
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Epoch', fontsize=10)

    plt.suptitle('Training Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'summary.png', dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhaseFlow model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/yanjie_huang/LLPS/phase_diagram/phase_diagram_original_scale.csv",
        help="Path to phase diagram CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--missing_threshold",
        type=int,
        default=-1,
        help="Missing value threshold for split (-1 to disable). "
             "Train uses missing_0 to missing_{threshold}, val/test uses missing_{threshold+1} to 15"
    )
    # Override config options
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device: str,
    epoch: int,
    config: dict,
    logger,
):
    """Train for one epoch."""
    model.train()

    # Metrics
    loss_meter = AverageMeter("Loss")
    flow_loss_meter = AverageMeter("Flow Loss")
    lm_loss_meter = AverageMeter("LM Loss")

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        phase_values = batch['phase_values'].to(device)
        phase_mask = batch['phase_mask'].to(device)
        seq_len = batch['seq_len'].to(device)

        # Create labels for LM task
        labels = create_labels_for_lm(
            input_ids,
            pad_token_id=AminoAcidTokenizer.PAD_ID,
            som_token_id=AminoAcidTokenizer.SOM_ID,
            eos_token_id=AminoAcidTokenizer.EOS_ID,
        )

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(enabled=config['training'].get('use_amp', True)):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                phase=phase_values,
                phase_mask=phase_mask,
                seq_len=seq_len,
                labels=labels,
                flow_weight=config['training']['flow_loss_weight'],
                lm_weight=config['training']['lm_loss_weight'],
            )
            loss = outputs['loss']

        # Backward pass
        if config['training'].get('use_amp', True):
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training'].get('max_grad_norm', 1.0)
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training'].get('max_grad_norm', 1.0)
            )
            optimizer.step()

        scheduler.step()

        # Update metrics
        batch_size = input_ids.shape[0]
        loss_meter.update(loss.item(), batch_size)
        flow_loss_meter.update(outputs['flow_loss'].item(), batch_size)
        lm_loss_meter.update(outputs['lm_loss'].item(), batch_size)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_meter.avg:.4f}",
            'flow': f"{flow_loss_meter.avg:.4f}",
            'lm': f"{lm_loss_meter.avg:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}",
        })

    logger.info(
        f"Epoch {epoch} - Train Loss: {loss_meter.avg:.4f}, "
        f"Flow: {flow_loss_meter.avg:.4f}, LM: {lm_loss_meter.avg:.4f}"
    )

    return {
        'loss': loss_meter.avg,
        'flow_loss': flow_loss_meter.avg,
        'lm_loss': lm_loss_meter.avg,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: str,
    config: dict,
    logger,
):
    """Validate the model."""
    model.eval()

    loss_meter = AverageMeter("Loss")
    flow_loss_meter = AverageMeter("Flow Loss")
    lm_loss_meter = AverageMeter("LM Loss")
    perplexity_meter = AverageMeter("Perplexity")

    # For phase prediction metrics
    all_pred_phase = []
    all_target_phase = []
    all_phase_mask = []

    for batch in tqdm(val_loader, desc="Validation"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        phase_values = batch['phase_values'].to(device)
        phase_mask = batch['phase_mask'].to(device)
        seq_len = batch['seq_len'].to(device)

        labels = create_labels_for_lm(
            input_ids,
            pad_token_id=AminoAcidTokenizer.PAD_ID,
            som_token_id=AminoAcidTokenizer.SOM_ID,
            eos_token_id=AminoAcidTokenizer.EOS_ID,
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            phase=phase_values,
            phase_mask=phase_mask,
            seq_len=seq_len,
            labels=labels,
            flow_weight=config['training']['flow_loss_weight'],
            lm_weight=config['training']['lm_loss_weight'],
        )

        batch_size = input_ids.shape[0]
        loss_meter.update(outputs['loss'].item(), batch_size)
        flow_loss_meter.update(outputs['flow_loss'].item(), batch_size)
        lm_loss_meter.update(outputs['lm_loss'].item(), batch_size)
        perplexity_meter.update(outputs['perplexity'].item(), batch_size)

        # Generate phase diagrams for evaluation
        # DDPM uses DDIM sampling; flow matching uses euler
        sampling_kwargs = {}
        if hasattr(model, 'diffusion_type') and model.diffusion_type == 'ddpm':
            sampling_kwargs = dict(
                num_steps=config.get('sampling', {}).get('sampling_steps', 50),
                use_ddim=config.get('sampling', {}).get('use_ddim', True),
            )
        else:
            sampling_kwargs = dict(method='euler')

        pred_phase = model.generate_phase(
            input_ids, attention_mask, seq_len,
            **sampling_kwargs,
        )

        all_pred_phase.append(pred_phase.cpu())
        all_target_phase.append(phase_values.cpu())
        all_phase_mask.append(phase_mask.cpu())

    # Compute phase prediction metrics
    pred_phase = torch.cat(all_pred_phase, dim=0)
    target_phase = torch.cat(all_target_phase, dim=0)
    phase_mask = torch.cat(all_phase_mask, dim=0)

    phase_metrics = compute_metrics(pred_phase, target_phase, phase_mask)

    logger.info(
        f"Validation - Loss: {loss_meter.avg:.4f}, "
        f"Flow: {flow_loss_meter.avg:.4f}, LM: {lm_loss_meter.avg:.4f}, "
        f"Perplexity: {perplexity_meter.avg:.2f}"
    )
    logger.info(
        f"Phase Metrics - MSE: {phase_metrics['mse']:.4f}, "
        f"MAE: {phase_metrics['mae']:.4f}, RMSE: {phase_metrics['rmse']:.4f}, "
        f"Pearson: {phase_metrics['pearson']:.4f}, Spearman: {phase_metrics['spearman']:.4f}"
    )

    return {
        'loss': loss_meter.avg,
        'flow_loss': flow_loss_meter.avg,
        'lm_loss': lm_loss_meter.avg,
        'perplexity': perplexity_meter.avg,
        **phase_metrics,
    }


@torch.no_grad()
def evaluate_test(
    model: nn.Module,
    test_loader,
    device: str,
    config: dict,
    logger,
    output_dir: Path,
):
    """Evaluate the model on test set."""
    model.eval()

    loss_meter = AverageMeter("Loss")
    flow_loss_meter = AverageMeter("Flow Loss")
    lm_loss_meter = AverageMeter("LM Loss")
    perplexity_meter = AverageMeter("Perplexity")

    # For phase prediction metrics
    all_pred_phase = []
    all_target_phase = []
    all_phase_mask = []

    for batch in tqdm(test_loader, desc="Test Evaluation"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        phase_values = batch['phase_values'].to(device)
        phase_mask = batch['phase_mask'].to(device)
        seq_len = batch['seq_len'].to(device)

        labels = create_labels_for_lm(
            input_ids,
            pad_token_id=AminoAcidTokenizer.PAD_ID,
            som_token_id=AminoAcidTokenizer.SOM_ID,
            eos_token_id=AminoAcidTokenizer.EOS_ID,
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            phase=phase_values,
            phase_mask=phase_mask,
            seq_len=seq_len,
            labels=labels,
            flow_weight=config['training']['flow_loss_weight'],
            lm_weight=config['training']['lm_loss_weight'],
        )

        batch_size = input_ids.shape[0]
        loss_meter.update(outputs['loss'].item(), batch_size)
        flow_loss_meter.update(outputs['flow_loss'].item(), batch_size)
        lm_loss_meter.update(outputs['lm_loss'].item(), batch_size)
        perplexity_meter.update(outputs['perplexity'].item(), batch_size)

        # Generate phase diagrams for evaluation
        sampling_kwargs = {}
        if hasattr(model, 'diffusion_type') and model.diffusion_type == 'ddpm':
            sampling_kwargs = dict(
                num_steps=config.get('sampling', {}).get('sampling_steps', 50),
                use_ddim=config.get('sampling', {}).get('use_ddim', True),
            )
        else:
            sampling_kwargs = dict(method='euler')

        pred_phase = model.generate_phase(
            input_ids, attention_mask, seq_len,
            **sampling_kwargs,
        )

        all_pred_phase.append(pred_phase.cpu())
        all_target_phase.append(phase_values.cpu())
        all_phase_mask.append(phase_mask.cpu())

    # Compute phase prediction metrics
    pred_phase = torch.cat(all_pred_phase, dim=0)
    target_phase = torch.cat(all_target_phase, dim=0)
    phase_mask = torch.cat(all_phase_mask, dim=0)

    phase_metrics = compute_metrics(pred_phase, target_phase, phase_mask)

    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 60)
    logger.info(
        f"Test - Loss: {loss_meter.avg:.4f}, "
        f"Flow: {flow_loss_meter.avg:.4f}, LM: {lm_loss_meter.avg:.4f}, "
        f"Perplexity: {perplexity_meter.avg:.2f}"
    )
    logger.info(
        f"Phase Metrics - MSE: {phase_metrics['mse']:.4f}, "
        f"MAE: {phase_metrics['mae']:.4f}, RMSE: {phase_metrics['rmse']:.4f}, "
        f"Pearson: {phase_metrics['pearson']:.4f}, Spearman: {phase_metrics['spearman']:.4f}"
    )
    logger.info("=" * 60)

    # Save test metrics to file
    test_results = {
        'loss': loss_meter.avg,
        'flow_loss': flow_loss_meter.avg,
        'lm_loss': lm_loss_meter.avg,
        'perplexity': perplexity_meter.avg,
        **phase_metrics,
    }

    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump({k: float(v) for k, v in test_results.items()}, f, indent=2)
    logger.info(f"Test results saved to: {results_path}")

    return test_results


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # Use default config
        config = {
            'model': {
                'dim': 256,
                'depth': 6,
                'heads': 8,
                'dim_head': 32,
                'vocab_size': 64,
                'phase_dim': 16,
                'max_seq_len': 32,
                'dropout': 0.1,
            },
            'training': {
                'batch_size': 64,
                'lr': 1e-4,
                'weight_decay': 0.01,
                'epochs': 100,
                'warmup_steps': 1000,
                'flow_loss_weight': 1.0,
                'lm_loss_weight': 1.0,
                'max_grad_norm': 1.0,
                'use_amp': True,
            },
            'sampling': {
                'ode_steps': 20,
                'temperature': 1.0,
            },
            'data': {
                'train_ratio': 0.9,
                'val_ratio': 0.05,
                'num_workers': 4,
            }
        }

    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['lr'] = args.lr
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.dim:
        config['model']['dim'] = args.dim
    if args.depth:
        config['model']['depth'] = args.depth

    # Setup output directory (use config filename)
    # Auto-switch output dir based on model type, unless user explicitly passed --output_dir
    config_name = Path(args.config).stem  # e.g., "bs512_lr0.0032_flow0.8_20260122"
    base_dir = args.output_dir
    if args.output_dir == "outputs":
        if config['model'].get('diffusion_type', 'flow_matching') == 'ddpm':
            base_dir = "outputs_ddpm"
        elif config['model'].get('use_set_encoder', False):
            base_dir = "outputs_set"
    output_dir = Path(base_dir) / f"output_{config_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = get_logger("train", log_file=str(output_dir / "train.log"))
    logger.info(f"Output directory: {output_dir}")

    # Save config
    save_config(config, str(output_dir / "config.yaml"))

    # Initialize CUDA context before creating DataLoader
    # This prevents CUBLAS_STATUS_NOT_INITIALIZED errors
    if args.device == 'cuda':
        torch.cuda.init()
        torch.cuda.set_device(0)
        # Warm up cuBLAS
        _ = torch.zeros(1).cuda()
        logger.info("CUDA context initialized")

    # Create tokenizer
    tokenizer = AminoAcidTokenizer()

    # Create dataloaders with hardcoded val/test sets
    logger.info("Loading data...")
    logger.info(f"Using missing threshold split: threshold={args.missing_threshold}")
    logger.info("Using hardcoded val/test sets from phase_diagram/val_set.csv and test_set.csv")

    # Training set: use all data filtered by missing_threshold (no splitting)
    train_loader = create_dataloader(
        args.data_path,
        batch_size=config['training']['batch_size'],
        split='all',  # Use all filtered data for training (no split)
        num_workers=config['data']['num_workers'],
        tokenizer=tokenizer,
        max_seq_len=config['model']['max_seq_len'],
        use_npz=False,
        normalize_phase=False,
        missing_threshold=args.missing_threshold,
    )

    # Validation set: hardcoded from val_set.csv
    val_loader = create_dataloader(
        '/data/yanjie_huang/LLPS/phase_diagram/val_set.csv',
        batch_size=config['training']['batch_size'],
        split='all',  # Use all data in this file
        num_workers=config['data']['num_workers'],
        tokenizer=tokenizer,
        max_seq_len=config['model']['max_seq_len'],
        use_npz=False,
        normalize_phase=False,
        missing_threshold=-1,  # Use all data from val_set.csv
    )

    # Test set: hardcoded from test_set.csv
    test_loader = create_dataloader(
        '/data/yanjie_huang/LLPS/phase_diagram/test_set.csv',
        batch_size=config['training']['batch_size'],
        split='all',  # Use all data in this file
        num_workers=config['data']['num_workers'],
        tokenizer=tokenizer,
        max_seq_len=config['model']['max_seq_len'],
        use_npz=False,
        normalize_phase=False,
        missing_threshold=-1,  # Use all data from test_set.csv
    )

    logger.info(f"Train samples (missing <= {args.missing_threshold}): {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating model...")
    model = PhaseFlow(
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        dim_head=config['model']['dim_head'],
        vocab_size=config['model']['vocab_size'],
        phase_dim=config['model']['phase_dim'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=config['model']['dropout'],
        use_set_encoder=config['model'].get('use_set_encoder', False),
        diffusion_type=config['model'].get('diffusion_type', 'flow_matching'),
        num_timesteps=config['model'].get('num_timesteps', 1000),
        beta_schedule=config['model'].get('beta_schedule', 'cosine'),
        use_ot_coupling=config['model'].get('use_ot_coupling', False),
        use_quadratic_weighting=config['model'].get('use_quadratic_weighting', True),
    )
    model = model.to(args.device)

    num_params = count_parameters(model)
    logger.info(f"Model parameters: {format_number(num_params)} ({num_params:,})")

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    # Create scheduler
    num_training_steps = len(train_loader) * config['training']['epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=num_training_steps,
        min_lr_ratio=0.1,
    )

    # Create scaler for mixed precision
    scaler = GradScaler(enabled=config['training'].get('use_amp', True))

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = load_checkpoint(
            args.resume, model, optimizer, scheduler, device=args.device
        )
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training'].get('early_stopping_patience', 10),
        mode='min'
    )

    # Initialize training history for visualization
    history = {
        'train_loss': [], 'val_loss': [],
        'train_flow_loss': [], 'val_flow_loss': [],
        'train_lm_loss': [], 'val_lm_loss': [],
        'perplexity': [],
        'mse': [], 'mae': [], 'rmse': [],
        'pearson': [], 'spearman': [],
    }

    # Create visualization directory using config filename
    config_name = Path(args.config).stem  # e.g., "bs2048_lr0.0064_flow10_20260122"
    visual_dir = Path("/data/yanjie_huang/LLPS/predictor/PhaseFlow/visual_training") / config_name
    visual_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Visualization directory: {visual_dir}")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            args.device, epoch, config, logger,
        )

        # Validate
        val_metrics = validate(model, val_loader, args.device, config, logger)

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_flow_loss'].append(train_metrics['flow_loss'])
        history['val_flow_loss'].append(val_metrics['flow_loss'])
        history['train_lm_loss'].append(train_metrics['lm_loss'])
        history['val_lm_loss'].append(val_metrics['lm_loss'])
        history['perplexity'].append(val_metrics['perplexity'])
        history['mse'].append(val_metrics['mse'])
        history['mae'].append(val_metrics['mae'])
        history['rmse'].append(val_metrics['rmse'])
        history['pearson'].append(val_metrics['pearson'])
        history['spearman'].append(val_metrics['spearman'])

        # Update training curves
        plot_training_curves(history, visual_dir)

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'],
                str(output_dir / "best_model.pt"),
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                config=config,
            )
            logger.info(f"Saved best model with val_loss={best_val_loss:.4f}")

        # Save periodic checkpoint
        if epoch % config['training'].get('save_every', 10) == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'],
                str(output_dir / f"checkpoint_epoch{epoch}.pt"),
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                config=config,
            )

        # Early stopping check
        if early_stopping(val_metrics['loss']):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Save final model
    save_checkpoint(
        model, optimizer, epoch, val_metrics['loss'],
        str(output_dir / "final_model.pt"),
        scheduler=scheduler,
        best_val_loss=best_val_loss,
        config=config,
    )

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    # Evaluate on test set if using missing_split
    if args.missing_threshold >= 0:
        logger.info("\nEvaluating on test set (incomplete phase diagrams)...")
        test_metrics = evaluate_test(model, test_loader, args.device, config, logger, output_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
