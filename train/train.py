"""
Training script for PhaseFlow.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

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
        default="/data4/huangyanjie/LLPS/phase_diagram/phase_diagram.csv",
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
        pred_phase = model.generate_phase(
            input_ids, attention_mask, seq_len,
            method='euler'  # 使用简单的 Euler 进行快速评估
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
        f"MAE: {phase_metrics['mae']:.4f}, RMSE: {phase_metrics['rmse']:.4f}"
    )

    return {
        'loss': loss_meter.avg,
        'flow_loss': flow_loss_meter.avg,
        'lm_loss': lm_loss_meter.avg,
        'perplexity': perplexity_meter.avg,
        **phase_metrics,
    }


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

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
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

    # Create dataloaders
    logger.info("Loading data...")
    train_loader = create_dataloader(
        args.data_path,
        batch_size=config['training']['batch_size'],
        split='train',
        num_workers=config['data']['num_workers'],
        tokenizer=tokenizer,
        max_seq_len=config['model']['max_seq_len'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        seed=args.seed,
        use_npz=True,  # 优先使用NPZ文件快速加载
        normalize_phase=False,  # 数据已经是[-1,1]范围
    )

    val_loader = create_dataloader(
        args.data_path,
        batch_size=config['training']['batch_size'],
        split='val',
        num_workers=config['data']['num_workers'],
        tokenizer=tokenizer,
        max_seq_len=config['model']['max_seq_len'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        seed=args.seed,
        use_npz=True,
        normalize_phase=False,
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

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

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            args.device, epoch, config, logger
        )

        # Validate
        val_metrics = validate(model, val_loader, args.device, config, logger)

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


if __name__ == "__main__":
    main()
