#!/usr/bin/env python3
"""
可视化 PhaseFlow 模型在测试集上的真实 vs 预测相图。
随机抽取 N 条序列，横向排列，每个相图独立归一化到 [0,1]。

Usage:
    conda activate phaseflow
    cd /data/yanjie_huang/LLPS/predictor/PhaseFlow
    python infer/visualize_phase_diagrams.py --model outputs_set/output_set_flow32_lm5_missing15 --n 8 --gpu 2
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phaseflow import PhaseFlow
from phaseflow.tokenizer import AminoAcidTokenizer
from phaseflow.utils import set_seed

TEST_PATH = '/data/yanjie_huang/LLPS/phase_diagram/test_set.csv'
PHASE_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]
MAX_SEQ_LEN = 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model output directory')
    parser.add_argument('--n', type=int, default=8, help='Number of sequences to visualize')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    return parser.parse_args()


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model = PhaseFlow(
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        dim_head=config['model']['dim_head'],
        vocab_size=config['model']['vocab_size'],
        phase_dim=config['model']['phase_dim'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=0.0,
        use_set_encoder=config['model'].get('use_set_encoder', False),
        diffusion_type=config['model'].get('diffusion_type', 'flow_matching'),
        num_timesteps=config['model'].get('num_timesteps', 1000),
        beta_schedule=config['model'].get('beta_schedule', 'cosine'),
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device).eval()
    return model, config


def normalize_grid(grid):
    """Per-sample min-max normalize to [0, 1]."""
    vmin, vmax = grid.min(), grid.max()
    if vmax - vmin < 1e-8:
        return np.full_like(grid, 0.5)
    return (grid - vmin) / (vmax - vmin)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load test data
    df = pd.read_csv(TEST_PATH)
    tokenizer = AminoAcidTokenizer()

    # Tokenize
    seqs = df['AminoAcidSequence'].tolist()
    tokens_list = [tokenizer.encode_sequence(s) for s in seqs]
    padded = torch.zeros(len(seqs), MAX_SEQ_LEN, dtype=torch.long)
    attn_mask = torch.zeros(len(seqs), MAX_SEQ_LEN, dtype=torch.bool)
    for i, t in enumerate(tokens_list):
        t_tensor = torch.tensor(t, dtype=torch.long)
        L = min(len(t), MAX_SEQ_LEN)
        padded[i, :L] = t_tensor[:L]
        attn_mask[i, :L] = True

    # Target values
    target = df[PHASE_COLS].values.astype(np.float32)
    mask = ~np.isnan(target)
    target_filled = np.nan_to_num(target, nan=0.0)

    # Random sample N sequences (prefer those with all 16 values)
    full_indices = np.where(mask.all(axis=1))[0]
    if len(full_indices) >= args.n:
        indices = np.random.choice(full_indices, size=args.n, replace=False)
    else:
        indices = np.random.choice(len(seqs), size=args.n, replace=False)
    indices = sorted(indices)

    # Load model and predict
    ckpt_path = os.path.join(args.model, 'best_model.pt')
    model, config = load_model(ckpt_path, device)

    is_ddpm = hasattr(model, 'diffusion_type') and model.diffusion_type == 'ddpm'

    bt = padded[indices].to(device)
    bm = attn_mask[indices].to(device)
    with torch.no_grad():
        if is_ddpm:
            pred = model.generate_phase(bt, bm, seq_len=None, num_steps=50, use_ddim=True)
        else:
            pred = model.generate_phase(bt, bm, seq_len=None, method='euler')
    pred = pred.cpu().numpy()

    # Get targets for selected indices
    true_vals = target_filled[indices]
    true_masks = mask[indices]
    selected_seqs = [seqs[i] for i in indices]

    # Layout: 2 rows (True / Predicted), N columns
    fig, axes = plt.subplots(2, args.n, figsize=(2.8 * args.n, 6.5))
    if args.n == 1:
        axes = axes.reshape(2, 1)

    row_labels = ['C1', 'C2', 'C3', 'C4']
    col_labels = ['L1', 'L2', 'L3', 'L4']
    norm01 = Normalize(vmin=0, vmax=1)

    for col_idx in range(args.n):
        seq = selected_seqs[col_idx]
        true_grid = true_vals[col_idx].reshape(4, 4)
        pred_grid = pred[col_idx].reshape(4, 4)
        m = true_masks[col_idx].reshape(4, 4)

        true_norm = normalize_grid(true_grid)
        pred_norm = normalize_grid(pred_grid)

        # --- True (top row) ---
        ax = axes[0, col_idx]
        ax.imshow(true_norm, cmap='RdBu_r', norm=norm01, aspect='equal')
        ax.set_xticks(range(4))
        ax.set_xticklabels(col_labels, fontsize=7)
        ax.set_yticks(range(4))
        ax.set_yticklabels(row_labels, fontsize=7)
        for i in range(4):
            for j in range(4):
                if m[i, j]:
                    ax.text(j, i, f'{true_grid[i,j]:.2f}', ha='center', va='center',
                            fontsize=6.5, color='black', fontweight='bold')
                else:
                    ax.text(j, i, 'NaN', ha='center', va='center',
                            fontsize=6.5, color='gray')
        # Sequence as title
        if len(seq) > 12:
            title_seq = seq[:6] + '..' + seq[-4:]
        else:
            title_seq = seq
        ax.set_title(f'{title_seq}\n(len={len(seq)})', fontsize=7.5, pad=4)

        # Row label
        if col_idx == 0:
            ax.set_ylabel('True', fontsize=11, fontweight='bold')

        # --- Predicted (bottom row) ---
        ax = axes[1, col_idx]
        im = ax.imshow(pred_norm, cmap='RdBu_r', norm=norm01, aspect='equal')
        ax.set_xticks(range(4))
        ax.set_xticklabels(col_labels, fontsize=7)
        ax.set_yticks(range(4))
        ax.set_yticklabels(row_labels, fontsize=7)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{pred_grid[i,j]:.2f}', ha='center', va='center',
                        fontsize=6.5, color='black', fontweight='bold')

        if col_idx == 0:
            ax.set_ylabel('Predicted', fontsize=11, fontweight='bold')

    # Colorbar
    fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.35)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm01, cmap='RdBu_r'),
                        cax=cbar_ax)
    cbar.set_label('Normalized PSSI\n(per-sample min-max)', fontsize=9)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Low', '', 'Mid', '', 'High'])

    model_name = Path(args.model).name.replace('output_', '')
    fig.suptitle(f'Phase Diagram: True vs Predicted ({model_name})\nValues = raw PSSI, Colors = per-sample normalized',
                 fontsize=12, fontweight='bold', y=1.02)

    output_path = args.output or f'infer/phase_diagram_visualization_{model_name}.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {output_path}')
    plt.close()


if __name__ == '__main__':
    main()
