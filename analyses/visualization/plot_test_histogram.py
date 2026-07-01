#!/usr/bin/env python3
"""
用 flow32_lm1_missing15 模型预测测试集，逐样本计算指标，画直方图。

Usage:
    python analyses/visualization/plot_test_histogram.py --checkpoint outputs_set/output_set_flow32_missing15/best_model.pt --test_path data/test_set.csv
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from phaseflow import PhaseFlow
from phaseflow.tokenizer import AminoAcidTokenizer
from phaseflow.utils import set_seed

DEFAULT_CKPT = ROOT / 'outputs_set' / 'output_set_flow32_missing15' / 'best_model.pt'
DEFAULT_TEST_PATH = ROOT / 'data' / 'test_set.csv'
DEFAULT_OUTPUT = ROOT / 'results' / 'evaluation' / 'test_histogram_flow32_m15.png'
PHASE_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]

COLOR_GT  = (114/255, 138/255, 185/255)
COLOR_GEN = (245/255, 181/255, 191/255)
COLOR_RND = (245/255, 221/255, 181/255)
MEAN_LINE = (100/255, 100/255, 100/255)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_CKPT))
    parser.add_argument('--test_path', type=str, default=str(DEFAULT_TEST_PATH))
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument('--font_path', type=str, default=None)
    args = parser.parse_args()

    if args.font_path:
        font_path = Path(args.font_path)
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            plt.rcParams['font.family'] = fm.FontProperties(fname=str(font_path)).get_name()

    checkpoint_path = Path(args.checkpoint)
    test_path = Path(args.test_path)
    output_path = Path(args.output)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not test_path.is_absolute():
        test_path = ROOT / test_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = PhaseFlow(
        dim=config['model']['dim'], depth=config['model']['depth'],
        heads=config['model']['heads'], dim_head=config['model']['dim_head'],
        vocab_size=config['model']['vocab_size'], phase_dim=config['model']['phase_dim'],
        max_seq_len=config['model']['max_seq_len'], dropout=0.0,
        use_set_encoder=config['model'].get('use_set_encoder', False),
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device).eval()

    # Load test set
    df = pd.read_csv(test_path)
    target = df[PHASE_COLS].values.astype(np.float32)
    mask = ~np.isnan(target)
    target_filled = np.nan_to_num(target, nan=0.0)

    tokenizer = AminoAcidTokenizer()
    tokens = tokenizer.batch_encode(df['AminoAcidSequence'].tolist(), max_len=32, return_tensors=True)
    attn_mask = (tokens != tokenizer.PAD_ID).long()

    # Predict
    preds = []
    for i in range(0, len(df), 64):
        end = min(i + 64, len(df))
        with torch.no_grad():
            pred = model.generate_phase(
                tokens[i:end].to(device), attn_mask[i:end].to(device),
                seq_len=None, method='euler')
        preds.append(pred.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    print(f'Predicted {pred.shape[0]} samples')

    # Per-sample metrics
    per_sample_spearman = []
    per_sample_mse = []
    for i in range(len(df)):
        v = mask[i]
        if v.sum() >= 2:
            rho, _ = spearmanr(pred[i][v], target_filled[i][v])
            per_sample_spearman.append(rho if not np.isnan(rho) else 0.0)
        else:
            per_sample_spearman.append(np.nan)
        if v.sum() >= 1:
            per_sample_mse.append(float(((pred[i][v] - target_filled[i][v]) ** 2).mean()))
        else:
            per_sample_mse.append(np.nan)

    # Per-sample mean prediction → global mean spearman is one number,
    # but we can show per-sample |pred_mean - target_mean| as "mean error"
    per_sample_mean_err = []
    for i in range(len(df)):
        v = mask[i]
        if v.sum() >= 1:
            per_sample_mean_err.append(abs(pred[i][v].mean() - target_filled[i][v].mean()))
        else:
            per_sample_mean_err.append(np.nan)

    sp_arr = np.array(per_sample_spearman)
    mse_arr = np.array(per_sample_mse)
    me_arr = np.array(per_sample_mean_err)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor='white')
    panels = [
        (sp_arr,  COLOR_GT,  'Per-Sample Spearman ρ', 'Spearman ρ'),
        (mse_arr, COLOR_GEN, 'Per-Sample MSE',        'MSE'),
        (me_arr,  COLOR_RND, 'Per-Sample |Mean Error|','|Pred Mean − Target Mean|'),
    ]

    for ax, (data, color, title, xlabel) in zip(axes, panels):
        valid = data[~np.isnan(data)]
        ax.set_facecolor('white')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.hist(valid, bins=30, color=color, alpha=0.75, edgecolor='white')
        m = np.mean(valid)
        ax.axvline(m, color=MEAN_LINE, linestyle='--', linewidth=1.5)
        ax.text(m, ax.get_ylim()[1] * 0.92, f'mean={m:.4f}',
                ha='center', fontsize=10, fontweight='bold', color=(80/255, 80/255, 80/255))
        ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
        ax.set_ylabel('Count', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=11)

    plt.suptitle('flow32 lm1 (missing15) — Test Set (n=500)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
