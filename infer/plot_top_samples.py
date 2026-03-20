#!/usr/bin/env python3
"""
输出 flow32_lm1_missing15 在测试集上 Spearman Top5 和 MSE Top5 的 GT vs Pred 相图对比。
(已修改为：每个图单独进行 Min-Max 归一化后再显示)

Usage:
    cd /data/yanjie_huang/LLPS/predictor/PhaseFlow
    python infer/plot_top_samples.py --gpu 0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

font_path = '/data/yanjie_huang/fonts/arial.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()

CKPT = '/data/yanjie_huang/LLPS/predictor/PhaseFlow/outputs_set/output_set_flow32_missing15/best_model.pt'
TEST_PATH = '/data/yanjie_huang/LLPS/phase_diagram/test_set.csv'
OUT_DIR = Path('/data/yanjie_huang/LLPS/predictor/PhaseFlow/outputs_set/output_set_flow32_missing15')
PHASE_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]

COLOR_GT  = (114/255, 138/255, 185/255)  # #728ab9
COLOR_PRED = (245/255, 181/255, 191/255)  # #f5b5bf


def plot_phase_diagram(gt, pred, mask, seq, sp, mse, save_path):
    """画单个样本的 4x4 GT vs Pred 相图对比 (归一化版本)。"""
    v = mask.astype(bool)

    # --- 1. 对 GT 进行 Min-Max 归一化 (仅基于 mask 区域) ---
    gt_valid = gt[v]
    gt_min, gt_max = gt_valid.min(), gt_valid.max()
    gt_norm = gt.copy()
    if gt_max > gt_min:
        gt_norm[v] = (gt_valid - gt_min) / (gt_max - gt_min)
    else:
        gt_norm[v] = 0.5  # 如果所有值都相同，给个中间值避免除零错误

    # --- 2. 对 Pred 进行 Min-Max 归一化 (基于整个预测矩阵) ---
    p_min, p_max = pred.min(), pred.max()
    pred_norm = np.zeros_like(pred)
    if p_max > p_min:
        pred_norm = (pred - p_min) / (p_max - p_min)
    else:
        pred_norm.fill(0.5)

    gt_grid = gt_norm.reshape(4, 4)
    pred_grid = pred_norm.reshape(4, 4)
    mask_grid = mask.reshape(4, 4)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='white')

    # 归一化后，色标固定在 0 到 1 之间
    vmin, vmax = 0.0, 1.0

    labels = ['Light 1', 'Light 2', 'Light 3', 'Light 4']
    conc_labels = ['Conc 1', 'Conc 2', 'Conc 3', 'Conc 4']

    # --- 绘制 GT ---
    ax = axes[0]
    gt_display = np.where(mask_grid, gt_grid, np.nan)
    im = ax.imshow(gt_display, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
    for r in range(4):
        for c in range(4):
            if mask_grid[r, c]:
                ax.text(c, r, f'{gt_grid[r,c]:.2f}', ha='center', va='center', fontsize=9)
            else:
                ax.text(c, r, 'NaN', ha='center', va='center', fontsize=8, color='gray')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(conc_labels, fontsize=8)
    ax.set_title('Ground Truth (Norm)', fontsize=13, fontweight='bold')

    # --- 绘制 Pred ---
    ax = axes[1]
    im = ax.imshow(pred_grid, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
    for r in range(4):
        for c in range(4):
            ax.text(c, r, f'{pred_grid[r,c]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(conc_labels, fontsize=8)
    ax.set_title('Predicted (Norm)', fontsize=13, fontweight='bold')

    # --- 绘制 Scatter ---
    ax = axes[2]
    ax.set_facecolor('white')
    ax.grid(alpha=0.3, linestyle='--')
    # 散点图展示归一化后的数据
    ax.scatter(gt_norm[v], pred_norm[v], color=COLOR_PRED, s=50, alpha=0.8, edgecolors='white', linewidths=0.5)
    
    # 坐标系限制在 [-0.1, 1.1] 留白
    lims = [-0.1, 1.1]
    ax.plot(lims, lims, '--', color='gray', linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('GT PSSI (Norm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Pred PSSI (Norm)', fontsize=11, fontweight='bold')
    ax.set_title('GT vs Pred (Norm)', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 标题保留原始的评估指标
    fig.suptitle(f'{seq}\nSpearman={sp:.4f}  MSE={mse:.4f}', fontsize=11, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    # Load model
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = PhaseFlow(
        dim=cfg['model']['dim'], depth=cfg['model']['depth'],
        heads=cfg['model']['heads'], dim_head=cfg['model']['dim_head'],
        vocab_size=cfg['model']['vocab_size'], phase_dim=cfg['model']['phase_dim'],
        max_seq_len=cfg['model']['max_seq_len'], dropout=0.0,
        use_set_encoder=cfg['model'].get('use_set_encoder', False),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device).eval()

    # Load test set
    df = pd.read_csv(TEST_PATH)
    target = df[PHASE_COLS].values.astype(np.float32)
    mask = ~np.isnan(target)
    target_filled = np.nan_to_num(target, nan=0.0)
    seqs = df['AminoAcidSequence'].tolist()

    tokenizer = AminoAcidTokenizer()
    tokens = tokenizer.batch_encode(seqs, max_len=32, return_tensors=True)
    attn_mask = (tokens != tokenizer.PAD_ID).long()

    # Predict
    preds = []
    for i in range(0, len(df), 64):
        end = min(i + 64, len(df))
        with torch.no_grad():
            p = model.generate_phase(tokens[i:end].to(device), attn_mask[i:end].to(device),
                                     seq_len=None, method='euler')
        preds.append(p.cpu().numpy())
    pred = np.concatenate(preds, axis=0)

    # Per-sample metrics (Metrics are computed on ORIGINAL RAW values)
    sp_list, mse_list = [], []
    for i in range(len(df)):
        v = mask[i]
        if v.sum() >= 2:
            rho, _ = spearmanr(pred[i][v], target_filled[i][v])
            sp_list.append(rho if not np.isnan(rho) else 0.0)
        else:
            sp_list.append(np.nan)
        if v.sum() >= 1:
            mse_list.append(float(((pred[i][v] - target_filled[i][v]) ** 2).mean()))
        else:
            mse_list.append(np.nan)

    sp_arr = np.array(sp_list)
    mse_arr = np.array(mse_list)

    # Top 5 Spearman
    top5_sp = np.argsort(sp_arr)[::-1][:5]
    # Top 5 MSE (lowest)
    top5_mse = np.argsort(mse_arr)[:5]

    out_sp = OUT_DIR / 'top5_spearman'
    out_mse = OUT_DIR / 'top5_mse'
    out_sp.mkdir(exist_ok=True)
    out_mse.mkdir(exist_ok=True)

    print('Top 5 Spearman:')
    for rank, idx in enumerate(top5_sp):
        print(f'  #{rank+1} idx={idx} seq={seqs[idx]} sp={sp_arr[idx]:.4f} mse={mse_arr[idx]:.4f}')
        plot_phase_diagram(target_filled[idx], pred[idx], mask[idx].astype(float),
                           seqs[idx], sp_arr[idx], mse_arr[idx],
                           out_sp / f'rank{rank+1}_idx{idx}.png')

    print('\nTop 5 MSE (lowest):')
    for rank, idx in enumerate(top5_mse):
        print(f'  #{rank+1} idx={idx} seq={seqs[idx]} sp={sp_arr[idx]:.4f} mse={mse_arr[idx]:.4f}')
        plot_phase_diagram(target_filled[idx], pred[idx], mask[idx].astype(float),
                           seqs[idx], sp_arr[idx], mse_arr[idx],
                           out_mse / f'rank{rank+1}_idx{idx}.png')

    print(f'\nSaved to {out_sp} and {out_mse}')


if __name__ == '__main__':
    main()