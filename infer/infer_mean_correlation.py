#!/usr/bin/env python3
"""
PhaseFlow 推理序列，评估预测均值与真实均值的相关性
使用 tokenizer 版本模型

Usage:
    # 评估 test_set.csv (500样本)
    python infer_mean_correlation.py --checkpoint <path_to_checkpoint> --use_test_set

    # 评估所有完整相图序列
    python infer_mean_correlation.py --checkpoint <path_to_checkpoint>
"""
import argparse
import sys
sys.path.insert(0, '/data/yanjie_huang/LLPS/predictor/PhaseFlow_WJX_Test')

import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint
from phaseflow import PhaseFlow
from phaseflow.utils import set_seed
from phaseflow.tokenizer import AminoAcidTokenizer
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Config
parser = argparse.ArgumentParser(description='PhaseFlow inference for mean correlation')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--data_path', type=str, default="/data/yanjie_huang/LLPS/phase_diagram/phase_diagram_original_scale.csv")
parser.add_argument('--output_dir', type=str, default="/data/yanjie_huang/LLPS/predictor/PhaseFlow_WJX_Test/infer/comparison")
parser.add_argument('--gpu', type=int, default=7, help='GPU device ID')
parser.add_argument('--use_test_set', action='store_true', help='Use test_set.csv instead of full dataset')
parser.add_argument('--test_set_path', type=str, default="/data/yanjie_huang/LLPS/phase_diagram/test_set.csv")
args = parser.parse_args()

CHECKPOINT = args.checkpoint
DATA_PATH = args.data_path
OUTPUT_DIR = args.output_dir

SEED = 42
set_seed(SEED)
device = torch.device(f'cuda:{args.gpu}')
print(f"Device: {device}")
print(f"Checkpoint: {CHECKPOINT}")

# Load model
print("Loading model...")
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
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
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device).eval()

# Load data from CSV
print("Loading data from CSV...")
df = pd.read_csv(DATA_PATH)
phase_cols = [f'group_{i//4+1}{i%4+1}' for i in range(16)]

if args.use_test_set:
    # 使用 test_set.csv
    print(f"Using test_set.csv: {args.test_set_path}")
    test_df = pd.read_csv(args.test_set_path)
    sequences = test_df['AminoAcidSequence'].tolist()
    ground_truth = test_df[phase_cols].values.astype(np.float32)  # (N, 16)
    print(f"Test set samples: {len(sequences)}")
else:
    # 找完整相图的序列 (missing=0)
    phase_data = df[phase_cols].values.astype(np.float32)
    phase_mask = ~np.isnan(phase_data)
    complete_mask = phase_mask.all(axis=1)
    complete_indices = np.where(complete_mask)[0]

    print(f"Complete samples: {len(complete_indices)}")

    # 获取所有完整序列
    sequences = df['AminoAcidSequence'].iloc[complete_indices].tolist()
    ground_truth = phase_data[complete_indices]  # (N, 16)

print(f"Ground truth range: [{ground_truth.min():.3f}, {ground_truth.max():.3f}]")
print(f"Ground truth mean per sample range: [{ground_truth.mean(axis=1).min():.3f}, {ground_truth.mean(axis=1).max():.3f}]")

# 准备 tokenizer
tokenizer = AminoAcidTokenizer()
tokens = tokenizer.batch_encode(sequences, max_len=None, return_tensors=True)  # (B, L)
attention_mask = torch.ones(tokens.shape[0], tokens.shape[1], dtype=torch.long, device=device)
tokens = tokens.to(device)

print(f"Tokens shape: {tokens.shape}")

# 批量推理
print("\nGenerating predictions for all complete samples...")
batch_size = 64
num_samples = len(sequences)
num_batches = (num_samples + batch_size - 1) // batch_size

all_predictions = []

for batch_idx in range(num_batches):
    start = batch_idx * batch_size
    end = min(start + batch_size, num_samples)

    batch_tokens = tokens[start:end]
    batch_mask = attention_mask[start:end]
    batch_num = end - start

    x_init = torch.randn(batch_num, model.phase_dim, device=device)
    phase_mask_tensor = torch.ones(batch_num, model.phase_dim, device=device)

    def ode_func(t, x):
        t_batch = torch.full((batch_num,), t.item() if t.dim() == 0 else t, device=device)
        return model.forward_flow(batch_tokens, batch_mask, x, phase_mask_tensor, t_batch, None, None)

    with torch.no_grad():
        prediction = odeint(ode_func, x_init, torch.tensor([0.0, 1.0], device=device), method='euler')[-1]
    all_predictions.append(prediction.cpu().numpy())

    if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
        print(f"  Batch {batch_idx + 1}/{num_batches}")

# 合并所有预测
prediction = np.concatenate(all_predictions, axis=0)
print(f"\nPrediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")

# 计算每个样本的16维均值
gt_means = ground_truth.mean(axis=1)  # (N,)
pred_means = prediction.mean(axis=1)   # (N,)

print(f"GT mean per sample range: [{gt_means.min():.3f}, {gt_means.max():.3f}]")
print(f"Pred mean per sample range: [{pred_means.min():.3f}, {pred_means.max():.3f}]")

# 计算 Spearman 相关系数
spearman_corr, p_value = spearmanr(gt_means, pred_means)
pearson_corr = np.corrcoef(gt_means, pred_means)[0, 1]
mse = np.mean((pred_means - gt_means) ** 2)
mae = np.mean(np.abs(pred_means - gt_means))

print("\n" + "="*50)
print("Mean-based Correlation Metrics:")
print("="*50)
print(f"  Spearman correlation: {spearman_corr:.4f} (p={p_value:.2e})")
print(f"  Pearson correlation: {pearson_corr:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  MAE: {mae:.4f}")
print("="*50)

# 绘制散点图
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(gt_means, pred_means, alpha=0.3, s=5)

# 对角线
min_val = min(gt_means.min(), pred_means.min())
max_val = max(gt_means.max(), pred_means.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

ax.set_xlabel('True Mean PSSI (16 groups)')
ax.set_ylabel('Predicted Mean PSSI (16 groups)')
ax.set_title(f'PhaseFlow (flow32) - Mean PSSI Correlation\nSpearman: {spearman_corr:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/mean_correlation_scatter.png', dpi=150, bbox_inches='tight')
print(f"\nSaved scatter plot to {OUTPUT_DIR}/mean_correlation_scatter.png")

# 保存结果
indices = np.arange(len(sequences))  # 默认使用所有索引
np.savez(f'{OUTPUT_DIR}/mean_correlation_results.npz',
         gt_means=gt_means,
         pred_means=pred_means,
         gt_full=ground_truth,
         pred_full=prediction,
         indices=indices)

print("\nDone!")
