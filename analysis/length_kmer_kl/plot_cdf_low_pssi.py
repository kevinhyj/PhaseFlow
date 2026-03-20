#!/usr/bin/env python3
"""
绘制序列长度的累积分布函数（CDF）- 仅 mean PSSI < -0.5 的序列
随机序列采样相同条数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager

BASE_SIZE = 14
font_manager.fontManager.addfont('/data/yanjie_huang/fonts/arial.ttf')
plt.rcParams.update({
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
    'font.size': BASE_SIZE * 2,
    'axes.titlesize': BASE_SIZE * 4,
    'axes.labelsize': BASE_SIZE * 4,
    'xtick.labelsize': BASE_SIZE * 3.2,
    'ytick.labelsize': BASE_SIZE * 3.2,
    'legend.fontsize': BASE_SIZE * 1.8 * 1.2
})

COLOR_GT  = (114/255, 138/255, 185/255)
COLOR_GEN = (245/255, 181/255, 191/255)
COLOR_RND = (245/255, 221/255, 181/255)

# --- 加载数据并筛选 mean PSSI < -0.5 ---
print("加载数据...")
gt_df = pd.read_csv('outputs/dataset_groundtruth/missing_0.csv')
gen1x_df = pd.read_csv('outputs/dataset_1x/sequences.csv')
random_df = pd.read_csv('outputs/dataset_random/random_sequences.csv')

pssi_cols = [c for c in gt_df.columns if c.startswith('group_')]
gt_df['mean_pssi'] = gt_df[pssi_cols].mean(axis=1)
low_mask = gt_df['mean_pssi'] < -0.5
n_low = low_mask.sum()
print(f"筛选 mean PSSI < -0.5: {n_low} 条")

gt_low = gt_df[low_mask]
gen_low = gen1x_df[low_mask.values]

# 随机序列采样相同条数
np.random.seed(42)
rand_low = random_df.sample(n=n_low, random_state=42)

gt_lengths = gt_low['AminoAcidSequence'].str.len().values
gen_lengths = gen_low['generated_sequence'].str.len().values
rand_lengths = rand_low['AminoAcidSequence'].str.len().values

print(f"Groundtruth: {len(gt_lengths)} 条, 长度 {gt_lengths.min()}-{gt_lengths.max()}")
print(f"Generated:   {len(gen_lengths)} 条, 长度 {gen_lengths.min()}-{gen_lengths.max()}")
print(f"Random:      {len(rand_lengths)} 条, 长度 {rand_lengths.min()}-{rand_lengths.max()}")

# --- 绘制 CDF ---
fig, axes = plt.subplots(1, 2, figsize=(30, 15), facecolor='white')

# 图1: Groundtruth vs Generated
ax = axes[0]
ax.set_facecolor('white')
sorted_gt = np.sort(gt_lengths)
sorted_gen = np.sort(gen_lengths)
cdf_gt = np.arange(1, len(sorted_gt) + 1) / len(sorted_gt)
cdf_gen = np.arange(1, len(sorted_gen) + 1) / len(sorted_gen)

ax.plot(sorted_gt, cdf_gt, label='Groundtruth', color=COLOR_GT, linewidth=8, alpha=0.95)
ax.plot(sorted_gen, cdf_gen, label='PhaseFlow', color=COLOR_GEN, linewidth=8, alpha=0.95, linestyle='--')
ax.set_xlabel('Sequence Length (AA)')
ax.set_ylabel('CDF')
ax.set_title(f'CDF: Groundtruth vs PhaseFlow\n(mean PSSI < -0.5, n={n_low})', pad=20, fontweight='normal')
ax.legend(loc='upper left', framealpha=0.8, edgecolor='gray', fancybox=False)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', width=4, length=10)
for spine in ax.spines.values():
    spine.set_linewidth(4)
ax.set_xlim(0, 25)

# 图2: Groundtruth vs Random
ax = axes[1]
ax.set_facecolor('white')
sorted_rand = np.sort(rand_lengths)
cdf_rand = np.arange(1, len(sorted_rand) + 1) / len(sorted_rand)

ax.plot(sorted_gt, cdf_gt, label='Groundtruth', color=COLOR_GT, linewidth=8, alpha=0.95)
ax.plot(sorted_rand, cdf_rand, label='Random', color=COLOR_RND, linewidth=8, alpha=0.95, linestyle='--')
ax.set_xlabel('Sequence Length (AA)')
ax.set_ylabel('CDF')
ax.set_title(f'CDF: Groundtruth vs Random\n(mean PSSI < -0.5, n={n_low})', pad=20, fontweight='normal')
ax.legend(loc='upper left', framealpha=0.8, edgecolor='gray', fancybox=False)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', width=4, length=10)
for spine in ax.spines.values():
    spine.set_linewidth(4)
ax.set_xlim(0, 25)

plt.tight_layout(pad=1.0)
plt.subplots_adjust(wspace=0.25)

output_path = 'outputs/cdf_length_comparison_low_pssi.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n保存到 {output_path}")

print("\n序列长度统计:")
print(f"Groundtruth:  mean={gt_lengths.mean():.2f}, median={np.median(gt_lengths):.0f}, std={gt_lengths.std():.2f}")
print(f"Generated:    mean={gen_lengths.mean():.2f}, median={np.median(gen_lengths):.0f}, std={gen_lengths.std():.2f}")
print(f"Random:       mean={rand_lengths.mean():.2f}, median={np.median(rand_lengths):.0f}, std={rand_lengths.std():.2f}")
