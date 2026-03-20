#!/usr/bin/env python3
"""
绘制序列长度的累积分布函数（CDF）

对比 Groundtruth, Generated 1x, Random 的序列长度分布
颜色：groundtruth=#728ab9, generated=#f5b5bf, random=#f5ddb5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager

# 字体配置（按RNA脚本比例，但整体缩小）
BASE_SIZE = 14  # 从17缩小到14
font_manager.fontManager.addfont('/data/yanjie_huang/fonts/arial.ttf')
plt.rcParams.update({
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
    'font.size': BASE_SIZE * 2,
    'axes.titlesize': BASE_SIZE * 4,
    'axes.labelsize': BASE_SIZE * 4,  # 和标题一样大
    'xtick.labelsize': BASE_SIZE * 3.2,
    'ytick.labelsize': BASE_SIZE * 3.2,
    'legend.fontsize': BASE_SIZE * 1.8 * 1.2  # 图例字体保持不变（相对放大）
})

# 颜色定义（保持原来的配色）
COLOR_GT  = (114/255, 138/255, 185/255)  # #728ab9
COLOR_GEN = (245/255, 181/255, 191/255)  # #f5b5bf
COLOR_RND = (245/255, 221/255, 181/255)  # #f5ddb5

# 加载数据
print("加载数据...")
gt_df = pd.read_csv('outputs/dataset_groundtruth/missing_0.csv')
gen1x_df = pd.read_csv('outputs/dataset_1x/sequences.csv')
random_df = pd.read_csv('outputs/dataset_random/random_sequences.csv')

gt_lengths = gt_df['AminoAcidSequence'].str.len().values
gen1x_lengths = gen1x_df['generated_sequence'].str.len().values
random_lengths = random_df['AminoAcidSequence'].str.len().values

print(f"Groundtruth: {len(gt_lengths)} 条, 长度 {gt_lengths.min()}-{gt_lengths.max()}")
print(f"Generated 1x: {len(gen1x_lengths)} 条, 长度 {gen1x_lengths.min()}-{gen1x_lengths.max()}")
print(f"Random: {len(random_lengths)} 条, 长度 {random_lengths.min()}-{random_lengths.max()}")

# 绘制 CDF
fig, axes = plt.subplots(1, 2, figsize=(30, 15), facecolor='white')

# 图1: Groundtruth vs Generated 1x
ax = axes[0]
ax.set_facecolor('white')
sorted_gt = np.sort(gt_lengths)
sorted_gen1x = np.sort(gen1x_lengths)
cdf_gt = np.arange(1, len(sorted_gt) + 1) / len(sorted_gt)
cdf_gen1x = np.arange(1, len(sorted_gen1x) + 1) / len(sorted_gen1x)

ax.plot(sorted_gt, cdf_gt, label='Groundtruth', color=COLOR_GT, linewidth=8, alpha=0.95)
ax.plot(sorted_gen1x, cdf_gen1x, label='PhaseFlow', color=COLOR_GEN, linewidth=8, alpha=0.95, linestyle='--')
ax.set_xlabel('Sequence Length (AA)')
ax.set_ylabel('CDF')
ax.set_title('CDF: Groundtruth vs PhaseFlow', pad=20, fontweight='normal')
ax.legend(loc='upper left', framealpha=0.8, edgecolor='gray', fancybox=False)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', width=4, length=10)
for spine in ax.spines.values():
    spine.set_linewidth(4)
ax.set_xlim(0, 25)

# 图2: Groundtruth vs Random
ax = axes[1]
ax.set_facecolor('white')
sorted_random = np.sort(random_lengths)
cdf_random = np.arange(1, len(sorted_random) + 1) / len(sorted_random)

ax.plot(sorted_gt, cdf_gt, label='Groundtruth', color=COLOR_GT, linewidth=8, alpha=0.95)
ax.plot(sorted_random, cdf_random, label='Random', color=COLOR_RND, linewidth=8, alpha=0.95, linestyle='--')
ax.set_xlabel('Sequence Length (AA)')
ax.set_ylabel('CDF')
ax.set_title('CDF: Groundtruth vs Random', pad=20, fontweight='normal')
ax.legend(loc='upper left', framealpha=0.8, edgecolor='gray', fancybox=False)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', width=4, length=10)
for spine in ax.spines.values():
    spine.set_linewidth(4)
ax.set_xlim(0, 25)

plt.tight_layout(pad=1.0)
plt.subplots_adjust(wspace=0.25)

output_path = 'outputs/cdf_length_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ 保存到 {output_path}")

# 统计信息
print("\n序列长度统计:")
print(f"Groundtruth:    mean={gt_lengths.mean():.2f}, median={np.median(gt_lengths):.0f}, std={gt_lengths.std():.2f}")
print(f"Generated 1x:   mean={gen1x_lengths.mean():.2f}, median={np.median(gen1x_lengths):.0f}, std={gen1x_lengths.std():.2f}")
print(f"Random:         mean={random_lengths.mean():.2f}, median={np.median(random_lengths):.0f}, std={random_lengths.std():.2f}")
