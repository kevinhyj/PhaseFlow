#!/usr/bin/env python3
"""
UMAP 可视化：参考RNA生成模型风格
下采样1000条，点放大8倍，子图为方形
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from scipy.sparse import csr_matrix
from scipy.stats import entropy
from umap import UMAP
from sklearn.preprocessing import normalize

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


def load_kmer_features(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    return csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])


def compute_kl_divergence(p_matrix, q_matrix, eps=1e-10):
    """
    计算两个数据集的 KL 散度
    先计算每个 k-mer 的平均频率分布，再计算 KL(P || Q)
    """
    p_mean = np.asarray(p_matrix.mean(axis=0)).flatten() + eps
    q_mean = np.asarray(q_matrix.mean(axis=0)).flatten() + eps
    p_mean /= p_mean.sum()
    q_mean /= q_mean.sum()
    kl_pq = entropy(p_mean, q_mean)
    kl_qp = entropy(q_mean, p_mean)
    return kl_pq, kl_qp


print("Loading k-mer features...")
gt    = load_kmer_features('outputs/dataset_groundtruth/kmer_features.npz')
gen1x = load_kmer_features('outputs/dataset_1x/kmer_features.npz')
rand  = load_kmer_features('outputs/dataset_random/kmer_features.npz')

# 下采样到1500条
print("Subsampling to 1500...")
np.random.seed(42)
n_sample = 1000  # 每类1000条，GT+Gen=2000
gt_idx = np.random.choice(gt.shape[0], n_sample, replace=False)
gen_idx = np.random.choice(gen1x.shape[0], n_sample, replace=False)
gt_sub = gt[gt_idx]
gen_sub = gen1x[gen_idx]

# 下采样random（同样1500条用于画图）
rand_idx = np.random.choice(rand.shape[0], n_sample, replace=False)
rand_sub = rand[rand_idx]

# KL 散度（用全量数据计算）
print("Computing KL divergences...")
kl_gt_gen, kl_gen_gt = compute_kl_divergence(gt, gen1x)
kl_gt_rnd, kl_rnd_gt = compute_kl_divergence(gt, rand)
print(f"  KL(GT || Gen1x) = {kl_gt_gen:.4f},  KL(Gen1x || GT) = {kl_gen_gt:.4f}")
print(f"  KL(GT || Random) = {kl_gt_rnd:.4f}, KL(Random || GT) = {kl_rnd_gt:.4f}")

# --- UMAP 1: groundtruth + 1x ---
print("Running UMAP 1 (groundtruth + phaseflow)...")
combined1 = normalize(np.vstack([gt_sub.toarray(), gen_sub.toarray()]), norm='l2')
emb1 = UMAP(n_neighbors=30, min_dist=0.01, spread=0.5, n_components=2,
            metric='cosine', random_state=42, n_jobs=1).fit_transform(combined1)
gt_emb1  = emb1[:gt_sub.shape[0]]
gen_emb1 = emb1[gt_sub.shape[0]:]

# --- UMAP 2: groundtruth + random ---
print("Running UMAP 2 (groundtruth + random)...")
combined2 = normalize(np.vstack([gt_sub.toarray(), rand_sub.toarray()]), norm='l2')
emb2 = UMAP(n_neighbors=30, min_dist=0.01, spread=0.5, n_components=2,
            metric='cosine', random_state=42, n_jobs=1).fit_transform(combined2)
gt_emb2   = emb2[:gt_sub.shape[0]]
rand_emb2 = emb2[gt_sub.shape[0]:]

# --- 画图 ---
print("Plotting...")
# 参考RNA脚本的布局方式：画布足够宽，子图自然接近方形
fig, axes = plt.subplots(1, 2, figsize=(36, 18), facecolor='white')

# 子图1: groundtruth vs phaseflow
ax = axes[0]
ax.set_facecolor('white')
ax.scatter(gt_emb1[:, 0], gt_emb1[:, 1],
           c=[COLOR_GT], alpha=0.8, s=800, edgecolors='none', label='Groundtruth')
ax.scatter(gen_emb1[:, 0], gen_emb1[:, 1],
           c=[COLOR_GEN], alpha=0.8, s=800, edgecolors='none', label='PhaseFlow')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('Groundtruth vs PhaseFlow', pad=20, fontweight='normal')
ax.legend(loc='upper left', framealpha=0.8, edgecolor='gray', fancybox=False)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', width=4, length=10)
for spine in ax.spines.values():
    spine.set_linewidth(4)
ax.text(0.98, 0.04,
        f'KL(Gen‖GT) = {kl_gen_gt:.4f}',
        transform=ax.transAxes, fontsize=BASE_SIZE * 2.0,
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))

# 子图2: groundtruth vs random
ax = axes[1]
ax.set_facecolor('white')
ax.scatter(gt_emb2[:, 0], gt_emb2[:, 1],
           c=[COLOR_GT], alpha=0.8, s=800, edgecolors='none', label='Groundtruth')
ax.scatter(rand_emb2[:, 0], rand_emb2[:, 1],
           c=[COLOR_RND], alpha=0.8, s=800, edgecolors='none', label='Random')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('Groundtruth vs Random', pad=20, fontweight='normal')
ax.legend(loc='upper left', framealpha=0.8, edgecolor='gray', fancybox=False)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', width=4, length=10)
for spine in ax.spines.values():
    spine.set_linewidth(4)
ax.text(0.98, 0.04,
        f'KL(Rand‖GT) = {kl_rnd_gt:.4f}',
        transform=ax.transAxes, fontsize=BASE_SIZE * 2.0,
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))

# 直接用subplots_adjust控制，不用tight_layout
plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, wspace=0.25)

out = 'outputs/umap_comparison.png'
plt.savefig(out, dpi=300, pad_inches=1.5, facecolor='white')
print(f"✓ Saved to {out}")
