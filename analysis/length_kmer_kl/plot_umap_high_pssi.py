#!/usr/bin/env python3
"""
UMAP 可视化：仅针对 mean PSSI > 0.3 的序列
随机序列采样相同条数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from scipy.sparse import csr_matrix
from scipy.stats import entropy
from umap import UMAP
from sklearn.preprocessing import normalize

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


def load_kmer_features(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    return csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])


def compute_kl_divergence(p_matrix, q_matrix, eps=1e-10):
    p_mean = np.asarray(p_matrix.mean(axis=0)).flatten() + eps
    q_mean = np.asarray(q_matrix.mean(axis=0)).flatten() + eps
    p_mean /= p_mean.sum()
    q_mean /= q_mean.sum()
    kl_pq = entropy(p_mean, q_mean)
    kl_qp = entropy(q_mean, p_mean)
    return kl_pq, kl_qp


# --- 筛选 mean PSSI > 0.3 的序列索引 ---
print("Filtering sequences with mean PSSI > 0.3...")

gt_df = pd.read_csv('outputs/dataset_groundtruth/missing_0.csv')
pssi_cols = [c for c in gt_df.columns if c.startswith('group_')]
gt_df['mean_pssi'] = gt_df[pssi_cols].mean(axis=1)
high_pssi_mask = gt_df['mean_pssi'] > 0.3
high_pssi_idx = np.where(high_pssi_mask.values)[0]
n_high = len(high_pssi_idx)
print(f"  Found {n_high} sequences with mean PSSI > 0.3")

# --- 加载 k-mer 特征并按索引筛选 ---
print("Loading k-mer features...")
gt_all    = load_kmer_features('outputs/dataset_groundtruth/kmer_features.npz')
gen1x_all = load_kmer_features('outputs/dataset_1x/kmer_features.npz')
rand_all  = load_kmer_features('outputs/dataset_random/kmer_features.npz')

gt_high  = gt_all[high_pssi_idx]
gen_high = gen1x_all[high_pssi_idx]

np.random.seed(42)
rand_idx = np.random.choice(rand_all.shape[0], n_high, replace=False)
rand_sub = rand_all[rand_idx]

print(f"  GT: {gt_high.shape[0]}, Gen: {gen_high.shape[0]}, Random: {rand_sub.shape[0]}")

# --- KL 散度 ---
print("Computing KL divergences...")
kl_gt_gen, kl_gen_gt = compute_kl_divergence(gt_high, gen_high)
kl_gt_rnd, kl_rnd_gt = compute_kl_divergence(gt_high, rand_sub)
print(f"  KL(GT || Gen) = {kl_gt_gen:.4f},  KL(Gen || GT) = {kl_gen_gt:.4f}")
print(f"  KL(GT || Rand) = {kl_gt_rnd:.4f}, KL(Rand || GT) = {kl_rnd_gt:.4f}")

# --- UMAP 1: groundtruth + phaseflow ---
print("Running UMAP 1 (groundtruth + phaseflow, high PSSI)...")
combined1 = normalize(np.vstack([gt_high.toarray(), gen_high.toarray()]), norm='l2')
emb1 = UMAP(n_neighbors=30, min_dist=0.01, spread=0.5, n_components=2,
            metric='cosine', random_state=42, n_jobs=1).fit_transform(combined1)
gt_emb1  = emb1[:n_high]
gen_emb1 = emb1[n_high:]

# --- UMAP 2: groundtruth + random ---
print("Running UMAP 2 (groundtruth + random, high PSSI)...")
combined2 = normalize(np.vstack([gt_high.toarray(), rand_sub.toarray()]), norm='l2')
emb2 = UMAP(n_neighbors=30, min_dist=0.01, spread=0.5, n_components=2,
            metric='cosine', random_state=42, n_jobs=1).fit_transform(combined2)
gt_emb2   = emb2[:n_high]
rand_emb2 = emb2[n_high:]

# --- 画图 ---
print("Plotting...")
fig, axes = plt.subplots(1, 2, figsize=(36, 18), facecolor='white')

ax = axes[0]
ax.set_facecolor('white')
ax.scatter(gt_emb1[:, 0], gt_emb1[:, 1],
           c=[COLOR_GT], alpha=0.8, s=800, edgecolors='none', label='Groundtruth')
ax.scatter(gen_emb1[:, 0], gen_emb1[:, 1],
           c=[COLOR_GEN], alpha=0.8, s=800, edgecolors='none', label='PhaseFlow')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title(f'Groundtruth vs PhaseFlow\n(mean PSSI > 0.3, n={n_high})', pad=20, fontweight='normal')
ax.legend(loc='upper left', framealpha=0.8, edgecolor='gray', fancybox=False)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', width=4, length=10)
for spine in ax.spines.values():
    spine.set_linewidth(4)
ax.text(0.98, 0.04,
        f'KL(Gen\u2016GT) = {kl_gen_gt:.4f}',
        transform=ax.transAxes, fontsize=BASE_SIZE * 2.0,
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))

ax = axes[1]
ax.set_facecolor('white')
ax.scatter(gt_emb2[:, 0], gt_emb2[:, 1],
           c=[COLOR_GT], alpha=0.8, s=800, edgecolors='none', label='Groundtruth')
ax.scatter(rand_emb2[:, 0], rand_emb2[:, 1],
           c=[COLOR_RND], alpha=0.8, s=800, edgecolors='none', label='Random')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title(f'Groundtruth vs Random\n(mean PSSI > 0.3, n={n_high})', pad=20, fontweight='normal')
ax.legend(loc='upper left', framealpha=0.8, edgecolor='gray', fancybox=False)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', width=4, length=10)
for spine in ax.spines.values():
    spine.set_linewidth(4)
ax.text(0.98, 0.04,
        f'KL(Rand\u2016GT) = {kl_rnd_gt:.4f}',
        transform=ax.transAxes, fontsize=BASE_SIZE * 2.0,
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08, wspace=0.25)

out = 'outputs/umap_comparison_high_pssi.png'
plt.savefig(out, dpi=300, pad_inches=1.5, facecolor='white')
print(f"Saved to {out}")
