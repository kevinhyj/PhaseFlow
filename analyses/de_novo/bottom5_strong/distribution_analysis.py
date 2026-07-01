#!/usr/bin/env python3
"""
De novo vs GT 分布对比 — Bottom 5% 强相分离 (Strong LLPS)

对比三组：
1. De Novo ITO (top-1 per phase)
2. De Novo Gen (1 per phase, sampled from candidates)
3. GT Bottom 5% (原始序列)

特征：疏水性、芳香族、电荷、极性、GRAVY、AA 组成
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from scipy.special import rel_entr


def kl_divergence_continuous(p_vals, q_vals, n_bins=50):
    lo = min(min(p_vals), min(q_vals))
    hi = max(max(p_vals), max(q_vals))
    bins = np.linspace(lo - 1e-8, hi + 1e-8, n_bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, density=True)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    return float(np.sum(rel_entr(p_hist, q_hist)))


# ============================================================================
# Feature computation
# ============================================================================

KD_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

HYDROPHOBIC = set('AVILMFWP')
AROMATIC = set('FWY')
CHARGED = set('RKDE')
POSITIVE = set('RK')
NEGATIVE = set('DE')
POLAR = set('STNQ')


def compute_features(seq):
    n = len(seq)
    if n == 0:
        return {}
    counts = Counter(seq)

    gravy = sum(KD_SCALE.get(aa, 0) for aa in seq) / n
    hydro_frac = sum(1 for aa in seq if aa in HYDROPHOBIC) / n
    arom_frac = sum(1 for aa in seq if aa in AROMATIC) / n
    charged_frac = sum(1 for aa in seq if aa in CHARGED) / n
    pos_charge = sum(1 for aa in seq if aa in POSITIVE)
    neg_charge = sum(1 for aa in seq if aa in NEGATIVE)
    net_charge = (pos_charge - neg_charge) / n
    polar_frac = sum(1 for aa in seq if aa in POLAR) / n

    f_frac = counts.get('F', 0) / n
    w_frac = counts.get('W', 0) / n
    y_frac = counts.get('Y', 0) / n

    return {
        'length': n,
        'gravy': gravy,
        'hydrophobic_frac': hydro_frac,
        'aromatic_frac': arom_frac,
        'charged_frac': charged_frac,
        'net_charge_per_res': net_charge,
        'polar_frac': polar_frac,
        'F_frac': f_frac,
        'W_frac': w_frac,
        'Y_frac': y_frac,
    }


def compute_aa_composition(seqs):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    all_freqs = []
    for seq in seqs:
        n = len(seq)
        if n == 0:
            continue
        counts = Counter(seq)
        freqs = [counts.get(aa, 0) / n for aa in AA]
        all_freqs.append(freqs)
    return np.array(all_freqs), list(AA)


def generate_random_seqs(n=509, seed=42):
    rng = np.random.RandomState(seed)
    AA = list('ACDEFGHIKLMNPQRSTVWY')
    seqs = []
    for _ in range(n):
        length = rng.randint(5, 21)
        seqs.append(''.join(rng.choice(AA, size=length)))
    return seqs


# ============================================================================
# Plotting
# ============================================================================

def plot_feature_comparison(feat_dict, feature_names, output_path, kl_dict=None):
    n_feat = len(feature_names)
    fig, axes = plt.subplots(2, (n_feat + 1) // 2, figsize=(20, 10))
    axes = axes.flatten()

    colors = [
        (245/255, 221/255, 181/255),  # Random 浅橙
        (181/255, 211/255, 185/255),  # ITO 鼠尾草绿
        (245/255, 181/255, 191/255),  # Gen 粉
        (114/255, 138/255, 185/255),  # GT 蓝灰
    ]
    labels = list(feat_dict.keys())

    for i, feat in enumerate(feature_names):
        ax = axes[i]
        ax.set_facecolor('white')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        data = [feat_dict[label][feat] for label in labels]
        parts = ax.violinplot(data, positions=[1, 2, 3, 4.5],
                              showmeans=True, showmedians=True)
        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[j])
            pc.set_alpha(0.6)
        parts['cmeans'].set_color('black')
        parts['cmedians'].set_color('gray')

        ax.axvline(3.75, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xticks([1, 2, 3, 4.5])
        ax.set_xticklabels(labels, fontsize=8.5)

        title = feat
        if kl_dict and feat in kl_dict:
            kl_rnd, kl_ito, kl_gen = kl_dict[feat]
            title += f'\nKL: Rnd={kl_rnd:.3f}, ITO={kl_ito:.3f}, Gen={kl_gen:.3f}'
        ax.set_title(title, fontsize=10)

        for j, (pos, d) in enumerate(zip([1, 2, 3, 4.5], data)):
            ax.text(pos, ax.get_ylim()[1] * 0.95, f'{np.mean(d):.3f}',
                    ha='center', fontsize=8, color=colors[j])

    for i in range(n_feat, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('LLPS Key Feature Distribution: De Novo vs GT (Bottom 5% Strong LLPS)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {output_path}')


def plot_aa_composition(comp_dict, aa_list, output_path):
    labels = list(comp_dict.keys())
    n_aa = len(aa_list)
    x = np.arange(n_aa)
    n_groups = len(labels)
    width = 0.18
    colors = [
        (181/255, 211/255, 185/255),  # ITO
        (245/255, 181/255, 191/255),  # Gen
        (245/255, 221/255, 181/255),  # Random
        (114/255, 138/255, 185/255),  # GT
    ]

    fig, ax = plt.subplots(figsize=(16, 5), facecolor='white')
    ax.set_facecolor('white')
    offsets = np.linspace(-(n_groups-1)/2, (n_groups-1)/2, n_groups) * width
    for i, label in enumerate(labels):
        means = comp_dict[label].mean(axis=0)
        ax.bar(x + offsets[i], means, width, label=label, color=colors[i], alpha=0.85, edgecolor='none')

    ax.set_xticks(x)
    ax.set_xticklabels(aa_list, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Amino Acid Composition: De Novo vs GT (Bottom 5% Strong LLPS)', fontsize=13)
    ax.legend(fontsize=10, edgecolor='gray', fancybox=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {output_path}')


# ============================================================================
# Main
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    OUT_DIR = os.path.join(SCRIPT_DIR, 'distribution_analysis')
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    ito = pd.read_csv(os.path.join(SCRIPT_DIR, 'de_novo_ito_results/ito_top1.csv'))
    gen = pd.read_csv(os.path.join(SCRIPT_DIR, 'de_novo_results/candidates.csv'))
    gt = pd.read_csv(os.path.join(SCRIPT_DIR, 'ground_truth.csv'))

    denovo_ito_seqs = ito['sequence'].tolist()
    gt_seqs = gt['gt_sequence'].tolist()

    gen_sampled = gen.groupby('target_phase_idx').apply(
        lambda x: x.sample(1, random_state=42)
    ).reset_index(drop=True)
    denovo_gen_seqs = gen_sampled['sequence'].tolist()

    random_seqs = generate_random_seqs(n=len(denovo_ito_seqs), seed=42)

    print(f'De novo ITO:        {len(denovo_ito_seqs)}')
    print(f'De novo Gen (1/ph): {len(denovo_gen_seqs)}')
    print(f'Random:             {len(random_seqs)}')
    print(f'GT Bottom 5%:       {len(gt_seqs)}')

    groups = {
        'Random':      random_seqs,
        'De Novo ITO': denovo_ito_seqs,
        'De Novo Gen': denovo_gen_seqs,
        'GT Bottom5%': gt_seqs,
    }

    feature_names = [
        'gravy', 'hydrophobic_frac', 'aromatic_frac',
        'charged_frac', 'net_charge_per_res', 'polar_frac',
        'F_frac', 'W_frac', 'Y_frac', 'length',
    ]

    feat_dict = {}
    for label, seqs in groups.items():
        feats = [compute_features(s) for s in seqs]
        feat_dict[label] = {k: [f[k] for f in feats] for k in feature_names}

    # Print stats with KL divergence
    gt_label = 'GT Bottom5%'
    print('\n' + '=' * 90)
    print(f'{"Feature":<22} {"De Novo ITO":>14} {"De Novo Gen":>14} {"GT Bottom5%":>14} {"KL(ITO||GT)":>12} {"KL(Gen||GT)":>12}')
    print('=' * 90)
    for feat in feature_names:
        vals = [np.mean(feat_dict[l][feat]) for l in feat_dict]
        kl_ito = kl_divergence_continuous(feat_dict['De Novo ITO'][feat], feat_dict[gt_label][feat])
        kl_gen = kl_divergence_continuous(feat_dict['De Novo Gen'][feat], feat_dict[gt_label][feat])
        print(f'{feat:<22} {vals[0]:>14.4f} {vals[1]:>14.4f} {vals[2]:>14.4f} {kl_ito:>12.4f} {kl_gen:>12.4f}')
    print('=' * 90)

    # Save stats
    stats_rows = []
    for feat in feature_names:
        row = {'feature': feat}
        for label in feat_dict:
            row[f'{label}_mean'] = np.mean(feat_dict[label][feat])
            row[f'{label}_std'] = np.std(feat_dict[label][feat])
        row['KL_ITO_vs_GT'] = kl_divergence_continuous(feat_dict['De Novo ITO'][feat], feat_dict[gt_label][feat])
        row['KL_Gen_vs_GT'] = kl_divergence_continuous(feat_dict['De Novo Gen'][feat], feat_dict[gt_label][feat])
        stats_rows.append(row)
    pd.DataFrame(stats_rows).to_csv(os.path.join(OUT_DIR, 'feature_stats.csv'), index=False)

    # Compute KL for plotting
    kl_dict = {}
    for feat in feature_names:
        kl_rnd = kl_divergence_continuous(feat_dict['Random'][feat], feat_dict[gt_label][feat])
        kl_ito = kl_divergence_continuous(feat_dict['De Novo ITO'][feat], feat_dict[gt_label][feat])
        kl_gen = kl_divergence_continuous(feat_dict['De Novo Gen'][feat], feat_dict[gt_label][feat])
        kl_dict[feat] = (kl_rnd, kl_ito, kl_gen)

    # Plot features
    plot_feature_comparison(feat_dict, feature_names,
                            os.path.join(OUT_DIR, 'feature_violin.png'),
                            kl_dict=kl_dict)

    # AA composition
    comp_dict = {}
    for label, seqs in groups.items():
        comp, aa_list = compute_aa_composition(seqs)
        comp_dict[label] = comp
    plot_aa_composition(comp_dict, aa_list,
                        os.path.join(OUT_DIR, 'aa_composition.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
