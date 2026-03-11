#!/usr/bin/env python3
"""Plot comprehensive LM evaluation results from lm_evaluation.json.

Generates lm_comprehensive_eval.png with key metrics for all models.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = '/data/yanjie_huang/LLPS/predictor/PhaseFlow'
RESULTS_PATH = f'{BASE_DIR}/infer/lm_evaluation_results/lm_evaluation.json'
OUT_PATH = f'{BASE_DIR}/infer/lm_evaluation_results/lm_comprehensive_eval.png'

with open(RESULTS_PATH) as f:
    data = json.load(f)

# Model ordering
flow_weights = ['flow32', 'flow5', 'flow1', 'flow0']
missings = ['m0', 'm3', 'm7', 'm11']

set_models = [f'set_{g}_{m}' for g in flow_weights for m in missings]
legacy_models = [f'{g}_{m}' for g in flow_weights for m in missings]

set_models = [m for m in set_models if m in data]
legacy_models = [m for m in legacy_models if m in data]

group_colors = {'flow32': '#E74C3C', 'flow5': '#E67E22', 'flow1': '#3498DB', 'flow0': '#2ECC71'}

def get_group(name):
    return name.replace('set_', '').rsplit('_', 1)[0]

def short_label(name):
    return name.replace('set_', 's_')

# ======================== Main Figure: 3×2 layout ========================
fig, axes = plt.subplots(3, 2, figsize=(22, 16))
fig.suptitle('PhaseFlow LM Evaluation: Set Encoder vs Legacy', fontsize=18, fontweight='bold', y=0.98)

metrics = [
    ('roundtrip_mean_spearman', 'Self-RT Mean Spearman', (-0.15, 0.6)),
    ('kl_div', 'AA KL Divergence', (0, 0.025)),
    ('len_mean', 'Mean Sequence Length', (12, 20)),
]

for col_idx, (title, models) in enumerate([
    ('Set Encoder', set_models),
    ('Legacy', legacy_models),
]):
    if not models:
        continue

    colors = [group_colors[get_group(m)] for m in models]
    labels = [short_label(m) for m in models]
    x = np.arange(len(models))

    for row_idx, (metric, ylabel, ylim) in enumerate(metrics):
        ax = axes[row_idx, col_idx]
        vals = []
        for m in models:
            v = data[m].get(metric, 0)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 0
            vals.append(v)

        ax.bar(x, vals, 0.6, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(ylabel)
        if row_idx == 0:
            ax.set_title(f'{title}\n{ylabel}')
        else:
            ax.set_title(ylabel)
        ax.set_ylim(ylim)
        ax.axhline(y=0, color='black', linewidth=0.3)
        for i in [4, 8, 12]:
            if i < len(models):
                ax.axvline(x=i - 0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

        # Add value labels for RT
        if metric == 'roundtrip_mean_spearman':
            for i, v in enumerate(vals):
                if abs(v) > 0.01:
                    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved to {OUT_PATH}')

# ======================== Direct comparison plot ========================
OUT_PATH2 = f'{BASE_DIR}/infer/lm_evaluation_results/lm_set_vs_legacy.png'

compare_metrics = [
    ('roundtrip_mean_spearman', 'Self-RT Mean Spearman'),
    ('roundtrip_spearman', 'Self-RT Flat Spearman'),
    ('kl_div', 'AA KL Divergence'),
    ('len_mean', 'Mean Length'),
    ('len_gt20_pct', '>20aa %'),
    ('novelty', 'Novelty'),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('Set Encoder vs Legacy: LM Metrics Comparison', fontsize=14, fontweight='bold')

for ax_idx, (metric, label) in enumerate(compare_metrics):
    ax = axes[ax_idx // 3, ax_idx % 3]
    x = np.arange(len(flow_weights) * len(missings))
    bar_w = 0.35

    set_vals = []
    leg_vals = []
    tick_labels = []
    for g in flow_weights:
        for m in missings:
            sname = f'set_{g}_{m}'
            lname = f'{g}_{m}'
            sv = data[sname].get(metric, 0) if sname in data else 0
            lv = data[lname].get(metric, 0) if lname in data else 0
            if sv is None or (isinstance(sv, float) and np.isnan(sv)):
                sv = 0
            if lv is None or (isinstance(lv, float) and np.isnan(lv)):
                lv = 0
            # Novelty stored as fraction, display as percentage
            if metric == 'novelty':
                sv *= 100
                lv *= 100
            set_vals.append(sv)
            leg_vals.append(lv)
            tick_labels.append(f'{g}_{m}')

    ax.bar(x - bar_w/2, set_vals, bar_w, label='Set Encoder', color='#E67E22', alpha=0.8, edgecolor='white')
    ax.bar(x + bar_w/2, leg_vals, bar_w, label='Legacy', color='#3498DB', alpha=0.8, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=5)
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend(fontsize=8)
    for i in [4, 8, 12]:
        ax.axvline(x=i - 0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH2, dpi=150, bbox_inches='tight')
print(f'Saved to {OUT_PATH2}')
