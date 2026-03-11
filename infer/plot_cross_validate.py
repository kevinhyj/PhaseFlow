#!/usr/bin/env python3
"""Visualize Seq2Phase cross-validation results vs PhaseFlow self-RT.

Handles both Set Encoder (per-position velocity) and Legacy (single token) models.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = '/data/yanjie_huang/LLPS/predictor/PhaseFlow'
RESULTS_PATH = f'{BASE_DIR}/infer/lm_evaluation_results/cross_validate_results.json'
OUT_PATH = f'{BASE_DIR}/infer/lm_evaluation_results/cross_validate.png'

with open(RESULTS_PATH) as f:
    data = json.load(f)

# Model groups
flow_weights = ['flow32', 'flow5', 'flow1', 'flow0']
missings = ['m0', 'm3', 'm7', 'm11']

set_models = [f'set_{g}_{m}' for g in flow_weights for m in missings]
legacy_models = [f'{g}_{m}' for g in flow_weights for m in missings]

# Filter to available models
set_models = [m for m in set_models if m in data]
legacy_models = [m for m in legacy_models if m in data]

def get_vals(models):
    self_rt = [data[m]['self_RT_mean'] for m in models]
    cross_rt_fl = [data[m]['cross_RT_flat'] for m in models]
    cross_rt_mn = [data[m]['cross_RT_mean'] for m in models]
    delta = [data[m]['cross_RT_mean'] - data[m]['self_RT_mean'] for m in models]
    return self_rt, cross_rt_fl, cross_rt_mn, delta

group_colors = {'flow32': '#E74C3C', 'flow5': '#E67E22', 'flow1': '#3498DB', 'flow0': '#2ECC71'}

def get_group(name):
    # strip 'set_' prefix if present
    n = name.replace('set_', '')
    return n.rsplit('_', 1)[0]

def short_label(name):
    return name.replace('set_', 's_')

# ======================== Main Figure: 3×2 layout ========================
fig, axes = plt.subplots(3, 2, figsize=(20, 18))
fig.suptitle('Seq2Phase Cross-Validation: Set Encoder vs Legacy', fontsize=18, fontweight='bold', y=0.98)

for col_idx, (title, models) in enumerate([
    ('Set Encoder (Per-Position Velocity)', set_models),
    ('Legacy (Single Token)', legacy_models),
]):
    if not models:
        continue

    self_rt, cross_rt_fl, cross_rt_mn, delta = get_vals(models)
    colors = [group_colors[get_group(m)] for m in models]
    labels = [short_label(m) for m in models]
    x = np.arange(len(models))
    bar_w = 0.35

    # --- Row 0: Self-RT vs Cross-RT (Mean Spearman) ---
    ax = axes[0, col_idx]
    ax.bar(x - bar_w/2, self_rt, bar_w, label='Self-RT (PhaseFlow)', color='#95A5A6', edgecolor='white', linewidth=0.5)
    ax.bar(x + bar_w/2, cross_rt_mn, bar_w, label='Cross-RT (Seq2Phase)', color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean Spearman ρ')
    ax.set_title(f'{title}\nSelf-RT vs Cross-RT (Mean Spearman)')
    ax.legend(loc='upper left', fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    for i in [4, 8, 12]:
        if i < len(models):
            ax.axvline(x=i - 0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_ylim(-0.15, 0.6)

    # --- Row 1: Delta (Cross - Self) ---
    ax = axes[1, col_idx]
    bar_colors_delta = ['#2ECC71' if d > 0 else '#E74C3C' for d in delta]
    ax.bar(x, delta, 0.6, color=bar_colors_delta, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Δ (Cross_RT - Self_RT)')
    ax.set_title(f'Delta: Cross-RT minus Self-RT')
    ax.axhline(y=0, color='black', linewidth=1)
    for i in [4, 8, 12]:
        if i < len(models):
            ax.axvline(x=i - 0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    # Annotate flow0 region
    flow0_start = next((i for i, m in enumerate(models) if 'flow0' in m), None)
    if flow0_start is not None and flow0_start + 1 < len(delta):
        ax.annotate('flow0: LM learned\nphase→seq but\nflow head untrained',
                    xy=(flow0_start + 1, delta[flow0_start + 1]),
                    xytext=(flow0_start - 2, max(delta) * 0.7),
                    fontsize=7, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1', edgecolor='#2ECC71'))

    # --- Row 2: Grouped summary ---
    ax = axes[2, col_idx]
    group_self = []
    group_cross = []
    for g in flow_weights:
        prefix = f'set_{g}' if col_idx == 0 else g
        g_models = [f'{prefix}_{m}' for m in missings if f'{prefix}_{m}' in data]
        if g_models:
            group_self.append(np.mean([data[m]['self_RT_mean'] for m in g_models]))
            group_cross.append(np.mean([data[m]['cross_RT_mean'] for m in g_models]))
        else:
            group_self.append(0)
            group_cross.append(0)

    gx = np.arange(len(flow_weights))
    ax.bar(gx - bar_w/2, group_self, bar_w, label='Self-RT (avg)', color='#95A5A6', edgecolor='white')
    ax.bar(gx + bar_w/2, group_cross, bar_w, label='Cross-RT (avg)',
           color=[group_colors[g] for g in flow_weights], edgecolor='white')
    ax.set_xticks(gx)
    ax.set_xticklabels([f'flow_w={w}' for w in ['32', '5', '1', '0']], fontsize=10)
    ax.set_ylabel('Mean Spearman ρ (group avg)')
    ax.set_title(f'Group Average')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(-0.1, 0.6)
    for i, (s, c) in enumerate(zip(group_self, group_cross)):
        ax.text(i - bar_w/2, s + 0.01, f'{s:.3f}', ha='center', va='bottom', fontsize=8, color='#555')
        ax.text(i + bar_w/2, c + 0.01, f'{c:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Add legend for flow weight colors
patches = [mpatches.Patch(color=group_colors[g], label=g) for g in flow_weights]
fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=10, frameon=True,
           bbox_to_anchor=(0.5, 0.005))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved to {OUT_PATH}')


# ======================== Bonus: Set vs Legacy direct comparison ========================
OUT_PATH2 = f'{BASE_DIR}/infer/lm_evaluation_results/cross_validate_set_vs_legacy.png'

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Set Encoder vs Legacy: Cross-RT Comparison', fontsize=14, fontweight='bold')

for ax_idx, (metric, label) in enumerate([
    ('cross_RT_mean', 'Cross-RT Mean Spearman'),
    ('cross_RT_flat', 'Cross-RT Flat Spearman'),
    ('self_RT_mean', 'Self-RT Mean Spearman'),
]):
    ax = axes[ax_idx]
    x = np.arange(len(flow_weights) * len(missings))
    bar_w = 0.35

    set_vals = []
    leg_vals = []
    tick_labels = []
    for g in flow_weights:
        for m in missings:
            sname = f'set_{g}_{m}'
            lname = f'{g}_{m}'
            set_vals.append(data[sname][metric] if sname in data else 0)
            leg_vals.append(data[lname][metric] if lname in data else 0)
            tick_labels.append(f'{g}_{m}')

    ax.bar(x - bar_w/2, set_vals, bar_w, label='Set Encoder', color='#3498DB', alpha=0.8, edgecolor='white')
    ax.bar(x + bar_w/2, leg_vals, bar_w, label='Legacy', color='#E74C3C', alpha=0.8, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.5)
    for i in [4, 8, 12]:
        ax.axvline(x=i - 0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

    # Mark winners
    wins_set = sum(1 for s, l in zip(set_vals, leg_vals) if s > l and (s != 0 or l != 0))
    wins_leg = sum(1 for s, l in zip(set_vals, leg_vals) if l > s and (s != 0 or l != 0))
    ax.text(0.98, 0.02, f'Set wins: {wins_set}/16, Legacy wins: {wins_leg}/16',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUT_PATH2, dpi=150, bbox_inches='tight')
print(f'Saved to {OUT_PATH2}')
