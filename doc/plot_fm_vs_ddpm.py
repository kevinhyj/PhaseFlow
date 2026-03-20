#!/usr/bin/env python3
"""
双向分组柱状图: FlowMatching vs DDPM (m15 full grid)
左边 PPL (越低越好), 右边 Spearman (越高越好, Flat/Mean 叠加)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Arial font
font_path = '/data/yanjie_huang/fonts/arial.ttf'
arial_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = arial_prop.get_name()

# ============================================================
# Data: fw={1,5,32} × lm={0,1,5,32}, 12 configs × 2 systems
# Order: fw=32 lm=5 first, then fw desc → lm asc
# ============================================================

configs = [
    # (label,       FM_flat, FM_mean, FM_ppl,  DDPM_flat, DDPM_mean, DDPM_ppl)
    ('fw=32, lm=5',  0.4127,  0.5781,   8.71,   0.1489,   0.2605,    8.66),
    ('fw=32, lm=0',  0.4167,  0.5826,  31.97,   0.2038,   0.3371,   30.82),
    ('fw=32, lm=1',  0.4133,  0.5593,   8.25,   0.1653,   0.2773,    8.27),
    ('fw=32, lm=32', 0.4024,  0.5752,   9.63,   0.1593,   0.2420,    8.59),
    ('fw=5,  lm=0',  0.4110,  0.5515,  32.84,   0.1957,   0.3335,   31.48),
    ('fw=5,  lm=1',  0.3847,  0.5061,   8.80,   0.1634,   0.2718,    8.90),
    ('fw=5,  lm=5',  0.4013,  0.5657,   9.57,   0.1454,   0.2017,    8.58),
    ('fw=5,  lm=32', 0.3537,  0.4993,   8.46,   0.1318,   0.1764,    8.78),
    ('fw=1,  lm=0',  0.4140,  0.5637,  32.49,   0.1916,   0.3165,   31.77),
    ('fw=1,  lm=1',  0.3761,  0.5697,   9.52,   0.1118,   0.1214,    9.47),
    ('fw=1,  lm=5',  0.3622,  0.5160,   8.65,   0.1372,   0.1719,    8.56),
    ('fw=1,  lm=32', 0.3353,  0.5113,   8.64,   0.1195,   0.2245,    8.43),
]

labels = [c[0] for c in configs]
n = len(configs)

# Unpack
fm_flat  = np.array([c[1] for c in configs])
fm_mean  = np.array([c[2] for c in configs])
fm_ppl   = np.array([c[3] for c in configs])
ddpm_flat  = np.array([c[4] for c in configs])
ddpm_mean  = np.array([c[5] for c in configs])
ddpm_ppl   = np.array([c[6] for c in configs])

# Colors (morandi style)
FM_DEEP   = '#728ab9'   # FlowMatching Flat (deep blue-gray)
FM_LIGHT  = '#a8bbd6'   # FlowMatching Mean (light blue-gray)
FM_PPL    = '#8da3c7'   # FlowMatching PPL
DDPM_DEEP  = '#50b9ae'  # DDPM Flat (deep teal)
DDPM_LIGHT = '#8fd4cc'  # DDPM Mean (light teal)
DDPM_PPL   = '#6ec7bc'  # DDPM PPL

# Layout: each config gets 2 bars (FM on top, DDPM below), with gap between configs
bar_h = 0.35
gap = 0.25
group_h = 2 * bar_h + gap

y_positions_fm = []
y_positions_ddpm = []
for i in range(n):
    y_base = (n - 1 - i) * group_h
    y_positions_fm.append(y_base + bar_h / 2 + 0.02)
    y_positions_ddpm.append(y_base - bar_h / 2 - 0.02)

y_fm = np.array(y_positions_fm)
y_ddpm = np.array(y_positions_ddpm)

fig, ax = plt.subplots(figsize=(16, 12))

# === LEFT SIDE: PPL (negative direction) ===
# Normalize PPL for display: use negative values
ppl_scale = 0.015  # scale PPL to be comparable with Spearman visually

ax.barh(y_fm, -fm_ppl * ppl_scale, height=bar_h,
        color=FM_PPL, edgecolor='#5a5a5a', linewidth=0.6, zorder=3)
ax.barh(y_ddpm, -ddpm_ppl * ppl_scale, height=bar_h,
        color=DDPM_PPL, edgecolor='#5a5a5a', linewidth=0.6, zorder=3)

# PPL value labels
for i in range(n):
    ax.text(-fm_ppl[i] * ppl_scale - 0.008, y_fm[i],
            f'{fm_ppl[i]:.1f}', ha='right', va='center', fontsize=7.5,
            color='#3a3a3a', fontweight='bold')
    ax.text(-ddpm_ppl[i] * ppl_scale - 0.008, y_ddpm[i],
            f'{ddpm_ppl[i]:.1f}', ha='right', va='center', fontsize=7.5,
            color='#3a3a3a', fontweight='bold')

# === RIGHT SIDE: Spearman (positive direction) ===
# Mean Spearman (light, outer layer)
ax.barh(y_fm, fm_mean, height=bar_h,
        color=FM_LIGHT, edgecolor='#5a5a5a', linewidth=0.6, zorder=3)
ax.barh(y_ddpm, ddpm_mean, height=bar_h,
        color=DDPM_LIGHT, edgecolor='#5a5a5a', linewidth=0.6, zorder=3)

# Flat Spearman (deep, inner layer, overlaid)
ax.barh(y_fm, fm_flat, height=bar_h,
        color=FM_DEEP, edgecolor='#5a5a5a', linewidth=0.6, zorder=4)
ax.barh(y_ddpm, ddpm_flat, height=bar_h,
        color=DDPM_DEEP, edgecolor='#5a5a5a', linewidth=0.6, zorder=4)

# Spearman value labels (show Mean value at the end)
for i in range(n):
    ax.text(fm_mean[i] + 0.008, y_fm[i],
            f'{fm_mean[i]:.3f}', ha='left', va='center', fontsize=7.5,
            color='#3a3a3a', fontweight='bold')
    ax.text(ddpm_mean[i] + 0.008, y_ddpm[i],
            f'{ddpm_mean[i]:.3f}', ha='left', va='center', fontsize=7.5,
            color='#3a3a3a', fontweight='bold')

# === Y axis labels ===
y_label_pos = [(y_fm[i] + y_ddpm[i]) / 2 for i in range(n)]
ax.set_yticks(y_label_pos)
ax.set_yticklabels(labels, fontsize=10)

# Highlight first config
ax.get_yticklabels()[0].set_fontweight('bold')
ax.get_yticklabels()[0].set_fontsize(11)

# Add FM/DDPM sub-labels
for i in range(n):
    ax.text(-0.005, y_fm[i], 'FM', ha='right', va='center', fontsize=6.5,
            color=FM_DEEP, fontweight='bold')
    ax.text(-0.005, y_ddpm[i], 'DDPM', ha='right', va='center', fontsize=6.5,
            color=DDPM_DEEP, fontweight='bold')

# === Center line ===
ax.axvline(x=0, color='#3a3a3a', linewidth=1.2, zorder=5)

# === Grid ===
ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# === X axis ===
# Create custom x ticks: left side shows PPL values, right side shows Spearman
ppl_ticks = [5, 10, 15, 20, 25, 30, 35]
spear_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

all_ticks = [-p * ppl_scale for p in ppl_ticks] + [0] + spear_ticks
all_labels_x = [f'{p}' for p in ppl_ticks] + ['0'] + [f'{s:.1f}' for s in spear_ticks]
ax.set_xticks(all_ticks)
ax.set_xticklabels(all_labels_x, fontsize=9)

# X axis range
ax.set_xlim(-37 * ppl_scale, 0.68)

# === Arrows for "better" direction ===
arrow_y = (n - 1) * group_h + bar_h + 0.6

# PPL arrow (left = better)
ax.annotate('', xy=(-35 * ppl_scale, arrow_y), xytext=(-10 * ppl_scale, arrow_y),
            arrowprops=dict(arrowstyle='->', color='#5a5a5a', lw=2))
ax.text((-35 * ppl_scale + -10 * ppl_scale) / 2, arrow_y + 0.15,
        '← Better (Lower PPL)', ha='center', va='bottom', fontsize=10,
        color='#5a5a5a', fontweight='bold')

# Spearman arrow (right = better)
ax.annotate('', xy=(0.62, arrow_y), xytext=(0.30, arrow_y),
            arrowprops=dict(arrowstyle='->', color='#5a5a5a', lw=2))
ax.text((0.62 + 0.30) / 2, arrow_y + 0.15,
        'Better (Higher ρ) →', ha='center', va='bottom', fontsize=10,
        color='#5a5a5a', fontweight='bold')

# === X axis labels ===
ax.text(-18.5 * ppl_scale, -1.2, 'Perplexity', ha='center', fontsize=11,
        fontweight='bold', color='#5a5a5a')
ax.text(0.35, -1.2, 'Spearman ρ', ha='center', fontsize=11,
        fontweight='bold', color='#5a5a5a')

# === Legend ===
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=FM_DEEP, edgecolor='#5a5a5a', label='Flow Matching — Flat ρ'),
    Patch(facecolor=FM_LIGHT, edgecolor='#5a5a5a', label='Flow Matching — Mean ρ'),
    Patch(facecolor=DDPM_DEEP, edgecolor='#5a5a5a', label='DDPM — Flat ρ'),
    Patch(facecolor=DDPM_LIGHT, edgecolor='#5a5a5a', label='DDPM — Mean ρ'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9.5,
          framealpha=0.9, edgecolor='#cccccc')

# === Title ===
ax.set_title('Flow Matching vs DDPM: Seq→Phase Performance (missing_threshold=15)',
             fontsize=14, fontweight='bold', pad=35)

# === Separator lines between configs ===
for i in range(n - 1):
    sep_y = (y_ddpm[i] + y_fm[i + 1]) / 2 - 0.05
    ax.axhline(y=sep_y, color='#e0e0e0', linewidth=0.5, zorder=0)

# Highlight fw=32 lm=5 row
from matplotlib.patches import FancyBboxPatch
highlight_y = y_ddpm[0] - bar_h / 2 - 0.05
highlight_h = y_fm[0] - y_ddpm[0] + bar_h + 0.1
rect = FancyBboxPatch((-37 * ppl_scale, highlight_y), 37 * ppl_scale + 0.68, highlight_h,
                       boxstyle='round,pad=0.02', facecolor='#fff8e1', edgecolor='#f5c842',
                       linewidth=1.5, alpha=0.3, zorder=1)
ax.add_patch(rect)

plt.tight_layout()
output_path = '/data/yanjie_huang/LLPS/predictor/PhaseFlow/doc/fm_vs_ddpm_comparison.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved to {output_path}')
plt.close()
