#!/usr/bin/env python3
"""Plot ITO benchmark from run_full_testset.log (auto-parsed, no hardcoded data)."""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

font_path = '/data/yanjie_huang/fonts/arial.ttf'
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

OUT = Path(__file__).parent
LOG = OUT / 'run_full_testset.log'
FM_COLOR   = '#728ab9'
DDPM_COLOR = '#50b9ae'

# ── Parse log ────────────────────────────────────────────────────────────────
pattern = re.compile(
    r'Round\s+(\d+)/\d+\s+t=([\d.]+)s\s+Spearman=([\d.+-]+)±([\d.]+)'
)

fm_times, fm_spearman, fm_std = [], [], []
ddpm_times, ddpm_spearman, ddpm_std = [], [], []

current = None
with open(LOG) as f:
    for line in f:
        if 'Flow Matching benchmark' in line:
            current = 'fm'
        elif 'DDPM benchmark' in line:
            current = 'ddpm'
        m = pattern.search(line)
        if m and current:
            t, sp, sd = float(m.group(2)), float(m.group(3)), float(m.group(4))
            if current == 'fm':
                fm_times.append(t)
                fm_spearman.append(sp)
                fm_std.append(sd)
            else:
                ddpm_times.append(t)
                ddpm_spearman.append(sp)
                ddpm_std.append(sd)

fm_times   = np.array(fm_times)
fm_spearman = np.array(fm_spearman)
fm_std      = np.array(fm_std)
ddpm_times  = np.array(ddpm_times)
ddpm_spearman = np.array(ddpm_spearman)
ddpm_std      = np.array(ddpm_std)

n_fm   = len(fm_times)
n_ddpm = len(ddpm_times)
print(f'Parsed: FM {n_fm} rounds, DDPM {n_ddpm} rounds')

fm_avg   = fm_times.mean()
ddpm_avg = ddpm_times.mean() if n_ddpm > 0 else 0

# ── Figure 1: Time curve ────────────────────────────────────────────────────
n_common = min(n_fm, n_ddpm) if n_ddpm > 0 else n_fm
rounds = np.arange(1, n_common + 1)

fig, ax = plt.subplots(figsize=(6, 6))

fm_cum_min   = np.cumsum(fm_times[:n_common]) / 60
ax.plot(rounds, fm_cum_min, color=FM_COLOR, linewidth=5,
        marker='o', markersize=8, markevery=2, label='Flow Matching')

if n_ddpm > 0:
    ddpm_cum_min = np.cumsum(ddpm_times[:n_common]) / 60
    ax.plot(rounds, ddpm_cum_min, color=DDPM_COLOR, linewidth=5,
            marker='s', markersize=8, markevery=2, label='DDPM')

mid = n_common // 2
ax.text(rounds[mid] - 4, fm_cum_min[mid] + 0.4,
        f'avg {fm_avg:.2f}s/round',
        fontsize=20, color=FM_COLOR, fontweight='bold', ha='center', va='bottom')
if n_ddpm > 0:
    ax.text(rounds[mid] - 4, ddpm_cum_min[mid] + 22.0,
            f'avg {ddpm_avg:.2f}s/round',
            fontsize=20, color=DDPM_COLOR, fontweight='bold', ha='center', va='bottom')

ax.set_xlabel('Round', fontsize=22, fontweight='bold')
ax.set_ylabel('Cumulative Time (min)', fontsize=22, fontweight='bold')
ax.set_title('Inference-Time Optimization:\nCumulative Time', fontsize=22, fontweight='bold', pad=12)
ax.tick_params(labelsize=18)
ax.legend(fontsize=18, framealpha=0.9, edgecolor='#cccccc')
ax.set_xlim(1, n_common + 3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
fig.savefig(OUT / 'benchmark_time_curve.png', dpi=300, bbox_inches='tight', facecolor='white')
print('Saved benchmark_time_curve.png')
plt.close()

# ── Figure 2: Spearman bar (FM only) ────────────────────────────────────────
checkpoints = [r for r in [1, 5, 10, 20, 50] if r <= n_fm]
idxs = [r - 1 for r in checkpoints]
vals = fm_spearman[idxs]
errs = fm_std[idxs]

fig, ax = plt.subplots(figsize=(6, 6))

x = np.arange(len(checkpoints))
width = 0.4

bars = ax.bar(x, vals, width,
              color=FM_COLOR, edgecolor='none', linewidth=0)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
            f'{h:.3f}', ha='center', va='bottom', fontsize=18,
            color=FM_COLOR, fontweight='bold')

ax.set_xlabel('Rounds', fontsize=22, fontweight='bold')
ax.set_ylabel('Spearman ρ', fontsize=22, fontweight='bold')
ax.set_title('Inference-Time Optimization', fontsize=22, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels([str(r) for r in checkpoints], fontsize=18)
ax.tick_params(labelsize=18)
ax.set_ylim(0, max(vals) * 1.25)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
fig.savefig(OUT / 'benchmark_spearman_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
print('Saved benchmark_spearman_bar.png')
plt.close()
