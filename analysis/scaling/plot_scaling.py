#!/usr/bin/env python3
"""
PhaseFlow Model Size Scaling — 收集结果并画 scaling curve。

Usage:
    python analysis/scaling/plot_scaling.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = '/data/yanjie_huang/fonts/arial.ttf'
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(BASE, '..', '..'))
OUTPUT_SET = os.path.join(PROJECT, 'outputs_set')

# ── 定义所有 scaling 配置 ──────────────────────────────────────────────────
CONFIGS = [
    # (label, output_dir_name, dim, depth, heads, group)
    # ── Small models ──
    ('S4\n(64×3)',          'output_scale_s4_dim64_d3',         64,   3, 4,  'deep'),
    ('S3\n(64×4)',          'output_scale_s3_dim64_d4',         64,   4, 4,  'deep'),
    ('S1\n(128×4)',         'output_scale_s1_dim128_d4',        128,  4, 4,  'deep'),
    ('S2\n(192×5)',         'output_scale_s2_dim192_d5',        192,  5, 6,  'deep'),
    # ── Baseline ──
    ('Baseline\n(256×6)',   'output_set_flow32_missing15',      256,  6, 8,  'deep'),
    # ── Large models ──
    ('A\n(320×8)',          'output_scale_a_dim320_d8',         320,  8, 10, 'deep'),
    ('B\n(384×8)',          'output_scale_b_dim384_d8',         384,  8, 12, 'deep'),
    ('C\n(512×8)',          'output_scale_c_dim512_d8',         512,  8, 16, 'deep'),
    ('D-sh\n(640×6)',       'output_scale_d_shallow_dim640_d6', 640,  6, 20, 'shallow'),
    ('E-sh\n(768×6)',       'output_scale_e_shallow_dim768_d6', 768,  6, 24, 'shallow'),
]

# 莫兰迪色系
COLOR_DEEP    = (114/255, 138/255, 185/255)  # 蓝灰
COLOR_SHALLOW = (245/255, 181/255, 191/255)  # 粉
COLOR_BASE    = (181/255, 211/255, 185/255)  # 鼠尾草绿
MEAN_LINE     = (100/255, 100/255, 100/255)


def estimate_params(dim, depth, heads):
    """粗略估算参数量 (与 model.py 对齐)"""
    # token embedding
    p = 32 * dim
    # set phase encoder: value_mlp + pos_emb
    p += dim * dim * 2 + dim * 2 + 16 * dim
    # time encoder
    p += dim * dim * 4 + dim * 4 + dim * dim + dim
    # transformer layers
    for _ in range(depth):
        # attention: qkv + out
        p += 3 * dim * dim + dim * dim + 4 * dim  # qkv, out, norms
        # feedforward (SwiGLU): 3 linear layers, ff_mult=4
        ff_dim = dim * 4
        p += dim * ff_dim + dim * ff_dim + ff_dim * dim + 2 * dim
    # output heads
    p += dim * 32 + dim * 16 + dim * (dim // 4) + (dim // 4)
    return p


def collect_results():
    """收集所有已完成实验的结果"""
    results = []
    for label, dirname, dim, depth, heads, group in CONFIGS:
        json_path = os.path.join(OUTPUT_SET, dirname, 'test_results.json')
        params = estimate_params(dim, depth, heads)
        entry = {
            'label': label, 'dim': dim, 'depth': depth, 'heads': heads,
            'group': group, 'params': params, 'params_m': params / 1e6,
        }
        if os.path.exists(json_path):
            with open(json_path) as f:
                metrics = json.load(f)
            entry.update(metrics)
            entry['available'] = True
        else:
            entry['available'] = False
        results.append(entry)
    return results


def plot_scaling(results):
    avail = [r for r in results if r['available']]
    if len(avail) < 2:
        print(f'Only {len(avail)} results available, need at least 2 to plot.')
        # 打印已有的
        for r in results:
            status = '✓' if r['available'] else '✗'
            print(f"  {status} {r['label'].replace(chr(10),' ')} ({r['params_m']:.1f}M)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

    metrics_list = [
        ('spearman', 'Spearman ρ', True),
        ('mse',      'MSE',        False),
        ('perplexity','Perplexity', False),
    ]

    for ax, (metric, ylabel, higher_better) in zip(axes, metrics_list):
        ax.set_facecolor('white')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for group, color, marker, ms in [('deep', COLOR_DEEP, 'o', 10), ('shallow', COLOR_SHALLOW, 's', 10)]:
            pts = [r for r in avail if r['group'] == group]
            if not pts:
                continue
            xs = [r['params_m'] for r in pts]
            ys = [r[metric] for r in pts]
            labels_txt = [r['label'].replace('\n', ' ') for r in pts]

            ax.plot(xs, ys, color=color, marker=marker, markersize=ms,
                    linewidth=2.5, alpha=0.85, label=f'{"Deep" if group == "deep" else "Shallow (d=6)"}',
                    zorder=3)

            for x, y, lab in zip(xs, ys, labels_txt):
                ax.annotate(lab, (x, y), textcoords='offset points',
                            xytext=(0, 12), ha='center', fontsize=7.5, color='gray')

        # 标注 baseline
        base = [r for r in avail if r['dim'] == 256]
        if base:
            bx, by = base[0]['params_m'], base[0][metric]
            ax.scatter([bx], [by], color=COLOR_SHALLOW, s=300, zorder=5,
                       edgecolors='none', linewidths=0, marker='*')

        ax.set_xlabel('Parameters (M)', fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=10, framealpha=0.9, edgecolor='#cccccc')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_title('Model Size vs Spearman', fontsize=14, fontweight='bold')
    axes[1].set_title('Model Size vs MSE', fontsize=14, fontweight='bold')
    axes[2].set_title('Model Size vs Perplexity', fontsize=14, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(BASE, 'scaling_curve.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved → {out_path}')


def print_table(results):
    print(f"\n{'Config':<20} {'Params':>8} {'Spearman':>10} {'MSE':>8} {'PPL':>8} {'Status'}")
    print('-' * 65)
    for r in results:
        label = r['label'].replace('\n', ' ')
        params = f"{r['params_m']:.1f}M"
        if r['available']:
            sp = f"{r['spearman']:.4f}"
            mse = f"{r['mse']:.4f}"
            ppl = f"{r['perplexity']:.2f}"
            status = '✓'
        else:
            sp = mse = ppl = '—'
            status = '✗ (not found)'
        print(f"{label:<20} {params:>8} {sp:>10} {mse:>8} {ppl:>8} {status}")


if __name__ == '__main__':
    results = collect_results()
    print_table(results)
    plot_scaling(results)
