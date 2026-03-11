#!/usr/bin/env python3
"""
计算 IDR vs 非 IDR 区域分数分布，生成三张对比分布图
每个模型一张图，显示 IDR 和非 IDR 的分数分布
"""

import os
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 12

WINDOW_SIZE = 10
OUTPUT_DIR = '/data/yanjie_huang/LLPS/comparision/idr_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_results(filepath):
    data = {}
    with open(filepath) as f:
        for line in f:
            item = json.loads(line)
            data[item['entry']] = {
                'sequence': item['sequence'],
                'scores': np.array(item['scores'])
            }
    return data

def load_human_idr(filepath):
    df = pd.read_csv(filepath)
    idr_dict = defaultdict(list)
    for _, row in df.iterrows():
        if pd.isna(row['start']) or pd.isna(row['end']):
            continue
        entry = row['Entry']
        start = int(row['start'])
        end = int(row['end'])
        idr_dict[entry].append((start, end))
    return idr_dict

def get_idr_window_positions(idr_regions, n_windows):
    idr_windows = set()
    for start, end in idr_regions:
        win_start = max(start, 1)
        win_end = min(end - WINDOW_SIZE + 1, n_windows)
        if win_start <= win_end:
            for pos in range(win_start, win_end + 1):
                if 1 <= pos <= n_windows:
                    idr_windows.add(pos)
    return idr_windows

def get_non_idr_window_positions(idr_regions, n_windows):
    non_idr_windows = set()
    for pos in range(1, n_windows + 1):
        window_start = pos
        window_end = pos + WINDOW_SIZE - 1
        overlaps = False
        for idr_start, idr_end in idr_regions:
            if not (window_end < idr_start or idr_end < window_start):
                overlaps = True
                break
        if not overlaps:
            non_idr_windows.add(pos)
    return non_idr_windows

def collect_all_scores(sequence, scores, idr_regions):
    """收集所有窗口的 IDR 和非 IDR 分数"""
    seq_len = len(sequence)
    n_windows = len(scores)
    if seq_len < WINDOW_SIZE:
        return [], []

    idr_windows = get_idr_window_positions(idr_regions, n_windows)
    non_idr_windows = get_non_idr_window_positions(idr_regions, n_windows)

    idr_scores = [scores[pos - 1] for pos in idr_windows]
    non_idr_scores = [scores[pos - 1] for pos in non_idr_windows]

    return idr_scores, non_idr_scores

def main():
    print("=" * 60)
    print("IDR vs Non-IDR 分数分布分析")
    print("=" * 60)

    # 加载数据
    print("\n1. 加载数据...")
    nn_results = load_results('/data/yanjie_huang/LLPS/sliding_window/10aa/nn_windows.jsonl')
    xgb_results = load_results('/data/yanjie_huang/LLPS/sliding_window/10aa/xgb_windows.jsonl')
    pf_results = load_results('/data/yanjie_huang/LLPS/sliding_window/10aa/pf_windows.jsonl')
    idr_dict = load_human_idr('/data/yanjie_huang/LLPS/data/human_idr.csv')

    print(f"   NN: {len(nn_results)}, XGBoost: {len(xgb_results)}, PhaseFlow: {len(pf_results)}")
    print(f"   human_idr entries: {len(idr_dict)}")

    common_entries = set(nn_results.keys()) & set(xgb_results.keys()) & set(pf_results.keys())
    entries_with_idr = common_entries & set(idr_dict.keys())
    print(f"   Common with IDR: {len(entries_with_idr)}")

    # 收集所有分数
    print("\n2. 收集分数...")
    nn_idr_scores = []
    nn_non_idr_scores = []
    xgb_idr_scores = []
    xgb_non_idr_scores = []
    pf_idr_scores = []
    pf_non_idr_scores = []

    for entry in entries_with_idr:
        idr_regions = idr_dict[entry]

        # NN
        idr_s, non_idr_s = collect_all_scores(
            nn_results[entry]['sequence'], nn_results[entry]['scores'], idr_regions)
        nn_idr_scores.extend(idr_s)
        nn_non_idr_scores.extend(non_idr_s)

        # XGBoost
        idr_s, non_idr_s = collect_all_scores(
            xgb_results[entry]['sequence'], xgb_results[entry]['scores'], idr_regions)
        xgb_idr_scores.extend(idr_s)
        xgb_non_idr_scores.extend(non_idr_s)

        # PhaseFlow
        idr_s, non_idr_s = collect_all_scores(
            pf_results[entry]['sequence'], pf_results[entry]['scores'], idr_regions)
        pf_idr_scores.extend(idr_s)
        pf_non_idr_scores.extend(non_idr_s)

    print(f"   NN: IDR={len(nn_idr_scores)}, Non-IDR={len(nn_non_idr_scores)}")
    print(f"   XGBoost: IDR={len(xgb_idr_scores)}, Non-IDR={len(xgb_non_idr_scores)}")
    print(f"   PhaseFlow: IDR={len(pf_idr_scores)}, Non-IDR={len(pf_non_idr_scores)}")

    # 转换为 numpy
    nn_idr = np.array(nn_idr_scores)
    nn_non_idr = np.array(nn_non_idr_scores)
    xgb_idr = np.array(xgb_idr_scores)
    xgb_non_idr = np.array(xgb_non_idr_scores)
    pf_idr = np.array(pf_idr_scores)
    pf_non_idr = np.array(pf_non_idr_scores)

    # ========== 图1: NN Baseline 分布对比 ==========
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.hist(nn_idr, bins=50, alpha=0.6, label=f'IDR (n={len(nn_idr):,})', color='#e74c3c', density=True)
    ax1.hist(nn_non_idr, bins=50, alpha=0.6, label=f'Non-IDR (n={len(nn_non_idr):,})', color='#3498db', density=True)
    ax1.axvline(nn_idr.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'IDR mean={nn_idr.mean():.4f}')
    ax1.axvline(nn_non_idr.mean(), color='#3498db', linestyle='--', linewidth=2, label=f'Non-IDR mean={nn_non_idr.mean():.4f}')
    ax1.set_xlabel('Window Score')
    ax1.set_ylabel('Density')
    ax1.set_title('NN Baseline: IDR vs Non-IDR Score Distribution')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(f'{OUTPUT_DIR}/fig1_nn_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\n   Fig1 saved: fig1_nn_distribution.png")

    # ========== 图2: XGBoost Baseline 分布对比 ==========
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(xgb_idr, bins=50, alpha=0.6, label=f'IDR (n={len(xgb_idr):,})', color='#e74c3c', density=True)
    ax2.hist(xgb_non_idr, bins=50, alpha=0.6, label=f'Non-IDR (n={len(xgb_non_idr):,})', color='#3498db', density=True)
    ax2.axvline(xgb_idr.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'IDR mean={xgb_idr.mean():.4f}')
    ax2.axvline(xgb_non_idr.mean(), color='#3498db', linestyle='--', linewidth=2, label=f'Non-IDR mean={xgb_non_idr.mean():.4f}')
    ax2.set_xlabel('Window Score')
    ax2.set_ylabel('Density')
    ax2.set_title('XGBoost Baseline: IDR vs Non-IDR Score Distribution')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(f'{OUTPUT_DIR}/fig2_xgboost_distribution.png', dpi=150, bbox_inches='tight')
    print(f"   Fig2 saved: fig2_xgboost_distribution.png")

    # ========== 图3: PhaseFlow 分布对比 ==========
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.hist(pf_idr, bins=50, alpha=0.6, label=f'IDR (n={len(pf_idr):,})', color='#e74c3c', density=True)
    ax3.hist(pf_non_idr, bins=50, alpha=0.6, label=f'Non-IDR (n={len(pf_non_idr):,})', color='#3498db', density=True)
    ax3.axvline(pf_idr.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'IDR mean={pf_idr.mean():.4f}')
    ax3.axvline(pf_non_idr.mean(), color='#3498db', linestyle='--', linewidth=2, label=f'Non-IDR mean={pf_non_idr.mean():.4f}')
    ax3.set_xlabel('Window Score')
    ax3.set_ylabel('Density')
    ax3.set_title('PhaseFlow: IDR vs Non-IDR Score Distribution')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(f'{OUTPUT_DIR}/fig3_phaseflow_distribution.png', dpi=150, bbox_inches='tight')
    print(f"   Fig3 saved: fig3_phaseflow_distribution.png")

    # 保存汇总统计
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)

    stats = []
    for name, idr, non_idr in [
        ('NN', nn_idr, nn_non_idr),
        ('XGBoost', xgb_idr, xgb_non_idr),
        ('PhaseFlow', pf_idr, pf_non_idr)
    ]:
        diff = idr.mean() - non_idr.mean()
        t_stat, t_pval = None, None
        try:
            from scipy.stats import ttest_ind, mannwhitneyu
            t_stat, t_pval = mannwhitneyu(idr, non_idr, alternative='two-sided')
        except:
            pass

        print(f"\n{name}:")
        print(f"  IDR mean: {idr.mean():.4f} ± {idr.std():.4f}")
        print(f"  Non-IDR mean: {non_idr.mean():.4f} ± {non_idr.std():.4f}")
        print(f"  Difference: {diff:.4f}")
        if t_pval is not None:
            print(f"  Mann-Whitney U p-value: {t_pval:.2e}")

        stats.append({
            'model': name,
            'n_idr_windows': len(idr),
            'n_non_idr_windows': len(non_idr),
            'idr_mean': idr.mean(),
            'idr_std': idr.std(),
            'non_idr_mean': non_idr.mean(),
            'non_idr_std': non_idr.std(),
            'difference': diff
        })

    # 保存统计 CSV
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(f'{OUTPUT_DIR}/distribution_summary.csv', index=False)
    print(f"\n统计汇总: {OUTPUT_DIR}/distribution_summary.csv")

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("完成!")

if __name__ == '__main__':
    main()
