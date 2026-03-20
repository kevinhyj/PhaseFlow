#!/usr/bin/env python3
"""
序列多样性分析：Bottom 5% 强相分离相图 (Strong LLPS, PSSI最负=相分离最强)

对每个目标相图生成 N 条候选序列，分析：
1. 序列唯一性（去重率）
2. 编辑距离分布
3. AA 组成方差
4. 预测相图一致性（Round-trip）

Usage:
    cd /data/yanjie_huang/LLPS/predictor/PhaseFlow
    python "analysis/de novo/bottom5_strong/diversity_analysis.py" --gpu 0 --n_samples 100
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phaseflow import PhaseFlow, AminoAcidTokenizer

# ============================================================================
# Config
# ============================================================================

CHECKPOINT = "/data/yanjie_huang/LLPS/predictor/PhaseFlow/outputs_set/output_set_flow32_missing15/best_model.pt"
MISSING0_CSV = "/data/yanjie_huang/LLPS/phase_diagram/by_missing/missing_0.csv"
GROUP_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'diversity_results')


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model = PhaseFlow(
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        dim_head=config['model']['dim_head'],
        vocab_size=config['model']['vocab_size'],
        phase_dim=config['model']['phase_dim'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=0.0,
        use_set_encoder=config['model'].get('use_set_encoder', False),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    return model


def edit_distance(s1, s2):
    n, m = len(s1), len(s2)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[m]


def pairwise_edit_distances(seqs, max_pairs=500):
    n = len(seqs)
    if n < 2:
        return []
    pairs = min(max_pairs, n * (n - 1) // 2)
    dists = []
    seen = set()
    while len(dists) < pairs:
        i, j = np.random.randint(0, n, 2)
        if i == j or (i, j) in seen:
            continue
        seen.add((i, j))
        seen.add((j, i))
        dists.append(edit_distance(seqs[i], seqs[j]))
    return dists


def aa_composition(seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    counts = Counter(seq)
    total = len(seq)
    if total == 0:
        return np.zeros(20)
    return np.array([counts.get(aa, 0) / total for aa in AA])


def analyze_one_phase(model, tokenizer, phase_tensor, n_samples, gen_batch, device):
    """对单个相图生成 n_samples 条序列并分析多样性。"""
    all_seqs = []
    for start in range(0, n_samples, gen_batch):
        end = min(start + gen_batch, n_samples)
        batch_phase = phase_tensor.unsqueeze(0).expand(end - start, -1).contiguous()
        with torch.no_grad():
            _, seqs = model.generate_sequence(batch_phase, tokenizer, max_len=25, temperature=1.0)
        all_seqs.extend(seqs)

    all_seqs = [s for s in all_seqs if len(s) > 0]
    if len(all_seqs) < 2:
        return None

    # 1. 唯一性
    unique_seqs = set(all_seqs)
    uniqueness = len(unique_seqs) / len(all_seqs)

    # 2. 编辑距离
    edit_dists = pairwise_edit_distances(all_seqs)
    mean_edit = np.mean(edit_dists) if edit_dists else 0
    median_edit = np.median(edit_dists) if edit_dists else 0

    # 3. AA 组成方差
    compositions = np.array([aa_composition(s) for s in all_seqs])
    aa_var = compositions.var(axis=0).mean()

    # 4. 长度分布
    lengths = [len(s) for s in all_seqs]
    len_mean = np.mean(lengths)
    len_std = np.std(lengths)

    # 5. Round-trip: 预测相图
    input_ids = tokenizer.batch_encode(all_seqs, max_len=32).to(device)
    attention_mask = (input_ids != tokenizer.PAD_ID).long()
    seq_lens = torch.tensor([len(s) for s in all_seqs], device=device)

    pred_phases = []
    for start in range(0, len(all_seqs), gen_batch):
        end = min(start + gen_batch, len(all_seqs))
        with torch.no_grad():
            pred = model.generate_phase(
                input_ids[start:end], attention_mask[start:end],
                seq_lens[start:end], method='euler')
        pred_phases.append(pred.cpu())
    pred_phases = torch.cat(pred_phases, dim=0).numpy()

    target_np = phase_tensor.cpu().numpy()
    corrs = []
    for i in range(len(all_seqs)):
        c, _ = spearmanr(pred_phases[i], target_np)
        corrs.append(float(c) if not np.isnan(c) else 0.0)

    pred_stds = pred_phases.std(axis=1)

    return {
        'n_generated': len(all_seqs),
        'n_unique': len(unique_seqs),
        'uniqueness': uniqueness,
        'mean_edit_dist': mean_edit,
        'median_edit_dist': median_edit,
        'aa_var': aa_var,
        'len_mean': len_mean,
        'len_std': len_std,
        'rt_spearman_mean': np.mean(corrs),
        'rt_spearman_std': np.std(corrs),
        'pred_std_mean': pred_stds.mean(),
        'pred_phase_var': pred_phases.var(axis=0).mean(),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=100, help='每个相图生成多少条')
    parser.add_argument('--n_phases', type=int, default=None, help='分析多少个相图（None=全部）')
    parser.add_argument('--gen_batch', type=int, default=50)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(CHECKPOINT, device)
    tokenizer = AminoAcidTokenizer()

    # 加载 Bottom 5% 强相分离相图 (PSSI最负=相分离最强)
    df = pd.read_csv(MISSING0_CSV)
    phase_data = df[GROUP_COLS].values.astype(np.float32)
    mean_pssi = phase_data.mean(axis=1)
    q5 = np.percentile(mean_pssi, 5)
    weak_idx = np.where(mean_pssi <= q5)[0]

    if args.n_phases is not None:
        weak_idx = weak_idx[:args.n_phases]

    print(f"Bottom 5% 强相分离: {len(weak_idx)} 条, mean_pssi <= {q5:.4f}")
    print(f"每个相图生成 {args.n_samples} 条序列")

    # 逐个相图分析
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    for idx in tqdm(weak_idx, desc="Analyzing diversity"):
        phase_tensor = torch.tensor(phase_data[idx], dtype=torch.float32, device=device)
        result = analyze_one_phase(model, tokenizer, phase_tensor, args.n_samples, args.gen_batch, device)
        if result is not None:
            result['phase_idx'] = int(idx)
            result['mean_pssi'] = float(mean_pssi[idx])
            result['original_seq'] = df['AminoAcidSequence'].iloc[idx]
            results.append(result)

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, 'diversity_results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\n结果保存: {csv_path}")

    # ================================================================
    # 汇总统计
    # ================================================================
    print("\n" + "=" * 60)
    print(f"序列多样性分析 (Bottom 5%, n_samples={args.n_samples})")
    print("=" * 60)
    print(f"分析相图数: {len(df_results)}")
    print(f"\n{'指标':<25} {'Mean':>10} {'Std':>10} {'Median':>10}")
    print("-" * 55)
    for col, label in [
        ('uniqueness', '唯一率'),
        ('mean_edit_dist', '平均编辑距离'),
        ('aa_var', 'AA组成方差'),
        ('len_mean', '平均长度'),
        ('len_std', '长度标准差'),
        ('rt_spearman_mean', 'RT Spearman'),
        ('pred_std_mean', '预测相图 per-sample std'),
        ('pred_phase_var', '预测相图跨样本方差'),
    ]:
        vals = df_results[col]
        print(f"{label:<22} {vals.mean():>10.4f} {vals.std():>10.4f} {vals.median():>10.4f}")
    print("=" * 60)

    # ================================================================
    # 可视化
    # ================================================================
    # 统一莫兰迪色系
    COLOR_GT  = (114/255, 138/255, 185/255)  # #728ab9 蓝灰
    COLOR_GEN = (245/255, 181/255, 191/255)  # #f5b5bf 粉
    COLOR_RND = (245/255, 221/255, 181/255)  # #f5ddb5 浅橙
    COLOR_ITO = (181/255, 211/255, 185/255)  # #b5d3b9 鼠尾草绿
    COLOR_LAV = (185/255, 172/255, 210/255)  # #b9acd2 薰衣草
    COLOR_GRY = (180/255, 180/255, 180/255)  # #b4b4b4 灰
    MEAN_LINE = (100/255, 100/255, 100/255)  # 深灰均值线

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')

    ax = axes[0, 0]
    ax.hist(df_results['uniqueness'], bins=30, color=COLOR_GT, alpha=0.7, edgecolor='white')
    ax.set_xlabel('Uniqueness Rate')
    ax.set_ylabel('Count')
    ax.set_title(f'Sequence Uniqueness (mean={df_results["uniqueness"].mean():.3f})')
    ax.axvline(df_results['uniqueness'].mean(), color=MEAN_LINE, linestyle='--')

    ax = axes[0, 1]
    ax.hist(df_results['mean_edit_dist'], bins=30, color=COLOR_GEN, alpha=0.7, edgecolor='white')
    ax.set_xlabel('Mean Edit Distance')
    ax.set_ylabel('Count')
    ax.set_title(f'Pairwise Edit Distance (mean={df_results["mean_edit_dist"].mean():.2f})')
    ax.axvline(df_results['mean_edit_dist'].mean(), color=MEAN_LINE, linestyle='--')

    ax = axes[0, 2]
    ax.hist(df_results['aa_var'], bins=30, color=COLOR_RND, alpha=0.7, edgecolor='white')
    ax.set_xlabel('AA Composition Variance')
    ax.set_ylabel('Count')
    ax.set_title(f'AA Composition Variance (mean={df_results["aa_var"].mean():.5f})')
    ax.axvline(df_results['aa_var'].mean(), color=MEAN_LINE, linestyle='--')

    ax = axes[1, 0]
    ax.hist(df_results['rt_spearman_mean'], bins=30, color=COLOR_ITO, alpha=0.7, edgecolor='white')
    ax.set_xlabel('Mean RT Spearman')
    ax.set_ylabel('Count')
    ax.set_title(f'Round-Trip Spearman (mean={df_results["rt_spearman_mean"].mean():.3f})')
    ax.axvline(df_results['rt_spearman_mean'].mean(), color=MEAN_LINE, linestyle='--')

    ax = axes[1, 1]
    ax.hist(df_results['pred_std_mean'], bins=30, color=COLOR_LAV, alpha=0.7, edgecolor='white')
    ax.set_xlabel('Pred Phase Per-Sample Std')
    ax.set_ylabel('Count')
    ax.set_title(f'Pred Phase Std (mean={df_results["pred_std_mean"].mean():.3f})')
    ax.axvline(df_results['pred_std_mean'].mean(), color=MEAN_LINE, linestyle='--')

    ax = axes[1, 2]
    ax.scatter(df_results['mean_pssi'], df_results['uniqueness'], alpha=0.3, s=10, color=COLOR_GRY)
    ax.set_xlabel('Mean PSSI (more negative = stronger LLPS)')
    ax.set_ylabel('Uniqueness Rate')
    ax.set_title('Uniqueness vs PSSI Strength')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'diversity_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表保存: {fig_path}")


if __name__ == '__main__':
    main()
