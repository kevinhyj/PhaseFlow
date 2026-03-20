#!/usr/bin/env python3
"""
验证 LM Log-Likelihood 的区分度。

三组对比：
1. 自然序列 + 正确相图 → LL 应该最高
2. 自然序列 + 错误相图（打乱） → LL 应该下降
3. 随机序列 + 任意相图 → LL 应该最低

Usage:
    cd /data/yanjie_huang/LLPS/predictor/PhaseFlow
    python pipeline/test_ll_discrimination.py --gpu 0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from phaseflow import PhaseFlow, AminoAcidTokenizer, PhaseDataset
from phaseflow.data import collate_fn

# ============================================================================
# Config
# ============================================================================

CHECKPOINT = "/data/yanjie_huang/LLPS/predictor/PhaseFlow/outputs_set/output_set_flow32_missing15/best_model.pt"
TEST_CSV = "/data/yanjie_huang/LLPS/phase_diagram/test_set.csv"
BATCH_SIZE = 64


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


def generate_random_sequences(n, min_len=5, max_len=20):
    """生成随机氨基酸序列。"""
    aa = "ACDEFGHIKLMNPQRSTVWY"
    seqs = []
    for _ in range(n):
        length = np.random.randint(min_len, max_len + 1)
        seq = ''.join(np.random.choice(list(aa), size=length))
        seqs.append(seq)
    return seqs


def compute_ll_for_batch(model, tokenizer, sequences, phase, phase_mask, device):
    """对一批序列计算 LL。"""
    input_ids = tokenizer.batch_encode(sequences, max_len=32).to(device)
    attention_mask = (input_ids != tokenizer.PAD_ID).long()
    phase = phase.to(device)
    phase_mask = phase_mask.to(device)
    ll = model.compute_sequence_log_likelihood(input_ids, attention_mask, phase, phase_mask)
    return ll.cpu().numpy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = load_model(CHECKPOINT, device)
    tokenizer = AminoAcidTokenizer()

    # Load test set
    dataset = PhaseDataset(TEST_CSV, tokenizer=tokenizer, max_seq_len=32, split='all')
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Collect all data
    all_seqs = []
    all_phase = []
    all_mask = []
    for batch in loader:
        all_seqs.extend(batch['sequences'])
        all_phase.append(batch['phase_values'])
        all_mask.append(batch['phase_mask'])
    all_phase = torch.cat(all_phase, dim=0)  # (N, 16)
    all_mask = torch.cat(all_mask, dim=0)    # (N, 16)
    n = len(all_seqs)
    print(f"Test set: {n} sequences")

    # ================================================================
    # Group 1: 自然序列 + 正确相图
    # ================================================================
    print("\n[1/3] 自然序列 + 正确相图...")
    ll_correct = []
    for i in range(0, n, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n)
        seqs = all_seqs[i:end]
        phase = all_phase[i:end]
        mask = all_mask[i:end]
        ll = compute_ll_for_batch(model, tokenizer, seqs, phase, mask, device)
        ll_correct.append(ll)
    ll_correct = np.concatenate(ll_correct)

    # ================================================================
    # Group 2: 自然序列 + 错误相图（随机打乱）
    # ================================================================
    print("[2/3] 自然序列 + 错误相图（打乱）...")
    perm = np.random.permutation(n)
    # 确保没有配到自己
    for i in range(n):
        if perm[i] == i:
            swap = (i + 1) % n
            perm[i], perm[swap] = perm[swap], perm[i]
    shuffled_phase = all_phase[perm]
    shuffled_mask = all_mask[perm]

    ll_wrong = []
    for i in range(0, n, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n)
        seqs = all_seqs[i:end]
        phase = shuffled_phase[i:end]
        mask = shuffled_mask[i:end]
        ll = compute_ll_for_batch(model, tokenizer, seqs, phase, mask, device)
        ll_wrong.append(ll)
    ll_wrong = np.concatenate(ll_wrong)

    # ================================================================
    # Group 3: 随机序列 + 任意相图
    # ================================================================
    print("[3/3] 随机序列 + 任意相图...")
    random_seqs = generate_random_sequences(n)
    ll_random = []
    for i in range(0, n, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n)
        seqs = random_seqs[i:end]
        phase = all_phase[i:end]
        mask = all_mask[i:end]
        ll = compute_ll_for_batch(model, tokenizer, seqs, phase, mask, device)
        ll_random.append(ll)
    ll_random = np.concatenate(ll_random)

    # ================================================================
    # Group 4 & 5: 两组划分分别做交叉配对 + 二分类
    # ================================================================
    mean_pssi = all_phase.numpy().mean(axis=1)  # (N,)
    sorted_idx = np.argsort(mean_pssi)  # 从小(强)到大(弱)

    cross_results = {}  # 收集画图数据

    for split_name, top_k in [('Top 25%', 125), ('Top 10%', 50)]:
        print(f"\n{'='*60}")
        print(f"[{split_name}] 强/弱交叉配对 + 二分类 (n={top_k} per group)")
        print(f"{'='*60}")

        s_idx = sorted_idx[:top_k]        # PSSI 最小 = 强相分离
        w_idx = sorted_idx[-top_k:]       # PSSI 最大 = 弱相分离

        s_seqs = [all_seqs[i] for i in s_idx]
        w_seqs = [all_seqs[i] for i in w_idx]
        s_phase = all_phase[s_idx]
        s_mask  = all_mask[s_idx]
        w_phase = all_phase[w_idx]
        w_mask  = all_mask[w_idx]

        print(f"  强 (Strong LLPS): mean_pssi=[{mean_pssi[s_idx].min():.3f}, {mean_pssi[s_idx].max():.3f}] (均值 {mean_pssi[s_idx].mean():.3f})")
        print(f"  弱 (Weak LLPS):   mean_pssi=[{mean_pssi[w_idx].min():.3f}, {mean_pssi[w_idx].max():.3f}] (均值 {mean_pssi[w_idx].mean():.3f})")

        # --- 交叉配对 ---
        perm1 = np.random.permutation(top_k)
        perm2 = np.random.permutation(top_k)
        perm3 = np.random.permutation(top_k)
        perm4 = np.random.permutation(top_k)

        ll_ss = compute_ll_for_batch(model, tokenizer, s_seqs, s_phase[perm1], s_mask[perm1], device)
        ll_sw = compute_ll_for_batch(model, tokenizer, s_seqs, w_phase[perm2], w_mask[perm2], device)
        ll_ww = compute_ll_for_batch(model, tokenizer, w_seqs, w_phase[perm3], w_mask[perm3], device)
        ll_ws = compute_ll_for_batch(model, tokenizer, w_seqs, s_phase[perm4], s_mask[perm4], device)

        print(f"\n  {'组别':<28} {'Mean LL':>10} {'Std':>10} {'Median':>10}")
        print(f"  {'-'*58}")
        print(f"  {'强序列 + 强相图(正确)':<24} {ll_ss.mean():>10.4f} {ll_ss.std():>10.4f} {np.median(ll_ss):>10.4f}")
        print(f"  {'强序列 + 弱相图(错误)':<24} {ll_sw.mean():>10.4f} {ll_sw.std():>10.4f} {np.median(ll_sw):>10.4f}")
        print(f"  {'弱序列 + 弱相图(正确)':<24} {ll_ww.mean():>10.4f} {ll_ww.std():>10.4f} {np.median(ll_ww):>10.4f}")
        print(f"  {'弱序列 + 强相图(错误)':<24} {ll_ws.mean():>10.4f} {ll_ws.std():>10.4f} {np.median(ll_ws):>10.4f}")

        diff_ss_sw = ll_ss.mean() - ll_sw.mean()
        diff_ww_ws = ll_ww.mean() - ll_ws.mean()
        print(f"\n  强序列: 正确 - 错误 = {diff_ss_sw:+.4f}")
        print(f"  弱序列: 正确 - 错误 = {diff_ww_ws:+.4f}")

        # --- 广义相图二分类 ---
        avg_s_phase = s_phase.mean(dim=0, keepdim=True)
        avg_w_phase = w_phase.mean(dim=0, keepdim=True)
        avg_mask = torch.ones(1, 16)

        all_seqs_sw = s_seqs + w_seqs
        labels_sw = np.array([1] * top_k + [0] * top_k)  # 1=强, 0=弱

        phase_s_exp = avg_s_phase.expand(len(all_seqs_sw), -1)
        mask_s_exp  = avg_mask.expand(len(all_seqs_sw), -1)
        ll_with_s = compute_ll_for_batch(model, tokenizer, all_seqs_sw, phase_s_exp, mask_s_exp, device)

        phase_w_exp = avg_w_phase.expand(len(all_seqs_sw), -1)
        mask_w_exp  = avg_mask.expand(len(all_seqs_sw), -1)
        ll_with_w = compute_ll_for_batch(model, tokenizer, all_seqs_sw, phase_w_exp, mask_w_exp, device)

        pred_strong = ll_with_s > ll_with_w
        correct = (pred_strong == labels_sw.astype(bool))
        acc_total  = correct.mean()
        acc_strong = correct[:top_k].mean()
        acc_weak   = correct[top_k:].mean()

        print(f"\n  广义相图二分类:")
        print(f"    广义强相图 mean_pssi: {avg_s_phase.mean():.4f}")
        print(f"    广义弱相图 mean_pssi: {avg_w_phase.mean():.4f}")
        print(f"    总准确率:   {acc_total:.4f} ({int(correct.sum())}/{len(correct)})")
        print(f"    强序列准确率: {acc_strong:.4f} ({int(correct[:top_k].sum())}/{top_k})")
        print(f"    弱序列准确率: {acc_weak:.4f} ({int(correct[top_k:].sum())}/{top_k})")

        # 保存中间结果
        cross_results[split_name] = {
            'll_ss': ll_ss, 'll_sw': ll_sw, 'll_ww': ll_ww, 'll_ws': ll_ws,
            'diff_ss_sw': diff_ss_sw, 'diff_ww_ws': diff_ww_ws,
            'acc_total': acc_total, 'acc_strong': acc_strong, 'acc_weak': acc_weak,
            'top_k': top_k,
        }

    # ================================================================
    # 总体结果汇总
    # ================================================================
    print("\n" + "=" * 60)
    print("Log-Likelihood 区分度验证")
    print("=" * 60)
    print(f"{'组别':<30} {'Mean LL':>10} {'Std':>10} {'Median':>10}")
    print("-" * 60)
    print(f"{'自然序列 + 正确相图':<26} {ll_correct.mean():>10.4f} {ll_correct.std():>10.4f} {np.median(ll_correct):>10.4f}")
    print(f"{'自然序列 + 错误相图':<26} {ll_wrong.mean():>10.4f} {ll_wrong.std():>10.4f} {np.median(ll_wrong):>10.4f}")
    print(f"{'随机序列 + 任意相图':<26} {ll_random.mean():>10.4f} {ll_random.std():>10.4f} {np.median(ll_random):>10.4f}")
    print("=" * 60)

    diff_12 = ll_correct.mean() - ll_wrong.mean()
    diff_13 = ll_correct.mean() - ll_random.mean()
    diff_23 = ll_wrong.mean() - ll_random.mean()
    print(f"\n差异分析:")
    print(f"  正确 - 随机打乱: {diff_12:+.4f}")
    print(f"  正确 - 随机序列: {diff_13:+.4f}")
    print(f"  随机打乱 - 随机序列: {diff_23:+.4f}")

    # ================================================================
    # 可视化：1×3 三个 panel
    # ================================================================
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.font_manager as fm

    font_path = '/data/yanjie_huang/fonts/arial.ttf'
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    matplotlib.rcParams['font.size'] = 11

    # 统一色系 (与 length_kmer_kl 一致)
    COLOR_GT  = (114/255, 138/255, 185/255)  # #728ab9 蓝灰
    COLOR_GEN = (245/255, 181/255, 191/255)  # #f5b5bf 粉
    COLOR_RND = (245/255, 221/255, 181/255)  # #f5ddb5 浅橙

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor='white')
    for ax in axes:
        ax.set_facecolor('white')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # --- Panel A: 三组总体对比 (violin) ---
    ax = axes[0]
    data_groups = [ll_correct, ll_wrong, ll_random]
    labels_groups = ['Natural+\nCorrect Phase', 'Natural+\nShuffled Phase', 'Random Seq+\nAny Phase']
    colors_3 = [COLOR_GT, COLOR_GEN, COLOR_RND]
    parts = ax.violinplot(data_groups, positions=[1, 2, 3], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_3[i])
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('gray')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels_groups, fontsize=9)
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('(A) Overall LL Discrimination', fontweight='bold')
    for i, d in enumerate(data_groups):
        ax.text(i + 1, d.mean() + 0.02, f'{d.mean():.3f}', ha='center', fontsize=9, fontweight='bold')

    # --- Panel B: 交叉配对 grouped bar (Top 25% vs Top 10%) ---
    ax = axes[1]
    GREEN, RED = COLOR_GT, COLOR_GEN
    split_names = list(cross_results.keys())
    x = np.array([0, 2.0])  # Strong Seq, Weak Seq
    width = 0.35
    gap = 0.03

    offsets = np.array([-1.5*width - gap/2, -0.5*width - gap/2,
                         0.5*width + gap/2,  1.5*width + gap/2])
    bar_colors = [GREEN, RED, GREEN, RED]
    bar_hatches = ['', '', '///', '///']

    r25 = cross_results['Top 25%']
    r10 = cross_results['Top 10%']
    vals_strong = [r25['ll_ss'].mean(), r25['ll_sw'].mean(), r10['ll_ss'].mean(), r10['ll_sw'].mean()]
    vals_weak   = [r25['ll_ww'].mean(), r25['ll_ws'].mean(), r10['ll_ww'].mean(), r10['ll_ws'].mean()]

    for gi, (gx, vals) in enumerate(zip(x, [vals_strong, vals_weak])):
        for bi, (off, val, col, hatch) in enumerate(zip(offsets, vals, bar_colors, bar_hatches)):
            ax.bar(gx + off, val, width, color=col, alpha=0.85,
                   edgecolor='white' if not hatch else 'gray', linewidth=0.8, hatch=hatch)
            ax.text(gx + off, val + 0.005, f'{val:.3f}', ha='center', fontsize=7, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GREEN, alpha=0.85, label='Correct Phase'),
        Patch(facecolor=RED, alpha=0.85, label='Wrong Phase'),
        Patch(facecolor='white', edgecolor='gray', hatch='///', label='Top 10%'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right',
              edgecolor='gray', fancybox=False)
    ax.set_xticks(x)
    ax.set_xticklabels(['Strong LLPS Seq', 'Weak LLPS Seq'], fontsize=10)
    ax.set_ylabel('Mean Log-Likelihood')
    ax.set_title('(B) Cross-Pairing (Top 25% vs Top 10%)', fontweight='bold')
    all_bar_vals = vals_strong + vals_weak
    ax.set_ylim(min(all_bar_vals) - 0.06, max(all_bar_vals) + 0.04)

    # --- Panel C: 二分类准确率 ---
    ax = axes[2]
    x = np.arange(2)
    width = 0.22
    ORANGE, BLUE = COLOR_RND, (160/255, 180/255, 210/255)  # 浅橙, 浅蓝灰

    strong_accs = [r25['acc_strong'] * 100, r10['acc_strong'] * 100]
    weak_accs   = [r25['acc_weak'] * 100,   r10['acc_weak'] * 100]
    total_accs  = [r25['acc_total'] * 100,  r10['acc_total'] * 100]

    b1 = ax.bar(x - width, strong_accs, width, color=ORANGE, alpha=0.85, edgecolor='white', label='Strong Seq Acc')
    b2 = ax.bar(x,         total_accs,  width, color=BLUE,   alpha=0.85, edgecolor='white', label='Total Acc')
    b3 = ax.bar(x + width, weak_accs,   width, color=GREEN,  alpha=0.85, edgecolor='white', label='Weak Seq Acc')

    for b in list(b1) + list(b2) + list(b3):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                f'{b.get_height():.1f}%', ha='center', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Top 25%\n(n={r25["top_k"]})', f'Top 10%\n(n={r10["top_k"]})'], fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(C) Phase Diagram Binary Classification', fontweight='bold')
    ax.legend(fontsize=8.5, loc='lower right', edgecolor='gray', fancybox=False)
    ax.set_ylim(50, 105)
    ax.axhline(y=50, color='gray', linewidth=0.5, linestyle='--')

    plt.tight_layout()
    fig_path = '/data/yanjie_huang/LLPS/predictor/PhaseFlow/analysis/loglikelihood/ll_discrimination.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {fig_path}")


if __name__ == '__main__':
    main()
