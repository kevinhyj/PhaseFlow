#!/usr/bin/env python3
"""
PhaseFlow De Novo 序列设计管线

Step 1: 从 missing_0 选 Top 5% 弱相分离相图 (Weak LLPS, PSSI最正=相分离最弱, 509条) + 噪声增强 → 1000 条目标
Step 2: 每个目标相图 LM 生成 10 条候选 → 10,000 条
Step 3: 生物学合理性过滤 (长度/合法AA/低复杂度/去重/新颖性)
Step 4: Flow 打分 (generate_phase → Spearman) → 排序取 Top-K
Step 5: 输出两份 CSV (过滤后全量 + 打分后 Top-K)

Usage:
    cd /data/yanjie_huang/LLPS/predictor/PhaseFlow
    python "analysis/de novo/de_novo_pipeline.py" --gpu 5
    python "analysis/de novo/de_novo_pipeline.py" --gpu 5 --n_candidates 5 --n_target 100  # 快速测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from scipy.stats import spearmanr
from datetime import datetime

from phaseflow import PhaseFlow, AminoAcidTokenizer

# ============================================================================
# Constants
# ============================================================================

GROUP_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')


# ============================================================================
# Model loading
# ============================================================================

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


# ============================================================================
# Step 1: Select target phase diagrams
# ============================================================================

def select_target_phases(phase_data, top_percent=5, n_target=1000,
                         noise_scale=0.05, seed=42):
    """从 missing_0 选 Top X% 强相分离相图，噪声增强到 n_target 条。"""
    rng = np.random.RandomState(seed)

    # 用 mean PSSI 排序，取 Top X%
    mean_pssi = phase_data.mean(axis=1)
    threshold = np.percentile(mean_pssi, 100 - top_percent)
    strong_idx = np.where(mean_pssi >= threshold)[0]
    strong_phases = phase_data[strong_idx]
    n_strong = len(strong_phases)

    print(f"  Top {top_percent}% threshold: mean_pssi >= {threshold:.4f}")
    print(f"  Selected {n_strong} strong phase diagrams")

    if n_strong >= n_target:
        # 够了，直接截断
        target_phases = strong_phases[:n_target]
        source_idx = strong_idx[:n_target]
        is_augmented = np.zeros(n_target, dtype=bool)
    else:
        # 不够，噪声增强补齐
        n_extra = n_target - n_strong
        extra_idx = rng.choice(n_strong, n_extra)
        noise = noise_scale * rng.randn(n_extra, phase_data.shape[1])
        augmented = strong_phases[extra_idx] + noise

        target_phases = np.concatenate([strong_phases, augmented], axis=0)
        source_idx = np.concatenate([strong_idx, strong_idx[extra_idx]])
        is_augmented = np.concatenate([
            np.zeros(n_strong, dtype=bool),
            np.ones(n_extra, dtype=bool),
        ])
        print(f"  Augmented {n_extra} phases (noise_scale={noise_scale})")

    print(f"  Total target phases: {len(target_phases)}")
    return target_phases, source_idx, is_augmented


# ============================================================================
# Step 2: Batch generation
# ============================================================================

def generate_candidates(model, tokenizer, target_phases, n_candidates=10,
                        batch_size=32, temperature=1.0, max_len=25, device='cpu'):
    """为每个目标相图生成 n_candidates 条候选序列。"""
    all_seqs = []
    all_phase_idx = []

    for i in tqdm(range(len(target_phases)), desc="Step 2: Generating"):
        phase_tensor = torch.tensor(
            target_phases[i], dtype=torch.float32, device=device
        )
        seqs_for_this = []

        for start in range(0, n_candidates, batch_size):
            actual_batch = min(batch_size, n_candidates - start)
            batch_phase = phase_tensor.unsqueeze(0).expand(actual_batch, -1).contiguous()

            with torch.no_grad():
                _, seqs = model.generate_sequence(
                    batch_phase, tokenizer,
                    max_len=max_len,
                    temperature=temperature,
                )
            seqs_for_this.extend(seqs)

        all_seqs.extend(seqs_for_this)
        all_phase_idx.extend([i] * len(seqs_for_this))

    return all_seqs, all_phase_idx


# ============================================================================
# Step 3: Biological validity filter
# ============================================================================

def is_biologically_valid(seq):
    """只检查生物学合理性，不过滤相分离特征。"""
    if len(seq) < 10:
        return False
    if not all(c in VALID_AA for c in seq):
        return False
    # 单AA占比 > 40%
    counts = Counter(seq)
    if max(counts.values()) / len(seq) > 0.4:
        return False
    # 连续4个相同AA
    for aa in VALID_AA:
        if aa * 4 in seq:
            return False
    return True


def filter_candidates(all_seqs, all_phase_idx, training_seqs=None):
    """过滤 + 去重 + 新颖性检查。"""
    # Step 3a: 生物学合理性
    valid = [(s, idx) for s, idx in zip(all_seqs, all_phase_idx)
             if is_biologically_valid(s)]
    n_bio = len(valid)

    # Step 3b: 去重（同一序列只保留第一次出现）
    seen = set()
    deduped = []
    for s, idx in valid:
        if s not in seen:
            seen.add(s)
            deduped.append((s, idx))
    n_dedup = len(deduped)

    # Step 3c: 训练集新颖性
    if training_seqs is not None:
        novel = [(s, idx) for s, idx in deduped if s not in training_seqs]
        n_novel = len(novel)
    else:
        novel = deduped
        n_novel = n_dedup

    seqs_out = [s for s, _ in novel]
    idx_out = [idx for _, idx in novel]

    print(f"  Biological validity: {len(all_seqs)} → {n_bio}")
    print(f"  Deduplication:       {n_bio} → {n_dedup}")
    print(f"  Novelty check:      {n_dedup} → {n_novel}")

    return seqs_out, idx_out


# ============================================================================
# Step 4: Flow scoring
# ============================================================================

def score_candidates(model, tokenizer, seqs, phase_idx, target_phases,
                     batch_size=64, device='cpu'):
    """用 generate_phase() 打分，计算 Spearman(预测, 目标)。"""
    input_ids = tokenizer.batch_encode(seqs, max_len=32).to(device)
    attention_mask = (input_ids != tokenizer.PAD_ID).long()
    seq_lens = torch.tensor([len(s) for s in seqs], device=device)

    all_pred = []
    for start in tqdm(range(0, len(seqs), batch_size), desc="Step 4: Scoring"):
        end = min(start + batch_size, len(seqs))
        with torch.no_grad():
            pred = model.generate_phase(
                input_ids[start:end],
                attention_mask[start:end],
                seq_lens[start:end],
                method='euler',
            )
        all_pred.append(pred.cpu())

    pred_phases = torch.cat(all_pred, dim=0).numpy()  # (N, 16)

    # Spearman per candidate
    corrs = []
    for i in range(len(seqs)):
        target = target_phases[phase_idx[i]]
        c, _ = spearmanr(pred_phases[i], target)
        corrs.append(float(c) if not np.isnan(c) else -1.0)

    return np.array(corrs), pred_phases


# ============================================================================
# Step 5: Output
# ============================================================================

def save_results(seqs, phase_idx, target_phases, corrs, pred_phases,
                 source_idx, is_augmented, top_k, output_dir):
    """保存两份 CSV：过滤后全量 + 打分后 Top-K。"""
    os.makedirs(output_dir, exist_ok=True)

    # === 全量（带打分） ===
    rows = []
    for i in range(len(seqs)):
        row = {
            'sequence': seqs[i],
            'length': len(seqs[i]),
            'target_phase_idx': phase_idx[i],
            'source_data_idx': int(source_idx[phase_idx[i]]),
            'is_augmented': bool(is_augmented[phase_idx[i]]),
            'spearman': corrs[i],
        }
        for k in range(16):
            row[f'target_pssi_{k}'] = target_phases[phase_idx[i]][k]
        for k in range(16):
            row[f'pred_pssi_{k}'] = pred_phases[i][k]
        row['mean_pred_pssi'] = float(pred_phases[i].mean())
        rows.append(row)

    df_all = pd.DataFrame(rows)
    df_all = df_all.sort_values('spearman', ascending=False).reset_index(drop=True)

    # 过滤后全量
    path_all = os.path.join(output_dir, 'candidates_filtered.csv')
    df_all.to_csv(path_all, index=False)
    print(f"\n  Saved all {len(df_all)} candidates → {path_all}")

    # Top-K
    df_topk = df_all.head(top_k).reset_index(drop=True)
    path_topk = os.path.join(output_dir, 'candidates_scored_topK.csv')
    df_topk.to_csv(path_topk, index=False)
    print(f"  Saved Top-{top_k} candidates → {path_topk}")

    return df_all, df_topk


def print_summary(df_all, df_topk, funnel_stats):
    """打印漏斗统计和结果摘要。"""
    print("\n" + "=" * 60)
    print("De Novo Pipeline Summary")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n--- Funnel ---")
    for step, count in funnel_stats:
        print(f"  {step}: {count:,}")

    print(f"\n--- All Candidates (n={len(df_all)}) ---")
    print(f"  Spearman  mean={df_all['spearman'].mean():.4f}  "
          f"median={df_all['spearman'].median():.4f}  "
          f"min={df_all['spearman'].min():.4f}  "
          f"max={df_all['spearman'].max():.4f}")
    print(f"  Length    mean={df_all['length'].mean():.1f}  "
          f"median={df_all['length'].median():.0f}")

    print(f"\n--- Top-K (n={len(df_topk)}) ---")
    print(f"  Spearman  mean={df_topk['spearman'].mean():.4f}  "
          f"median={df_topk['spearman'].median():.4f}  "
          f"min={df_topk['spearman'].min():.4f}  "
          f"max={df_topk['spearman'].max():.4f}")
    print(f"  Length    mean={df_topk['length'].mean():.1f}  "
          f"median={df_topk['length'].median():.0f}")
    print(f"  Mean pred PSSI: {df_topk['mean_pred_pssi'].mean():.4f}")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PhaseFlow De Novo Pipeline')
    parser.add_argument('--checkpoint', type=str,
                        default='/data/yanjie_huang/LLPS/predictor/PhaseFlow/outputs_set/output_set_flow32_missing15/best_model.pt')
    parser.add_argument('--input_csv', type=str,
                        default='/data/yanjie_huang/LLPS/phase_diagram/by_missing/missing_0.csv')
    parser.add_argument('--output_dir', type=str,
                        default='/data/yanjie_huang/LLPS/predictor/PhaseFlow/analysis/de novo/top5_weak/de_novo_results')
    parser.add_argument('--top_percent', type=float, default=5.0)
    parser.add_argument('--n_candidates', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_len', type=int, default=25)
    parser.add_argument('--gen_batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)
    tokenizer = AminoAcidTokenizer()

    # Load data
    print(f"\nLoading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    phase_data = df[GROUP_COLS].values.astype(np.float32)
    training_seqs = set(df['AminoAcidSequence'].values)
    print(f"  Loaded {len(df)} sequences, {len(training_seqs)} unique")

    # Step 1: Select Top 5% phase diagrams (no augmentation)
    print(f"\n{'='*60}")
    print("Step 1: Select target phase diagrams")
    print(f"{'='*60}")
    mean_pssi = phase_data.mean(axis=1)
    threshold = np.percentile(mean_pssi, 100 - args.top_percent)
    strong_idx = np.where(mean_pssi >= threshold)[0]
    target_phases = phase_data[strong_idx]
    print(f"  Top {args.top_percent}% threshold: mean_pssi >= {threshold:.4f}")
    print(f"  Selected {len(target_phases)} phase diagrams")

    # Step 2: Generate
    print(f"\n{'='*60}")
    print(f"Step 2: Generate {args.n_candidates} candidates per phase")
    print(f"{'='*60}")
    all_seqs, all_phase_idx = generate_candidates(
        model, tokenizer, target_phases,
        n_candidates=args.n_candidates,
        batch_size=args.gen_batch_size,
        temperature=args.temperature,
        max_len=args.max_len,
        device=device,
    )
    n_generated = len(all_seqs)
    print(f"  Generated {n_generated:,} candidates")

    # Step 3: Filter
    print(f"\n{'='*60}")
    print("Step 3: Biological validity filter")
    print(f"{'='*60}")
    filtered_seqs, filtered_idx = filter_candidates(
        all_seqs, all_phase_idx, training_seqs=training_seqs,
    )
    n_filtered = len(filtered_seqs)

    # Step 4: Save (no Flow scoring)
    print(f"\n{'='*60}")
    print("Step 4: Save results")
    print(f"{'='*60}")
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for i in range(len(filtered_seqs)):
        row = {
            'sequence': filtered_seqs[i],
            'length': len(filtered_seqs[i]),
            'target_phase_idx': filtered_idx[i],
            'source_data_idx': int(strong_idx[filtered_idx[i]]),
        }
        for k in range(16):
            row[f'target_pssi_{k}'] = target_phases[filtered_idx[i]][k]
        row['mean_target_pssi'] = float(target_phases[filtered_idx[i]].mean())
        rows.append(row)

    df_out = pd.DataFrame(rows)
    output_path = os.path.join(args.output_dir, 'candidates.csv')
    df_out.to_csv(output_path, index=False)
    print(f"  Saved {len(df_out):,} candidates → {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("De Novo Pipeline Summary (v2 — no Flow scoring)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n--- Funnel ---")
    print(f"  Target phases: {len(target_phases)}")
    print(f"  Generated candidates: {n_generated:,}")
    print(f"  After bio filter + dedup + novelty: {n_filtered:,}")
    print(f"\n--- Output (n={len(df_out)}) ---")
    print(f"  Length  mean={df_out['length'].mean():.1f}  median={df_out['length'].median():.0f}")
    print(f"  Unique target phases covered: {df_out['target_phase_idx'].nunique()}")
    print(f"  Candidates per phase: mean={len(df_out)/df_out['target_phase_idx'].nunique():.1f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
