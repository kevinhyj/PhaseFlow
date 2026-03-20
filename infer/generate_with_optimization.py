#!/usr/bin/env python3
"""
PhaseFlow Inference-Time Optimization

对每个输入相图生成 N 个候选序列，用 seq→phase 预测相图并计算相关性，
选择相关性最高的序列作为最终输出。

Usage:
    # 测试单个相图
    python generate_with_optimization.py --n_candidates 10 --num_samples 1 --gpu 0

    # 处理前500个相图
    python generate_with_optimization.py --n_candidates 100 --num_samples 500 --gpu 0

    # 处理全部相图
    python generate_with_optimization.py --n_candidates 100 --gpu 0
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

# Add PhaseFlow to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phaseflow import PhaseFlow, AminoAcidTokenizer


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: str, device: torch.device):
    """Load PhaseFlow model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
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
    print(f"  dim={config['model']['dim']}, depth={config['model']['depth']}, "
          f"use_set_encoder={config['model'].get('use_set_encoder', False)}")
    return model, config


def generate_and_score_candidates(
    model, tokenizer, target_phase: torch.Tensor,
    n_candidates: int, gen_batch_size: int, score_batch_size: int,
    temperature: float, max_len: int, device: torch.device
):
    """
    为单个相图生成 N 个候选序列，预测相图并计算相关性，返回最优序列。

    Args:
        target_phase: (16,) 目标相图（已在 device 上）

    Returns:
        best_seq: 最优序列字符串
        best_corr: 最高 Spearman 相关性
        mean_corr: 所有候选的平均相关性
        std_corr: 所有候选的相关性标准差
        best_pred_phase: (16,) 最优序列预测的相图
        all_seqs: 所有候选序列列表
        all_corrs: 所有候选的相关性列表
    """
    # === Step 1: 批量生成 N 个候选序列 ===
    phase_batch = target_phase.unsqueeze(0).expand(gen_batch_size, -1)  # (B, 16)
    all_seqs = []

    for start in range(0, n_candidates, gen_batch_size):
        end = min(start + gen_batch_size, n_candidates)
        actual_batch = end - start
        batch_phase = target_phase.unsqueeze(0).expand(actual_batch, -1).contiguous()

        with torch.no_grad():
            _, seqs = model.generate_sequence(
                batch_phase, tokenizer,
                max_len=max_len,
                temperature=temperature,
                top_k=None,
                top_p=None,
            )
        all_seqs.extend(seqs)

    # === Step 2: Tokenize 所有候选序列 ===
    input_ids = tokenizer.batch_encode(all_seqs, max_len=32).to(device)
    attention_mask = (input_ids != tokenizer.PAD_ID).long()
    seq_lens = torch.tensor([len(s) for s in all_seqs], device=device)

    # === Step 3: 批量预测相图 ===
    all_pred_phases = []
    for start in range(0, len(all_seqs), score_batch_size):
        end = min(start + score_batch_size, len(all_seqs))
        with torch.no_grad():
            pred = model.generate_phase(
                input_ids[start:end],
                attention_mask[start:end],
                seq_lens[start:end],
                method='euler',
            )
        all_pred_phases.append(pred.cpu())

    pred_phases = torch.cat(all_pred_phases, dim=0).numpy()  # (N, 16)
    target_np = target_phase.cpu().numpy()

    # === Step 4: 计算每个候选的 Spearman 相关性 ===
    all_corrs = []
    for i in range(len(all_seqs)):
        if len(all_seqs[i]) == 0:
            all_corrs.append(-1.0)
            continue
        corr, _ = spearmanr(pred_phases[i], target_np)
        all_corrs.append(float(corr) if not np.isnan(corr) else -1.0)

    # === Step 5: 选择最优 ===
    best_idx = int(np.argmax(all_corrs))
    best_seq = all_seqs[best_idx]
    best_corr = all_corrs[best_idx]
    best_pred_phase = pred_phases[best_idx]
    mean_corr = float(np.mean(all_corrs))
    std_corr = float(np.std(all_corrs))

    return best_seq, best_corr, mean_corr, std_corr, best_pred_phase, all_seqs, all_corrs


def main():
    parser = argparse.ArgumentParser(description='PhaseFlow Inference-Time Optimization')
    parser.add_argument('--checkpoint', type=str,
                        default='../outputs_set/output_set_flow32_missing15/best_model.pt')
    parser.add_argument('--input_csv', type=str,
                        default='/data/yanjie_huang/LLPS/phase_diagram/by_missing/missing_0.csv')
    parser.add_argument('--output_dir', type=str,
                        default='../generated_from_missing0/outputs/dataset_optimized')
    parser.add_argument('--n_candidates', type=int, default=100,
                        help='Number of candidate sequences per phase diagram')
    parser.add_argument('--gen_batch_size', type=int, default=32,
                        help='Batch size for sequence generation')
    parser.add_argument('--score_batch_size', type=int, default=64,
                        help='Batch size for seq→phase scoring')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_len', type=int, default=25)
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of phase diagrams to process (None = all)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model, config = load_model(args.checkpoint, device)
    tokenizer = AminoAcidTokenizer()

    # Load input data
    print(f"\nLoading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    GROUP_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]
    phase_data = df[GROUP_COLS].values.astype(np.float32)
    original_sequences = df['AminoAcidSequence'].values

    if args.num_samples is not None:
        phase_data = phase_data[:args.num_samples]
        original_sequences = original_sequences[:args.num_samples]

    n_total = len(phase_data)
    print(f"Processing {n_total} phase diagrams, {args.n_candidates} candidates each")

    # Main loop
    results = []
    for i in tqdm(range(n_total), desc="Optimizing"):
        target_phase = torch.tensor(phase_data[i], dtype=torch.float32, device=device)

        best_seq, best_corr, mean_corr, std_corr, best_pred_phase, _, all_corrs = \
            generate_and_score_candidates(
                model, tokenizer, target_phase,
                n_candidates=args.n_candidates,
                gen_batch_size=args.gen_batch_size,
                score_batch_size=args.score_batch_size,
                temperature=args.temperature,
                max_len=args.max_len,
                device=device,
            )

        row = {
            'original_sequence': original_sequences[i],
            'optimized_sequence': best_seq,
            'length': len(best_seq),
            'best_correlation': best_corr,
            'mean_correlation': mean_corr,
            'std_correlation': std_corr,
            'original_idx': i,
        }
        # Input phase values
        for k in range(16):
            row[f'pssi_{k}'] = phase_data[i][k]
        # Predicted phase values for best sequence
        for k in range(16):
            row[f'pred_pssi_{k}'] = best_pred_phase[k]

        results.append(row)

    # Save results
    df_out = pd.DataFrame(results)
    output_csv = output_dir / 'sequences.csv'
    df_out.to_csv(output_csv, index=False)
    print(f"\n✓ Saved {len(df_out):,} sequences to {output_csv}")

    # Print summary stats
    print("\n" + "=" * 60)
    print("Optimization Summary")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total sequences: {len(df_out):,}")
    print(f"n_candidates per phase: {args.n_candidates}")
    print(f"\nBest Correlation (Spearman):")
    print(f"  Mean:   {df_out['best_correlation'].mean():.4f}")
    print(f"  Median: {df_out['best_correlation'].median():.4f}")
    print(f"  Min:    {df_out['best_correlation'].min():.4f}")
    print(f"  Max:    {df_out['best_correlation'].max():.4f}")
    print(f"\nMean Correlation (across candidates):")
    print(f"  Mean:   {df_out['mean_correlation'].mean():.4f}")
    print(f"  Median: {df_out['mean_correlation'].median():.4f}")
    print(f"\nSequence Length:")
    print(f"  Mean:   {df_out['length'].mean():.2f}")
    print(f"  Median: {df_out['length'].median():.0f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
