#!/usr/bin/env python3
"""
PhaseFlow Inference-Time Optimization for Top 5% Phase Diagrams

对 Top 5% 弱相分离相图 (Weak LLPS, PSSI最正=相分离最弱)，每个生成 200 个候选序列，
用 Flow 反向预测相图 + Spearman 打分，每个相图取 Top-1。

Usage:
    cd /data/yanjie_huang/LLPS/predictor/PhaseFlow
    python "analysis/de novo/de_novo_ito.py" --gpu 5
    python "analysis/de novo/de_novo_ito.py" --gpu 5 --n_candidates 50 --num_samples 10  # 快速测试
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

GROUP_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')


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


def is_biologically_valid(seq):
    if len(seq) < 10:
        return False
    if not all(c in VALID_AA for c in seq):
        return False
    counts = Counter(seq)
    if max(counts.values()) / len(seq) > 0.4:
        return False
    for aa in VALID_AA:
        if aa * 4 in seq:
            return False
    return True


def optimize_one_phase(model, tokenizer, target_phase, n_candidates,
                       gen_batch_size, score_batch_size, temperature,
                       max_len, device):
    """对单个相图生成 N 个候选，打分后返回 top-1。"""
    # Generate
    all_seqs = []
    for start in range(0, n_candidates, gen_batch_size):
        actual_batch = min(gen_batch_size, n_candidates - start)
        batch_phase = target_phase.unsqueeze(0).expand(actual_batch, -1).contiguous()
        with torch.no_grad():
            _, seqs = model.generate_sequence(
                batch_phase, tokenizer,
                max_len=max_len, temperature=temperature,
            )
        all_seqs.extend(seqs)

    # Filter biologically invalid
    valid_seqs = [s for s in all_seqs if is_biologically_valid(s)]
    if len(valid_seqs) == 0:
        return None

    # Score with Flow
    input_ids = tokenizer.batch_encode(valid_seqs, max_len=32).to(device)
    attention_mask = (input_ids != tokenizer.PAD_ID).long()
    seq_lens = torch.tensor([len(s) for s in valid_seqs], device=device)

    all_pred = []
    for start in range(0, len(valid_seqs), score_batch_size):
        end = min(start + score_batch_size, len(valid_seqs))
        with torch.no_grad():
            pred = model.generate_phase(
                input_ids[start:end],
                attention_mask[start:end],
                seq_lens[start:end],
                method='euler',
            )
        all_pred.append(pred.cpu())

    pred_phases = torch.cat(all_pred, dim=0).numpy()
    target_np = target_phase.cpu().numpy()

    corrs = []
    for i in range(len(valid_seqs)):
        c, _ = spearmanr(pred_phases[i], target_np)
        corrs.append(float(c) if not np.isnan(c) else -1.0)

    best_idx = int(np.argmax(corrs))
    return {
        'best_seq': valid_seqs[best_idx],
        'best_corr': corrs[best_idx],
        'mean_corr': float(np.mean(corrs)),
        'std_corr': float(np.std(corrs)),
        'best_pred_phase': pred_phases[best_idx],
        'n_generated': len(all_seqs),
        'n_valid': len(valid_seqs),
    }


def main():
    parser = argparse.ArgumentParser(description='De Novo ITO for Top 5%')
    parser.add_argument('--checkpoint', type=str,
                        default='/data/yanjie_huang/LLPS/predictor/PhaseFlow/outputs_set/output_set_flow32_missing15/best_model.pt')
    parser.add_argument('--input_csv', type=str,
                        default='/data/yanjie_huang/LLPS/phase_diagram/by_missing/missing_0.csv')
    parser.add_argument('--output_dir', type=str,
                        default='/data/yanjie_huang/LLPS/predictor/PhaseFlow/analysis/de novo/top5_weak/de_novo_ito_results')
    parser.add_argument('--top_percent', type=float, default=5.0)
    parser.add_argument('--n_candidates', type=int, default=200)
    parser.add_argument('--gen_batch_size', type=int, default=32)
    parser.add_argument('--score_batch_size', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_len', type=int, default=25)
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Only process first N phases (for testing)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    tokenizer = AminoAcidTokenizer()

    # Load data & select Top 5%
    print(f"\nLoading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    phase_data = df[GROUP_COLS].values.astype(np.float32)

    mean_pssi = phase_data.mean(axis=1)
    threshold = np.percentile(mean_pssi, 100 - args.top_percent)
    strong_idx = np.where(mean_pssi >= threshold)[0]
    target_phases = phase_data[strong_idx]

    if args.num_samples is not None:
        strong_idx = strong_idx[:args.num_samples]
        target_phases = target_phases[:args.num_samples]

    print(f"  Top {args.top_percent}% threshold: mean_pssi >= {threshold:.4f}")
    print(f"  Target phases: {len(target_phases)}")
    print(f"  Candidates per phase: {args.n_candidates}")

    # Main loop
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    n_failed = 0

    for i in tqdm(range(len(target_phases)), desc="ITO"):
        target = torch.tensor(target_phases[i], dtype=torch.float32, device=device)
        res = optimize_one_phase(
            model, tokenizer, target,
            n_candidates=args.n_candidates,
            gen_batch_size=args.gen_batch_size,
            score_batch_size=args.score_batch_size,
            temperature=args.temperature,
            max_len=args.max_len,
            device=device,
        )
        if res is None:
            n_failed += 1
            continue

        row = {
            'sequence': res['best_seq'],
            'length': len(res['best_seq']),
            'source_data_idx': int(strong_idx[i]),
            'best_spearman': res['best_corr'],
            'mean_spearman': res['mean_corr'],
            'std_spearman': res['std_corr'],
            'n_generated': res['n_generated'],
            'n_valid': res['n_valid'],
        }
        for k in range(16):
            row[f'target_pssi_{k}'] = target_phases[i][k]
        for k in range(16):
            row[f'pred_pssi_{k}'] = res['best_pred_phase'][k]
        row['mean_target_pssi'] = float(target_phases[i].mean())
        row['mean_pred_pssi'] = float(res['best_pred_phase'].mean())
        results.append(row)

    # Save
    df_out = pd.DataFrame(results)
    output_path = os.path.join(args.output_dir, 'ito_top1.csv')
    df_out.to_csv(output_path, index=False)

    # Summary
    print("\n" + "=" * 60)
    print(f"ITO Summary (Top {args.top_percent}%, {args.n_candidates} candidates/phase)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target phases: {len(target_phases)}")
    print(f"Successful: {len(df_out)}  Failed: {n_failed}")
    print(f"\nBest Spearman (top-1 per phase):")
    print(f"  Mean:   {df_out['best_spearman'].mean():.4f}")
    print(f"  Median: {df_out['best_spearman'].median():.4f}")
    print(f"  Min:    {df_out['best_spearman'].min():.4f}")
    print(f"  Max:    {df_out['best_spearman'].max():.4f}")
    print(f"\nMean Spearman (across all candidates):")
    print(f"  Mean:   {df_out['mean_spearman'].mean():.4f}")
    print(f"\nSequence Length:")
    print(f"  Mean:   {df_out['length'].mean():.1f}")
    print(f"  Median: {df_out['length'].median():.0f}")
    print(f"\nSaved → {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
