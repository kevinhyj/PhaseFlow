#!/usr/bin/env python3
"""
PhaseFlow De Novo 序列设计管线 — Bottom 5% 强相分离 (Strong LLPS)

Step 1: 从 missing_0 选 Bottom 5% 强相分离相图 (~509条, PSSI最负=相分离最强)
Step 2: 每个目标相图 LM 生成 20 条候选
Step 3: 生物学合理性过滤 (长度/合法AA/低复杂度/去重/新颖性)
Step 4: 输出 CSV

Usage:
    cd /data/yanjie_huang/LLPS/predictor/PhaseFlow
    python "analysis/de novo/bottom5_strong/de_novo_pipeline.py" --gpu 0
    python "analysis/de novo/bottom5_strong/de_novo_pipeline.py" --gpu 0 --n_candidates 5 --num_samples 100  # 快速测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
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
# Generation
# ============================================================================

def generate_candidates(model, tokenizer, target_phases, n_candidates=20,
                        batch_size=32, temperature=1.0, max_len=25, device='cpu'):
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
# Biological validity filter
# ============================================================================

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


def filter_candidates(all_seqs, all_phase_idx, training_seqs=None):
    valid = [(s, idx) for s, idx in zip(all_seqs, all_phase_idx)
             if is_biologically_valid(s)]
    n_bio = len(valid)

    seen = set()
    deduped = []
    for s, idx in valid:
        if s not in seen:
            seen.add(s)
            deduped.append((s, idx))
    n_dedup = len(deduped)

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
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PhaseFlow De Novo Pipeline — Bottom 5%')
    parser.add_argument('--checkpoint', type=str,
                        default='/data/yanjie_huang/LLPS/predictor/PhaseFlow/outputs_set/output_set_flow32_missing15/best_model.pt')
    parser.add_argument('--input_csv', type=str,
                        default='/data/yanjie_huang/LLPS/phase_diagram/by_missing/missing_0.csv')
    parser.add_argument('--output_dir', type=str,
                        default='/data/yanjie_huang/LLPS/predictor/PhaseFlow/analysis/de novo/bottom5_strong/de_novo_results')
    parser.add_argument('--bottom_percent', type=float, default=5.0)
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

    model = load_model(args.checkpoint, device)
    tokenizer = AminoAcidTokenizer()

    # Load data
    print(f"\nLoading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    phase_data = df[GROUP_COLS].values.astype(np.float32)
    training_seqs = set(df['AminoAcidSequence'].values)
    print(f"  Loaded {len(df)} sequences, {len(training_seqs)} unique")

    # Step 1: Select Bottom 5% phase diagrams
    print(f"\n{'='*60}")
    print("Step 1: Select target phase diagrams (Bottom 5%)")
    print(f"{'='*60}")
    mean_pssi = phase_data.mean(axis=1)
    threshold = np.percentile(mean_pssi, args.bottom_percent)
    weak_idx = np.where(mean_pssi <= threshold)[0]
    target_phases = phase_data[weak_idx]
    print(f"  Bottom {args.bottom_percent}% threshold: mean_pssi <= {threshold:.4f}")
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

    # Step 4: Save
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
            'source_data_idx': int(weak_idx[filtered_idx[i]]),
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
    print("De Novo Pipeline Summary — Bottom 5%")
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
