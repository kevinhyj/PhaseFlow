#!/usr/bin/env python3
"""
PhaseFlow Phase→Sequence Generation from missing_0.csv

从 missing_0.csv 的完整相图数据生成序列，用于数据增强。

Usage:
    python generate_sequences.py --n_samples 1 --output_dir outputs/dataset_1x --gpu 0
    python generate_sequences.py --n_samples 5 --output_dir outputs/dataset_5x --gpu 0
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
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

    print(f"Model config:")
    print(f"  dim={config['model']['dim']}, depth={config['model']['depth']}")
    print(f"  heads={config['model']['heads']}, use_set_encoder={config['model'].get('use_set_encoder', False)}")

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
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model, config


def load_missing0_data(csv_path: str):
    """Load missing_0.csv data."""
    print(f"Loading data from {csv_path}...")

    df = pd.read_csv(csv_path)
    GROUP_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]

    phase_data = df[GROUP_COLS].values  # (10165, 16), no NaN
    original_sequences = df['AminoAcidSequence'].values

    print(f"Loaded {len(df)} samples")
    print(f"  Phase data shape: {phase_data.shape}")
    print(f"  Phase value range: [{phase_data.min():.3f}, {phase_data.max():.3f}]")
    print(f"  Sequence length range: [{min(len(s) for s in original_sequences)}, {max(len(s) for s in original_sequences)}]")

    return phase_data, original_sequences, GROUP_COLS


def generate_sequences_batch(
    model, tokenizer, phase_data, original_sequences,
    n_samples_per_phase, batch_size, max_len, temperature, device
):
    """Generate sequences from phase diagrams in batches."""
    all_results = []
    n_total = len(phase_data)

    print(f"\nGenerating {n_samples_per_phase} sequence(s) per phase diagram...")
    print(f"Total: {n_total} × {n_samples_per_phase} = {n_total * n_samples_per_phase} sequences")

    for sample_idx in range(n_samples_per_phase):
        print(f"\n--- Generation round {sample_idx + 1}/{n_samples_per_phase} ---")

        for i in tqdm(range(0, n_total, batch_size), desc=f"Round {sample_idx + 1}"):
            end_idx = min(i + batch_size, n_total)
            batch_phases = phase_data[i:end_idx]  # (B, 16)
            batch_orig_seqs = original_sequences[i:end_idx]

            # Convert to tensor
            phase_tensor = torch.tensor(batch_phases, dtype=torch.float32, device=device)

            # Generate sequences
            with torch.no_grad():
                _, generated_seqs = model.generate_sequence(
                    phase_tensor, tokenizer,
                    max_len=max_len,
                    temperature=temperature,
                    top_k=None,
                    top_p=None,
                )

            # Record results
            for j, gen_seq in enumerate(generated_seqs):
                result = {
                    'original_sequence': batch_orig_seqs[j],
                    'generated_sequence': gen_seq,
                    'generation_idx': sample_idx,
                    'original_idx': i + j,
                    'length': len(gen_seq),
                }

                # Add 16 PSSI values
                for k in range(16):
                    result[f'pssi_{k}'] = batch_phases[j][k]

                # Add statistics
                result['mean_pssi'] = batch_phases[j].mean()
                result['std_pssi'] = batch_phases[j].std()
                result['min_pssi'] = batch_phases[j].min()
                result['max_pssi'] = batch_phases[j].max()

                all_results.append(result)

    return all_results


def compute_statistics(df: pd.DataFrame, original_sequences: np.ndarray):
    """Compute generation statistics."""
    stats = []
    stats.append("=" * 80)
    stats.append("PhaseFlow Generation Statistics")
    stats.append("=" * 80)
    stats.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stats.append(f"Total sequences: {len(df):,}")
    stats.append("")

    # Sequence length distribution
    stats.append("Sequence Length Distribution:")
    length_counts = df['length'].value_counts().sort_index()
    for length, count in length_counts.items():
        pct = 100 * count / len(df)
        stats.append(f"  Length {length:>2}: {count:>6,} ({pct:>5.2f}%)")
    stats.append(f"  Mean: {df['length'].mean():.2f}")
    stats.append(f"  Median: {df['length'].median():.0f}")
    stats.append("")

    # Amino acid composition
    all_aa = ''.join(df['generated_sequence'])
    aa_counts = Counter(all_aa)
    total_aa = len(all_aa)

    stats.append("Amino Acid Composition (Top 10):")
    for aa, count in aa_counts.most_common(10):
        pct = 100 * count / total_aa
        stats.append(f"  {aa}: {count:>7,} ({pct:>5.2f}%)")
    stats.append("")

    # Aromatic content
    aromatic = sum(all_aa.count(aa) for aa in ['F', 'W', 'Y'])
    stats.append(f"Aromatic content (F+W+Y): {100*aromatic/total_aa:.2f}%")
    stats.append(f"W content: {100*all_aa.count('W')/total_aa:.2f}%")
    stats.append("")

    # Uniqueness
    unique_count = df['generated_sequence'].nunique()
    stats.append("Uniqueness:")
    stats.append(f"  Unique sequences: {unique_count:,}")
    stats.append(f"  Duplicates: {len(df) - unique_count:,}")
    stats.append(f"  Uniqueness ratio: {100 * unique_count / len(df):.2f}%")
    stats.append("")

    # Top sequences
    top_seqs = df['generated_sequence'].value_counts().head(5)
    stats.append("Top 5 Most Frequent Sequences:")
    for seq, count in top_seqs.items():
        pct = 100 * count / len(df)
        stats.append(f"  {seq[:30]:<30} {count:>5} ({pct:>5.2f}%)")
    stats.append("")

    # PSSI statistics
    stats.append("PSSI Statistics:")
    stats.append(f"  Mean PSSI range: [{df['mean_pssi'].min():.4f}, {df['mean_pssi'].max():.4f}]")
    stats.append(f"  Mean PSSI median: {df['mean_pssi'].median():.4f}")
    stats.append(f"  Std PSSI range: [{df['std_pssi'].min():.4f}, {df['std_pssi'].max():.4f}]")
    stats.append("")

    # Novelty (overlap with original sequences)
    original_set = set(original_sequences)
    generated_set = set(df['generated_sequence'].unique())
    overlap = len(generated_set & original_set)
    novelty = 1 - overlap / len(generated_set)

    stats.append("Novelty:")
    stats.append(f"  Overlap with original: {overlap:,} / {len(generated_set):,}")
    stats.append(f"  Novelty ratio: {100 * novelty:.2f}%")
    stats.append("")

    stats.append("=" * 80)

    return '\n'.join(stats)


def main():
    parser = argparse.ArgumentParser(description='Generate sequences from missing_0.csv phase diagrams')
    parser.add_argument('--checkpoint', type=str,
                        default='../outputs_set/output_set_flow1_missing11/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--input_csv', type=str,
                        default='/data/yanjie_huang/LLPS/phase_diagram/by_missing/missing_0.csv',
                        help='Path to missing_0.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory (e.g., outputs/dataset_1x)')
    parser.add_argument('--n_samples', type=int, required=True,
                        help='Number of sequences to generate per phase diagram (1 or 5)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for generation')
    parser.add_argument('--max_len', type=int, default=25,
                        help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_model(args.checkpoint, device)
    tokenizer = AminoAcidTokenizer()

    # Load data
    phase_data, original_sequences, group_cols = load_missing0_data(args.input_csv)

    # Generate sequences
    results = generate_sequences_batch(
        model, tokenizer, phase_data, original_sequences,
        args.n_samples, args.batch_size, args.max_len, args.temperature, device
    )

    # Save results
    df_out = pd.DataFrame(results)
    output_csv = output_dir / 'sequences.csv'
    df_out.to_csv(output_csv, index=False)
    print(f"\n✓ Saved {len(df_out):,} sequences to {output_csv}")

    # Compute and save statistics
    stats_text = compute_statistics(df_out, original_sequences)
    stats_file = output_dir / 'generation_stats.txt'
    with open(stats_file, 'w') as f:
        f.write(stats_text)
    print(f"✓ Saved statistics to {stats_file}")

    # Print statistics
    print("\n" + stats_text)

    print("\n" + "=" * 80)
    print("Generation completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
