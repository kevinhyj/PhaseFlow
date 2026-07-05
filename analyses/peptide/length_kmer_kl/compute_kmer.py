#!/usr/bin/env python3
"""
计算蛋白质序列的 1-mer, 2-mer, 3-mer 频率特征

Usage:
    python compute_kmer.py --input <csv> --seq_col <col> --output <csv>

Example:
    python compute_kmer.py \
        --input outputs/dataset_1x/sequences.csv \
        --seq_col generated_sequence \
        --output outputs/dataset_1x/kmer_features.csv
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

AA = list('ACDEFGHIKLMNPQRSTVWY')


def get_all_kmers(k):
    return [''.join(p) for p in product(AA, repeat=k)]


def compute_kmer_freq(seq, k, all_kmers):
    """计算序列的 k-mer 频率向量"""
    counts = {km: 0 for km in all_kmers}
    n = len(seq) - k + 1
    if n <= 0:
        return counts
    for i in range(n):
        km = seq[i:i+k]
        if km in counts:
            counts[km] += 1
    total = sum(counts.values())
    if total > 0:
        return {km: v / total for km, v in counts.items()}
    return counts


def compute_all_kmer_features(sequences, ks=(1, 2, 3)):
    """为所有序列计算 k-mer 特征，返回 DataFrame"""
    all_kmers_by_k = {k: get_all_kmers(k) for k in ks}
    all_cols = []
    for k in ks:
        all_cols.extend([f'{k}mer_{km}' for km in all_kmers_by_k[k]])

    rows = []
    for seq in tqdm(sequences, desc='Computing k-mer features'):
        row = {}
        for k in ks:
            freq = compute_kmer_freq(seq, k, all_kmers_by_k[k])
            for km, v in freq.items():
                row[f'{k}mer_{km}'] = v
        rows.append(row)

    return pd.DataFrame(rows, columns=all_cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--seq_col', required=True, help='Column name for sequences')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--ks', nargs='+', type=int, default=[1, 2, 3],
                        help='k values (default: 1 2 3)')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    sequences = df[args.seq_col].astype(str).tolist()
    print(f"  {len(sequences)} sequences")

    feat_df = compute_all_kmer_features(sequences, ks=args.ks)

    # 拼接原始序列列
    result = pd.concat([df[[args.seq_col]].reset_index(drop=True), feat_df], axis=1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    n_cols = feat_df.shape[1]
    print(f"✓ Saved {len(result)} rows × {n_cols} features → {args.output}")
    print(f"  1-mer: {len(get_all_kmers(1))} | 2-mer: {len(get_all_kmers(2))} | 3-mer: {len(get_all_kmers(3))}")


if __name__ == '__main__':
    main()
