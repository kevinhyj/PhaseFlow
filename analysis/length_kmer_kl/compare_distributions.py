#!/usr/bin/env python3
"""Compare missing_0 original sequences vs generated sequences."""

import pandas as pd
import numpy as np
from collections import Counter

# Load data
m0 = pd.read_csv('/data/yanjie_huang/LLPS/phase_diagram/by_missing/missing_0.csv')
gen1x = pd.read_csv('outputs/dataset_1x/sequences.csv')

print("=" * 80)
print("missing_0 原始序列 vs 生成序列分布对比")
print("=" * 80)

# 1. Sequence length
orig_lens = m0['AminoAcidSequence'].str.len()
gen_lens = gen1x['generated_sequence'].str.len()

print("\n【序列长度】")
print(f"原始: mean={orig_lens.mean():.2f}, median={orig_lens.median():.0f}, range=[{orig_lens.min()}, {orig_lens.max()}]")
print(f"生成: mean={gen_lens.mean():.2f}, median={gen_lens.median():.0f}, range=[{gen_lens.min()}, {gen_lens.max()}]")

# 2. AA composition
AA = list('ACDEFGHIKLMNPQRSTVWY')
orig_all = ''.join(m0['AminoAcidSequence'])
gen_all = ''.join(gen1x['generated_sequence'])

print("\n【氨基酸组成】")
print(f"{'AA':<4} {'原始%':>8} {'生成%':>8} {'差值':>8}")
print("-" * 32)

aa_diffs = []
for aa in AA:
    o_pct = 100 * orig_all.count(aa) / len(orig_all)
    g_pct = 100 * gen_all.count(aa) / len(gen_all)
    diff = g_pct - o_pct
    aa_diffs.append((aa, o_pct, g_pct, diff))
    print(f"{aa:<4} {o_pct:>8.2f} {g_pct:>8.2f} {diff:>+8.2f}")

# Top differences
aa_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
print("\n【最大差异 Top 5】")
for aa, o, g, d in aa_diffs[:5]:
    print(f"  {aa}: {d:+.2f}% (原始{o:.2f}% → 生成{g:.2f}%)")


# 3. Property groups
hydrophobic = list('VILMFYWAC')
charged_pos = list('KRH')
charged_neg = list('DE')
polar = list('STNQ')
special = list('GP')

def aa_group_pct(seq_str, group):
    return 100 * sum(seq_str.count(aa) for aa in group) / len(seq_str)

print("\n【氨基酸性质分组】")
groups = [
    ('疏水性 (VILMFYWAC)', hydrophobic),
    ('正电荷 (KRH)',       charged_pos),
    ('负电荷 (DE)',        charged_neg),
    ('极性 (STNQ)',        polar),
    ('特殊 (GP)',          special),
]
for name, grp in groups:
    o = aa_group_pct(orig_all, grp)
    g = aa_group_pct(gen_all, grp)
    print(f"  {name:<22}: 原始{o:.1f}% → 生成{g:.1f}% ({g-o:+.1f}%)")

# 4. Uniqueness
orig_set = set(m0['AminoAcidSequence'])
gen_set = set(gen1x['generated_sequence'])
overlap = len(gen_set & orig_set)

print("\n【多样性与新颖性】")
print(f"  原始唯一序列: {len(orig_set):,}")
print(f"  生成唯一序列: {len(gen_set):,} / {len(gen1x):,} ({100*len(gen_set)/len(gen1x):.2f}%)")
print(f"  与原始重叠: {overlap} 条")
print(f"  新颖性: {100*(1 - overlap/len(gen_set)):.2f}%")

# 5. Length distribution detail
print("\n【长度分布对比 (5-20)】")
print(f"{'长度':<6} {'原始':>8} {'原始%':>7} {'生成':>8} {'生成%':>7}")
print("-" * 40)
for l in range(5, 21):
    oc = (orig_lens == l).sum()
    gc = (gen_lens == l).sum()
    op = 100 * oc / len(orig_lens)
    gp = 100 * gc / len(gen_lens)
    print(f"{l:<6} {oc:>8,} {op:>7.2f}% {gc:>8,} {gp:>7.2f}%")

print("\n" + "=" * 80)
