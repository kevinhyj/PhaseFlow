#!/usr/bin/env python3
"""
生成纯随机蛋白质序列数据集

长度在 5-20 AA 之间均匀随机，氨基酸也完全随机
"""

import numpy as np
import pandas as pd
from pathlib import Path

AA = list('ACDEFGHIKLMNPQRSTVWY')
N = 10165  # 与 dataset_1x 数量一致

np.random.seed(42)

random_sequences = []
for _ in range(N):
    length = np.random.randint(5, 21)  # 5 到 20（含）
    seq = ''.join(np.random.choice(AA, size=length))
    random_sequences.append(seq)

output_dir = Path('outputs/dataset_random')
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame({
    'AminoAcidSequence': random_sequences,
    'length': [len(s) for s in random_sequences],
})
df.to_csv(output_dir / 'random_sequences.csv', index=False)

print(f"✓ 生成 {N} 条随机序列")
print(f"长度分布:\n{df['length'].value_counts().sort_index()}")
print(f"平均长度: {df['length'].mean():.2f}")

