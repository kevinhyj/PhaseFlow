#!/usr/bin/env python3
"""
创建 Bottom 5% 强相分离序列的 ground_truth.csv (PSSI最负=相分离最强)

从 missing_0.csv 提取 mean_pssi 最低的 5% 序列，
输出格式与 top5/ground_truth.csv 一致。

Usage:
    cd /data/yanjie_huang/LLPS/predictor/PhaseFlow
    python "analysis/de novo/bottom5_strong/create_ground_truth.py"
"""

import numpy as np
import pandas as pd
import os

GROUP_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = '/data/yanjie_huang/LLPS/phase_diagram/by_missing/missing_0.csv'
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'ground_truth.csv')


def main():
    df = pd.read_csv(INPUT_CSV)
    phase_data = df[GROUP_COLS].values.astype(np.float32)
    mean_pssi = phase_data.mean(axis=1)

    threshold = np.percentile(mean_pssi, 5)
    weak_idx = np.where(mean_pssi <= threshold)[0]

    print(f"Bottom 5% threshold: mean_pssi <= {threshold:.4f}")
    print(f"Selected {len(weak_idx)} sequences")

    rows = []
    for idx in weak_idx:
        row = {
            'source_data_idx': int(idx),
            'gt_sequence': df['AminoAcidSequence'].iloc[idx],
        }
        for k in range(16):
            row[f'gt_pssi_{k}'] = phase_data[idx][k]
        rows.append(row)

    gt_df = pd.DataFrame(rows)
    gt_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(gt_df)} rows → {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
