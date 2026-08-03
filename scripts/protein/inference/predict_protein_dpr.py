"""
PhaseFlow 滑动窗口预测脚本
对 IDR 序列进行相位图预测，支持多窗口大小

Usage:
    python scripts/protein/inference/predict_protein_dpr.py
    python scripts/protein/inference/predict_protein_dpr.py --input artifacts/data/protein/idr_sequences.xlsx --output runs/idp_phaseflow_profiles.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 添加 PhaseFlow 到路径
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protein.workflows.evaluation import (
    DEFAULT_PHASEFLOW_ROOT,
    DEFAULT_PHASEFLOW_CHECKPOINT,
    PhaseFlowFusionConfig,
    PhaseFlowWindowScorer,
    local_contrast_profile,
)


def read_sequences_from_excel(excel_path: str | Path) -> list[dict]:
    """从 Excel 文件中读取 IDR 序列"""
    df = pd.read_excel(excel_path)

    sequences = []
    for idx, row in df.iterrows():
        for col in df.columns:
            val = str(row[col])
            # 检查是否是氨基酸序列（只包含标准氨基酸字符，长度 > 20）
            if len(val) > 20 and set(val.upper()).issubset(set('ACDEFGHIKLMNPQRSTVWY')):
                # 生成蛋白质 ID
                protein_id = f"IDR_{idx:03d}"
                sequences.append({
                    'id': protein_id,
                    'sequence': val.upper(),
                    'length': len(val)
                })
                break  # 每行只取第一个序列

    return sequences


def main():
    parser = argparse.ArgumentParser(description='PhaseFlow 滑动窗口预测 IDR 序列相位图')
    parser.add_argument('--input', type=str, default='artifacts/data/protein/idr_sequences.xlsx',
                        help='输入 Excel 文件路径')
    parser.add_argument('--output', type=str, default='runs/idp_phaseflow_profiles.jsonl',
                        help='输出 JSONL 文件路径')
    parser.add_argument('--csv', type=str, default=None,
                        help='输出 CSV 文件路径（包含 id, sequence, pssi_16d, pssi_mean）')
    parser.add_argument('--phaseflow-root', type=str, default=None,
                        help=f'PhaseFlow 根目录 (默认: {DEFAULT_PHASEFLOW_ROOT})')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help=f'PhaseFlow checkpoint 路径 (默认: {DEFAULT_PHASEFLOW_CHECKPOINT})')
    parser.add_argument('--window-sizes', type=str, default='20',
                        help='滑动窗口大小，逗号分隔 (默认: 20)')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (默认: auto)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批处理大小 (默认: 64)')
    args = parser.parse_args()

    # 解析窗口大小
    window_sizes = tuple(int(x.strip()) for x in args.window_sizes.split(',') if x.strip())

    # 解析路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # PhaseFlow 配置
    phaseflow_root = Path(args.phaseflow_root) if args.phaseflow_root else DEFAULT_PHASEFLOW_ROOT
    checkpoint = Path(args.checkpoint) if args.checkpoint else DEFAULT_PHASEFLOW_CHECKPOINT

    if not phaseflow_root.exists():
        print(f"错误: PhaseFlow 根目录不存在: {phaseflow_root}")
        return
    if not checkpoint.exists():
        print(f"错误: PhaseFlow checkpoint 不存在: {checkpoint}")
        return

    # 创建 PhaseFlow 配置
    config = PhaseFlowFusionConfig(
        phaseflow_root=phaseflow_root,
        checkpoint=checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        window_sizes=window_sizes,
    )

    # 加载 PhaseFlow 模型
    print("加载 PhaseFlow 模型...")
    scorer = PhaseFlowWindowScorer(config)
    print(f"  设备: {scorer.device}")
    print(f"  窗口大小: {window_sizes}")
    print(f"  批处理大小: {args.batch_size}")

    # 读取序列
    print(f"\n读取序列: {input_path}")
    sequences = read_sequences_from_excel(input_path)
    print(f"  找到 {len(sequences)} 条序列")

    # 预测
    print("\n开始预测...")
    results = []
    for i, item in enumerate(sequences):
        seq_id = item['id']
        sequence = item['sequence']
        length = item['length']

        # PhaseFlow 滑动窗口预测 - 残基级别 profile
        profile, used_windows = scorer.score_sequence(sequence)

        # 生成 16 维 PSSI 向量
        pssi_16d = scorer.score_sequence_global_pssi(sequence, window_size=20)

        # 计算统计量
        mean_score = float(np.mean(profile))
        max_score = float(np.max(profile))
        min_score = float(np.min(profile))

        # 找出高分区域 (top 10%)
        threshold = np.percentile(profile, 90)
        high_score_indices = np.where(profile >= threshold)[0]

        # 聚类高分区域
        regions = []
        if len(high_score_indices) > 0:
            start = high_score_indices[0]
            prev = high_score_indices[0]
            for idx in high_score_indices[1:]:
                if idx - prev > 5:  # gap > 5
                    regions.append({'start': int(start), 'end': int(prev)})
                    start = idx
                prev = idx
            regions.append({'start': int(start), 'end': int(prev)})

        result = {
            'id': seq_id,
            'length': length,
            'window_sizes': list(used_windows),
            'profile': [float(x) for x in profile],
            'pssi_16d': [float(x) for x in pssi_16d],
            'pssi_mean': float(np.mean(pssi_16d)),
            'stats': {
                'mean': mean_score,
                'max': max_score,
                'min': min_score,
                'std': float(np.std(profile))
            },
            'high_score_regions': regions,
            'high_score_threshold': float(threshold),
        }
        results.append(result)

        print(f"  [{i+1}/{len(sequences)}] {seq_id}: length={length}, "
              f"mean={mean_score:.3f}, max={max_score:.3f}, "
              f"high_regions={len(regions)}")

    # 保存结果
    print(f"\n保存结果: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 保存 CSV（简洁版）
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv as csv_lib
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv_lib.DictWriter(f, fieldnames=['id', 'sequence', 'pssi_16d', 'pssi_mean'])
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'id': result['id'],
                    'sequence': sequences[[s['id'] for s in sequences].index(result['id'])]['sequence'],
                    'pssi_16d': json.dumps(result['pssi_16d']),
                    'pssi_mean': result['pssi_mean'],
                })
        print(f"CSV 保存: {csv_path}")

    print(f"\n完成! 共处理 {len(results)} 条序列")
    print(f"JSONL 输出: {output_path}")


if __name__ == '__main__':
    main()
