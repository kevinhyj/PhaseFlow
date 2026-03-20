#!/usr/bin/env python3
"""
蛋白质序列 K-mer 特征提取工具

计算 1-mer, 2-mer, 3-mer 频率并拼接成一个向量

Usage:
    python compute_kmer_features.py \
        --input outputs/dataset_1x/sequences.csv \
        --seq_col generated_sequence \
        --output outputs/dataset_1x/kmer_features.npz
"""

import argparse
import itertools
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm


def generate_all_kmers(k: int) -> List[str]:
    """生成所有可能的k-mer（20种氨基酸）"""
    AA = list('ACDEFGHIKLMNPQRSTVWY')
    return [''.join(p) for p in itertools.product(AA, repeat=k)]


def build_kmer_index(min_k: int = 1, max_k: int = 3) -> Tuple[Dict[str, int], List[str]]:
    """
    构建全局k-mer索引映射

    Returns:
        (kmer_to_index, kmer_names)
        - 1-mer: 20 个
        - 2-mer: 400 个
        - 3-mer: 8000 个
        - 总计: 8420 个特征
    """
    all_kmers = []
    for k in range(min_k, max_k + 1):
        all_kmers.extend(generate_all_kmers(k))

    kmer_to_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}
    return kmer_to_index, all_kmers


def count_kmers_in_sequence(seq: str, min_k: int = 1, max_k: int = 3) -> Dict[str, int]:
    """
    统计序列中所有k-mer的频率

    Args:
        seq: 蛋白质序列
        min_k: 最小k值
        max_k: 最大k值

    Returns:
        k-mer计数字典
    """
    seq = seq.upper()
    all_counts = {}

    for i in range(len(seq)):
        for k in range(min_k, min(max_k + 1, len(seq) - i + 1)):
            kmer = seq[i:i + k]

            # 只统计标准氨基酸
            if all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in kmer):
                all_counts[kmer] = all_counts.get(kmer, 0) + 1

    return all_counts


def build_sparse_matrix(
    sequences: List[str],
    kmer_to_index: Dict[str, int],
    n_features: int,
    min_k: int = 1,
    max_k: int = 3
) -> csr_matrix:
    """
    从序列列表构建稀疏矩阵

    Args:
        sequences: 序列列表
        kmer_to_index: k-mer到索引的映射
        n_features: 特征总数（8420）
        min_k, max_k: k值范围

    Returns:
        CSR格式的稀疏矩阵 (n_sequences, 8420)
    """
    row_indices = []
    col_indices = []
    data = []

    for seq_idx, seq in enumerate(tqdm(sequences, desc="Computing k-mers")):
        kmer_counts = count_kmers_in_sequence(seq, min_k=min_k, max_k=max_k)

        # 归一化为频率
        total = sum(kmer_counts.values())
        if total > 0:
            for kmer, count in kmer_counts.items():
                if kmer in kmer_to_index:
                    row_indices.append(seq_idx)
                    col_indices.append(kmer_to_index[kmer])
                    data.append(count / total)  # 频率

    # 构建COO矩阵
    coo = coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(sequences), n_features),
        dtype=np.float32
    )

    # 转换为CSR格式
    return coo.tocsr()


def save_sparse_matrix(
    sparse_matrix: csr_matrix,
    sequence_ids: List[str],
    kmer_names: List[str],
    output_file: Path
):
    """保存稀疏矩阵及元数据到.npz文件"""
    np.savez_compressed(
        output_file,
        data=sparse_matrix.data,
        indices=sparse_matrix.indices,
        indptr=sparse_matrix.indptr,
        shape=sparse_matrix.shape,
        sequence_ids=np.array(sequence_ids, dtype=object),
        kmer_names=np.array(kmer_names, dtype=object)
    )


def load_sparse_matrix(npz_file: Path) -> Tuple[csr_matrix, List[str], List[str]]:
    """从.npz文件加载稀疏矩阵及元数据"""
    data = np.load(npz_file, allow_pickle=True)

    sparse_matrix = csr_matrix(
        (data['data'], data['indices'], data['indptr']),
        shape=data['shape']
    )

    sequence_ids = data['sequence_ids'].tolist()
    kmer_names = data['kmer_names'].tolist()

    return sparse_matrix, sequence_ids, kmer_names


def main():
    parser = argparse.ArgumentParser(
        description='蛋白质序列 K-mer 特征提取工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s -i dataset_1x/sequences.csv -s generated_sequence -o dataset_1x/kmer_features.npz
  %(prog)s -i dataset_groundtruth/missing_0.csv -s AminoAcidSequence -o dataset_groundtruth/kmer_features.npz
        """
    )

    parser.add_argument('-i', '--input', required=True, type=Path,
                        help='输入CSV文件路径')
    parser.add_argument('-s', '--seq-col', required=True, type=str,
                        help='序列列名')
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help='输出.npz文件路径')
    parser.add_argument('--min-k', type=int, default=1,
                        help='最小k值 (默认: 1)')
    parser.add_argument('--max-k', type=int, default=3,
                        help='最大k值 (默认: 3)')

    args = parser.parse_args()

    # 参数验证
    if not args.input.exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return 1

    print("="*80)
    print("蛋白质序列 K-mer 特征提取工具")
    print("="*80)
    print(f"输入文件: {args.input}")
    print(f"序列列名: {args.seq_col}")
    print(f"输出文件: {args.output}")
    print(f"k值范围: {args.min_k}-{args.max_k}")
    print("="*80)

    # 步骤1: 构建k-mer索引
    print("\n[1/3] 构建k-mer索引...")
    start_time = time.time()
    kmer_to_index, kmer_names = build_kmer_index(min_k=args.min_k, max_k=args.max_k)
    n_features = len(kmer_names)
    print(f"  1-mer: {20}")
    print(f"  2-mer: {400}")
    print(f"  3-mer: {8000}")
    print(f"  总特征数: {n_features}")
    print(f"  耗时: {time.time() - start_time:.2f}秒")

    # 步骤2: 加载序列
    print("\n[2/3] 加载序列...")
    start_time = time.time()
    df = pd.read_csv(args.input)
    sequences = df[args.seq_col].astype(str).tolist()
    n_sequences = len(sequences)
    print(f"  序列数: {n_sequences:,}")
    print(f"  耗时: {time.time() - start_time:.2f}秒")

    # 步骤3: 计算k-mer特征并构建稀疏矩阵
    print("\n[3/3] 计算k-mer特征...")
    start_time = time.time()
    sparse_matrix = build_sparse_matrix(
        sequences,
        kmer_to_index,
        n_features,
        min_k=args.min_k,
        max_k=args.max_k
    )

    # 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_sparse_matrix(sparse_matrix, sequences, kmer_names, args.output)

    # 统计信息
    n_nonzero = sparse_matrix.nnz
    sparsity = 1 - (n_nonzero / (n_sequences * n_features))
    file_size = args.output.stat().st_size / 1024 / 1024

    print(f"  矩阵形状: {sparse_matrix.shape}")
    print(f"  非零元素: {n_nonzero:,}")
    print(f"  稀疏度: {sparsity*100:.2f}%")
    print(f"  文件大小: {file_size:.2f} MB")
    print(f"  耗时: {time.time() - start_time:.2f}秒")

    # 总结
    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"输出文件: {args.output}")
    print(f"\n加载方法:")
    print(f"  from scipy.sparse import csr_matrix")
    print(f"  import numpy as np")
    print(f"  data = np.load('{args.output.name}', allow_pickle=True)")
    print(f"  matrix = csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])")
    print(f"  sequence_ids = data['sequence_ids']")
    print(f"  kmer_names = data['kmer_names']")
    print("="*80)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
