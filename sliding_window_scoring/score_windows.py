#!/usr/bin/env python3
"""
Slide 20aa window on protein sequences and score with PhaseFlow model.
Each window is tokenized, then predicted by PhaseFlow (16-dim PSSI via ODE), then averaged.

Usage:
    python score_windows.py                         # Run with default settings
    python score_windows.py --gpu 0                 # Use GPU 0
    python score_windows.py --window-batch 128      # Process 128 windows per batch
"""
import argparse
import pandas as pd
import numpy as np
import torch
import json
import pickle
import logging
import sys
import os
from pathlib import Path
from tqdm import tqdm
from torchdiffeq import odeint

# Add PhaseFlow to path
sys.path.insert(0, '/data/yanjie_huang/LLPS/predictor/PhaseFlow_WJX_Test')
from phaseflow import PhaseFlow
from phaseflow.tokenizer import AminoAcidTokenizer

# 标准氨基酸集合
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ============================================================================
# 配置
# ============================================================================

DATA_DIR = Path('/data/yanjie_huang/LLPS/data')
UNIQUE_PROTEINS_FILE = DATA_DIR / 'unique_proteins.csv'
PHASEFLOW_MODEL_PATH = Path('/data/yanjie_huang/LLPS/predictor/PhaseFlow_WJX_Test/outputs/misssing0_output_bs2048_lr0.0008_flow32_20260123/best_model.pt')
OUTPUT_DIR = Path('/data/yanjie_huang/LLPS/predictor/PhaseFlow_WJX_Test/sliding_window_scoring')
OUTPUT_JSONL = OUTPUT_DIR / 'protein_window_scores.jsonl'
INDEX_FILE = OUTPUT_DIR / 'protein_window_index.pkl'
WINDOW_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配置日志
LOG_FILE = OUTPUT_DIR / 'sliding_window_scoring.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# PhaseFlow 滑动窗口打分器
# ============================================================================

class SlidingWindowScorer:
    """PhaseFlow 滑动窗口打分器"""

    def __init__(self, device='cuda:0', window_batch_size=64):
        self.device = torch.device(device)
        self.window_batch_size = window_batch_size
        self.model = None
        self.tokenizer = None

    def load_models(self):
        """加载 PhaseFlow 模型"""
        logger.info("Loading PhaseFlow model...")

        # 加载 checkpoint
        checkpoint = torch.load(PHASEFLOW_MODEL_PATH, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})

        # 获取模型配置
        model_config = config.get('model', {})

        logger.info(f"  Model config: dim={model_config.get('dim', 256)}, depth={model_config.get('depth', 6)}")

        # 创建模型
        self.model = PhaseFlow(
            dim=model_config.get('dim', 256),
            depth=model_config.get('depth', 6),
            heads=model_config.get('heads', 8),
            dim_head=model_config.get('dim_head', 32),
            vocab_size=model_config.get('vocab_size', 64),
            phase_dim=model_config.get('phase_dim', 16),
            max_seq_len=model_config.get('max_seq_len', 32),
            dropout=0.0,  # 推理时不使用 dropout
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device).eval()

        # 创建 tokenizer
        self.tokenizer = AminoAcidTokenizer()

        epoch = checkpoint.get('epoch', 'unknown')
        logger.info(f"  PhaseFlow model loaded (epoch {epoch}) on {self.device}")

    def predict_pssi(self, sequences):
        """
        用 PhaseFlow 模型预测 16 维 PSSI (通过 ODE 积分)

        Args:
            sequences: list of str, 蛋白质序列列表
        Returns:
            predictions: (N, 16) numpy array
        """
        batch_size = len(sequences)

        # Tokenize sequences
        tokens = self.tokenizer.batch_encode(sequences, max_len=None, return_tensors=True)
        tokens = tokens.to(self.device)
        attention_mask = torch.ones(batch_size, tokens.shape[1], dtype=torch.long, device=self.device)

        # 初始化噪声
        x_init = torch.randn(batch_size, self.model.phase_dim, device=self.device)
        phase_mask = torch.ones(batch_size, self.model.phase_dim, device=self.device)

        # ODE 函数
        def ode_func(t, x):
            t_batch = torch.full((batch_size,), t.item() if t.dim() == 0 else t, device=self.device)
            return self.model.forward_flow(tokens, attention_mask, x, phase_mask, t_batch, None, None)

        # ODE 积分 (从 t=0 到 t=1)
        with torch.no_grad():
            prediction = odeint(
                ode_func,
                x_init,
                torch.tensor([0.0, 1.0], device=self.device),
                method='euler'
            )[-1]  # 取最终状态

        return prediction.cpu().numpy()

    def score_protein(self, sequence):
        """
        对单个蛋白质序列进行滑动窗口打分

        Args:
            sequence: str, 蛋白质序列
        Returns:
            scores: list of float, 每个窗口的平均 PSSI 分数
        """
        seq_len = len(sequence)
        n_windows = max(0, seq_len - WINDOW_SIZE + 1)

        if n_windows <= 0:
            return []

        # 收集所有窗口序列
        window_seqs = [sequence[i:i+WINDOW_SIZE] for i in range(n_windows)]

        # 分批处理
        all_scores = []
        for batch_start in range(0, n_windows, self.window_batch_size):
            batch_end = min(batch_start + self.window_batch_size, n_windows)
            batch_seqs = window_seqs[batch_start:batch_end]

            # 预测 16 维 PSSI
            predictions = self.predict_pssi(batch_seqs)  # (batch, 16)

            # 取 16 维的平均作为窗口分数
            batch_scores = predictions.mean(axis=1).tolist()
            all_scores.extend(batch_scores)

        return all_scores


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Slide window scoring with PhaseFlow')
    parser.add_argument('--gpu', type=int, default=7, help='GPU device ID (default: 7)')
    parser.add_argument('--window-batch', type=int, default=64,
                        help='Number of windows to process per batch (default: 64)')
    parser.add_argument('--protein-batch', type=int, default=1,
                        help='Number of proteins to process before writing (default: 1)')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    logger.info("=" * 60)
    logger.info("Sliding Window PhaseFlow Scoring")
    logger.info(f"Window size: {WINDOW_SIZE} aa")
    logger.info(f"Device: {device}")
    logger.info(f"Window batch size: {args.window_batch}")
    logger.info("=" * 60)

    # 初始化打分器
    scorer = SlidingWindowScorer(device=device, window_batch_size=args.window_batch)
    scorer.load_models()

    # 读取蛋白质数据
    logger.info(f"Loading {UNIQUE_PROTEINS_FILE}...")
    df = pd.read_csv(UNIQUE_PROTEINS_FILE)
    logger.info(f"Total proteins: {len(df)}")

    # 准备序列数据 (跳过包含非标准氨基酸的序列)
    seq_data = []
    skipped_count = 0
    for _, row in df.iterrows():
        seq = str(row['Sequence']).upper()
        if len(seq) >= WINDOW_SIZE:
            # 检查是否包含非标准氨基酸
            if all(aa in STANDARD_AA for aa in seq):
                seq_data.append({
                    'entry': row['Entry'],
                    'sequence': seq
                })
            else:
                skipped_count += 1

    logger.info(f"Proteins >= {WINDOW_SIZE} aa: {len(seq_data)}")
    logger.info(f"Skipped (non-standard AA): {skipped_count}")

    # 估计总窗口数
    est_windows = sum(len(s['sequence']) - WINDOW_SIZE + 1 for s in seq_data)
    logger.info(f"Estimated total windows: {est_windows:,}")

    # 处理并保存
    index = []
    with open(OUTPUT_JSONL, 'w') as f:
        for item in tqdm(seq_data, desc="Processing proteins"):
            entry = item['entry']
            sequence = item['sequence']

            # 打分
            scores = scorer.score_protein(sequence)

            if len(scores) > 0:
                # 写入 JSONL
                json_line = json.dumps({
                    'entry': entry,
                    'sequence': sequence,
                    'length': len(sequence),
                    'scores': scores
                })
                f.write(json_line + '\n')

                # 记录索引
                index.append({
                    'entry': entry,
                    'length': len(sequence),
                    'n_windows': len(scores)
                })

    # 保存索引
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(index, f)

    logger.info("=" * 60)
    logger.info(f"Done! Output: {OUTPUT_JSONL}")
    logger.info(f"Index: {INDEX_FILE}")
    logger.info(f"Proteins processed: {len(index)}")
    logger.info(f"Total windows: {sum(i['n_windows'] for i in index):,}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
