#!/usr/bin/env python3
"""
评估 outputs_set/ 下所有 PhaseFlow Set Encoder 模型的 seq→phase 性能。

指标:
  1. Flattened Spearman: 所有样本的 pred/target 展平为一个长向量, 算一次 spearmanr
  2. Mean Spearman: 每个样本 16 维取均值 → N 个标量, spearmanr(pred_means, target_means)
  3. MSE: 有效位置的均方误差

Usage:
    conda activate phaseflow
    cd /data/yanjie_huang/LLPS/predictor/PhaseFlow
    python infer/eval_seq2phase.py --gpu 0
"""
import sys
import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phaseflow import PhaseFlow
from phaseflow.tokenizer import AminoAcidTokenizer
from phaseflow.utils import set_seed

TEST_PATH = '/data/yanjie_huang/LLPS/phase_diagram/test_set.csv'
OUTPUTS_SET_DIR = Path(__file__).resolve().parent.parent / 'outputs_set'
PHASE_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]
MAX_SEQ_LEN = 32  # 必须与模型的 max_seq_len 一致, 否则 phase_start_idx 对齐出错


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', type=str, default=None,
                        help='JSON output path (default: infer/lm_evaluation_results/set_seq2phase.json)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Random seeds to average over (default: [42])')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only evaluate models whose name contains this substring')
    parser.add_argument('--dir', type=str, default=None,
                        help='Model directory to scan (default: outputs_set). '
                             'E.g. --dir outputs_ddpm')
    return parser.parse_args()


def load_model(ckpt_path: str, device: torch.device):
    """从 checkpoint 加载模型, 返回 eval 模式的 PhaseFlow."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
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
        diffusion_type=config['model'].get('diffusion_type', 'flow_matching'),
        num_timesteps=config['model'].get('num_timesteps', 1000),
        beta_schedule=config['model'].get('beta_schedule', 'cosine'),
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device).eval()
    return model


def predict_phase(model, tokens: torch.Tensor, attn_mask: torch.Tensor,
                  device: torch.device, batch_size: int = 64) -> np.ndarray:
    """seq→phase 推理, 自动根据 diffusion_type 选择采样方式."""
    N = tokens.shape[0]
    all_preds = []

    # DDPM 用 DDIM 50步, flow matching 用 euler
    is_ddpm = hasattr(model, 'diffusion_type') and model.diffusion_type == 'ddpm'

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        bt = tokens[i:end].to(device)
        bm = attn_mask[i:end].to(device)
        with torch.no_grad():
            if is_ddpm:
                pred = model.generate_phase(bt, bm, seq_len=None,
                                            num_steps=50, use_ddim=True)
            else:
                pred = model.generate_phase(bt, bm, seq_len=None, method='euler')
        all_preds.append(pred.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def compute_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray):
    """
    计算:
      sp_flat: 展平 Spearman — 所有有效值展平后算 spearmanr
      sp_mean: 均值 Spearman — 每样本有效值取均值, 样本间 spearmanr
    """
    valid = mask.astype(bool)

    # 1. Flattened Spearman
    p_flat = pred[valid]
    t_flat = target[valid]
    sp_flat = float('nan')
    if len(p_flat) > 1:
        sp_flat, _ = spearmanr(p_flat, t_flat)

    # 2. Mean Spearman: 每个样本有效位置取均值
    pred_means = []
    target_means = []
    for i in range(pred.shape[0]):
        v = valid[i]
        if v.sum() >= 1:
            pred_means.append(pred[i][v].mean())
            target_means.append(target[i][v].mean())
    sp_mean = float('nan')
    if len(pred_means) > 1:
        sp_mean, _ = spearmanr(pred_means, target_means)

    # 3. MSE: 有效位置的均方误差
    mse = float(((p_flat - t_flat) ** 2).mean()) if len(p_flat) > 0 else float('nan')

    return float(sp_flat), float(sp_mean), float(mse)


def discover_models(outputs_dir: Path):
    """扫描 outputs_set/ 下所有 best_model.pt, 返回 {name: path} 按名称排序."""
    models = {}
    for d in sorted(outputs_dir.iterdir()):
        if not d.is_dir():
            continue
        ckpt = d / 'best_model.pt'
        if ckpt.exists():
            # output_set_flow32_missing0 → set_flow32_m0
            name = d.name.replace('output_set_', 'set_').replace('missing', 'm')
            models[name] = str(ckpt)
    return models


def parse_model_name(name: str):
    """解析模型名 → (weight_type, weight_value, missing_value).
    set_flow32_m0 → ('flow', 32, 0)
    set_lm5_m11  → ('lm', 5, 11)
    """
    m = re.match(r'set_(flow|lm)(\d+)_m(\d+)', name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None


def print_table(results: dict):
    """按 weight_type 分组, 打印二维表格 (flow_w/lm_w × missing)."""
    missing_vals = sorted(set(
        v for _, _, v in (parse_model_name(n) for n in results) if v is not None
    ))

    for wtype in ['flow', 'lm']:
        weight_vals = sorted(set(
            wv for wt, wv, _ in (parse_model_name(n) for n in results) if wt == wtype
        ))
        if not weight_vals:
            continue

        # 表头
        col_w = 14
        header = f"{'weight':<16}" + "".join(f"{'m' + str(m):>{col_w}}" for m in missing_vals)
        sep = "-" * len(header)
        print(f"\n{'=' * len(header)}")
        print(f"  {wtype}_weight sweep  (Flat Spearman / Mean Spearman / MSE)")
        print(f"{'=' * len(header)}")
        print(header)
        print(sep)

        for wv in weight_vals:
            row_flat = f"  {wtype}={wv:<8} Flat"
            row_mean = f"  {'':>10} Mean"
            row_mse  = f"  {'':>10} MSE "
            for mv in missing_vals:
                key = f"set_{wtype}{wv}_m{mv}"
                if key in results:
                    r = results[key]
                    row_flat += f"{r['sp_flat']:>{col_w}.4f}"
                    row_mean += f"{r['sp_mean']:>{col_w}.4f}"
                    row_mse  += f"{r['mse']:>{col_w}.4f}"
                else:
                    row_flat += f"{'---':>{col_w}}"
                    row_mean += f"{'---':>{col_w}}"
                    row_mse  += f"{'---':>{col_w}}"
            print(row_flat)
            print(row_mean)
            print(row_mse)
            print(sep)


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")

    # 加载测试集
    print(f"Loading test set: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)
    target = test_df[PHASE_COLS].values.astype(np.float32)
    mask = (~np.isnan(target)).astype(np.float32)
    target_filled = np.nan_to_num(target, nan=0.0)
    print(f"  Samples: {len(test_df)}, NaN ratio: {np.isnan(target).mean():.4f}")

    # Tokenize — 必须用 max_len=MAX_SEQ_LEN, 否则 phase_start_idx 对齐出错
    tokenizer = AminoAcidTokenizer()
    sequences = test_df['AminoAcidSequence'].tolist()
    tokens = tokenizer.batch_encode(sequences, max_len=MAX_SEQ_LEN, return_tensors=True)
    attn_mask = (tokens != tokenizer.PAD_ID).long()
    print(f"  Tokens shape: {tokens.shape}, attn_mask valid ratio: {attn_mask.float().mean():.4f}")

    # 发现所有模型
    if args.dir:
        scan_dir = Path(__file__).resolve().parent.parent / args.dir
    else:
        scan_dir = OUTPUTS_SET_DIR
    models = discover_models(scan_dir)
    if args.filter:
        models = {k: v for k, v in models.items() if args.filter in k}
    print(f"\nFound {len(models)} models in {scan_dir}\n")

    # 逐个评估，对每个模型跑多个 seed 取平均
    results = {}
    for idx, (name, ckpt_path) in enumerate(models.items()):
        print(f"[{idx+1}/{len(models)}] {name} ... ", end='', flush=True)
        model = load_model(ckpt_path, device)

        # 多个 seed 的结果
        sp_flat_list = []
        sp_mean_list = []
        mse_list = []

        for seed in args.seeds:
            set_seed(seed)  # 固定随机种子
            pred = predict_phase(model, tokens, attn_mask, device, args.batch_size)
            sp_flat, sp_mean, mse = compute_metrics(pred, target_filled, mask)
            sp_flat_list.append(sp_flat)
            sp_mean_list.append(sp_mean)
            mse_list.append(mse)

        # 计算平均值和标准差
        sp_flat_avg = np.mean(sp_flat_list)
        sp_flat_std = np.std(sp_flat_list)
        sp_mean_avg = np.mean(sp_mean_list)
        sp_mean_std = np.std(sp_mean_list)
        mse_avg = np.mean(mse_list)
        mse_std = np.std(mse_list)

        print(f"Flat={sp_flat_avg:.4f}±{sp_flat_std:.4f}  Mean={sp_mean_avg:.4f}±{sp_mean_std:.4f}  MSE={mse_avg:.4f}±{mse_std:.4f}")
        results[name] = {
            'sp_flat': float(sp_flat_avg),
            'sp_flat_std': float(sp_flat_std),
            'sp_mean': float(sp_mean_avg),
            'sp_mean_std': float(sp_mean_std),
            'mse': float(mse_avg),
            'mse_std': float(mse_std),
        }
        del model
        torch.cuda.empty_cache()

    # 保存 JSON
    out_dir = Path(__file__).resolve().parent / 'lm_evaluation_results'
    out_dir.mkdir(exist_ok=True)
    out_path = args.output or str(out_dir / 'set_seq2phase.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # 打印表格
    print_table(results)


if __name__ == '__main__':
    main()
