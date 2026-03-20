#!/usr/bin/env python3
"""
评估 PhaseFlow 模型的 phase→seq 性能 (AA KL Divergence + Round-Trip Spearman)。

指标:
  1. AA KL Divergence: 生成序列的 AA 频率 vs 训练集 AA 频率, KL(gen || real)
  2. Round-Trip Flattened Spearman: phase→seq→phase, 所有有效值展平后 spearmanr
  3. Round-Trip Mean Spearman: phase→seq→phase, 每样本均值后 spearmanr

与 eval_seq2phase.py 结构一致: 自动发现模型, 固定 seed, 支持 --dir 切换目录。

Usage:
    conda activate phaseflow
    cd /data/yanjie_huang/LLPS/predictor/PhaseFlow
    python infer/eval_phase2seq.py --gpu 0
    python infer/eval_phase2seq.py --gpu 0 --dir outputs_ddpm
    python infer/eval_phase2seq.py --gpu 0 --filter flow32
"""
import sys
import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, entropy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phaseflow import PhaseFlow
from phaseflow.tokenizer import AminoAcidTokenizer
from phaseflow.utils import set_seed

TEST_PATH = '/data/yanjie_huang/LLPS/phase_diagram/test_set.csv'
TRAIN_PATH = '/data/yanjie_huang/LLPS/phase_diagram/phase_diagram_original_scale.csv'
OUTPUTS_SET_DIR = Path(__file__).resolve().parent.parent / 'outputs_set'
PHASE_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
MAX_SEQ_LEN = 32

# --- PLACEHOLDER_REMAINING ---


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', type=str, default=None,
                        help='JSON output path (default: infer/lm_evaluation_results/phase2seq.json)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Random seeds to average over (default: [42])')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only evaluate models whose name contains this substring')
    parser.add_argument('--dir', type=str, default=None,
                        help='Model directory to scan (default: outputs_set). '
                             'E.g. --dir outputs_ddpm')
    return parser.parse_args()


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
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


# ── AA KL ────────────────────────────────────────────────────────────────────

def aa_frequency(sequences):
    """Compute 20-dim AA frequency distribution."""
    total = Counter()
    for seq in sequences:
        total.update(seq)
    freq = np.zeros(20)
    n = sum(total[aa] for aa in AMINO_ACIDS)
    if n == 0:
        return freq
    for i, aa in enumerate(AMINO_ACIDS):
        freq[i] = total[aa] / n
    return freq


def kl_divergence(p, q, epsilon=1e-10):
    """KL(P || Q) with smoothing."""
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(entropy(p, q))


# ── Phase→Seq generation ────────────────────────────────────────────────────

def generate_sequences(model, tokenizer, phase_values: torch.Tensor,
                       device: torch.device, batch_size: int = 64,
                       max_len: int = 25, temperature: float = 1.0):
    """Generate one sequence per phase diagram. Returns list[str]."""
    N = phase_values.shape[0]
    all_seqs = []
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        phase_batch = phase_values[i:end].to(device)
        with torch.no_grad():
            _, seqs = model.generate_sequence(
                phase_batch, tokenizer,
                max_len=max_len, temperature=temperature,
            )
        all_seqs.extend(seqs)
    return all_seqs


# ── Seq→Phase prediction (for round-trip) ────────────────────────────────────

def predict_phase(model, tokenizer, sequences, device, batch_size=64):
    """seq→phase, returns (N, 16) numpy. Auto-dispatches FM/DDPM."""
    input_ids = tokenizer.batch_encode(sequences, max_len=MAX_SEQ_LEN, return_tensors=True).to(device)
    attn_mask = (input_ids != tokenizer.PAD_ID).long()

    is_ddpm = hasattr(model, 'diffusion_type') and model.diffusion_type == 'ddpm'
    N = len(sequences)
    all_preds = []
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        bt = input_ids[i:end]
        bm = attn_mask[i:end]
        with torch.no_grad():
            if is_ddpm:
                pred = model.generate_phase(bt, bm, seq_len=None,
                                            num_steps=50, use_ddim=True)
            else:
                pred = model.generate_phase(bt, bm, seq_len=None, method='euler')
        all_preds.append(pred.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(gen_seqs, model, tokenizer, device,
                    target_phase, target_mask, real_aa_freq, batch_size):
    """
    Compute AA KL + Round-Trip Spearman from generated sequences.

    Args:
        gen_seqs: list[str], one per test sample
        target_phase: (N, 16) numpy, NaN filled with 0
        target_mask: (N, 16) numpy, 1=valid
        real_aa_freq: (20,) numpy, training set AA frequency
    """
    # Filter empty sequences, track valid indices
    valid_idx = [i for i, s in enumerate(gen_seqs) if len(s) > 0]
    valid_seqs = [gen_seqs[i] for i in valid_idx]

    results = {}

    if len(valid_seqs) == 0:
        results['kl_div'] = float('nan')
        results['rt_flat'] = float('nan')
        results['rt_mean'] = float('nan')
        return results

    # 1. AA KL Divergence
    gen_aa_freq = aa_frequency(valid_seqs)
    results['kl_div'] = kl_divergence(gen_aa_freq, real_aa_freq)

    # 2. Round-Trip: seq→phase on generated sequences
    pred_phase = predict_phase(model, tokenizer, valid_seqs, device, batch_size)
    rt_target = target_phase[valid_idx]
    rt_mask = target_mask[valid_idx]

    # RT Flattened Spearman
    valid_flat = rt_mask.astype(bool).flatten()
    p_flat = pred_phase.flatten()[valid_flat]
    t_flat = rt_target.flatten()[valid_flat]
    rt_flat = float('nan')
    if len(p_flat) > 1:
        rt_flat, _ = spearmanr(p_flat, t_flat)
    results['rt_flat'] = float(rt_flat)

    # RT Mean Spearman
    pred_means = []
    target_means = []
    for i in range(pred_phase.shape[0]):
        v = rt_mask[i].astype(bool)
        if v.sum() >= 1:
            pred_means.append(pred_phase[i][v].mean())
            target_means.append(rt_target[i][v].mean())
    rt_mean = float('nan')
    if len(pred_means) > 1:
        rt_mean, _ = spearmanr(pred_means, target_means)
    results['rt_mean'] = float(rt_mean)

    return results


# ── Model discovery & table printing ─────────────────────────────────────────

def discover_models(outputs_dir: Path):
    models = {}
    for d in sorted(outputs_dir.iterdir()):
        if not d.is_dir():
            continue
        ckpt = d / 'best_model.pt'
        if ckpt.exists():
            name = d.name.replace('output_set_', 'set_').replace('missing', 'm')
            models[name] = str(ckpt)
    return models


def parse_model_name(name: str):
    m = re.match(r'set_(flow|lm)(\d+)_m(\d+)', name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None


# --- PLACEHOLDER_TABLE ---


def print_table(results: dict):
    """按 weight_type 分组, 打印二维表格."""
    missing_vals = sorted(set(
        v for _, _, v in (parse_model_name(n) for n in results) if v is not None
    ))

    for wtype in ['flow', 'lm']:
        weight_vals = sorted(set(
            wv for wt, wv, _ in (parse_model_name(n) for n in results) if wt == wtype
        ))
        if not weight_vals:
            continue

        col_w = 14
        header = f"{'weight':<16}" + "".join(f"{'m' + str(m):>{col_w}}" for m in missing_vals)
        sep = "-" * len(header)
        print(f"\n{'=' * len(header)}")
        print(f"  {wtype}_weight sweep  (AA KL / RT_Flat / RT_Mean)")
        print(f"{'=' * len(header)}")
        print(header)
        print(sep)

        for wv in weight_vals:
            row_kl   = f"  {wtype}={wv:<8} KL  "
            row_flat = f"  {'':>10} RT_F"
            row_mean = f"  {'':>10} RT_M"
            for mv in missing_vals:
                key = f"set_{wtype}{wv}_m{mv}"
                if key in results:
                    r = results[key]
                    row_kl   += f"{r['kl_div']:>{col_w}.4f}"
                    row_flat += f"{r['rt_flat']:>{col_w}.4f}"
                    row_mean += f"{r['rt_mean']:>{col_w}.4f}"
                else:
                    row_kl   += f"{'---':>{col_w}}"
                    row_flat += f"{'---':>{col_w}}"
                    row_mean += f"{'---':>{col_w}}"
            print(row_kl)
            print(row_flat)
            print(row_mean)
            print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────

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
    test_phases = torch.tensor(target_filled, dtype=torch.float32)
    print(f"  Samples: {len(test_df)}, NaN ratio: {np.isnan(target).mean():.4f}")

    # 加载训练集 AA 频率 (参考分布)
    print(f"Loading training set for AA reference: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    real_aa_freq = aa_frequency(train_df['AminoAcidSequence'].values)
    print(f"  Training sequences: {len(train_df)}")

    tokenizer = AminoAcidTokenizer()

    # 发现所有模型
    if args.dir:
        scan_dir = Path(__file__).resolve().parent.parent / args.dir
    else:
        scan_dir = OUTPUTS_SET_DIR
    models = discover_models(scan_dir)
    if args.filter:
        models = {k: v for k, v in models.items() if args.filter in k}
    print(f"\nFound {len(models)} models in {scan_dir}\n")

    # 逐个评估
    results = {}
    for idx, (name, ckpt_path) in enumerate(models.items()):
        print(f"[{idx+1}/{len(models)}] {name} ... ", end='', flush=True)
        model = load_model(ckpt_path, device)

        kl_list = []
        rt_flat_list = []
        rt_mean_list = []

        for seed in args.seeds:
            set_seed(seed)
            gen_seqs = generate_sequences(
                model, tokenizer, test_phases, device,
                batch_size=args.batch_size,
            )
            m = compute_metrics(
                gen_seqs, model, tokenizer, device,
                target_filled, mask, real_aa_freq, args.batch_size,
            )
            kl_list.append(m['kl_div'])
            rt_flat_list.append(m['rt_flat'])
            rt_mean_list.append(m['rt_mean'])

        kl_avg = np.nanmean(kl_list)
        rt_flat_avg = np.nanmean(rt_flat_list)
        rt_mean_avg = np.nanmean(rt_mean_list)

        print(f"KL={kl_avg:.4f}  RT_Flat={rt_flat_avg:.4f}  RT_Mean={rt_mean_avg:.4f}")
        results[name] = {
            'kl_div': float(kl_avg),
            'rt_flat': float(rt_flat_avg),
            'rt_mean': float(rt_mean_avg),
        }
        del model
        torch.cuda.empty_cache()

    # 保存 JSON
    out_dir = Path(__file__).resolve().parent / 'lm_evaluation_results'
    out_dir.mkdir(exist_ok=True)
    out_path = args.output or str(out_dir / 'phase2seq.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # 打印表格
    print_table(results)


if __name__ == '__main__':
    main()
