#!/usr/bin/env python3
"""
Inference-Time Optimization Benchmark: Flow Matching vs DDPM

每轮生成1个候选序列（phase→seq）并预测相图（seq→phase），
累积到n轮时从所有候选中选最优Spearman。测量时间和质量随轮次的变化。

Usage:
    python benchmark_optimization.py --gpu 0
    python benchmark_optimization.py --n_samples 10 --max_rounds 5 --gpu 0  # 快速测试
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from phaseflow import PhaseFlow, AminoAcidTokenizer

# ── Font ─────────────────────────────────────────────────────────────────────
font_path = '/data/yanjie_huang/fonts/arial.ttf'
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

OUT_DIR = Path(__file__).parent / 'benchmark_results'
GROUP_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]

FM_COLOR   = '#728ab9'
DDPM_COLOR = '#50b9ae'


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt['config']
    m    = cfg['model']
    # detect diffusion_type from config (try multiple paths) or fallback to path name
    diff_type = (cfg.get('diffusion_type')
                 or cfg.get('training', {}).get('diffusion_type')
                 or ('ddpm' if 'ddpm' in checkpoint_path else 'flow_matching'))
    model = PhaseFlow(
        dim=m['dim'], depth=m['depth'], heads=m['heads'], dim_head=m['dim_head'],
        vocab_size=m['vocab_size'], phase_dim=m['phase_dim'],
        max_seq_len=m['max_seq_len'], dropout=0.0,
        use_set_encoder=m.get('use_set_encoder', False),
        diffusion_type=diff_type,
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device).eval()
    print(f"  Loaded {checkpoint_path.split('/')[-2]}  [{diff_type}]"
          f"  dim={m['dim']} depth={m['depth']}")
    return model, diff_type


# ── Single-round helper ───────────────────────────────────────────────────────

def one_round(model, tokenizer, phase_batch: torch.Tensor,
              max_len: int, temperature: float,
              score_batch_size: int, device: torch.device):
    """
    Generate 1 candidate per sample, score them, return pred_phases.

    Args:
        phase_batch: (N, 16) target phase diagrams on device

    Returns:
        pred_phases: (N, 16) numpy array  — predicted phases for new candidates
        seqs: list[str] of length N
    """
    N = phase_batch.shape[0]

    # ── phase → seq ──
    with torch.no_grad():
        _, seqs = model.generate_sequence(
            phase_batch, tokenizer,
            max_len=max_len, temperature=temperature,
            top_k=None, top_p=None,
        )

    # ── seq → phase ──
    input_ids = tokenizer.batch_encode(seqs, max_len=32).to(device)
    attn_mask = (input_ids != tokenizer.PAD_ID).long()
    seq_lens  = torch.tensor([len(s) for s in seqs], device=device)

    all_preds = []
    for s in range(0, N, score_batch_size):
        e = min(s + score_batch_size, N)
        with torch.no_grad():
            pred = model.generate_phase(
                input_ids[s:e], attn_mask[s:e], seq_lens[s:e], method='euler'
            )
        all_preds.append(pred.cpu())

    pred_phases = torch.cat(all_preds, dim=0).numpy()  # (N, 16)
    return pred_phases, seqs


# ── Benchmark loop ────────────────────────────────────────────────────────────

def run_benchmark(model, tokenizer,
                  phase_data: np.ndarray,
                  max_rounds: int,
                  max_len: int, temperature: float,
                  score_batch_size: int, device: torch.device):
    """
    Returns:
        round_times:    (max_rounds,) per-round elapsed time (seconds)
        round_spearman: (max_rounds,) mean Spearman across samples at each round
        round_spearman_std: (max_rounds,) std across samples
    """
    N = len(phase_data)
    phase_batch = torch.tensor(phase_data, dtype=torch.float32, device=device)

    # accumulated candidate preds: list of lists
    # cand_preds[i] → list of (16,) arrays
    cand_preds = [[] for _ in range(N)]

    round_times    = []
    round_spearman = []
    round_spearman_std = []

    for r in range(max_rounds):
        t0 = time.time()
        pred_phases, _ = one_round(
            model, tokenizer, phase_batch,
            max_len, temperature, score_batch_size, device,
        )
        dt = time.time() - t0
        round_times.append(dt)

        # accumulate and compute best Spearman per sample
        sample_best = []
        for i in range(N):
            cand_preds[i].append(pred_phases[i])
            target = phase_data[i]
            best = -1.0
            for p in cand_preds[i]:
                c, _ = spearmanr(p, target)
                if not np.isnan(c):
                    best = max(best, float(c))
            sample_best.append(best)

        round_spearman.append(float(np.mean(sample_best)))
        round_spearman_std.append(float(np.std(sample_best)))

        print(f"  Round {r+1:3d}/{max_rounds}  t={dt:.2f}s  "
              f"Spearman={round_spearman[-1]:.4f}±{round_spearman_std[-1]:.4f}")

    return (np.array(round_times),
            np.array(round_spearman),
            np.array(round_spearman_std))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_time_curve(fm_times, ddpm_times, output_path: Path):
    max_rounds = len(fm_times)
    rounds = np.arange(1, max_rounds + 1)
    fm_cum   = np.cumsum(fm_times)
    ddpm_cum = np.cumsum(ddpm_times)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rounds, fm_cum,   color=FM_COLOR,   linewidth=2.5,
            marker='o', markersize=4, label='Flow Matching (fw=32, lm=1)')
    ax.plot(rounds, ddpm_cum, color=DDPM_COLOR, linewidth=2.5,
            marker='s', markersize=4, label='DDPM (fw=32, lm=1)')

    # annotate average per-round time
    ax.text(max_rounds * 0.6, fm_cum[-1]   * 0.92,
            f'avg {np.mean(fm_times):.2f}s/round',
            fontsize=13, color=FM_COLOR, fontweight='bold')
    ax.text(max_rounds * 0.6, ddpm_cum[-1] * 0.92,
            f'avg {np.mean(ddpm_times):.2f}s/round',
            fontsize=13, color=DDPM_COLOR, fontweight='bold')

    ax.set_xlabel('Round', fontsize=16, fontweight='bold')
    ax.set_ylabel('Cumulative Time (s)', fontsize=16, fontweight='bold')
    ax.set_title('Inference-Time Optimization: Cumulative Time', fontsize=20, fontweight='bold', pad=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14, framealpha=0.9, edgecolor='#cccccc')

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved → {output_path}')
    plt.close()


def plot_spearman_bar(fm_spearman, fm_std, ddpm_spearman, ddpm_std,
                     checkpoint_rounds: list, output_path: Path):
    """Bar chart at selected round checkpoints."""
    idxs  = [r - 1 for r in checkpoint_rounds]  # 0-indexed
    fm_v  = fm_spearman[idxs]
    fm_e  = fm_std[idxs]
    dd_v  = ddpm_spearman[idxs]
    dd_e  = ddpm_std[idxs]

    x     = np.arange(len(checkpoint_rounds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    b1 = ax.bar(x - width / 2, fm_v, width, yerr=fm_e, capsize=4,
                color=FM_COLOR,   edgecolor='#5a5a5a', linewidth=0.8,
                label='Flow Matching (fw=32, lm=1)', error_kw=dict(elinewidth=1.2))
    b2 = ax.bar(x + width / 2, dd_v, width, yerr=dd_e, capsize=4,
                color=DDPM_COLOR, edgecolor='#5a5a5a', linewidth=0.8,
                label='DDPM (fw=32, lm=1)',           error_kw=dict(elinewidth=1.2))

    # value labels
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=11, color=FM_COLOR, fontweight='bold')
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=11, color=DDPM_COLOR, fontweight='bold')

    ax.set_xlabel('Rounds (# candidates)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Mean Spearman ρ', fontsize=16, fontweight='bold')
    ax.set_title('Inference-Time Optimization: Spearman vs Rounds',
                 fontsize=20, fontweight='bold', pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in checkpoint_rounds], fontsize=14)
    ax.tick_params(labelsize=14)
    ax.set_ylim(0, max(fm_v.max(), dd_v.max()) * 1.2)
    ax.legend(fontsize=14, framealpha=0.9, edgecolor='#cccccc')

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved → {output_path}')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fm_checkpoint',   type=str,
                        default='/data/yanjie_huang/LLPS/predictor/PhaseFlow/outputs_set/output_set_flow32_missing15/best_model.pt')
    parser.add_argument('--ddpm_checkpoint', type=str,
                        default='/data/yanjie_huang/LLPS/predictor/PhaseFlow/outputs_ddpm/output_set_ddpm_flow32_missing15/best_model.pt')
    parser.add_argument('--input_csv',  type=str,
                        default='/data/yanjie_huang/LLPS/phase_diagram/test_set.csv')
    parser.add_argument('--n_samples',  type=int, default=None)
    parser.add_argument('--max_rounds', type=int, default=50)
    parser.add_argument('--max_len',    type=int, default=25)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--score_batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu',  type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    df = pd.read_csv(args.input_csv)
    phase_data = df[GROUP_COLS].values.astype(np.float32)
    if args.n_samples is not None:
        phase_data = phase_data[:args.n_samples]
    print(f'Samples: {len(phase_data)}, Rounds: {args.max_rounds}')

    # Load models
    print('\n[Flow Matching]')
    fm_model, fm_type = load_model(args.fm_checkpoint, device)
    print('\n[DDPM]')
    ddpm_model, ddpm_type = load_model(args.ddpm_checkpoint, device)

    # Warm-up
    print('\nWarming up...')
    dummy = torch.zeros(1, 16, device=device)
    tokenizer = AminoAcidTokenizer()
    with torch.no_grad():
        fm_model.generate_sequence(dummy, tokenizer, max_len=5, temperature=1.0)
        ddpm_model.generate_sequence(dummy, tokenizer, max_len=5, temperature=1.0)

    # Run FM benchmark
    print(f'\n=== Flow Matching benchmark ({args.max_rounds} rounds, {len(phase_data)} samples) ===')
    fm_times, fm_spearman, fm_std = run_benchmark(
        fm_model, tokenizer, phase_data,
        args.max_rounds, args.max_len,
        args.temperature, args.score_batch_size, device,
    )

    # Run DDPM benchmark
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f'\n=== DDPM benchmark ({args.max_rounds} rounds, {len(phase_data)} samples) ===')
    ddpm_times, ddpm_spearman, ddpm_std = run_benchmark(
        ddpm_model, tokenizer, phase_data,
        args.max_rounds, args.max_len,
        args.temperature, args.score_batch_size, device,
    )

    # Save raw data
    OUT_DIR.mkdir(exist_ok=True)
    rounds_arr = np.arange(1, args.max_rounds + 1)
    df_out = pd.DataFrame({
        'round':              rounds_arr,
        'fm_time':            fm_times,
        'fm_time_cum':        np.cumsum(fm_times),
        'fm_spearman':        fm_spearman,
        'fm_spearman_std':    fm_std,
        'ddpm_time':          ddpm_times,
        'ddpm_time_cum':      np.cumsum(ddpm_times),
        'ddpm_spearman':      ddpm_spearman,
        'ddpm_spearman_std':  ddpm_std,
    })
    csv_path = OUT_DIR / 'benchmark_optimization_results.csv'
    df_out.to_csv(csv_path, index=False)
    print(f'\nResults saved → {csv_path}')

    # Plot
    plot_time_curve(fm_times, ddpm_times, OUT_DIR / 'benchmark_time_curve.png')
    checkpoint_rounds = [1, 5, 10, 20, 50]
    checkpoint_rounds = [r for r in checkpoint_rounds if r <= args.max_rounds]
    plot_spearman_bar(fm_spearman, fm_std, ddpm_spearman, ddpm_std,
                      checkpoint_rounds, OUT_DIR / 'benchmark_spearman_bar.png')

    # Print summary
    print('\n' + '=' * 55)
    print(f'{"":>20} {"Flow Matching":>15} {"DDPM":>15}')
    print('-' * 55)
    print(f'{"Avg time/round":>20} {np.mean(fm_times):>14.2f}s {np.mean(ddpm_times):>14.2f}s')
    print(f'{"Final Spearman":>20} {fm_spearman[-1]:>15.4f} {ddpm_spearman[-1]:>15.4f}')
    print(f'{"Speedup":>20} {np.mean(ddpm_times)/np.mean(fm_times):>14.1f}x')
    print('=' * 55)


if __name__ == '__main__':
    main()
