#!/usr/bin/env python3
"""Detailed analysis of PhaseFlow→Seq2Phase cross-validation.

Picks representative models, generates sequences, gets Seq2Phase predictions,
and creates comprehensive visualizations comparing predicted vs original phase diagrams.

Environment: phaseflow (with fair-esm installed)
"""
import sys
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

PHASEFLOW_DIR = '/data/yanjie_huang/LLPS/predictor/PhaseFlow'
SEQ2PHASE_DIR = '/data/yanjie_huang/LLPS/seq2phase'
ESM2_MODEL_PATH = os.path.expanduser('~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')
SEQ2PHASE_CKPT = '/data/yanjie_huang/LLPS/seq2phase/outputs/v1/ablation_no_pos_emb_std2/best_model.pt'
TEST_CSV = '/data/yanjie_huang/LLPS/phase_diagram/test_set.csv'
OUT_DIR = os.path.join(PHASEFLOW_DIR, 'infer', 'lm_evaluation_results')

sys.path.insert(0, PHASEFLOW_DIR)
sys.path.insert(0, os.path.join(SEQ2PHASE_DIR, 'v1'))

# Representative models (best from each architecture)
SELECTED = {
    # Set Encoder (per-position velocity)
    'set_flow5_m7':   'outputs_set/output_set_flow5_missing7/best_model.pt',    # best Self-RT (0.506)
    'set_flow5_m11':  'outputs_set/output_set_flow5_missing11/best_model.pt',   # best Cross-RT (0.496)
    'set_flow1_m3':   'outputs_set/output_set_flow1_missing3/best_model.pt',    # high Cross-RT (0.491)
    'set_flow0_m3':   'outputs_set/output_set_flow0_missing3/best_model.pt',    # pure LM, high Cross-RT (0.495)
    # Legacy (single token)
    'flow1_m3':  'outputs/output_flow1_missing3_20260310/best_model.pt',   # best Self-RT legacy (0.523)
    'flow5_m11': 'outputs/output_flow5_missing11_20260310/best_model.pt',  # best Cross-RT legacy (0.462)
    'flow0_m3':  'outputs/output_flow0_missing3_20260310/best_model.pt',   # pure LM legacy
    'flow32_m3': 'outputs/output_flow32_missing3_20260310/best_model.pt',  # high flow weight
}

device = torch.device('cuda:0')
SEQ_LEN = 22
PHASE_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]


def load_phaseflow(ckpt_path):
    from phaseflow import PhaseFlow
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = PhaseFlow(
        dim=cfg['model']['dim'], depth=cfg['model']['depth'],
        heads=cfg['model']['heads'], dim_head=cfg['model']['dim_head'],
        vocab_size=cfg['model']['vocab_size'], phase_dim=cfg['model']['phase_dim'],
        max_seq_len=cfg['model']['max_seq_len'], dropout=0.0,
        use_set_encoder=cfg['model'].get('use_set_encoder', False),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model.to(device).eval()


def load_seq2phase():
    from model_ablation import Seq2PhaseModel
    ckpt = torch.load(SEQ2PHASE_CKPT, map_location=device, weights_only=False)
    model = Seq2PhaseModel(
        esm_dim=1280, d_model=128, n_heads=8,
        n_layers=2, ffn_expansion=4,
        dropout=0.348, seq_len=SEQ_LEN,
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model.to(device).eval()


def load_esm2():
    import esm
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])
    model_data = torch.load(ESM2_MODEL_PATH, map_location="cpu", weights_only=False)
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        "esm2_t33_650M_UR50D", model_data
    )
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return esm_model, batch_converter


def run_pipeline(pf_model, tokenizer, esm_model, batch_converter, s2p_model,
                 test_phases, n_test=500):
    """Generate sequences and get Seq2Phase predictions."""
    # Generate sequences
    all_seqs = []
    for st in range(0, n_test, 100):
        en = min(st + 100, n_test)
        with torch.no_grad():
            _, seqs = pf_model.generate_sequence(
                test_phases[st:en].to(device), tokenizer, max_len=32, temperature=1.0
            )
        all_seqs.extend(seqs)

    # ESM2 embeddings
    all_embeddings = []
    all_lengths = []
    for i in range(0, len(all_seqs), 64):
        batch_seqs = all_seqs[i:i+64]
        # Replace empty with single 'A' placeholder
        batch_seqs_safe = [s if len(s) > 0 else 'A' for s in batch_seqs]
        batch_data = [(str(k), seq) for k, seq in enumerate(batch_seqs_safe)]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens[:, :SEQ_LEN].to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        emb = results['representations'][33]
        if emb.size(1) < SEQ_LEN:
            pad = torch.zeros(emb.size(0), SEQ_LEN - emb.size(1), 1280, device=device)
            emb = torch.cat([emb, pad], dim=1)
        all_embeddings.append(emb.cpu())
        all_lengths.extend([len(s) for s in batch_seqs])
    embeddings = torch.cat(all_embeddings, 0)
    seq_lengths = torch.tensor(all_lengths)

    # Seq2Phase predictions
    all_preds = []
    for i in range(0, len(embeddings), 256):
        emb = embeddings[i:i+256].to(device)
        sl = seq_lengths[i:i+256].to(device)
        with torch.no_grad():
            out = s2p_model(emb, sl)
        all_preds.append(out['predictions'].cpu())
    preds = torch.cat(all_preds, 0).numpy()

    valid_mask = np.array([len(s) > 0 for s in all_seqs])
    return preds, all_seqs, valid_mask


def plot_comprehensive(all_data, test_pv, test_pm, test_df):
    """Create comprehensive visualization for Set Encoder + Legacy models."""
    n_models = len(all_data)
    model_names = list(all_data.keys())
    n_cols = min(n_models, 4)
    n_model_rows = (n_models + n_cols - 1) // n_cols

    # ====== Figure 1: Scatter + Distribution (per model, grid layout) ======
    fig, axes = plt.subplots(n_model_rows * 2, n_cols, figsize=(5 * n_cols, 5 * n_model_rows * 2))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1) if n_model_rows * 2 == 1 else axes.reshape(-1, 1)

    for idx, (name, (preds, seqs, valid)) in enumerate(all_data.items()):
        row_block = idx // n_cols  # which model-row
        col = idx % n_cols
        target = test_pv[:len(preds)]
        mask = test_pm[:len(preds)]
        vm = (mask.flatten() > 0) & valid.repeat(16)

        pred_flat = preds.flatten()[vm]
        tgt_flat = target.flatten()[vm]

        # Scatter plot
        ax = axes[row_block * 2, col]
        ax.scatter(tgt_flat, pred_flat, s=1, alpha=0.15, c='#3498DB', rasterized=True)
        lims = [min(tgt_flat.min(), pred_flat.min()) - 0.1,
                max(tgt_flat.max(), pred_flat.max()) + 0.1]
        ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.7, label='y=x')
        sp, _ = spearmanr(pred_flat, tgt_flat)
        pr, _ = pearsonr(pred_flat, tgt_flat)
        color = '#E67E22' if name.startswith('set_') else '#3498DB'
        ax.set_xlabel('Original PSSI')
        ax.set_ylabel('Seq2Phase Predicted PSSI')
        ax.set_title(f'{name}\nSpearman={sp:.3f}, Pearson={pr:.3f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Distribution comparison
        ax = axes[row_block * 2 + 1, col]
        ax.hist(tgt_flat, bins=50, alpha=0.6, density=True, label='Original', color='#2ECC71')
        ax.hist(pred_flat, bins=50, alpha=0.6, density=True, label='Predicted', color='#E74C3C')
        ax.set_xlabel('PSSI value')
        ax.set_ylabel('Density')
        ax.set_title(f'{name}: PSSI Distribution')
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_models, n_model_rows * n_cols):
        for r in range(2):
            axes[(idx // n_cols) * 2 + r, idx % n_cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cross_validate_scatter_dist.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved scatter + distribution plot")

    # ====== Figure 2: Per-position Spearman heatmap (4×4 grid) ======
    fig, axes = plt.subplots(n_model_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_model_rows))
    if n_model_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_model_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (name, (preds, seqs, valid)) in enumerate(all_data.items()):
        row = idx // n_cols
        col = idx % n_cols
        target = test_pv[:len(preds)]
        mask = test_pm[:len(preds)]
        valid_idx = np.where(valid)[0]

        sp_grid = np.full(16, np.nan)
        for p in range(16):
            vm = (mask[valid_idx, p] > 0)
            if vm.sum() > 10:
                sp_grid[p], _ = spearmanr(preds[valid_idx[vm], p], target[valid_idx[vm], p])

        sp_mat = sp_grid.reshape(4, 4)
        ax = axes[row, col]
        im = ax.imshow(sp_mat, cmap='RdYlGn', vmin=-0.1, vmax=0.5, aspect='equal')
        for i in range(4):
            for j in range(4):
                v = sp_mat[i, j]
                txt = f'{v:.3f}' if not np.isnan(v) else 'N/A'
                ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                        color='black' if v > 0.1 else 'white')
        ax.set_xticks(range(4))
        ax.set_xticklabels([f'C{j+1}' for j in range(4)])
        ax.set_yticks(range(4))
        ax.set_yticklabels([f'L{i+1}' for i in range(4)])
        ax.set_title(f'{name}\nPer-position Spearman')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for idx in range(n_models, n_model_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cross_validate_per_position.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved per-position heatmap")

    # ====== Figure 3: Example phase diagram comparisons (best model) ======
    best_name = max(all_data, key=lambda n: spearmanr(
        all_data[n][0].flatten()[(test_pm[:len(all_data[n][0])].flatten() > 0) & all_data[n][2].repeat(16)],
        test_pv[:len(all_data[n][0])].flatten()[(test_pm[:len(all_data[n][0])].flatten() > 0) & all_data[n][2].repeat(16)]
    )[0])
    preds_best, seqs_best, valid_best = all_data[best_name]
    target_best = test_pv[:len(preds_best)]
    mask_best = test_pm[:len(preds_best)]

    # Pick 6 examples: 3 low PSSI + 3 high PSSI
    valid_idx = np.where(valid_best)[0]
    target_means = (target_best[valid_idx] * mask_best[valid_idx]).sum(1) / (mask_best[valid_idx].sum(1) + 1e-6)
    sorted_idx = np.argsort(target_means)
    example_idx = np.concatenate([sorted_idx[:3], sorted_idx[-3:]])

    fig, axes = plt.subplots(6, 3, figsize=(14, 28))
    fig.suptitle(f'Example Phase Diagrams: Original vs {best_name} Predicted\n'
                 f'(Top 3: low PSSI, Bottom 3: high PSSI)', fontsize=14, y=0.995)

    for row, eidx in enumerate(example_idx):
        real_idx = valid_idx[eidx]
        orig = target_best[real_idx].reshape(4, 4)
        pred = preds_best[real_idx].reshape(4, 4)
        m = mask_best[real_idx].reshape(4, 4)

        orig_masked = np.where(m > 0, orig, np.nan)
        vmin = min(np.nanmin(orig_masked[~np.isnan(orig_masked)]),
                   np.nanmin(pred)) if (~np.isnan(orig_masked)).any() else pred.min()
        vmax = max(np.nanmax(orig_masked[~np.isnan(orig_masked)]),
                   np.nanmax(pred)) if (~np.isnan(orig_masked)).any() else pred.max()

        # Column 0: Original
        ax = axes[row, 0]
        im = ax.imshow(orig_masked, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(f'#{real_idx} Original (mean={np.nanmean(orig_masked):.2f})', fontsize=9)
        for i in range(4):
            for j in range(4):
                v = orig_masked[i, j]
                txt = f'{v:.2f}' if not np.isnan(v) else 'NaN'
                ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                        color='gray' if np.isnan(v) else 'black')

        # Column 1: Predicted
        ax = axes[row, 1]
        ax.imshow(pred, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(f'#{real_idx} Predicted (mean={pred.mean():.2f})', fontsize=9)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{pred[i,j]:.2f}', ha='center', va='center', fontsize=7)

        # Column 2: Difference + info
        diff = pred - orig
        diff_masked = np.where(m > 0, diff, np.nan)
        valid_diff = diff_masked[~np.isnan(diff_masked)]
        dmax = max(abs(valid_diff.min()), abs(valid_diff.max()), 0.3) if len(valid_diff) > 0 else 0.5
        ax = axes[row, 2]
        ax.imshow(diff_masked, cmap='RdBu_r', vmin=-dmax, vmax=dmax, aspect='equal')
        vm = m.flatten() > 0
        sp_val = spearmanr(pred.flatten()[vm], orig.flatten()[vm])[0] if vm.sum() > 2 else float('nan')
        seq = seqs_best[real_idx] if len(seqs_best[real_idx]) > 0 else '(empty)'
        ax.set_title(f'Diff | seq={seq[:15]}.. | ρ={sp_val:.2f}', fontsize=8)
        for i in range(4):
            for j in range(4):
                v = diff_masked[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=7)

        for c in range(3):
            axes[row, c].set_xticks(range(4))
            axes[row, c].set_xticklabels([f'C{j+1}' for j in range(4)], fontsize=7)
            axes[row, c].set_yticks(range(4))
            axes[row, c].set_yticklabels([f'L{i+1}' for i in range(4)], fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cross_validate_examples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved example comparisons")

    # ====== Figure 4: Mean PSSI scatter (per sample) for all models ======
    fig, axes = plt.subplots(n_model_rows, n_cols, figsize=(5 * n_cols, 5 * n_model_rows))
    if n_model_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_model_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (name, (preds, seqs, valid)) in enumerate(all_data.items()):
        row = idx // n_cols
        col = idx % n_cols
        target = test_pv[:len(preds)]
        mask = test_pm[:len(preds)]
        valid_idx = np.where(valid)[0]

        pred_mean = preds[valid_idx].mean(axis=1)
        tgt_mean = (target[valid_idx] * mask[valid_idx]).sum(axis=1) / (mask[valid_idx].sum(axis=1) + 1e-6)

        ax = axes[row, col]
        ax.scatter(tgt_mean, pred_mean, s=8, alpha=0.4, c='#3498DB', edgecolors='none')
        lims = [min(tgt_mean.min(), pred_mean.min()) - 0.1,
                max(tgt_mean.max(), pred_mean.max()) + 0.1]
        ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.7)
        sp, _ = spearmanr(pred_mean, tgt_mean)
        pr, _ = pearsonr(pred_mean, tgt_mean)
        ax.set_xlabel('Original Mean PSSI')
        ax.set_ylabel('Predicted Mean PSSI')
        ax.set_title(f'{name}\nMean PSSI per sample\nSpearman={sp:.3f}, Pearson={pr:.3f}')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')

    # Hide unused axes
    for idx in range(n_models, n_model_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cross_validate_mean_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved mean PSSI scatter")


def main():
    print("=" * 80)
    print("Detailed Cross-Validation Analysis")
    print("=" * 80)

    # Load test data
    test_df = pd.read_csv(TEST_CSV)
    test_pv = test_df[PHASE_COLS].values.astype(np.float32)
    test_pm = (~np.isnan(test_pv)).astype(np.float32)
    test_pv_filled = np.nan_to_num(test_pv, nan=0.0)
    test_phases = torch.tensor(test_pv_filled, dtype=torch.float32)
    print(f"Test set: {len(test_df)} samples")

    from phaseflow import AminoAcidTokenizer
    tokenizer = AminoAcidTokenizer()

    print("Loading ESM2...")
    esm_model, batch_converter = load_esm2()

    print("Loading Seq2Phase...")
    s2p_model = load_seq2phase()

    all_data = {}

    for name, ckpt_rel in SELECTED.items():
        ckpt_path = os.path.join(PHASEFLOW_DIR, ckpt_rel)
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {name}")
            continue

        print(f"\n--- {name} ---")
        pf_model = load_phaseflow(ckpt_path)
        preds, seqs, valid = run_pipeline(
            pf_model, tokenizer, esm_model, batch_converter, s2p_model, test_phases
        )
        all_data[name] = (preds, seqs, valid)

        del pf_model
        torch.cuda.empty_cache()

        # Quick stats
        valid_idx = np.where(valid)[0]
        vm = (test_pm[:len(preds)].flatten() > 0) & valid.repeat(16)
        sp, _ = spearmanr(preds.flatten()[vm], test_pv_filled[:len(preds)].flatten()[vm])
        print(f"  Valid: {valid.sum()}/{len(valid)}, Flat Spearman: {sp:.4f}")

    # Save raw predictions
    save_data = {}
    for name, (preds, seqs, valid) in all_data.items():
        save_data[name] = {
            'predictions': preds.tolist(),
            'sequences': seqs,
            'valid': valid.tolist(),
        }
    with open(os.path.join(OUT_DIR, 'cross_validate_detailed.json'), 'w') as f:
        json.dump(save_data, f)
    print(f"\nSaved raw predictions to {OUT_DIR}/cross_validate_detailed.json")

    visualize(all_data, test_pv_filled, test_pm, test_df)


def visualize(all_data, test_pv_filled, test_pm, test_df):
    print("\nGenerating visualizations...")
    plot_comprehensive(all_data, test_pv_filled, test_pm, test_df)
    print("\nDone!")


def load_and_visualize():
    """Load saved predictions and regenerate plots (no GPU needed)."""
    test_df = pd.read_csv(TEST_CSV)
    test_pv = test_df[PHASE_COLS].values.astype(np.float32)
    test_pm = (~np.isnan(test_pv)).astype(np.float32)
    test_pv_filled = np.nan_to_num(test_pv, nan=0.0)

    with open(os.path.join(OUT_DIR, 'cross_validate_detailed.json')) as f:
        save_data = json.load(f)

    all_data = {}
    for name, d in save_data.items():
        all_data[name] = (
            np.array(d['predictions']),
            d['sequences'],
            np.array(d['valid']),
        )
    visualize(all_data, test_pv_filled, test_pm, test_df)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true', help='Only regenerate plots from saved data')
    args = parser.parse_args()
    if args.plot_only:
        load_and_visualize()
    else:
        main()
