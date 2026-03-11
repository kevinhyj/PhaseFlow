#!/usr/bin/env python3
"""Cross-validate PhaseFlow generated sequences using Seq2Phase.

PhaseFlow(phase→seq) → ESM2 → Seq2Phase(seq→phase) → compare with original phase.

Environment: esm_project
"""
import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm

# Paths
PHASEFLOW_DIR = '/data/yanjie_huang/LLPS/predictor/PhaseFlow'
SEQ2PHASE_DIR = '/data/yanjie_huang/LLPS/seq2phase'
ESM2_MODEL_PATH = os.path.expanduser('~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')
SEQ2PHASE_CKPT = '/data/yanjie_huang/LLPS/seq2phase/outputs/v1/ablation_no_pos_emb_std2/best_model.pt'
TEST_CSV = '/data/yanjie_huang/LLPS/phase_diagram/test_set.csv'

sys.path.insert(0, PHASEFLOW_DIR)
sys.path.insert(0, os.path.join(SEQ2PHASE_DIR, 'v1'))

MODELS = {
    # === Set Encoder (per-position velocity) ===
    'set_flow32_m0':  'outputs_set/output_set_flow32_missing0/best_model.pt',
    'set_flow32_m3':  'outputs_set/output_set_flow32_missing3/best_model.pt',
    'set_flow32_m7':  'outputs_set/output_set_flow32_missing7/best_model.pt',
    'set_flow32_m11': 'outputs_set/output_set_flow32_missing11/best_model.pt',
    'set_flow5_m0':   'outputs_set/output_set_flow5_missing0/best_model.pt',
    'set_flow5_m3':   'outputs_set/output_set_flow5_missing3/best_model.pt',
    'set_flow5_m7':   'outputs_set/output_set_flow5_missing7/best_model.pt',
    'set_flow5_m11':  'outputs_set/output_set_flow5_missing11/best_model.pt',
    'set_flow1_m0':   'outputs_set/output_set_flow1_missing0/best_model.pt',
    'set_flow1_m3':   'outputs_set/output_set_flow1_missing3/best_model.pt',
    'set_flow1_m7':   'outputs_set/output_set_flow1_missing7/best_model.pt',
    'set_flow1_m11':  'outputs_set/output_set_flow1_missing11/best_model.pt',
    'set_flow0_m0':   'outputs_set/output_set_flow0_missing0/best_model.pt',
    'set_flow0_m3':   'outputs_set/output_set_flow0_missing3/best_model.pt',
    'set_flow0_m7':   'outputs_set/output_set_flow0_missing7/best_model.pt',
    'set_flow0_m11':  'outputs_set/output_set_flow0_missing11/best_model.pt',
    # === Legacy (single token) ===
    'flow32_m0':  'outputs/output_flow32_missing0_20260310/best_model.pt',
    'flow32_m3':  'outputs/output_flow32_missing3_20260310/best_model.pt',
    'flow32_m7':  'outputs/output_flow32_missing7_20260310/best_model.pt',
    'flow32_m11': 'outputs/output_flow32_missing11_20260310/best_model.pt',
    'flow5_m0':   'outputs/output_flow5_missing0_20260310/best_model.pt',
    'flow5_m3':   'outputs/output_flow5_missing3_20260310/best_model.pt',
    'flow5_m7':   'outputs/output_flow5_missing7_20260310/best_model.pt',
    'flow5_m11':  'outputs/output_flow5_missing11_20260310/best_model.pt',
    'flow1_m0':   'outputs/output_flow1_missing0_20260310/best_model.pt',
    'flow1_m3':   'outputs/output_flow1_missing3_20260310/best_model.pt',
    'flow1_m7':   'outputs/output_flow1_missing7_20260310/best_model.pt',
    'flow1_m11':  'outputs/output_flow1_missing11_20260310/best_model.pt',
    'flow0_m0':   'outputs/output_flow0_missing0_20260310/best_model.pt',
    'flow0_m3':   'outputs/output_flow0_missing3_20260310/best_model.pt',
    'flow0_m7':   'outputs/output_flow0_missing7_20260310/best_model.pt',
    'flow0_m11':  'outputs/output_flow0_missing11_20260310/best_model.pt',
}

# PhaseFlow self-RT results will be loaded from evaluate_lm results or set to None
SELF_RT = {
    # Legacy (from previous ANALYSIS.md)
    'flow32_m0': 0.348, 'flow32_m3': 0.434, 'flow32_m7': 0.430, 'flow32_m11': 0.262,
    'flow5_m0': 0.375, 'flow5_m3': 0.461, 'flow5_m7': 0.402, 'flow5_m11': 0.450,
    'flow1_m0': 0.451, 'flow1_m3': 0.491, 'flow1_m7': 0.470, 'flow1_m11': 0.485,
    'flow0_m0': 0.044, 'flow0_m3': 0.069, 'flow0_m7': 0.073, 'flow0_m11': -0.011,
    # Set encoder — will be populated after evaluate_lm runs
}

device = torch.device('cuda:0')
SEQ_LEN = 22


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
    # ablation_no_pos_emb_std2: config from config.txt (no 'args' key in checkpoint)
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
    # Load model weights directly, skip contact regression head
    model_data = torch.load(ESM2_MODEL_PATH, map_location="cpu", weights_only=False)
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        "esm2_t33_650M_UR50D", model_data
    )
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return esm_model, batch_converter


def extract_esm2_embeddings(sequences, esm_model, batch_converter, batch_size=64):
    """Extract ESM2 embeddings for a list of sequences."""
    all_embeddings = []
    all_lengths = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        batch_data = [(str(k), seq) for k, seq in enumerate(batch_seqs)]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens[:, :SEQ_LEN].to(device)

        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        emb = results['representations'][33]  # (B, L, 1280)

        if emb.size(1) < SEQ_LEN:
            pad = torch.zeros(emb.size(0), SEQ_LEN - emb.size(1), 1280, device=device)
            emb = torch.cat([emb, pad], dim=1)

        all_embeddings.append(emb.cpu())
        all_lengths.extend([len(s) for s in batch_seqs])

    return torch.cat(all_embeddings, 0), torch.tensor(all_lengths)


def predict_with_seq2phase(embeddings, seq_lengths, s2p_model, batch_size=256):
    """Run Seq2Phase prediction."""
    all_preds = []
    for i in range(0, len(embeddings), batch_size):
        emb = embeddings[i:i+batch_size].to(device)
        sl = seq_lengths[i:i+batch_size].to(device)
        with torch.no_grad():
            out = s2p_model(emb, sl)
        all_preds.append(out['predictions'].cpu())
    return torch.cat(all_preds, 0)


def main():
    print("=" * 80)
    print("Cross-Validation: PhaseFlow(phase→seq) → Seq2Phase(seq→phase)")
    print("=" * 80)

    # Load test data
    test_df = pd.read_csv(TEST_CSV)
    phase_cols = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]
    test_pv = test_df[phase_cols].values.astype(np.float32)
    test_pm = (~np.isnan(test_pv)).astype(np.float32)
    test_pv_filled = np.nan_to_num(test_pv, nan=0.0)
    test_phases = torch.tensor(test_pv_filled, dtype=torch.float32)
    test_masks = torch.tensor(test_pm, dtype=torch.float32)
    print(f"Test set: {len(test_df)} samples")

    # Load PhaseFlow tokenizer
    from phaseflow import AminoAcidTokenizer
    tokenizer = AminoAcidTokenizer()

    # Load ESM2
    print("Loading ESM2...")
    esm_model, batch_converter = load_esm2()

    # Load Seq2Phase
    print("Loading Seq2Phase...")
    s2p_model = load_seq2phase()

    # Try to load self-RT results from evaluate_lm output
    lm_results_path = os.path.join(PHASEFLOW_DIR, 'infer', 'lm_evaluation_results', 'lm_evaluation.json')
    if os.path.exists(lm_results_path):
        with open(lm_results_path) as f:
            lm_results = json.load(f)
        for name, r in lm_results.items():
            if name not in SELF_RT and 'roundtrip_mean_spearman' in r:
                SELF_RT[name] = r['roundtrip_mean_spearman']
        print(f"Loaded self-RT for {len(SELF_RT)} models from {lm_results_path}")

    results = {}

    for name, ckpt_rel in MODELS.items():
        ckpt_path = os.path.join(PHASEFLOW_DIR, ckpt_rel)
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {name}")
            continue

        print(f"\n--- {name} ---")

        # Load PhaseFlow
        pf_model = load_phaseflow(ckpt_path)

        # Generate 1 sequence per test sample
        all_seqs = []
        for st in range(0, 500, 100):
            en = min(st + 100, 500)
            with torch.no_grad():
                _, seqs = pf_model.generate_sequence(
                    test_phases[st:en].to(device), tokenizer, max_len=32, temperature=1.0
                )
            all_seqs.extend(seqs)

        del pf_model
        torch.cuda.empty_cache()

        # Filter valid sequences
        valid_idx = [i for i, s in enumerate(all_seqs) if len(s) > 0]
        valid_seqs = [all_seqs[i] for i in valid_idx]
        print(f"  Generated: {len(all_seqs)}, valid: {len(valid_seqs)}")

        if len(valid_seqs) < 10:
            print(f"  Too few valid sequences, skipping")
            continue

        # ESM2 embeddings
        embeddings, seq_lengths = extract_esm2_embeddings(valid_seqs, esm_model, batch_converter)

        # Seq2Phase predictions
        s2p_preds = predict_with_seq2phase(embeddings, seq_lengths, s2p_model).numpy()

        # Compare with original phase diagrams
        target = test_pv_filled[valid_idx]
        masks = test_pm[valid_idx]

        # Flattened Spearman
        vm = masks.flatten() > 0
        if vm.sum() > 10:
            flat_sp, _ = spearmanr(s2p_preds.flatten()[vm], target.flatten()[vm])
        else:
            flat_sp = float('nan')

        # Mean Spearman
        pred_mean = s2p_preds.mean(axis=1)
        target_mean = (target * masks).sum(axis=1) / (masks.sum(axis=1) + 1e-6)
        mean_sp, _ = spearmanr(pred_mean, target_mean)

        results[name] = {
            'n_valid': len(valid_seqs),
            'cross_RT_flat': float(flat_sp),
            'cross_RT_mean': float(mean_sp),
            'self_RT_mean': SELF_RT.get(name, float('nan')),
        }

        print(f"  Cross-RT (flat): {flat_sp:.4f}  |  Cross-RT (mean): {mean_sp:.4f}  |  Self-RT (mean): {SELF_RT.get(name, 0):.4f}")

    # Print comparison table
    print("\n\n" + "=" * 90)
    print("COMPARISON: PhaseFlow Self-RT vs Seq2Phase Cross-RT")
    print("=" * 90)
    print(f"{'Model':<14} {'Self_RT_Mn':>10} {'Cross_RT_Fl':>11} {'Cross_RT_Mn':>11} {'Delta':>8}")
    print("-" * 90)

    for group in ['set_flow32', 'set_flow5', 'set_flow1', 'set_flow0',
                  'flow32', 'flow5', 'flow1', 'flow0']:
        for name in MODELS:
            if name not in results or not name.startswith(group):
                continue
            r = results[name]
            delta = r['cross_RT_mean'] - r['self_RT_mean']
            print(f"{name:<14} {r['self_RT_mean']:>10.4f} {r['cross_RT_flat']:>11.4f} {r['cross_RT_mean']:>11.4f} {delta:>+8.4f}")
        print("-" * 90)

    # Save
    out_dir = os.path.join(PHASEFLOW_DIR, 'infer', 'lm_evaluation_results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'cross_validate_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir}/cross_validate_results.json")


if __name__ == '__main__':
    main()
