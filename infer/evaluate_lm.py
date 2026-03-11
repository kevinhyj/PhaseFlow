#!/usr/bin/env python3
"""Comprehensive LM (phase→seq) evaluation for PhaseFlow models.

Evaluates:
1. Perplexity (from test_results.json or recomputed)
2. Generated sequence quality (length, AA composition, KL divergence)
3. Diversity (unique ratio, edit distance)
4. Novelty (overlap with training set)
5. Conditional consistency (phase→seq→phase round-trip)
"""
import sys
sys.path.insert(0, '/data/yanjie_huang/LLPS/predictor/PhaseFlow')

import os
import json
import torch
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from scipy.stats import spearmanr, entropy
from phaseflow import PhaseFlow, AminoAcidTokenizer
from phaseflow.data import PhaseDataset, create_dataloader

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
BASE_DIR = '/data/yanjie_huang/LLPS/predictor/PhaseFlow'

MODELS = {
    # === Set Encoder (per-position velocity) ===
    # flow32 set
    'set_flow32_m0':  'outputs_set/output_set_flow32_missing0/best_model.pt',
    'set_flow32_m3':  'outputs_set/output_set_flow32_missing3/best_model.pt',
    'set_flow32_m7':  'outputs_set/output_set_flow32_missing7/best_model.pt',
    'set_flow32_m11': 'outputs_set/output_set_flow32_missing11/best_model.pt',
    # flow5 set
    'set_flow5_m0':   'outputs_set/output_set_flow5_missing0/best_model.pt',
    'set_flow5_m3':   'outputs_set/output_set_flow5_missing3/best_model.pt',
    'set_flow5_m7':   'outputs_set/output_set_flow5_missing7/best_model.pt',
    'set_flow5_m11':  'outputs_set/output_set_flow5_missing11/best_model.pt',
    # flow1 set
    'set_flow1_m0':   'outputs_set/output_set_flow1_missing0/best_model.pt',
    'set_flow1_m3':   'outputs_set/output_set_flow1_missing3/best_model.pt',
    'set_flow1_m7':   'outputs_set/output_set_flow1_missing7/best_model.pt',
    'set_flow1_m11':  'outputs_set/output_set_flow1_missing11/best_model.pt',
    # flow0 set (pure LM)
    'set_flow0_m0':   'outputs_set/output_set_flow0_missing0/best_model.pt',
    'set_flow0_m3':   'outputs_set/output_set_flow0_missing3/best_model.pt',
    'set_flow0_m7':   'outputs_set/output_set_flow0_missing7/best_model.pt',
    'set_flow0_m11':  'outputs_set/output_set_flow0_missing11/best_model.pt',
    # === Legacy (single token) ===
    # flow32 legacy
    'flow32_m0':  'outputs/output_flow32_missing0_20260310/best_model.pt',
    'flow32_m3':  'outputs/output_flow32_missing3_20260310/best_model.pt',
    'flow32_m7':  'outputs/output_flow32_missing7_20260310/best_model.pt',
    'flow32_m11': 'outputs/output_flow32_missing11_20260310/best_model.pt',
    # flow5 legacy
    'flow5_m0':   'outputs/output_flow5_missing0_20260310/best_model.pt',
    'flow5_m3':   'outputs/output_flow5_missing3_20260310/best_model.pt',
    'flow5_m7':   'outputs/output_flow5_missing7_20260310/best_model.pt',
    'flow5_m11':  'outputs/output_flow5_missing11_20260310/best_model.pt',
    # flow1 legacy
    'flow1_m0':   'outputs/output_flow1_missing0_20260310/best_model.pt',
    'flow1_m3':   'outputs/output_flow1_missing3_20260310/best_model.pt',
    'flow1_m7':   'outputs/output_flow1_missing7_20260310/best_model.pt',
    'flow1_m11':  'outputs/output_flow1_missing11_20260310/best_model.pt',
    # flow0 legacy (pure LM)
    'flow0_m0':   'outputs/output_flow0_missing0_20260310/best_model.pt',
    'flow0_m3':   'outputs/output_flow0_missing3_20260310/best_model.pt',
    'flow0_m7':   'outputs/output_flow0_missing7_20260310/best_model.pt',
    'flow0_m11':  'outputs/output_flow0_missing11_20260310/best_model.pt',
}

device = torch.device('cuda:0')


def load_model(ckpt_path):
    """Load a PhaseFlow model from checkpoint."""
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
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device).eval()
    return model, config


def aa_frequency(sequences):
    """Compute amino acid frequency distribution (20-dim)."""
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
    return entropy(p, q)


def levenshtein_distance(s1, s2):
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def evaluate_model(model_name, model, tokenizer, test_phases, test_masks,
                   train_sequences, real_aa_freq, n_samples=5):
    """Evaluate a single model's LM performance."""
    results = {}

    # === Generate sequences ===
    all_seqs = []
    batch_size = 100
    n_test = test_phases.shape[0]

    for sample_idx in range(n_samples):
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            phase_batch = test_phases[start:end].to(device)
            with torch.no_grad():
                _, seqs = model.generate_sequence(
                    phase_batch, tokenizer, max_len=32, temperature=1.0
                )
            all_seqs.extend(seqs)

    # Filter out empty sequences
    valid_seqs = [s for s in all_seqs if len(s) > 0]
    results['total_generated'] = len(all_seqs)
    results['valid_generated'] = len(valid_seqs)
    results['empty_ratio'] = 1 - len(valid_seqs) / max(len(all_seqs), 1)

    if len(valid_seqs) == 0:
        print(f"  WARNING: {model_name} generated 0 valid sequences!")
        return results

    # === 1. Length statistics ===
    lengths = [len(s) for s in valid_seqs]
    results['len_mean'] = np.mean(lengths)
    results['len_std'] = np.std(lengths)
    results['len_min'] = min(lengths)
    results['len_max'] = max(lengths)
    results['len_gt20_pct'] = 100 * sum(1 for l in lengths if l > 20) / len(lengths)
    results['len_5to20_pct'] = 100 * sum(1 for l in lengths if 5 <= l <= 20) / len(lengths)

    # === 2. AA composition ===
    gen_aa_freq = aa_frequency(valid_seqs)
    results['kl_div'] = kl_divergence(gen_aa_freq, real_aa_freq)

    # Most common AA
    max_aa_idx = np.argmax(gen_aa_freq)
    results['dominant_aa'] = AMINO_ACIDS[max_aa_idx]
    results['dominant_aa_pct'] = gen_aa_freq[max_aa_idx] * 100

    # Per-sequence single AA max ratio
    max_single_ratios = []
    for seq in valid_seqs:
        if len(seq) == 0:
            continue
        cnt = Counter(seq)
        max_ratio = max(cnt.values()) / len(seq)
        max_single_ratios.append(max_ratio)
    results['max_single_aa_ratio_mean'] = np.mean(max_single_ratios) * 100

    # === 3. Diversity ===
    unique_seqs = set(valid_seqs)
    results['unique_ratio'] = len(unique_seqs) / len(valid_seqs)
    results['n_unique'] = len(unique_seqs)

    # Top-5 most frequent sequences
    seq_counter = Counter(valid_seqs)
    top5 = seq_counter.most_common(5)
    results['top1_freq'] = top5[0][1] / len(valid_seqs) * 100 if top5 else 0
    results['top5_freq'] = sum(c for _, c in top5) / len(valid_seqs) * 100 if top5 else 0
    results['top1_seq'] = top5[0][0] if top5 else ''

    # Average pairwise edit distance (sample 200 pairs)
    if len(valid_seqs) >= 2:
        rng = np.random.RandomState(42)
        n_pairs = min(500, len(valid_seqs) * (len(valid_seqs) - 1) // 2)
        indices = rng.choice(len(valid_seqs), size=(n_pairs, 2), replace=True)
        distances = [levenshtein_distance(valid_seqs[i], valid_seqs[j])
                     for i, j in indices if i != j]
        results['avg_edit_distance'] = np.mean(distances) if distances else 0
    else:
        results['avg_edit_distance'] = 0

    # === 4. Novelty ===
    overlap = sum(1 for s in unique_seqs if s in train_sequences)
    results['novelty'] = 1 - overlap / max(len(unique_seqs), 1)
    results['overlap_count'] = overlap

    # === 5. Conditional consistency (round-trip) ===
    # Generate phase from generated sequences using the SAME model's flow direction
    # Pick first sample of each test case (500 sequences)
    roundtrip_seqs = all_seqs[:n_test]
    valid_rt_indices = [i for i, s in enumerate(roundtrip_seqs) if len(s) > 0]

    if len(valid_rt_indices) >= 10:
        rt_seqs = [roundtrip_seqs[i] for i in valid_rt_indices]
        rt_phases_input = test_phases[valid_rt_indices]
        rt_masks_input = test_masks[valid_rt_indices]

        # Tokenize generated sequences
        rt_input_ids = tokenizer.batch_encode(rt_seqs, max_len=32).to(device)
        rt_attention_mask = (rt_input_ids != tokenizer.PAD_ID).long()
        rt_seq_lens = torch.tensor([len(s) for s in rt_seqs], device=device)

        # Generate phase diagrams from generated sequences
        all_pred_phase = []
        for start in range(0, len(rt_seqs), batch_size):
            end = min(start + batch_size, len(rt_seqs))
            with torch.no_grad():
                pred_phase = model.generate_phase(
                    rt_input_ids[start:end],
                    rt_attention_mask[start:end],
                    rt_seq_lens[start:end],
                    method='euler',
                )
            all_pred_phase.append(pred_phase.cpu())

        pred_phase_all = torch.cat(all_pred_phase, dim=0).numpy()
        target_phase = rt_phases_input.numpy()
        masks = rt_masks_input.numpy()

        # Compute correlation on valid values
        valid_mask = masks.flatten() > 0
        if valid_mask.sum() > 10:
            flat_pred = pred_phase_all.flatten()[valid_mask]
            flat_target = target_phase.flatten()[valid_mask]
            rt_spearman, _ = spearmanr(flat_pred, flat_target)
            results['roundtrip_spearman'] = rt_spearman
        else:
            results['roundtrip_spearman'] = float('nan')

        # Per-sample mean correlation
        pred_means = pred_phase_all.mean(axis=1)
        target_means = (target_phase * masks).sum(axis=1) / (masks.sum(axis=1) + 1e-6)
        mean_sr, _ = spearmanr(pred_means, target_means)
        results['roundtrip_mean_spearman'] = mean_sr
    else:
        results['roundtrip_spearman'] = float('nan')
        results['roundtrip_mean_spearman'] = float('nan')

    return results


def main():
    print("=" * 80)
    print("PhaseFlow LM Performance Evaluation")
    print("=" * 80)

    tokenizer = AminoAcidTokenizer()

    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv('/data/yanjie_huang/LLPS/phase_diagram/test_set.csv')
    phase_cols = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]
    test_phase_values = test_df[phase_cols].values.astype(np.float32)
    test_phase_mask = (~np.isnan(test_phase_values)).astype(np.float32)
    test_phase_values = np.nan_to_num(test_phase_values, nan=0.0)
    test_phases = torch.tensor(test_phase_values, dtype=torch.float32)
    test_masks = torch.tensor(test_phase_mask, dtype=torch.float32)
    print(f"  Test set: {len(test_df)} samples")

    # Load training data for novelty check and AA frequency
    print("Loading training data for reference...")
    train_df = pd.read_csv('/data/yanjie_huang/LLPS/phase_diagram/phase_diagram_original_scale.csv')
    train_sequences = set(train_df['AminoAcidSequence'].values)
    real_aa_freq = aa_frequency(train_df['AminoAcidSequence'].values)
    print(f"  Training set: {len(train_sequences)} unique sequences")

    # Evaluate each model
    all_results = {}
    for model_name, ckpt_path in MODELS.items():
        full_path = os.path.join(BASE_DIR, ckpt_path)
        if not os.path.exists(full_path):
            print(f"\n  SKIP {model_name}: checkpoint not found at {full_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        model, config = load_model(full_path)

        # Load perplexity from test_results.json or training log
        ppl = None
        results_json = os.path.join(os.path.dirname(full_path), 'test_results.json')
        if os.path.exists(results_json):
            try:
                with open(results_json) as f:
                    test_results = json.load(f)
                ppl = test_results.get('perplexity', None)
            except json.JSONDecodeError:
                pass

        # Fallback: extract from training log
        if ppl is None:
            import glob, re
            output_dir_name = os.path.basename(os.path.dirname(full_path))
            config_name = output_dir_name.replace('output_', '')
            log_pattern = os.path.join(BASE_DIR, 'logs', f'train_{config_name}_*.log')
            log_files = sorted(glob.glob(log_pattern))
            for lf in log_files:
                try:
                    with open(lf) as f:
                        content = f.read()
                    # Find perplexity after TEST SET EVALUATION
                    test_idx = content.rfind('TEST SET EVALUATION')
                    if test_idx >= 0:
                        after = content[test_idx:]
                        m = re.search(r'Perplexity:\s*([0-9.]+)', after)
                        if m:
                            ppl = float(m.group(1))
                            break
                except Exception:
                    pass

        if ppl is not None:
            print(f"  Perplexity: {ppl:.2f}")
        else:
            print("  Perplexity: N/A")

        results = evaluate_model(
            model_name, model, tokenizer, test_phases, test_masks,
            train_sequences, real_aa_freq, n_samples=5
        )
        if ppl is not None:
            results['perplexity'] = ppl

        all_results[model_name] = results

        # Print per-model summary
        print(f"\n  --- {model_name} Summary ---")
        print(f"  Generated: {results.get('total_generated', 0)} total, "
              f"{results.get('valid_generated', 0)} valid, "
              f"{results.get('empty_ratio', 0)*100:.1f}% empty")
        print(f"  Length: mean={results.get('len_mean', 0):.1f}, "
              f"std={results.get('len_std', 0):.1f}, "
              f"range=[{results.get('len_min', 0)}, {results.get('len_max', 0)}], "
              f">20aa={results.get('len_gt20_pct', 0):.1f}%, "
              f"5-20aa={results.get('len_5to20_pct', 0):.1f}%")
        print(f"  AA KL divergence: {results.get('kl_div', 0):.4f}")
        print(f"  Dominant AA: {results.get('dominant_aa', '?')} "
              f"({results.get('dominant_aa_pct', 0):.1f}%)")
        print(f"  Max single AA ratio (mean): {results.get('max_single_aa_ratio_mean', 0):.1f}%")
        print(f"  Diversity: {results.get('unique_ratio', 0)*100:.1f}% unique "
              f"({results.get('n_unique', 0)}/{results.get('valid_generated', 0)})")
        print(f"  Top-1 seq freq: {results.get('top1_freq', 0):.2f}% ({results.get('top1_seq', '')[:20]})")
        print(f"  Top-5 seq freq: {results.get('top5_freq', 0):.2f}%")
        print(f"  Avg edit distance: {results.get('avg_edit_distance', 0):.1f}")
        print(f"  Novelty: {results.get('novelty', 0)*100:.1f}%")
        print(f"  Round-trip Spearman (flat): {results.get('roundtrip_spearman', float('nan')):.4f}")
        print(f"  Round-trip Spearman (mean): {results.get('roundtrip_mean_spearman', float('nan')):.4f}")

        del model
        torch.cuda.empty_cache()

    # === Print comparison table ===
    print("\n\n" + "=" * 120)
    print("COMPARISON TABLE")
    print("=" * 120)

    headers = ['Model', 'PPL', 'Len', '>20aa%', '5-20%', 'KL', 'Uniq%',
               'Top1%', 'EditDist', 'Novel%', 'RT_Sp', 'RT_Mean']
    header_line = f"{'Model':<14} {'PPL':>6} {'Len':>5} {'>20%':>5} {'5-20%':>5} "
    header_line += f"{'KL':>6} {'Uniq%':>6} {'Top1%':>6} {'EDist':>6} "
    header_line += f"{'Novel%':>7} {'RT_Sp':>7} {'RT_Mn':>7}"
    print(header_line)
    print("-" * 120)

    for name in MODELS:
        if name not in all_results:
            continue
        r = all_results[name]
        ppl_str = f"{r.get('perplexity', 0):.1f}" if r.get('perplexity') else "N/A"
        line = f"{name:<14} {ppl_str:>6} "
        line += f"{r.get('len_mean', 0):>5.1f} "
        line += f"{r.get('len_gt20_pct', 0):>5.1f} "
        line += f"{r.get('len_5to20_pct', 0):>5.1f} "
        line += f"{r.get('kl_div', 0):>6.4f} "
        line += f"{r.get('unique_ratio', 0)*100:>6.1f} "
        line += f"{r.get('top1_freq', 0):>6.2f} "
        line += f"{r.get('avg_edit_distance', 0):>6.1f} "
        line += f"{r.get('novelty', 0)*100:>7.1f} "
        rt_sp = r.get('roundtrip_spearman', float('nan'))
        rt_mn = r.get('roundtrip_mean_spearman', float('nan'))
        line += f"{rt_sp:>7.4f} " if not np.isnan(rt_sp) else f"{'N/A':>7} "
        line += f"{rt_mn:>7.4f}" if not np.isnan(rt_mn) else f"{'N/A':>7}"
        print(line)

    # Save results
    out_dir = os.path.join(BASE_DIR, 'infer', 'lm_evaluation_results')
    os.makedirs(out_dir, exist_ok=True)

    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_results = {
        name: {k: to_serializable(v) for k, v in results.items()}
        for name, results in all_results.items()
    }

    with open(os.path.join(out_dir, 'lm_evaluation.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to {out_dir}/lm_evaluation.json")


if __name__ == '__main__':
    main()
