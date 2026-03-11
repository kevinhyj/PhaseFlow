#!/usr/bin/env python3
"""Generate sequences from 4 flow5 models and plot length distribution histograms."""
import sys
sys.path.insert(0, '/data/yanjie_huang/LLPS/predictor/PhaseFlow_WJX_Test')

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phaseflow import PhaseFlow, AminoAcidTokenizer

MODELS = {
    'missing0': 'outputs/output_flow5_missing0_20260310/best_model.pt',
    'missing3': 'outputs/output_flow5_missing3_20260310/best_model.pt',
    'missing7': 'outputs/output_flow5_missing7_20260310/best_model.pt',
    'missing11': 'outputs/output_flow5_missing11_20260310/best_model.pt',
}

BASE_DIR = '/data/yanjie_huang/LLPS/predictor/PhaseFlow_WJX_Test'
NUM_SEQUENCES = 1000
BATCH_SIZE = 100
device = torch.device('cuda:7')
tokenizer = AminoAcidTokenizer()

# Target phase diagram (strong LLPS)
target_phase = torch.tensor([[-1.5] * 16], dtype=torch.float32, device=device)

results = {}

for name, ckpt_path in MODELS.items():
    print(f"\n=== {name} ===")
    full_path = f"{BASE_DIR}/{ckpt_path}"
    checkpoint = torch.load(full_path, map_location=device, weights_only=False)
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
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    all_seqs = []
    for i in range(0, NUM_SEQUENCES, BATCH_SIZE):
        batch_n = min(BATCH_SIZE, NUM_SEQUENCES - i)
        phase_batch = target_phase.repeat(batch_n, 1)
        with torch.no_grad():
            _, seqs = model.generate_sequence(phase_batch, tokenizer, max_len=32, temperature=1.0)
        all_seqs.extend(seqs)
        print(f"  Generated {len(all_seqs)}/{NUM_SEQUENCES}")

    lengths = [len(s) for s in all_seqs]
    results[name] = lengths
    print(f"  Mean length: {np.mean(lengths):.1f}, >20aa: {100*sum(1 for l in lengths if l > 20)/len(lengths):.1f}%")

    del model
    torch.cuda.empty_cache()

# Load real data length distribution
print("\nLoading real data distribution...")
real_df = pd.read_csv('/data/yanjie_huang/LLPS/phase_diagram/phase_diagram_original_scale.csv')
real_lengths = real_df['AminoAcidSequence'].str.len().values
print(f"Real data: {len(real_lengths)} sequences, mean={np.mean(real_lengths):.1f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

for idx, (name, lengths) in enumerate(results.items()):
    ax = axes[idx // 2][idx % 2]
    # Background: real data distribution (normalized to match generated count)
    ax.hist(real_lengths, bins=range(1, 35), color='gray', alpha=0.3, edgecolor='gray', linewidth=0.3,
            weights=np.ones(len(real_lengths)) * len(lengths) / len(real_lengths), label='Real data')
    # Foreground: generated distribution
    ax.hist(lengths, bins=range(1, 35), color=colors[idx], alpha=0.7, edgecolor='black', linewidth=0.5, label='Generated')
    mean_len = np.mean(lengths)
    gt20_pct = 100 * sum(1 for l in lengths if l > 20) / len(lengths)
    ax.axvline(x=20, color='red', linestyle='--', linewidth=1.5, label='20aa limit')
    ax.axvline(x=mean_len, color='orange', linestyle='--', linewidth=1.5, label=f'Mean={mean_len:.1f}')
    ax.set_title(f'Flow5 {name}\nMean={mean_len:.1f}, >20aa={gt20_pct:.1f}%', fontsize=12)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 34)

plt.suptitle('Flow5 Generated Sequence Length Distribution (N=1000)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE_DIR}/infer/flow5_length_distribution.png', dpi=150, bbox_inches='tight')
print(f"\nSaved to {BASE_DIR}/infer/flow5_length_distribution.png")
