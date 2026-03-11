"""
Step 1: PhaseFlow Sequence Generation

Generate 20,000 candidate sequences from target phase diagram using PhaseFlow LM.

Environment: phaseflow
Output: candidates_raw.csv (20,000 sequences)
"""

import sys
sys.path.insert(0, '.')

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from phaseflow import PhaseFlow, AminoAcidTokenizer
import os
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

CHECKPOINT_PATH = "outputs/missing_none_output_bs2048_lr0.0008_flow32_20260123/best_model.pt"
ORIGINAL_DATA = "/data/yanjie_huang/LLPS/phase_diagram/phase_diagram_original_scale.csv"
OUTPUT_DIR = "outputs/phaseflow_generation_v2"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidates_raw.csv")
STATS_FILE = os.path.join(OUTPUT_DIR, "generation_stats.txt")

# Generation parameters
TOTAL_SEQUENCES = 100000
BATCH_SIZE = 256
MAX_LEN = 20

# Sampling strategy - using real good phase diagrams
TOP_PERCENT = 5  # Top 5% good phase diagrams
NOISE_SCALE = 0.1  # Noise for augmentation

SAMPLING_STRATEGY = [
    {"type": "real", "count": 50000, "name": "real_good", "temperature": 1.0},      # 50% real samples
    {"type": "augmented", "count": 50000, "name": "augmented", "temperature": 1.0},  # 50% augmented
]

# ============================================================================
# Main Functions
# ============================================================================

def load_phaseflow_model(checkpoint_path, device):
    """Load PhaseFlow model from checkpoint."""
    print(f"Loading PhaseFlow model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    print(f"Model config: {config['model']}")

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
    model = model.to(device)
    model.eval()

    print("PhaseFlow model loaded successfully!")
    return model, config


def load_real_good_phase_diagrams(data_path, top_percent=5):
    """Load top N% good phase diagrams from original data."""
    print(f"Loading good phase diagrams from {data_path}...")

    df = pd.read_csv(data_path)
    pssi_cols = [c for c in df.columns if c.startswith('group_')]

    # Only keep rows with complete PSSI values
    df_valid = df.dropna(subset=pssi_cols)
    print(f"Valid samples with all 16 PSSI values: {len(df_valid)}")

    pssi_values = df_valid[pssi_cols].values
    mean_pssi = pssi_values.mean(axis=1)

    threshold = np.percentile(mean_pssi, top_percent)
    good_phases = pssi_values[mean_pssi <= threshold]

    print(f"Loaded {len(good_phases)} good phase diagrams (Top {top_percent}%, mean_pssi <= {threshold:.3f})")
    print(f"Phase mean range: [{good_phases.mean(axis=1).min():.3f}, {good_phases.mean(axis=1).max():.3f}]")

    return good_phases


def augment_phase_diagram(phase, noise_scale=0.1):
    """Augment phase diagram with Gaussian noise."""
    noise = np.random.normal(0, noise_scale, phase.shape)
    augmented = phase + noise
    # Clip to physical range
    augmented = np.clip(augmented, -2.17, 1.64)
    return augmented


def generate_sequences_batch(model, tokenizer, phase, batch_size, temperature, max_len, device):
    """Generate a batch of sequences from phase diagram."""
    phase_batch = phase.repeat(batch_size, 1).to(device)

    with torch.no_grad():
        _, generated_seqs = model.generate_sequence(
            phase=phase_batch,
            tokenizer=tokenizer,
            max_len=max_len,
            temperature=temperature,
            top_k=None,
            top_p=None,
        )

    return generated_seqs


def generate_all_sequences(model, tokenizer, good_phases, sampling_strategy, batch_size, max_len, device, noise_scale=0.1):
    """Generate all sequences according to sampling strategy using real phase diagrams."""
    all_sequences = []
    all_metadata = []
    all_target_phases = []

    np.random.seed(42)  # For reproducibility

    for strategy in sampling_strategy:
        strategy_type = strategy["type"]
        count = strategy["count"]
        name = strategy["name"]
        temperature = strategy.get("temperature", 1.0)

        print(f"\n{'='*60}")
        print(f"Generating {count} sequences with {name} ({strategy_type})")
        print(f"{'='*60}")

        num_batches = (count + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc=f"{name} generation"):
            # Select target phase for this batch
            if strategy_type == "real":
                # Use real phase diagram
                phase_idx = np.random.randint(0, len(good_phases))
                target_phase = good_phases[phase_idx].copy()
            elif strategy_type == "augmented":
                # Use augmented phase diagram
                phase_idx = np.random.randint(0, len(good_phases))
                target_phase = augment_phase_diagram(good_phases[phase_idx], noise_scale)
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")

            phase_tensor = torch.tensor([target_phase], dtype=torch.float32)

            sequences = generate_sequences_batch(
                model, tokenizer, phase_tensor, batch_size,
                temperature, max_len, device
            )

            all_sequences.extend(sequences)
            all_metadata.extend([{
                "temperature": temperature,
                "strategy": name,
                "batch_idx": batch_idx,
                "phase_idx": phase_idx if 'phase_idx' in locals() else -1
            }] * len(sequences))
            all_target_phases.extend([target_phase] * len(sequences))

    return all_sequences, all_metadata, all_target_phases


def save_results(sequences, metadata, target_phases, output_file, stats_file):
    """Save generated sequences to CSV and statistics to text file."""
    print(f"\nSaving {len(sequences)} sequences to {output_file}...")

    # Create DataFrame
    df = pd.DataFrame({
        "sequence": sequences,
        "length": [len(s) for s in sequences],
        "temperature": [m["temperature"] for m in metadata],
        "strategy": [m["strategy"] for m in metadata],
        "batch_idx": [m["batch_idx"] for m in metadata],
    })

    # Add target phase columns (use mean for statistics)
    target_phases_array = np.array(target_phases)
    df["target_mean_pssi"] = target_phases_array.mean(axis=1)
    df["target_std_pssi"] = target_phases_array.std(axis=1)

    # Save CSV
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    # Generate statistics
    target_phases_array = np.array(target_phases)

    stats = []
    stats.append("="*60)
    stats.append("PhaseFlow Generation Statistics (V2 - Real Phase Diagrams)")
    stats.append("="*60)
    stats.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stats.append(f"Total sequences: {len(sequences)}")
    stats.append(f"Source: Top {TOP_PERCENT}% real phase diagrams from original data")
    stats.append("")

    stats.append("Target Phase Statistics:")
    stats.append(f"  Mean PSSI range: [{target_phases_array.mean(axis=1).min():.3f}, {target_phases_array.mean(axis=1).max():.3f}]")
    stats.append(f"  Mean PSSI median: {np.median(target_phases_array.mean(axis=1)):.3f}")
    stats.append(f"  Std range: [{target_phases_array.std(axis=1).min():.3f}, {target_phases_array.std(axis=1).max():.3f}]")
    stats.append("")

    stats.append("Sampling Strategy:")
    for strategy_name in df["strategy"].unique():
        count = (df["strategy"] == strategy_name).sum()
        temp = df[df["strategy"] == strategy_name]["temperature"].iloc[0]
        stats.append(f"  {strategy_name} (T={temp}): {count} sequences")
    stats.append("")

    stats.append("Sequence Length Distribution:")
    stats.append(f"  Min: {df['length'].min()}")
    stats.append(f"  Max: {df['length'].max()}")
    stats.append(f"  Mean: {df['length'].mean():.2f}")
    stats.append(f"  Median: {df['length'].median():.0f}")
    stats.append("")

    stats.append("Unique Sequences:")
    unique_count = df["sequence"].nunique()
    stats.append(f"  Unique: {unique_count}")
    stats.append(f"  Duplicates: {len(sequences) - unique_count}")
    stats.append(f"  Uniqueness: {100 * unique_count / len(sequences):.2f}%")
    stats.append("")

    # Amino acid composition
    all_aa = ''.join(sequences)
    aa_counts = {aa: all_aa.count(aa) for aa in set(all_aa)}
    total_aa = len(all_aa)

    stats.append("Amino Acid Composition (Top 10):")
    sorted_aa = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for aa, count in sorted_aa:
        stats.append(f"  {aa}: {count} ({100*count/total_aa:.2f}%)")
    stats.append("")

    # Aromatic content
    aromatic = sum(all_aa.count(aa) for aa in ['F', 'W', 'Y'])
    stats.append(f"Aromatic content (F+W+Y): {100*aromatic/total_aa:.2f}%")
    stats.append(f"W content: {100*all_aa.count('W')/total_aa:.2f}%")
    stats.append("")

    stats.append("="*60)

    # Save statistics
    stats_text = '\n'.join(stats)
    with open(stats_file, 'w') as f:
        f.write(stats_text)

    print(f"Statistics saved to {stats_file}")
    print("\n" + stats_text)


def main():
    """Main execution function."""
    print("="*60)
    print("PhaseFlow Sequence Generation - Step 1 (V2)")
    print("Using Real Good Phase Diagrams from Original Data")
    print("="*60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    model, config = load_phaseflow_model(CHECKPOINT_PATH, device)
    tokenizer = AminoAcidTokenizer()

    # Load real good phase diagrams
    good_phases = load_real_good_phase_diagrams(ORIGINAL_DATA, TOP_PERCENT)

    # Generate sequences
    print(f"\nTarget: Generate {TOTAL_SEQUENCES} sequences")
    print(f"Strategy: {SAMPLING_STRATEGY}")
    print(f"Noise scale for augmentation: {NOISE_SCALE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max length: {MAX_LEN}")

    sequences, metadata, target_phases = generate_all_sequences(
        model, tokenizer, good_phases, SAMPLING_STRATEGY,
        BATCH_SIZE, MAX_LEN, device, NOISE_SCALE
    )

    # Save results
    save_results(sequences, metadata, target_phases, OUTPUT_FILE, STATS_FILE)

    print("\n" + "="*60)
    print("Step 1 Complete!")
    print(f"Generated {len(sequences)} sequences")
    print(f"Output: {OUTPUT_FILE}")
    print("="*60)


if __name__ == "__main__":
    main()
