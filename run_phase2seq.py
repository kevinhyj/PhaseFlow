import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import pandas as pd
from phaseflow import PhaseFlow, AminoAcidTokenizer

CHECKPOINT_PATH = "outputs/missing11_output_bs2048_lr0.0008_flow32_20260123/best_model.pt"
DATA_PATH = "/data/yanjie_huang/LLPS/phase_diagram/by_missing/missing_11.csv"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("Loading model...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
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
model = model.to(device)
model.eval()
print("Model loaded!")

print("Loading data...")
df = pd.read_csv(DATA_PATH)
GROUP_COLS = [f'group_{i}{j}' for i in range(1, 5) for j in range(1, 5)]
samples = df.head(5)
phase_data = samples[GROUP_COLS].fillna(0).values

tokenizer = AminoAcidTokenizer()

print("\n" + "="*60)
print("Phase Diagram -> Sequence Inference")
print("="*60)

for i, (idx, row) in enumerate(samples.iterrows()):
    true_seq = row['AminoAcidSequence']
    phase = phase_data[i:i+1]
    phase_tensor = torch.tensor(phase, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Create phase_mask (all valid, no missing)
        phase_mask = torch.ones(phase_tensor.shape, dtype=torch.long, device=device)
        _, generated_seqs = model.generate_sequence(
            phase_tensor, tokenizer, max_len=20, temperature=1.0
        )

    generated_seq = generated_seqs[0]

    print(f"\n[Sample {i+1}]")
    print(f"  True:      {true_seq}")
    print(f"  Generated: {generated_seq}")

    min_len = min(len(true_seq), len(generated_seq))
    matches = sum(1 for a, b in zip(true_seq[:min_len], generated_seq[:min_len]) if a == b)
    print(f"  Similarity: {matches}/{min_len} ({100*matches/min_len:.1f}%)")

print("\nDone!")
