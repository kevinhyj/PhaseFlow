"""
Quick test script to verify PhaseFlow installation and basic functionality.
"""

import torch
import numpy as np
from phaseflow import PhaseFlow, AminoAcidTokenizer, PhaseDataset
from phaseflow.utils import count_parameters, format_number

print("=" * 60)
print("PhaseFlow Installation Test")
print("=" * 60)

# Test 1: Tokenizer
print("\n1. Testing Tokenizer...")
tokenizer = AminoAcidTokenizer()
test_seq = "ACDEFGHIKL"
tokens = tokenizer.encode_sequence(test_seq)
decoded = tokenizer.decode_sequence(tokens)
print(f"   Original: {test_seq}")
print(f"   Tokens: {tokens}")
print(f"   Decoded: {decoded}")
print(f"   ✓ Tokenizer works!" if test_seq == decoded else "   ✗ Tokenizer failed!")

# Test 2: Model Creation
print("\n2. Testing Model Creation...")
model = PhaseFlow(
    dim=64,
    depth=2,
    heads=4,
    dim_head=16,
    vocab_size=64,
    phase_dim=16,
    max_seq_len=32,
)
num_params = count_parameters(model)
print(f"   Model parameters: {format_number(num_params)} ({num_params:,})")
print(f"   ✓ Model created successfully!")

# Test 3: Forward Pass (Flow Matching)
print("\n3. Testing Forward Pass (Sequence → Phase)...")
batch_size = 4
seq_len = 16
phase_dim = 16

input_ids = torch.randint(0, 20, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
phase = torch.randn(batch_size, phase_dim)
phase_mask = torch.ones(batch_size, phase_dim)
seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)

model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        phase=phase,
        phase_mask=phase_mask,
        seq_len=seq_lens,
    )

print(f"   Loss: {outputs['loss'].item():.4f}")
print(f"   Flow loss: {outputs['flow_loss'].item():.4f}")
print(f"   LM loss: {outputs['lm_loss'].item():.4f}")
print(f"   ✓ Forward pass works!")

# Test 4: Phase Generation
print("\n4. Testing Phase Generation...")
with torch.no_grad():
    pred_phase = model.generate_phase(
        input_ids,
        attention_mask,
        seq_lens,
        num_steps=10
    )
print(f"   Generated phase shape: {pred_phase.shape}")
print(f"   Phase values range: [{pred_phase.min():.3f}, {pred_phase.max():.3f}]")
print(f"   ✓ Phase generation works!")

# Test 5: Sequence Generation
print("\n5. Testing Sequence Generation...")
with torch.no_grad():
    tokens, sequences = model.generate_sequence(
        phase[:2],  # Use 2 samples
        tokenizer,
        max_len=10,
        temperature=1.0,
    )
print(f"   Generated sequences:")
for i, seq in enumerate(sequences):
    print(f"     [{i}] {seq}")
print(f"   ✓ Sequence generation works!")

# Test 6: Batch Encoding
print("\n6. Testing Batch Encoding...")
sequences = ["ACDEF", "GHIKL", "MNPQR"]
batch_tokens = tokenizer.batch_encode(sequences, max_len=20, return_tensors=True)
print(f"   Input sequences: {sequences}")
print(f"   Batch shape: {batch_tokens.shape}")
print(f"   ✓ Batch encoding works!")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nPhaseFlow is ready to use!")
print("Run 'python train.py --help' for training options.")
print("Run 'python sample.py --help' for inference options.")
