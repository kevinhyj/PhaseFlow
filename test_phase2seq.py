"""Test PhaseFlow phase2seq generation."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from phaseflow import PhaseFlow, AminoAcidTokenizer

# 使用最佳模型
CHECKPOINT_PATH = "outputs/output_set_encoder_bs2048_lr0.0008_flow32_20260128/best_model.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("Loading model...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
config = checkpoint.get('config', {})

print(f"Config: {config}")

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

tokenizer = AminoAcidTokenizer()

# 测试1: 强相分离相图 (所有值都很低)
print("\n" + "="*60)
print("Test 1: Strong LLPS phase diagram (all low values)")
print("="*60)
phase_strong = torch.tensor([[-2.0] * 16], dtype=torch.float32, device=device)

with torch.no_grad():
    _, generated_seqs = model.generate_sequence(
        phase_strong, tokenizer, max_len=20, temperature=1.0
    )

print(f"Generated sequence: {generated_seqs[0]}")

# 测试2: 弱相分离相图 (所有值都很高)
print("\n" + "="*60)
print("Test 2: Weak LLPS phase diagram (all high values)")
print("="*60)
phase_weak = torch.tensor([[1.0] * 16], dtype=torch.float32, device=device)

with torch.no_grad():
    _, generated_seqs = model.generate_sequence(
        phase_weak, tokenizer, max_len=20, temperature=1.0
    )

print(f"Generated sequence: {generated_seqs[0]}")

# 测试3: 混合相图
print("\n" + "="*60)
print("Test 3: Mixed phase diagram")
print("="*60)
phase_mixed = torch.tensor([[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, -1.2, -0.8,
                              -0.3, 0.2, 0.7, 1.2, -1.0, -0.5, 0.0, 0.5]],
                            dtype=torch.float32, device=device)

with torch.no_grad():
    _, generated_seqs = model.generate_sequence(
        phase_mixed, tokenizer, max_len=20, temperature=1.0
    )

print(f"Generated sequence: {generated_seqs[0]}")

# 测试4: 多次采样同一相图 (检查随机性)
print("\n" + "="*60)
print("Test 4: Multiple samples from same phase diagram")
print("="*60)
phase_test = torch.tensor([[-1.8] * 16], dtype=torch.float32, device=device)

print("Generating 5 sequences:")
for i in range(5):
    with torch.no_grad():
        _, generated_seqs = model.generate_sequence(
            phase_test, tokenizer, max_len=20, temperature=1.0
        )
    print(f"  {i+1}. {generated_seqs[0]}")

print("\nDone!")
