#!/usr/bin/env python3
"""Small phase-to-sequence generation demo."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phaseflow import AminoAcidTokenizer


PHASE_COLUMNS = [f"group_{i}{j}" for i in range(1, 5) for j in range(1, 5)]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sequences from target phase diagrams.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained PhaseFlow checkpoint.")
    parser.add_argument("--input_csv", default="artifacts/data/peptide/test_set.csv", help="CSV containing phase diagram columns.")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    from phaseflow import PhaseFlow

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    model = PhaseFlow(
        dim=model_config.get("dim", 256),
        depth=model_config.get("depth", 6),
        heads=model_config.get("heads", 8),
        dim_head=model_config.get("dim_head", 32),
        vocab_size=model_config.get("vocab_size", 32),
        phase_dim=model_config.get("phase_dim", 16),
        max_seq_len=model_config.get("max_seq_len", 32),
        dropout=0.0,
        use_set_encoder=model_config.get("use_set_encoder", False),
        diffusion_type=model_config.get("diffusion_type", "flow_matching"),
        num_timesteps=model_config.get("num_timesteps", 1000),
        beta_schedule=model_config.get("beta_schedule", "cosine"),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model.to(device).eval()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint)
    input_csv = Path(args.input_csv)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not input_csv.is_absolute():
        input_csv = root / input_csv

    model = load_model(str(checkpoint), args.device)
    tokenizer = AminoAcidTokenizer()

    df = pd.read_csv(input_csv).head(args.num_samples)
    phase_values = df[PHASE_COLUMNS].fillna(0.0).values
    phase_tensor = torch.tensor(phase_values, dtype=torch.float32, device=args.device)

    with torch.no_grad():
        _, generated = model.generate_sequence(
            phase_tensor,
            tokenizer,
            max_len=args.max_len,
            temperature=args.temperature,
        )

    for idx, sequence in enumerate(generated, start=1):
        print(f"{idx}\t{sequence}")


if __name__ == "__main__":
    main()
