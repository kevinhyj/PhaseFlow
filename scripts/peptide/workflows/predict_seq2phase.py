"""Predict 4x4 phase diagrams from amino acid sequences."""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from phaseflow import AminoAcidTokenizer


PHASE_COLUMNS = [f"group_{i}{j}" for i in range(1, 5) for j in range(1, 5)]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict phase diagrams from sequences.")
    parser.add_argument("--checkpoint", required=True, help="Path to a PhaseFlow checkpoint.")
    parser.add_argument("--input_file", required=True, help="Text or CSV file containing sequences.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--sequence_col", default="AminoAcidSequence", help="CSV sequence column name.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--method", default="euler", help="ODE method for flow matching models.")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Override max sequence length.")
    return parser.parse_args()


def read_sequences(path: str, sequence_col: str):
    input_path = Path(path)
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
        if sequence_col not in df.columns:
            raise ValueError(f"Column '{sequence_col}' not found in {input_path}")
        return df[sequence_col].dropna().astype(str).tolist()

    with open(input_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


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
    model = model.to(device).eval()
    return model, model_config


def main():
    args = parse_args()
    sequences = read_sequences(args.input_file, args.sequence_col)
    if not sequences:
        raise ValueError(f"No sequences found in {args.input_file}")

    model, model_config = load_model(args.checkpoint, args.device)
    tokenizer = AminoAcidTokenizer()
    max_seq_len = args.max_seq_len or model_config.get("max_seq_len", 32)

    rows = []
    for start in range(0, len(sequences), args.batch_size):
        batch = sequences[start:start + args.batch_size]
        input_ids = tokenizer.batch_encode(batch, max_len=max_seq_len, return_tensors=True).to(args.device)
        attention_mask = (input_ids != tokenizer.PAD_ID).long()
        seq_len = torch.tensor([len(tokenizer.build_input_sequence(seq)) for seq in batch], device=args.device)

        with torch.no_grad():
            if getattr(model, "diffusion_type", "flow_matching") == "ddpm":
                phase = model.generate_phase(input_ids, attention_mask, seq_len, num_steps=50, use_ddim=True)
            else:
                phase = model.generate_phase(input_ids, attention_mask, seq_len, method=args.method)

        for sequence, values in zip(batch, phase.cpu().tolist()):
            row = {"AminoAcidSequence": sequence}
            row.update({col: val for col, val in zip(PHASE_COLUMNS, values)})
            rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["AminoAcidSequence", *PHASE_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} predictions to {output_path}")


if __name__ == "__main__":
    main()
