"""
Sampling/inference script for PhaseFlow.

Supports:
- Generating phase diagrams from amino acid sequences
- Generating sequences from phase diagrams
- Batch processing from CSV/text files
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from phaseflow import PhaseFlow, AminoAcidTokenizer, PhaseDataset
from phaseflow.data import PhaseDataset
from phaseflow.utils import (
    load_config,
    load_checkpoint,
    visualize_phase_diagram,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from PhaseFlow model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['seq2phase', 'phase2seq', 'interactive'],
        default='seq2phase',
        help="Sampling mode"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input sequence (for seq2phase) or CSV file path"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input file with sequences (one per line) or phase values (CSV)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.csv",
        help="Output file path"
    )
    parser.add_argument(
        "--method",
        type=str,
        default='euler',
        choices=['euler', 'dopri5', 'midpoint', 'rk4'],
        help="ODE solver method for phase generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for sequence generation"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize generated phase diagrams"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> tuple:
    """Load model from checkpoint.

    Returns:
        Tuple of (model, config, tokenizer)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or use defaults
    config = checkpoint.get('config', {
        'model': {
            'dim': 256,
            'depth': 6,
            'heads': 8,
            'dim_head': 32,
            'vocab_size': 64,
            'phase_dim': 16,
            'max_seq_len': 64,
            'dropout': 0.0,
        }
    })

    # Create model
    model = PhaseFlow(
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        dim_head=config['model']['dim_head'],
        vocab_size=config['model']['vocab_size'],
        phase_dim=config['model']['phase_dim'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=0.0,  # No dropout during inference
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create tokenizer
    tokenizer = AminoAcidTokenizer()

    return model, config, tokenizer


@torch.no_grad()
def generate_phase_from_sequences(
    model: PhaseFlow,
    tokenizer: AminoAcidTokenizer,
    sequences: List[str],
    method: str = 'euler',
    batch_size: int = 32,
    device: str = 'cuda',
    max_seq_len: int = 64,
) -> np.ndarray:
    """Generate phase diagrams from sequences.

    Args:
        model: PhaseFlow model
        tokenizer: Tokenizer
        sequences: List of amino acid sequences
        method: ODE solver method ('euler', 'dopri5', 'midpoint', 'rk4')
        batch_size: Batch size
        device: Device
        max_seq_len: Maximum sequence length

    Returns:
        (N, 16) array of phase diagrams
    """
    model.eval()
    all_phases = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Generating phases"):
        batch_seqs = sequences[i:i + batch_size]

        # Encode sequences
        input_ids = tokenizer.batch_encode(
            batch_seqs,
            max_len=max_seq_len,
            return_tensors=True
        ).to(device)

        # Create attention mask
        attention_mask = (input_ids != tokenizer.PAD_ID).long()

        # Get sequence lengths
        seq_lens = attention_mask.sum(dim=1)

        # Generate phase diagrams
        phase = model.generate_phase(
            input_ids, attention_mask, seq_lens,
            method=method
        )

        all_phases.append(phase.cpu().numpy())

    return np.concatenate(all_phases, axis=0)


@torch.no_grad()
def generate_sequences_from_phases(
    model: PhaseFlow,
    tokenizer: AminoAcidTokenizer,
    phases: np.ndarray,
    max_len: int = 25,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    batch_size: int = 32,
    device: str = 'cuda',
) -> List[str]:
    """Generate sequences from phase diagrams.

    Args:
        model: PhaseFlow model
        tokenizer: Tokenizer
        phases: (N, 16) array of phase diagrams
        max_len: Maximum sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        batch_size: Batch size
        device: Device

    Returns:
        List of generated sequences
    """
    model.eval()
    all_sequences = []

    for i in tqdm(range(0, len(phases), batch_size), desc="Generating sequences"):
        batch_phases = phases[i:i + batch_size]
        phase_tensor = torch.tensor(batch_phases, dtype=torch.float32, device=device)

        # Generate sequences
        _, decoded = model.generate_sequence(
            phase_tensor,
            tokenizer,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        all_sequences.extend(decoded)

    return all_sequences


def interactive_mode(model, tokenizer, config, device, args):
    """Interactive sampling mode."""
    print("\n=== PhaseFlow Interactive Mode ===")
    print("Commands:")
    print("  seq2phase <sequence> - Generate phase diagram from sequence")
    print("  phase2seq <values>   - Generate sequence from phase (comma-separated)")
    print("  quit                 - Exit")
    print()

    max_seq_len = config['model']['max_seq_len']

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            break

        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()

        if command == 'seq2phase' and len(parts) > 1:
            sequence = parts[1].upper().strip()

            # Validate sequence
            valid_aa = set(AminoAcidTokenizer.AMINO_ACIDS)
            if not all(aa in valid_aa for aa in sequence):
                print(f"Error: Invalid amino acids in sequence")
                continue

            print(f"Generating phase diagram for: {sequence}")

            # Generate
            phases = generate_phase_from_sequences(
                model, tokenizer, [sequence],
                method=args.method,
                device=device,
                max_seq_len=max_seq_len,
            )

            phase = phases[0]
            print("\nGenerated phase diagram (4x4):")
            phase_grid = phase.reshape(4, 4)
            for row in phase_grid:
                print("  " + "  ".join(f"{v:6.3f}" for v in row))

            if args.visualize:
                visualize_phase_diagram(phase, title=f"Phase: {sequence[:10]}...")

        elif command == 'phase2seq' and len(parts) > 1:
            try:
                values = [float(v) for v in parts[1].split(',')]
                if len(values) != 16:
                    print("Error: Need exactly 16 comma-separated values")
                    continue
            except ValueError:
                print("Error: Could not parse phase values")
                continue

            print("Generating sequence from phase diagram...")

            phases = np.array([values], dtype=np.float32)
            sequences = generate_sequences_from_phases(
                model, tokenizer, phases,
                max_len=25,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )

            print(f"Generated sequence: {sequences[0]}")

        else:
            print("Unknown command. Type 'quit' to exit.")


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading model from {args.checkpoint}...")
    model, config, tokenizer = load_model(args.checkpoint, args.device)
    print("Model loaded successfully!")

    max_seq_len = config['model']['max_seq_len']

    if args.mode == 'interactive':
        interactive_mode(model, tokenizer, config, args.device, args)

    elif args.mode == 'seq2phase':
        # Sequence to phase diagram
        if args.input:
            sequences = [args.input]
        elif args.input_file:
            with open(args.input_file, 'r') as f:
                sequences = [line.strip() for line in f if line.strip()]
        else:
            print("Error: Provide --input or --input_file")
            return

        print(f"Generating phase diagrams for {len(sequences)} sequences...")
        phases = generate_phase_from_sequences(
            model, tokenizer, sequences,
            method=args.method,
            batch_size=args.batch_size,
            device=args.device,
            max_seq_len=max_seq_len,
        )

        # Save results
        columns = [f'group_{i//4+1}{i%4+1}' for i in range(16)]
        df = pd.DataFrame(phases, columns=columns)
        df.insert(0, 'AminoAcidSequence', sequences)
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

        # Visualize if requested
        if args.visualize and len(sequences) <= 10:
            output_dir = Path(args.output).parent / "visualizations"
            output_dir.mkdir(exist_ok=True)
            for i, (seq, phase) in enumerate(zip(sequences, phases)):
                save_path = str(output_dir / f"phase_{i}_{seq[:5]}.png")
                visualize_phase_diagram(
                    phase,
                    title=f"{seq[:15]}...",
                    save_path=save_path
                )
            print(f"Visualizations saved to {output_dir}")

    elif args.mode == 'phase2seq':
        # Phase diagram to sequence
        if args.input_file:
            df = pd.read_csv(args.input_file)
            # Assume CSV has group_XX columns
            phase_cols = [c for c in df.columns if c.startswith('group_')]
            phases = df[phase_cols].values.astype(np.float32)
        else:
            print("Error: Provide --input_file with phase values")
            return

        print(f"Generating sequences for {len(phases)} phase diagrams...")
        sequences = generate_sequences_from_phases(
            model, tokenizer, phases,
            max_len=25,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            batch_size=args.batch_size,
            device=args.device,
        )

        # Save results
        df_out = pd.DataFrame({
            'GeneratedSequence': sequences,
        })
        # Include original phase values
        for i, col in enumerate(phase_cols):
            df_out[col] = phases[:, i]
        df_out.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
