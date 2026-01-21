"""
Example usage of PhaseFlow for programmatic inference.
"""

import torch
import numpy as np
from phaseflow import PhaseFlow, AminoAcidTokenizer
from phaseflow.utils import load_checkpoint, visualize_phase_diagram


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load a trained PhaseFlow model."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    # Create model
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

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def predict_phase_diagram(
    model: PhaseFlow,
    tokenizer: AminoAcidTokenizer,
    sequence: str,
    device: str = 'cuda',
    num_steps: int = 20,
) -> np.ndarray:
    """
    Predict phase diagram from amino acid sequence.

    Args:
        model: Trained PhaseFlow model
        tokenizer: AminoAcidTokenizer
        sequence: Amino acid sequence string
        device: Device to run on
        num_steps: ODE integration steps

    Returns:
        (16,) array of phase diagram values
    """
    model.eval()

    # Encode sequence
    tokens = tokenizer.build_input_sequence(sequence)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)

    # Sequence length
    seq_len = torch.tensor([len(tokens)], dtype=torch.long, device=device)

    # Generate phase diagram
    with torch.no_grad():
        phase = model.generate_phase(
            input_ids,
            attention_mask,
            seq_len,
            num_steps=num_steps
        )

    return phase.cpu().numpy()[0]


def predict_sequence(
    model: PhaseFlow,
    tokenizer: AminoAcidTokenizer,
    phase_diagram: np.ndarray,
    device: str = 'cuda',
    max_len: int = 25,
    temperature: float = 1.0,
) -> str:
    """
    Predict amino acid sequence from phase diagram.

    Args:
        model: Trained PhaseFlow model
        tokenizer: AminoAcidTokenizer
        phase_diagram: (16,) array of phase values
        device: Device to run on
        max_len: Maximum sequence length
        temperature: Sampling temperature

    Returns:
        Predicted amino acid sequence
    """
    model.eval()

    # Convert to tensor
    phase_tensor = torch.tensor(
        phase_diagram[np.newaxis, :],
        dtype=torch.float32,
        device=device
    )

    # Generate sequence
    with torch.no_grad():
        _, sequences = model.generate_sequence(
            phase_tensor,
            tokenizer,
            max_len=max_len,
            temperature=temperature
        )

    return sequences[0]


def main():
    """Example usage."""
    # Configuration
    checkpoint_path = "outputs/run_xxx/best_model.pt"  # Update this path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model from {checkpoint_path}...")
    print(f"Using device: {device}\n")

    # Load model
    try:
        model, config = load_model(checkpoint_path, device)
    except FileNotFoundError:
        print("Checkpoint not found. Please train a model first using:")
        print("  python train.py --data_path /path/to/data.csv")
        return

    # Create tokenizer
    tokenizer = AminoAcidTokenizer()

    # Example 1: Predict phase diagram from sequence
    print("=" * 60)
    print("Example 1: Sequence → Phase Diagram")
    print("=" * 60)

    test_sequence = "ACDEFGHIKLMNPQRST"
    print(f"Input sequence: {test_sequence}")

    phase = predict_phase_diagram(
        model, tokenizer, test_sequence, device, num_steps=20
    )

    print(f"\nPredicted phase diagram (4x4):")
    phase_grid = phase.reshape(4, 4)
    for i, row in enumerate(phase_grid):
        print(f"  Row {i+1}: " + "  ".join(f"{v:7.3f}" for v in row))

    # Visualize (optional)
    try:
        visualize_phase_diagram(
            phase,
            title=f"Phase: {test_sequence}",
            save_path="example_phase.png"
        )
        print("\nVisualization saved to example_phase.png")
    except ImportError:
        print("\nInstall matplotlib to visualize: pip install matplotlib")

    # Example 2: Predict sequence from phase diagram
    print("\n" + "=" * 60)
    print("Example 2: Phase Diagram → Sequence")
    print("=" * 60)

    # Use the phase we just generated
    print("Using the phase diagram from Example 1...")

    predicted_seq = predict_sequence(
        model, tokenizer, phase, device, max_len=25, temperature=1.0
    )

    print(f"\nOriginal sequence:  {test_sequence}")
    print(f"Generated sequence: {predicted_seq}")

    # Compare
    if test_sequence == predicted_seq:
        print("✓ Perfect reconstruction!")
    else:
        print(f"Similarity: {sum(a == b for a, b in zip(test_sequence, predicted_seq))}/{len(test_sequence)} matches")

    # Example 3: Batch prediction
    print("\n" + "=" * 60)
    print("Example 3: Batch Prediction")
    print("=" * 60)

    sequences = [
        "ACDEFGHIKL",
        "MNPQRSTVWY",
        "FGHIKLMNPQ",
    ]

    print("Predicting phase diagrams for multiple sequences...")
    for seq in sequences:
        phase = predict_phase_diagram(model, tokenizer, seq, device)
        print(f"\n{seq}:")
        print(f"  Mean: {phase.mean():.3f}, Std: {phase.std():.3f}")
        print(f"  Range: [{phase.min():.3f}, {phase.max():.3f}]")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
