# PhaseFlow

Transfusion-based model for bidirectional prediction between amino acid sequences and phase diagrams using Flow Matching.

## Overview

PhaseFlow is a unified bidirectional model that can:
- **Forward**: Generate phase diagrams (4x4 grids) from amino acid sequences using Flow Matching
- **Backward**: Generate amino acid sequences from phase diagrams using language modeling

## Features

- **Unified Architecture**: Single model handles both directions
- **Flow Matching**: Uses CondOT path for smooth phase diagram generation
- **Transformer Backbone**: 6-layer transformer with rotary embeddings
- **Masked Training**: Handles missing PSSI values in phase diagrams
- **Efficient**: Small model (256 dim, ~7M parameters) for fast iteration

## Project Structure

```
PhaseFlow/
├── phaseflow/
│   ├── __init__.py         # Package initialization
│   ├── model.py            # Main PhaseFlow model
│   ├── transformer.py      # Transformer backbone
│   ├── data.py            # Dataset and DataLoader
│   ├── tokenizer.py       # Amino acid tokenizer
│   └── utils.py           # Utility functions
├── train.py               # Training script
├── sample.py              # Inference/sampling script
├── config/
│   └── default.yaml       # Default configuration
└── requirements.txt       # Dependencies
```

## Installation

1. Install dependencies:
```bash
cd /data4/huangyanjie/LLPS/predictor/PhaseFlow
pip install -r requirements.txt
```

2. Verify installation:
```bash
python -c "from phaseflow import PhaseFlow; print('PhaseFlow installed successfully!')"
```

## Usage

### Training

Train the model on your dataset:

```bash
python train.py \
    --data_path /data4/huangyanjie/LLPS/phase_diagram/phase_diagram.csv \
    --output_dir outputs \
    --config config/default.yaml \
    --batch_size 64 \
    --epochs 100
```

Key arguments:
- `--data_path`: Path to phase diagram CSV file
- `--output_dir`: Directory to save checkpoints and logs
- `--config`: Configuration file path
- `--resume`: Resume from checkpoint
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-4)
- `--epochs`: Number of epochs (default: 100)

### Inference

#### 1. Generate Phase Diagrams from Sequences

Single sequence:
```bash
python sample.py \
    --checkpoint outputs/run_xxx/best_model.pt \
    --mode seq2phase \
    --input "ACDEFGHIKLMNPQRSTVWY" \
    --output predictions.csv
```

Multiple sequences from file:
```bash
python sample.py \
    --checkpoint outputs/run_xxx/best_model.pt \
    --mode seq2phase \
    --input_file sequences.txt \
    --output predictions.csv \
    --batch_size 64
```

#### 2. Generate Sequences from Phase Diagrams

```bash
python sample.py \
    --checkpoint outputs/run_xxx/best_model.pt \
    --mode phase2seq \
    --input_file phase_values.csv \
    --output generated_sequences.csv \
    --temperature 1.0
```

#### 3. Interactive Mode

```bash
python sample.py \
    --checkpoint outputs/run_xxx/best_model.pt \
    --mode interactive
```

Then use commands:
- `seq2phase ACDEF` - Generate phase diagram from sequence
- `phase2seq 1.0,-1.0,0.5,...` - Generate sequence from phase (16 values)
- `quit` - Exit

### Visualization

Visualize generated phase diagrams:
```bash
python sample.py \
    --checkpoint outputs/run_xxx/best_model.pt \
    --mode seq2phase \
    --input "ACDEFGHIKLMN" \
    --output output.csv \
    --visualize
```

## Configuration

Edit `config/default.yaml` to customize:

```yaml
model:
  dim: 256              # Model dimension
  depth: 6              # Number of layers
  heads: 8              # Attention heads
  phase_dim: 16         # Phase diagram size (4x4)

training:
  batch_size: 64
  lr: 1e-4
  epochs: 100
  flow_loss_weight: 1.0 # Weight for flow matching loss
  lm_loss_weight: 1.0   # Weight for language modeling loss

sampling:
  ode_steps: 20         # ODE integration steps
  temperature: 1.0      # Sampling temperature
```

## Architecture Details

### Model Architecture

```
[SOS] [amino tokens...] [META] [shape] [SOM] [phase tokens...] [EOM] [EOS]
```

- **Embedding**: Token embeddings for amino acids, projection for phase diagrams
- **Time Encoding**: Sinusoidal time embeddings added to phase tokens (local addition)
- **Transformer**: 6 layers with rotary embeddings and RMS normalization
- **Attention**: Causal for sequence tokens, bidirectional within phase tokens
- **Heads**:
  - Language modeling head for sequence generation
  - Velocity head for flow matching

### Flow Matching

Uses CondOT path:
```python
x_t = (1-t) * x_0 + t * x_1
velocity = x_1 - x_0
```

- **Training**: Predict velocity at random timesteps
- **Inference**: ODE integration with Euler solver (20 steps)

### Loss Functions

1. **Flow Loss**: MSE on velocity prediction (masked for missing values)
2. **LM Loss**: Cross-entropy for next token prediction
3. **Total Loss**: Weighted sum of both losses

## Data Format

Input CSV should have:
- `AminoAcidSequence`: Amino acid sequence (5-20 residues)
- `group_11` to `group_44`: Phase diagram values (16 columns for 4x4 grid)
- Missing values: Empty cells or NaN

Example:
```csv
AminoAcidSequence,group_11,group_12,group_13,group_14,...
ACDEFGHIKL,1.0,-1.0,0.5,0.3,...
```

## Performance Tips

1. **Mixed Precision**: Enabled by default (`use_amp: true`)
2. **Gradient Clipping**: Prevents exploding gradients
3. **Learning Rate Schedule**: Warmup + cosine annealing
4. **Early Stopping**: Stops when validation loss plateaus
5. **Batch Size**: Increase for faster training (if GPU memory allows)

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `dim` or `depth` in config
- Enable gradient checkpointing (modify model.py)

### Poor Convergence
- Increase `warmup_steps`
- Adjust learning rate
- Check data normalization

### Missing Values
Model handles missing values by masking. Check `phase_mask` in outputs.

## Citation

If you use PhaseFlow, please cite:
```
[Add citation information here]
```

## License

[Add license information here]

## Contact

For questions or issues, please open an issue on the GitHub repository.
