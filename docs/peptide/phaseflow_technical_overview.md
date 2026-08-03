# PhaseFlow Peptide Technical Overview

PhaseFlow is a multimodal peptide model that relates amino-acid sequences to
16-position PSSI phase diagrams. It is designed for two complementary tasks:

1. Predict a phase diagram from a peptide sequence.
2. Generate a peptide sequence conditioned on a phase diagram.

The public implementation is research software. Outputs are model predictions,
not experimental measurements or design guarantees.

## Core Components

| Component | Source | Responsibility |
| --- | --- | --- |
| Tokenizer | `phaseflow/tokenizer.py` | Encodes standard amino acids and control tokens |
| Dataset | `phaseflow/data.py` | Loads sequences, PSSI values, masks, and train/validation/test splits |
| Model | `phaseflow/model.py` | Shared Transformer, phase encoder, Flow Matching or DDPM head, and LM head |
| Trainer | `scripts/peptide/workflows/train.py` | Optimization, checkpointing, metrics, and learning-curve exports |
| Predictor | `scripts/peptide/workflows/predict_seq2phase.py` | Batched sequence-to-phase inference |

## Modeling Directions

### Sequence To Phase

The model encodes peptide tokens and a noisy or partial phase representation.
For Flow Matching, it predicts a velocity field over the 16 PSSI values and
integrates it using the requested ODE method. For DDPM, it follows the model's
configured noise schedule. Both paths honor the PSSI validity mask during
training loss computation.

### Phase To Sequence

The model conditions its causal Transformer on the phase representation and
predicts the next token. Generation proceeds autoregressively. Random sampling
settings influence diversity and should be saved alongside generated results.

## Missing PSSI Measurements

PSSI inputs may be incomplete. The data loader maintains a binary phase mask;
the model's set encoder zeroes masked positions and the trainer computes phase
loss only on observed entries. When quadratic weighting is enabled, examples
with more observed phase positions receive greater phase-loss weight.

## Checkpoint Compatibility

A checkpoint is compatible only with the architecture and tokenizer settings
used to train it. At minimum, preserve the configuration fields controlling
vocabulary size, phase dimension, hidden width, depth, attention dimensions,
maximum sequence length, phase encoder selection, and diffusion mode. Always
load a checkpoint with its saved configuration where possible.

## Recommended Reporting

For each run, report source-data identity, split method and seed, complete
configuration, training device and software versions, checkpoint selection
criterion, phase prediction metrics on valid PSSI positions, and generation
sampling settings. Avoid comparing metrics from different splits or different
observed-PSSI masks without stating the difference.
