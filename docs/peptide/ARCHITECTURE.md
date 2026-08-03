# Short-Peptide Architecture

The peptide model is implemented by `phaseflow.peptide.model.PhaseFlow`. It combines a
shared causal Transformer backbone with two task-specific objectives: a
continuous phase-diagram objective and next-token prediction.

## Inputs And Representation

The `AminoAcidTokenizer` maps the 20 standard amino acids and six control
tokens into a vocabulary of 32 identifiers. A peptide input is represented as
a sequence with start and end markers. The model also receives a 16-value PSSI
vector and a binary validity mask for its 4x4 grid positions.

The conceptual multimodal layout is:

```text
[sequence tokens] [metadata markers] [phase tokens]
```

The Transformer receives causal attention for token generation while the phase
portion is encoded as a conditional modality. Exact token layout and masking
are defined by `PhaseFlow.forward_flow` and `PhaseFlow.forward_lm`.

## Phase Encoders

Two phase encoders are available.

| Encoder | Configuration | Behavior |
| --- | --- | --- |
| Linear encoder | `use_set_encoder: false` | Projects the complete 16-value vector to one phase token |
| Set encoder | `use_set_encoder: true` | Encodes each valid PSSI position as a token using value features and a learned positional embedding |

The set encoder multiplies invalid positions by the phase mask, so missing
PSSI values do not contribute a learned value representation.

## Sequence-To-Phase Objective

For `diffusion_type: flow_matching`, the model predicts a velocity field from
an interpolated noisy phase vector and integrates that field at inference time.
The trainer evaluates the phase loss only at valid grid positions. Optional
quadratic reliability weighting gives more influence to samples containing more
observed PSSI positions.

For `diffusion_type: ddpm`, the model initializes a configurable linear or
cosine noise schedule and uses the DDPM sampling path. The two modes share the
tokenization and Transformer backbone but use different phase-generation
procedures.

## Phase-To-Sequence Objective

The language-model head predicts the next amino-acid or control token from the
causal hidden states. During generation, `generate_sequence` starts from the
sequence marker and samples autoregressively until an end token or the supplied
maximum length is reached.

## Principal Configuration Fields

| Field | Meaning |
| --- | --- |
| `model.dim` | Hidden width of token, phase, and Transformer representations |
| `model.depth` | Number of Transformer blocks |
| `model.heads` and `model.dim_head` | Attention configuration |
| `model.phase_dim` | Number of PSSI positions; the public peptide data uses 16 |
| `model.max_seq_len` | Padded sequence length including tokenizer control tokens |
| `model.diffusion_type` | `flow_matching` or `ddpm` phase objective |
| `training.flow_loss_weight` | Weight applied to the phase-generation loss |
| `training.lm_loss_weight` | Weight applied to the language-model loss |

See `configs/peptide/peptide.yaml` for an executable baseline. Changes to
model dimensions, tokenization, or phase dimension invalidate checkpoint
compatibility and should be treated as a new experimental configuration.
