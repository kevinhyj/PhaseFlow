# Short-Peptide Workflow

The short-peptide PhaseFlow module learns a bidirectional relationship between
an amino-acid sequence and a 4x4 phase-separation score index (PSSI) diagram.
It supports sequence-to-phase prediction with Flow Matching or DDPM, and
phase-conditioned peptide generation with a causal language-model objective.

This workflow is separate from the protein LLPS and DPR pipeline. Its
public implementation lives in `phaseflow/`, while training and evaluation
entry points live in `scripts/peptide/workflows/`.

## Quick Start

Install the package and its runtime dependencies:

```bash
python -m pip install -e .
```

Place a peptide source table outside Git, for example at
`artifacts/data/peptide/phase_diagram_original_scale.csv`, then launch training:

```bash
bash scripts/peptide/train.sh \
  --config configs/peptide/peptide.yaml \
  --data artifacts/data/peptide/phase_diagram_original_scale.csv \
  --foreground
```

The launcher creates a time-stamped log and delegates optimization to
`scripts/peptide/workflows/train.py`. Pass `--help` to the launcher for all
available overrides.

## Data Contract

The source CSV requires one sequence column and 16 PSSI columns:

```text
AminoAcidSequence,group_11,group_12,...,group_44
ACDEFGHIKL,0.12,-0.08,...,0.31
```

`AminoAcidSequence` contains uppercase one-letter amino-acid sequences. The
`group_11` through `group_44` columns represent a row-major 4x4 PSSI grid.
Missing grid values may be encoded as `NaN`; the dataset loader derives a mask
and excludes missing positions from the phase loss.

When `phase_diagram.npz` is present beside the CSV, the loader can read the
preprocessed phase array while continuing to use the CSV for sequences. The
NPZ is an optional performance cache, not a source-of-truth replacement.

## Training

The canonical baseline is `configs/peptide/peptide.yaml`. It defines model,
training, sampling, and split defaults. Explicit validation and test tables can
be supplied with `--val` and `--test`; otherwise the Python trainer makes a
deterministic train/validation/test split according to the configuration.

```bash
python scripts/peptide/workflows/train.py \
  --config configs/peptide/peptide.yaml \
  --data_path artifacts/data/peptide/phase_diagram_original_scale.csv \
  --output_dir runs/peptide \
  --device cuda
```

Training outputs include the resolved configuration, checkpoints, metrics, and
learning curves. Keep them outside Git.

## Sequence-To-Phase Prediction

Prepare a text file with one sequence per line, or a CSV containing the
configured sequence column, then run:

```bash
python scripts/peptide/workflows/predict_seq2phase.py \
  --checkpoint artifacts/models/peptide/model.pt \
  --input_file examples/sequences.txt \
  --output runs/peptide/predicted_phases.csv \
  --device cuda
```

For Flow Matching models, `--method` selects the ODE solver. The output CSV
contains the input sequence and predicted values for all 16 PSSI positions.

## Phase-To-Sequence Generation

Use `examples/phase2seq_demo.py` for a minimal programmatic example. The model
generates tokens autoregressively from a supplied phase vector and phase mask.
Sampling controls such as temperature and maximum length should be recorded
with every generated candidate set.

## Documentation Map

- [Architecture](ARCHITECTURE.md)
- [Data loading and missing values](DATALOADER_OPTIMIZATION.md)
- [Script reference](scripts.md)
- [Technical overview](phaseflow_technical_overview.md)
- [Evaluation and analysis protocol](ANALYSIS.md)
- [Experimental record guidance](PHASEFLOW_EXPLORATION.md)
- [Documentation index](PHASEFLOW_INDEX.md)
