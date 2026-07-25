# Peptide Script Reference

Shell launchers in `scripts/peptide/` are convenience wrappers around Python
entry points in `research/peptide/experiments/`. They resolve repository-relative
paths, set `PYTHONPATH`, and expose common options. Use the Python entry points
directly when integrating PhaseFlow into another workflow.

## Training

```bash
bash scripts/peptide/train.sh \
  --config configs/peptide/default.yaml \
  --data artifacts/data/peptide/phase_diagram_original_scale.csv \
  --val path/to/validation.csv \
  --test path/to/test.csv \
  --output-dir runs/peptide \
  --foreground
```

| Option | Meaning |
| --- | --- |
| `--config` | YAML configuration path, or a file name under `configs/peptide/` |
| `--data` | Required source CSV unless `PHASEFLOW_DATA_PATH` is set |
| `--val`, `--test` | Optional explicit split files |
| `--output-dir` | Directory for checkpoints and run outputs |
| `--batch`, `--lr`, `--epochs` | Training overrides recorded by the trainer |
| `--threshold` | Missing-value threshold mode; use only with `by_missing/` inputs |
| `--foreground` | Stream logs in the current terminal instead of starting `nohup` |

## Resume Training

```bash
bash scripts/peptide/resume.sh 0 runs/peptide/best_model.pt
```

The launcher reads `PHASEFLOW_CONFIG`, `PHASEFLOW_DATA_PATH`, and
`PHASEFLOW_OUTPUT_DIR` when set. These variables are optional overrides, not
repository defaults.

## Sequence-To-Phase Inference

```bash
bash scripts/peptide/infer.sh \
  artifacts/models/peptide/model.pt \
  examples/sequences.txt \
  runs/peptide/predicted_phases.csv \
  0
```

The underlying entry point is
`research/peptide/experiments/predict_seq2phase.py`. It accepts text or CSV
sequence inputs and writes a CSV prediction table.

## Experiment Launchers

The remaining shell scripts launch predefined sweeps:

- `train_missing15_grid.sh` for missing-value threshold studies.
- `train_scaling.sh` for model-size studies.
- `run_ddpm_all.sh` for DDPM configuration studies.
- `kill_training.sh` for interactive termination of launchers started locally.

Review every shell script before use. Sweep scripts are research conveniences;
they do not replace a recorded experimental protocol.
