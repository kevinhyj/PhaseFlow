# Peptide Script Reference

Shell launchers in `scripts/peptide/` are convenience wrappers around the
installed PhaseFlow package. They resolve repository-relative paths and expose
common options. Review the launcher arguments before using them in a recorded
experiment.

## Training

```bash
bash scripts/peptide/train.sh \
  --config configs/peptide/peptide.yaml \
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
`scripts/peptide/workflows/predict_seq2phase.py`. It accepts text or CSV
sequence inputs and writes a CSV prediction table.

## Experiment Launchers

The remaining shell scripts launch predefined sweeps:

- `train_missing15_grid.sh` for missing-value threshold studies.
- `train_scaling.sh` for model-size studies.
- `run_ddpm_all.sh` for DDPM configuration studies.
- `kill_training.sh` for interactive termination of launchers started locally.

Review every shell script before use. Sweep scripts are research conveniences;
they do not replace a recorded experimental protocol.

## Mutation Figures

The mutation figure generators accept explicit, tabular inputs and write PNG,
PDF, SVG, and the exact plot-data CSV used to render each figure.

```bash
python scripts/peptide/figures/mutation/plot_mutation_metrics.py \
  --input artifacts/results/peptide/mutation_metrics.csv \
  --output-dir runs/figures/mutation_metrics

python scripts/peptide/figures/mutation/plot_mutation_scatter.py \
  --input artifacts/results/peptide/mutation_effects.csv \
  --output-dir runs/figures/mutation_scatter

python scripts/peptide/figures/mutation/plot_multi_mutation_dose.py \
  --input artifacts/results/peptide/multi_mutation_scores.csv \
  --output-dir runs/figures/multi_mutation_dose
```

`plot_mutation_metrics.py` requires `model`, `mean_auroc`, and `mean_auprc`.
`plot_mutation_scatter.py` requires `experimental_effect` plus the documented
comparator score columns.
`plot_multi_mutation_dose.py` requires `mutation`, `mutation_count`, and
`official_score`.
