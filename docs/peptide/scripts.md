# Script Reference

Shell launchers live in `scripts/`. They are thin wrappers around Python entrypoints in
`experiments/`.

## Training

```bash
bash scripts/peptide/train.sh \
  --config configs/peptide/default.yaml \
  --data /path/to/phase_diagram_original_scale.csv \
  --val /path/to/val_set.csv \
  --test /path/to/test_set.csv \
  --gpu 0
```

Useful options:

| Option | Meaning |
|---|---|
| `--config` | YAML config path or filename under `configs/` |
| `--data` | Training CSV |
| `--val` | Optional validation CSV |
| `--test` | Optional test CSV |
| `--output-dir` | Output root for checkpoints |
| `--threshold` | Missing-value threshold for `by_missing/missing_*.csv` datasets |
| `--foreground` | Run in foreground instead of `nohup` background mode |

If `--val` and `--test` are omitted, `research/peptide/experiments/train.py` creates train/val/test splits from
`--data`.

## Resume Training

```bash
bash scripts/resume.sh 0 outputs/run_xxx/best_model.pt
```

Environment overrides:

| Variable | Meaning |
|---|---|
| `PHASEFLOW_CONFIG` | Config path |
| `PHASEFLOW_DATA_PATH` | Training CSV path |
| `PHASEFLOW_OUTPUT_DIR` | Output directory |

## Seq2Phase Inference

```bash
bash scripts/peptide/infer.sh outputs/run_xxx/best_model.pt examples/sequences.txt artifacts/results/peptide/predicted_phases.csv 0
```

The underlying Python entrypoint is `research/peptide/experiments/predict_seq2phase.py`.

## Batch Experiments

- `scripts/train_missing15_grid.sh`: launch a weight grid over missing-threshold 15 configs.
- `scripts/train_scaling.sh`: launch model-size scaling configs.
- `scripts/run_ddpm_all.sh`: launch DDPM ablation configs.
- `scripts/peptide/kill_training.sh`: interactively terminate training processes.
