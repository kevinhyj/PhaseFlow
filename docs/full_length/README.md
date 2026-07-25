# Full-Length PhaseFlow

This directory documents the full-length protein workflow: protein-level LLPS
prediction, residue-level DPR prediction, data preparation, evaluation, and
publication figures.

## Public Layout

- `configs/full_length/llps.yaml`: LLPS training configuration.
- `configs/full_length/dpr.yaml`: DPR training configuration.
- `scripts/full_length/train_llps.py`: LLPS training entry point.
- `scripts/full_length/train_dpr.py`: DPR training entry point.
- `scripts/full_length/data/build_dataset.py`: validates a release dataset and
  writes a checksum manifest.
- `scripts/full_length/data/build_region_targets.py`: prepares DPR region
  targets from teacher profiles.
- `scripts/full_length/figures/`: benchmark, ablation, example-profile, and
  architecture figure generators.

## Artifact Contract

The source repository contains no model checkpoints, feature caches, profile
archives, or training tables. Two kinds of Hugging Face artifact are needed:
source data packages and regenerated training artifacts. Keep source packages
under `artifacts/data/full_length/`, derived feature/packing artifacts under
`artifacts/derived/full_length/`, and checkpoints under
`artifacts/models/full_length/`. The data packages use this layout:

```text
PhaseFlow-LLPS/
  data/proteins.parquet
  data/training_units.parquet
  benchmark/
  configs/
  metadata/
PhaseFlow-DPR/
  data/proteins.parquet
  data/training_units.parquet
  data/base_training_schedule.parquet
  data/region_supervision.parquet
  configs/
  metadata/
```

Validate a package before feature reconstruction or training:

```bash
python scripts/full_length/data/build_dataset.py \
  --task llps \
  --package-root artifacts/data/full_length/PhaseFlow-LLPS \
  --output runs/data/llps_manifest.json
```

See `reproduction.md` for the command order and `artifact_policy.md` for files
that must remain outside Git.
