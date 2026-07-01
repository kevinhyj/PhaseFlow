# Data Directory

This directory is reserved for small example data and documentation of external datasets.

Large LLPS phase-diagram CSV/NPZ files should not be committed to this repository. Pass their
paths explicitly to scripts, for example:

```bash
bash scripts/train.sh \
  --data /path/to/phase_diagram_original_scale.csv \
  --val /path/to/val_set.csv \
  --test /path/to/test_set.csv
```

Expected CSV columns:

- `AminoAcidSequence`
- `group_11` through `group_44` for the 4x4 PSSI phase diagram

For missing-threshold training, place split files under `by_missing/` with names such as
`missing_0.csv`, `missing_1.csv`, ..., `missing_15.csv`.
