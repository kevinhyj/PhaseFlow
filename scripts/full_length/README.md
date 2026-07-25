# Full-Length Scripts

## Training

- `train_llps.py`: full-length LLPS training entry point.
- `train_dpr.py`: full-length DPR training entry point.

## Data

- `data/build_dataset.py`: validate a release package and write a checksum
  manifest.
- `data/build_region_targets.py`: create DPR region supervision from teacher
  profiles supplied explicitly from the released artifact collection.

## Figures

The `figures/` directory contains portable figure generators for LLPS and DPR
benchmarks, ablations, DPR profiles, and the model architecture. Every script
uses explicit command-line input and output paths and produces PNG, PDF, SVG,
and its corresponding plot-data CSV.

## Feature And Teacher Utilities

`teacher/` and the feature modules in `phaseflow/full_length/features/` wrap
optional third-party tools. Supply their installation locations through the
documented command-line options or environment variables.
