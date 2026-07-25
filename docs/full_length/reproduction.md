# Full-Length Reproduction Guide

This guide describes the order of operations for reproducing a released
full-length PhaseFlow run. It does not prescribe a hardware environment; GPU
count, mixed precision, and worker settings are defined by the selected YAML
configuration.

## 1. Install The Source Package

```bash
python -m pip install -e ".[full_length,test]"
```

Install optional feature-generation dependencies only when the corresponding
feature utility is required. Consult the utility's `--help` output and the
upstream tool documentation before generating external structural features.

## 2. Obtain And Validate Source Data

Place the two released source packages at the locations below:

```text
artifacts/data/full_length/PhaseFlow-LLPS/
artifacts/data/full_length/PhaseFlow-DPR/
```

Run the package validator for both tasks and retain the generated manifests:

```bash
python scripts/full_length/data/build_dataset.py \
  --task llps \
  --package-root artifacts/data/full_length/PhaseFlow-LLPS \
  --output runs/data/llps_source_manifest.json

python scripts/full_length/data/build_dataset.py \
  --task dpr \
  --package-root artifacts/data/full_length/PhaseFlow-DPR \
  --output runs/data/dpr_source_manifest.json
```

Do not train from a package that fails validation. Correct the download or
obtain a new artifact copy before proceeding.

## 3. Provision Derived Inputs And Checkpoints

The training code reads derived tensors, not raw source tables. Place or
reconstruct the required derived artifacts below:

```text
artifacts/derived/full_length/llps/
artifacts/derived/full_length/dpr/
```

Place checkpoints at the paths referenced by the configuration, including:

```text
artifacts/models/full_length/llps/model.pt
artifacts/models/full_length/phaseflow/model.pt
```

The `configs/full_length/llps.yaml` and `configs/full_length/dpr.yaml` files
are the authoritative path contracts. Review each path before training. For
large distributed jobs, record the source-package manifests, configuration
copy, checkpoint checksums, and generated schedule audit in the experiment
directory.

## 4. Train LLPS

```bash
python scripts/full_length/train_llps.py \
  --config configs/full_length/llps.yaml
```

The run writes its outputs below `runs/llps/` by default. The canonical model
artifact is named `model.pt`; the paired checkpoint convenience copy is
`model.ckpt`.

## 5. Train DPR

```bash
python scripts/full_length/train_dpr.py \
  --config configs/full_length/dpr.yaml \
  --arm dpr
```

To inspect schedule construction without launching optimization, pass
`--make-schedule-only`. The resolved configuration and schedule audit are
written under the selected DPR run directory.

## 6. Evaluate And Render Figures

Run evaluation with explicit input and output paths, then render figures from
the resulting tabular exports or profile archive. For example:

```bash
python scripts/full_length/figures/plot_dpr_ablation.py \
  --input runs/dpr/reports/ablation_metrics.csv \
  --output-dir runs/figures/dpr_ablation
```

The figure directory contains the image assets and the exact values used to
draw them. Treat that CSV as the provenance record for the rendered panel.

## Reproducibility Checklist

- Validate each source package and save its manifest.
- Use a copied, immutable training configuration for every run.
- Record all checkpoint hashes and third-party feature-tool versions.
- Preserve generated schedules and training logs outside Git.
- Keep raw metrics, plot-data CSVs, and manuscript figures together.
