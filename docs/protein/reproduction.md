# Protein Reproduction Guide

This guide describes the order of operations for reproducing a released
protein PhaseFlow run. It does not prescribe a hardware environment; GPU
count, mixed precision, and worker settings are defined by the selected YAML
configuration.

## 1. Install The Source Package

```bash
python -m pip install -e ".[protein,test]"
```

Install optional feature-generation dependencies only when the corresponding
feature utility is required. Consult the utility's `--help` output and the
upstream tool documentation before generating external structural features.

## 2. Obtain And Validate Source Data

Place `PhaseFlow-LLPS` and `PhaseFlow-DPR` directly below one user-owned data
root. The release contains source tables and fixed training-protocol metadata.
It intentionally does not include embeddings, graphs, packed tensors, model
checkpoints, or run output. The three LLPS plan files fix the reference 8-GPU
sample order and must be identity-checked against `training_units.parquet`.

Run the package validator for both tasks and retain the generated manifests:

```bash
python scripts/protein/run.py validate-data \
  --task llps \
  --package-root /path/to/data-root/PhaseFlow-LLPS \
  --output runs/data/llps_source_manifest.json

python scripts/protein/run.py validate-data \
  --task dpr \
  --package-root /path/to/data-root/PhaseFlow-DPR \
  --output runs/data/dpr_source_manifest.json
```

Do not train from a package that fails validation. Correct the download or
obtain a new artifact copy before proceeding.

## 3. Rebuild Derived Inputs

Use separate user-owned work and output roots. The following dry run is the
authoritative stage map. It does not create files, download models, or launch
training:

Before starting a long run, inspect the complete portable workflow with three
explicit roots. This command only prints the resolved plan and does not create
features, download models, or start training:

```bash
python scripts/protein/run.py reproduce \
  --data-root /path/to/open_release \
  --work-root /path/to/derived_cache \
  --output-root /path/to/runs \
  --dry-run
```

The raw-data rebuild stages are validated progressively. Do not describe a
rebuild as numerically equivalent to the reference result until the generated
LLPS hidden sidecar and final benchmark report have passed their contracts.

For LLPS, generate one HDF5 feature cache per public protein with the released
feature utility. New graph caches use 96 neighbors and 32 edge features.
The evaluator also accepts the frozen historical 13-feature HDF5 schema: it
deterministically zero-pads edge attributes to the dimension declared in the
checkpoint before graph inference. Supply structure and Starling feature
directories when those modalities are part of the selected LLPS checkpoint;
missing modalities are represented explicitly in the cache and change the
experiment.

```bash
python scripts/protein/run.py build-features \
  --manifest /path/to/data-root/PhaseFlow-LLPS/data/proteins.parquet \
  --out-dir /path/to/work-root/llps/h5_features \
  --mode esm2 \
  --esm2-model-dir /path/to/models/esm2 \
  --graph-max-neighbors 96 \
  --graph-edge-dim 32
```

After an LLPS checkpoint has been trained or otherwise supplied, create the
immutable DPR input sidecar. The command rebuilds the 112-dimensional
biophysical node feature, obtains the frozen 256-dimensional LLPS residue
state, records sequence and file hashes, and validates the completed sidecar.

```bash
python scripts/protein/run.py build-dpr-sidecar \
  --feature-dir /path/to/work-root/dpr/features \
  --llps-checkpoint /path/to/llps/model.pt \
  --output-root /path/to/work-root/dpr/packed \
  --device cuda
```

The sidecar must validate before DPR training or evaluation. Never substitute a
different sequence, checkpoint, graph width, or hidden-state key in an existing
sidecar.

## 4. Train LLPS

Compile the locally generated LLPS feature caches into the offline cache
required by the fixed plan. The compiler validates every plan row by public
protein ID and sequence hash, preserves public `dataset_index` order, and then
copies the validated rank-local plans unchanged.

```bash
python scripts/protein/run.py compile-llps-inputs \
  --release-root /path/to/data-root/PhaseFlow-LLPS \
  --feature-root /path/to/work-root/llps/h5_features \
  --output-root /path/to/work-root/llps
```

The compiler writes `llps.yaml` beside the derived tree. It preserves the
released model and optimization settings while resolving the generated
`processed/` and `training/plan/` paths. Launch it directly:

```bash
torchrun --nproc_per_node=8 scripts/protein/run.py train-llps \
  --config /path/to/work-root/llps/llps.yaml
```

`--resume` accepts either a full training checkpoint or an initialization
checkpoint. With `training.strict_resume: false`, an initialization checkpoint
whose optimizer state does not match the current parameter groups restores
model weights only; the run records `resume_weights_only` in its log.

Until this source route completes the full LLPS and DPR benchmark gates, it
must not be described as numerically equivalent to the reference result.

```bash
python scripts/protein/run.py train-llps \
  --config configs/protein/llps.yaml
```

The run writes its outputs below `runs/llps/` by default. The canonical model
artifact is named `model.pt`; the paired checkpoint convenience copy is
`model.ckpt`.

## 5. Train DPR

```bash
python scripts/protein/run.py train-dpr \
  --config configs/protein/dpr.yaml \
  --arm dpr
```

The DPR source package includes the frozen base-training schedule. Preserve its
row order and record the selected sidecar manifest hash with every run.

## 6. Evaluate The Fixed Benchmarks

The released configurations define the only publication comparison contracts.
LLPS uses the frozen PPMC panel, the checkpoint SHA256 in `llps.yaml`, FP32
inference in stable feature-length order with batch size 8, the checkpoint's
`region_global_llps_score`, and a fixed 0.5 threshold. DPR uses the raw final checkpoint (not
EMA), BF16 inference, the frozen 121-protein PhasePro sidecar, and the
checkpoint SHA256 in `dpr.yaml`. Do not mix a training packed sidecar with the
PhasePro evaluation sidecar.

The LLPS evaluator checks AUROC, AUPRC, MCC at 0.5, and F1 at 0.5 against the
published references in `llps.yaml`. A maximum absolute difference of `1e-5`
is accepted for FP32 GPU inference; this is numerical agreement, not a claim
of bitwise identity.

```bash
python scripts/protein/run.py evaluate-llps \
  --config configs/protein/llps.yaml \
  --checkpoint /path/to/llps/best_single_model.pt \
  --feature-dir /path/to/derived-cache/llps/ppmc_h5 \
  --panel /path/to/data-root/PhaseFlow-LLPS/benchmark/panel_membership.csv \
  --output-root runs/llps/ppmc
```

```bash
python scripts/protein/run.py evaluate-phasepro \
  --config configs/protein/dpr.yaml \
  --checkpoint /path/to/dpr/update_000050.pt \
  --sidecar-root /path/to/derived-cache/dpr/phasepro_packed \
  --benchmark-root /path/to/data-root/PhaseFlow-DPR/evaluation/phasepro \
  --output-root runs/dpr/phasepro
```

The evaluator rejects a sidecar whose identities do not match the 121 frozen
benchmark proteins. Compare AUPRC, AUROC, and Spearman against the values in
`configs/protein/dpr.yaml` with the declared tolerance.

## 7. Render Figures

Run evaluation with explicit input and output paths, then render figures from
the resulting tabular exports or profile archive. For example:

```bash
python artifacts/results/protein/scripts/figures/plot_dpr_ablation_summary.py \
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
