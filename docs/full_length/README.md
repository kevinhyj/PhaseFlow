# Full-Length Protein Workflow

This directory documents the public PhaseFlow workflow for full-length proteins.
It covers protein-level liquid-liquid phase separation (LLPS) prediction,
residue-level droplet-promoting region (DPR) prediction, data contracts,
training, evaluation, and publication figures.

The full-length workflow has its own data and training-artifact contracts. It
uses residue-level sequence embeddings, physicochemical and disorder features,
structure-derived features, and graph features to produce protein-level LLPS
scores and DPR profiles. DPR training may additionally consume the released
PhaseFlow checkpoint specified by its configuration.

## Entry Points

| Purpose | Public entry point |
| --- | --- |
| LLPS training | `scripts/full_length/train_llps.py --config configs/full_length/llps.yaml` |
| DPR training | `scripts/full_length/train_dpr.py --config configs/full_length/dpr.yaml --arm dpr` |
| Source-package validation | `scripts/full_length/data/build_dataset.py` |
| DPR target construction | `scripts/full_length/data/build_region_targets.py` |
| Benchmark evaluation | `scripts/full_length/benchmark/evaluate_benchmarks.py` |
| Tables | `scripts/full_length/tables/generate_tables.py` |
| Publication figures | `scripts/full_length/figures/` |

The training configurations are the canonical descriptions of the released
LLPS and DPR runs. Do not edit them in place when running an experiment;
copy a configuration to a separate experiment directory and record the change.

## Artifact Contract

The Git repository deliberately excludes large or derived artifacts. A public
run requires three artifact classes, each distributed outside Git:

| Artifact class | Local destination | Contents |
| --- | --- | --- |
| Source datasets | `artifacts/data/full_length/` | Released tables, benchmark metadata, and dataset contracts |
| Derived training artifacts | `artifacts/derived/full_length/` | Reconstructed embeddings, graph tensors, packed arrays, and batch plans |
| Model checkpoints | `artifacts/models/full_length/` | Released LLPS and full PhaseFlow checkpoints |

Source datasets are immutable provenance records. Derived artifacts are
deterministic training inputs generated from source datasets and external model
components, but they are too large for the code repository. Training consumes
the derived-artifact paths in `configs/full_length/` rather than the source
tables directly.

The two source packages have the following required layouts:

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

## Validate Source Packages

Validate every downloaded source package before feature reconstruction or
training. The validator checks required tables, required columns, identifier
references, duplicate protein identifiers, and records SHA256 checksums in a
machine-readable manifest.

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

Keep these manifests with each experiment record. They identify the exact
source tables consumed by a reconstruction or training run.

## Training Inputs

The LLPS configuration expects an audited offline feature tree and index-only
batch plans below `artifacts/derived/full_length/llps/`. The DPR configuration
expects a packed tensor sidecar, a tier manifest, a PhasePro evaluation package,
and the LLPS and full PhaseFlow checkpoints. These paths are explicit in the
respective YAML files and should be checked before launching a distributed run.

Use the feature utilities in `phaseflow/full_length/features/` to reconstruct
sequence embeddings, biophysical features, and structure-derived inputs. The
third-party tools used by those utilities are optional dependencies and must be
installed separately. Generated artifacts remain outside Git.

For DPR pseudo-supervision, the target builder intentionally requires all
teacher inputs and output locations explicitly:

```bash
python scripts/full_length/data/build_region_targets.py \
  --teacher-scores artifacts/derived/full_length/dpr/teacher_profiles.h5 \
  --out artifacts/derived/full_length/dpr/region_targets.h5 \
  --report runs/data/dpr_region_targets_report.json \
  --policy stratified
```

This avoids silently substituting an unpublished teacher archive.

## Evaluation And Figures

Evaluation and figure generators never assume a local machine layout. Supply
their input and output paths explicitly. The figure scripts write PNG, PDF,
SVG, and an auditable CSV containing the values rendered in the figure.

The expected data semantics are as follows:

| Figure | Required input |
| --- | --- |
| LLPS benchmark | Tabular model-level LLPS metrics |
| LLPS ablation | One row per ablation arm, with values and optional confidence intervals |
| DPR benchmark | Tabular model-level DPR metrics |
| DPR ablation | Released multi-scale DPR summary table |
| DPR examples | NPZ residue profiles, with optional protein metadata and region intervals |
| Architecture | No data input; optional font path |

Use `--help` on any script for its complete interface. The scripts write only
to the requested output directory.

## Further Reading

- [Reproduction guide](reproduction.md)
- [Artifact policy](artifact_policy.md)
- [`configs/full_length/README.md`](../../configs/full_length/README.md)
- [`scripts/full_length/README.md`](../../scripts/full_length/README.md)
