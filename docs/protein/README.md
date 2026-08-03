# Protein Workflow

This directory documents the public PhaseFlow workflow for proteins.
It covers protein-level liquid-liquid phase separation (LLPS) prediction,
residue-level droplet-promoting region (DPR) prediction, data contracts,
training, evaluation, and publication figures.

The protein workflow has its own data and training-artifact contracts. It
uses residue-level sequence embeddings, physicochemical and disorder features,
structure-derived features, and graph features to produce protein-level LLPS
scores and DPR profiles. DPR training may additionally consume the released
PhaseFlow checkpoint specified by its configuration.

For a reader-first map of the model, including `tokenizer.py`, `model.py`, the
feature/graph boundary, and the scripts that own reproduction operations, see
[the protein architecture guide](architecture.md).

## Entry Points

| Purpose | Public entry point |
| --- | --- |
| LLPS training | `scripts/protein/run.py train-llps --config configs/protein/llps.yaml` |
| LLPS input compilation | `scripts/protein/run.py compile-llps-inputs` |
| DPR training | `scripts/protein/run.py train-dpr --config configs/protein/dpr.yaml --arm dpr` |
| Source-package validation | `scripts/protein/run.py validate-data` |
| DPR target construction | `scripts/protein/run.py region-targets` |
| PPMC LLPS evaluation | `scripts/protein/run.py evaluate-llps` |
| PhasePro DPR evaluation | `scripts/protein/run.py evaluate-phasepro` |
| Benchmark summary | `scripts/protein/analysis/evaluate_benchmarks.py` |
| Standalone DPR inference | `scripts/protein/inference/predict_protein_dpr.py` |
| Tables | `artifacts/results/protein/scripts/tables/generate_tables.py` |
| Publication figures | `artifacts/results/protein/scripts/figures/` |

The training configurations are the canonical descriptions of the released
LLPS and DPR runs. Do not edit them in place when running an experiment;
copy a configuration to a separate experiment directory and record the change.

## Artifact Contract

The Git repository deliberately excludes large or derived artifacts. A public
run requires three artifact classes, each distributed outside Git:

| Artifact class | Local destination | Contents |
| --- | --- | --- |
| Source datasets | `artifacts/data/protein/` | Released tables, benchmark metadata, and dataset contracts |
| Derived training artifacts | `artifacts/derived/protein/` | Reconstructed embeddings, graph tensors, packed arrays, and batch plans |
| Model checkpoints | `artifacts/models/protein/` | Released LLPS and full PhaseFlow checkpoints |

Source datasets are immutable provenance records. Derived artifacts are
deterministic training inputs generated from source datasets and external model
components, but they are too large for the code repository. Training consumes
the derived-artifact paths in `configs/protein/` rather than the source
tables directly.

The two source packages have the following required layouts:

```text
PhaseFlow-LLPS/
  data/proteins.parquet
  data/training_units.parquet
  data/training_plan/
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
python scripts/protein/run.py validate-data \
  --task llps \
  --package-root artifacts/data/protein/PhaseFlow-LLPS \
  --output runs/data/llps_source_manifest.json

python scripts/protein/run.py validate-data \
  --task dpr \
  --package-root artifacts/data/protein/PhaseFlow-DPR \
  --output runs/data/dpr_source_manifest.json
```

Keep these manifests with each experiment record. They identify the exact
source tables consumed by a reconstruction or training run.

## Training Inputs

The LLPS input compiler creates the audited offline feature tree and copies the
validated index-only batch plans below `artifacts/derived/protein/llps/`. The DPR configuration
expects a packed tensor sidecar, a tier manifest, a PhasePro evaluation package,
and the LLPS and full PhaseFlow checkpoints. These paths are explicit in the
respective YAML files and should be checked before launching a distributed run.

Use `scripts/protein/run.py build-features` and the public
`phaseflow.protein` API to reconstruct
sequence embeddings, biophysical features, and structure-derived inputs. The
third-party tools used by those utilities are optional dependencies and must be
installed separately. Generated artifacts remain outside Git.

For DPR pseudo-supervision, the target builder intentionally requires all
teacher inputs and output locations explicitly:

```bash
python scripts/protein/run.py region-targets \
  --teacher-scores artifacts/derived/protein/dpr/teacher_profiles.h5 \
  --out artifacts/derived/protein/dpr/region_targets.h5 \
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
- [Model architecture and tokenizer map](architecture.md)
- [Artifact policy](artifact_policy.md)
- [Curated final results](../../artifacts/results/protein/README.md)
- [`configs/protein/README.md`](../../configs/protein/README.md)
- [`scripts/protein/README.md`](../../scripts/protein/README.md)
