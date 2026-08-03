# Protein Reproduction Commands

This directory exposes one canonical LLPS-to-DPR command: `run.py`. The
`prepare.py`, `train.py`, and `evaluate.py` adapters group the same commands
for interactive use. Full training, evaluation, release, target-building, and
feature-generation workflows live in `workflows/`; `phaseflow/protein/` remains
limited to model structure and reusable core components. All paths are explicit;
derived caches, checkpoints, and run outputs belong outside the source checkout.

## Workflow

1. `run.py validate-data` validates a released `PhaseFlow-LLPS` or
   `PhaseFlow-DPR` source package and records its manifest checksums.
2. `run.py build-features` regenerates local sequence and optional structural
   feature caches from released source tables.
3. `run.py compile-llps-inputs` validates those LLPS caches and materializes the
   fixed-plan LLPS input tree.
4. `run.py train-llps` trains the LLPS stage.
5. `run.py region-targets` creates DPR region targets from explicitly
   supplied teacher profiles.
6. `run.py build-dpr-sidecar` materializes frozen LLPS hidden states and packed
   DPR inputs.
7. `run.py train-dpr` runs DPR stage 1; `run.py refine-dpr` runs the configured
   refinement stages.
8. `run.py evaluate-phasepro` evaluates a cached-hidden DPR checkpoint on the
   frozen PhasePro set.

`run.py reproduce` prints the complete path-derived stage map before any data
is created when supplied with explicit roots:

```bash
python scripts/protein/run.py reproduce \
  --data-root /path/to/open_release \
  --work-root /path/to/derived_cache \
  --output-root /path/to/runs \
  --dry-run
```

The commands and required artifact layouts are documented in
[`docs/protein/reproduction.md`](../../docs/protein/reproduction.md).

For example, `python scripts/protein/train.py llps --config
configs/protein/llps.yaml` delegates to `run.py train-llps` without changing
arguments or runtime behavior.

## Optional Tool Wrappers

`features/` contains shell adapters for AF3 and Protenix. Python command
implementations live in `workflows/`, because they are reproduction tools rather
than importable model-core APIs.

## Other Repository Utilities

Publication figures live in `artifacts/results/protein/scripts/figures/`;
manuscript tables live in `artifacts/results/protein/scripts/tables/`;
exploratory benchmark and threshold analyses live in `analysis/`; standalone
inference helpers live in `inference/`; wrappers around third-party teacher
programs live in `scripts/external/protein/teacher/`.
