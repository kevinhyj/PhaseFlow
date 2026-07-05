# PhaseFlow Full-Length

This directory documents the full-length protein portion of PhaseFlow: model
code, data/feature utilities, training and inference entry points, final
configuration summaries, and focused tests.

The code is integrated into the unified repository as the
`phaseflow.full_length` Python subpackage. Large model checkpoints, regenerated
feature caches, raw benchmark artifacts, and manuscript build artifacts are
intentionally not included in this GitHub tree.

## Contents

- `phaseflow/full_length/`: Python package for model, data, feature, training, inference, and evaluation code.
- `configs/full_length/`: final LLPS and DPR configuration summaries.
- `scripts/full_length/`: maintained data, feature, training, evaluation, benchmark, report, and audit entry points.
- `tests/full_length/`: focused unit tests for model, data, metrics, DPR variants, and inference paths.
- `docs/full_length/`: lightweight architecture, reproduction, and artifact-policy notes.

## Install

```bash
python -m pip install -e ".[full_length,test]"
```

Optional feature-generation dependencies:

```bash
python -m pip install -e ".[plm,starling]"
```

## Basic Checks

```bash
python -m pytest tests/full_length/test_imports.py tests/full_length/test_phaseflow_fusion.py
```

## Artifact Policy

This integrated repository keeps the full-length code and configuration files
but excludes model files and heavyweight artifacts. In particular, do not commit
checkpoint files such as `*.pt`, `*.pth`, or `*.ckpt`; keep them in external
storage or a release channel outside the normal Git history.

## Reproduction Notes

- DPR reproduction report: `docs/full_length/final/dpr_v6_rankp257_repro_report_20260617.md`
- Final DPR config: `configs/full_length/final_dpr.yaml`
- Final LLPS config: `configs/full_length/final_llps.yaml`
