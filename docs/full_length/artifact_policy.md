# PhaseFlow Artifact Policy

This paper repository keeps final artifacts needed for review, reuse, and
reproducibility. It intentionally excludes regenerated feature caches and raw
training run directories that are not appropriate for GitHub maintenance.

## Included

- Final checkpoints under `artifacts/model/checkpoints/` tracked with Git LFS.
- Resolved training config, summaries, model structure figures, and audit reports.
- Open LLPS/DPR data audit reports and machine-readable manifests.
- Benchmark reports, tables, selected profiles, ROC curves, and summary JSON/CSV files.
- Mutation benchmark outputs and paper assets.
- Ablation configs, runbooks, reports, scripts, and figures.
- Repository file list and SHA256 manifest in `manifests/`.

## Excluded Large Artifacts

| Artifact class | Reason |
| --- | --- |
| Raw DPR final-chain run products | Too large; summarized by included reports, tables, and figures |
| Raw LLPS ablation run directories | Too large; summarized by included tables and figures |
| Regenerated validation feature caches | Rebuildable from the data and feature-generation pipeline |
| Intermediate scratch outputs | Not part of the final reported model state |

The repository manifests are generated from this cleaned paper tree.
