# Peptide Evaluation And Analysis Protocol

This document defines a reproducible analysis protocol for the short-peptide
workflow. It intentionally does not preserve machine-specific experiment
snapshots or unversioned score tables. Store numerical results with the
corresponding external artifact release instead.

## Sequence-To-Phase Evaluation

Evaluate predicted PSSI values only at positions marked valid by the source
phase mask. Common summary metrics are mean squared error, mean absolute error,
root mean squared error, Pearson correlation, and Spearman correlation. Report
the aggregation level explicitly: per-position, per-peptide, or pooled.

Use a fixed held-out table when comparing models. If a random split is used,
record the source checksum, seed, train/validation/test ratios, and all
filtering rules.

## Phase-To-Sequence Evaluation

Sequence generation is stochastic. Report the checkpoint, phase input, phase
mask, seed, temperature, maximum length, sampling method, candidate count, and
any post-generation filter. Do not aggregate candidate-quality metrics without
also describing the candidate-selection rule.

Round-trip analyses can be informative, but they measure consistency within a
pipeline and do not by themselves validate experimental phase behavior. State
whether the scorer is the same model, a separately trained model, or an
external method.

## Ablations

When comparing architectural or objective variants:

- Use the same source table and fixed split for every arm.
- Keep parameter budget, training budget, and checkpoint-selection rule clear.
- Report uncertainty across independent seeds where practical.
- Separate model selection data from the final evaluation set.
- Save the raw per-example metrics used to draw figures or tables.

## Minimum Artifact Record

Each published analysis should contain the following files outside Git:

```text
config.yaml
source_manifest.json
checkpoint_manifest.json
metrics.csv
per_example_predictions.parquet
figure_plot_data.csv
command.txt
environment.txt
```

Large binary files and row-level prediction tables belong in the external
artifact repository, not in this source repository.
