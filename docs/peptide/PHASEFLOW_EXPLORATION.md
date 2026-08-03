# Peptide Experiment Guide

This guide is a concise protocol for developing a new short-peptide PhaseFlow
experiment. It replaces historical exploratory notes that referred to local
paths, transient runs, and unversioned results.

## Start From A Baseline

Copy `configs/peptide/peptide.yaml` into a dedicated experiment directory.
Change one hypothesis at a time, such as the phase encoder, diffusion mode,
loss weighting, or model width. Do not overwrite the canonical baseline.

## Define The Data Split First

Choose either a fixed external validation/test split or a deterministic random
split. Record the source checksum and split seed before optimization. For
missing-value studies, document the threshold and the exact set of
`by_missing/` source files included in every split.

## Run And Record

Use the Python trainer or `scripts/peptide/train.sh`. Preserve the resolved
configuration, command line, run log, selected checkpoint, and metrics. A
checkpoint name alone is not sufficient provenance.

## Compare Fairly

Use the same split, preprocessing, maximum sequence length, and evaluation
mask across compared arms. If a change alters a training budget or parameter
count, report that fact prominently. Select checkpoints using validation data;
evaluate the selected checkpoint once on held-out data.

## Promote Results

Before a result is used in documentation or a manuscript figure, export its
raw metrics and a plot-data table. Review labels, units, split identity, and
uncertainty. Compact, reviewed figures may be added to `figures/peptide/`;
large results and checkpoints must remain in the external artifact release.
