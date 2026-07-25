# Peptide Data Loading

`phaseflow.data.PhaseDataset` loads peptide sequences and their 4x4 PSSI phase
diagrams. This document describes its input contract, split behavior, missing
value semantics, and practical performance options.

## Source And Cache Files

The CSV source contains `AminoAcidSequence` and `group_11` through `group_44`.
The loader preserves missing PSSI measurements as a boolean mask and replaces
their stored tensor values with zero only as a placeholder. Model losses must
use the mask rather than interpreting those zeros as measurements.

An optional `phase_diagram.npz` file in the same directory can provide a
preprocessed `data` array with shape `(N, 16)`. It is used only when the row
order matches the CSV exactly. Keep the CSV with the NPZ cache so the loader
can read sequence strings and verify the intended dataset.

## Splitting Modes

With `missing_threshold: -1`, the loader creates a seeded random split using
the ratios in the configuration. With a non-negative threshold, it loads the
`by_missing/missing_<n>.csv` files up to that threshold before splitting.

The missing-threshold mode is useful for controlled experiments on PSSI
completeness. It is not a replacement for an externally defined held-out set;
provide explicit validation and test files when the experimental protocol
requires fixed splits.

## Batch Contents

Each sample provides token IDs, an attention mask, phase values, a phase mask,
the original sequence length, and the original sequence. `collate_fn` stacks
the tensor fields and preserves the sequence strings for logging or analysis.

## Performance Guidance

- Use the NPZ cache only after validating that it matches the source CSV.
- Choose `max_seq_len` large enough for sequence and control tokens, but avoid
  excessive padding.
- Set `num_workers` according to local storage throughput and available CPU
  memory; more workers do not help when input storage is saturated.
- Keep `pin_memory` enabled for CUDA training when host-to-device transfer is a
  meaningful bottleneck.
- Record data split seed, source checksum, cache checksum, and loader options
  with every experiment.

## Failure Modes

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Missing PSSI columns | Nonconforming source CSV | Supply all `group_11` through `group_44` columns |
| Unexpected sequence truncation | `max_seq_len` too small | Increase it in the configuration and retrain or reevaluate consistently |
| Inconsistent results across runs | Unrecorded split seed or source change | Pin the seed and persist source checksums |
| Incorrect phase values | Stale or misordered NPZ cache | Remove the cache and regenerate it from the current CSV |
