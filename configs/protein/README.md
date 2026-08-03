# Protein Configurations

- `llps.yaml` configures protein LLPS training.
- `dpr.yaml` configures protein DPR training.

Both configurations use paths rooted at `artifacts/` for data and models and
write generated outputs under `runs/`. They describe the released training
interfaces without embedding machine-specific paths. Their `reproduction`
sections are the benchmark contracts: LLPS fixes the PPMC checkpoint, panel,
precision, and threshold; DPR fixes the raw checkpoint variant, checksum,
BF16 precision, and metric tolerance for PhasePro.
