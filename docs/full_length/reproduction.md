# Full-Length Reproduction

1. Place `PhaseFlow-LLPS` and `PhaseFlow-DPR` under
   `artifacts/data/full_length/` and validate each with
   `scripts/full_length/data/build_dataset.py`.
2. Reconstruct embeddings, structural features, graph caches, and task-specific
   packed tensors from the source packages. Store these generated training
   artifacts below `artifacts/derived/full_length/`; this is the location
   consumed by the two training configurations.
3. Place released checkpoints below `artifacts/models/full_length/`. DPR
   training requires the LLPS and full PhaseFlow checkpoints listed in
   `configs/full_length/dpr.yaml`.
4. Run LLPS training with `scripts/full_length/train_llps.py --config
   configs/full_length/llps.yaml`.
5. Run DPR training with `scripts/full_length/train_dpr.py --config
   configs/full_length/dpr.yaml --arm dpr`.
6. Run evaluation utilities with explicit input and output paths.
7. Generate the publication figures with the scripts in
   `scripts/full_length/figures/`.

Each figure script writes PNG, PDF, SVG, and an auditable plot-data CSV. Use the CSV or NPZ exports
from the corresponding evaluation step as its input; the scripts do not rely
on a machine-specific directory layout.
