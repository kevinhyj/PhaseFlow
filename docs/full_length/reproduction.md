# Full-Length Reproduction

1. Place the LLPS and DPR dataset packages under
   `artifacts/data/full_length/` and validate them with
   `scripts/full_length/data/build_dataset.py`.
2. Reconstruct the required embeddings, structural features, and graph caches
   with the feature utilities in `phaseflow/full_length/features/`.
3. Place released checkpoints under `artifacts/models/full_length/`.
4. Run LLPS training with `scripts/full_length/train_llps.py --config
   configs/full_length/llps.yaml`.
5. Run DPR training with `scripts/full_length/train_dpr.py --config
   configs/full_length/dpr.yaml --arm dpr`.
6. Run evaluation utilities with explicit input and output paths.
7. Generate the publication figures with the scripts in
   `scripts/full_length/figures/`.

Each figure script writes PNG, PDF, and SVG assets. Use the CSV or NPZ exports
from the corresponding evaluation step as its input; the scripts do not rely
on a machine-specific directory layout.
