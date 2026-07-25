# Full-Length Artifact Policy

Git tracks source code, configuration, documentation, tests, and compact
publication figures. It never tracks model checkpoints, raw datasets,
feature caches, profile archives, training logs, or regenerated benchmark
outputs.

Keep release data under `artifacts/data/full_length/` and model checkpoints
under `artifacts/models/full_length/`. These directories are intentionally
ignored except for their README files. The repository ignores `.pt`, `.pth`,
`.ckpt`, `.safetensors`, `.bin`, `.onnx`, `.h5`, `.parquet`, `.npz`, and
other generated binary artifacts.

Figure scripts accept input and output paths explicitly. Store regenerated
figures under `runs/figures/`; only reviewed compact figures belong under
`figures/full_length/`.
