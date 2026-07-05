# Data

This directory is reserved for datasets and generated feature stores downloaded
from Hugging Face or produced locally.

Suggested layout:

```text
artifacts/data/
  peptide/                 Short-peptide phase-diagram CSV/NPZ data
  full_length/             Full-length manifests, feature stores, and benchmarks
```

Model checkpoints should not be stored here. Use `artifacts/models/` for local
model downloads.
