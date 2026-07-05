# Models

This directory is reserved for local model checkpoints downloaded from Hugging
Face or produced during reproduction.

Suggested layout:

```text
artifacts/models/
  peptide/                 Short-peptide PhaseFlow checkpoints
  full_length/
    llps/                  Full-length LLPS checkpoints
    dpr/                   Full-length DPR checkpoints
```

Model files are intentionally ignored by Git. Keep published weights in the
corresponding Hugging Face model repositories and download them here when
running training, inference, or benchmark scripts.
