# Protein Artifact Policy

The PhaseFlow Git repository contains source code, configuration, tests,
documentation, and compact reviewed figures. It must remain free of model
weights, raw datasets, feature caches, profile archives, packed tensors,
training logs, and regenerated benchmark outputs.

## Storage Classes

| Class | Location | Version-control policy |
| --- | --- | --- |
| Source code and documentation | Repository tree | Tracked in Git |
| Source datasets | `artifacts/data/protein/` | Distributed externally; ignored by Git |
| Derived training artifacts | `artifacts/derived/protein/` | Distributed or regenerated externally; ignored by Git |
| Model checkpoints | `artifacts/models/protein/` | Distributed externally; ignored by Git |
| Run outputs | `runs/` | Local experiment output; ignored by Git |

## Prohibited Git Content

Do not commit checkpoints, `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.bin`,
`.onnx`, `.h5`, `.hdf5`, `.parquet`, `.npz`, `.npy`, archives, cached feature
tensors, or training logs. The repository `.gitignore` enforces this policy for
the common artifact locations and binary formats.

## Publication Figures

Figure generators accept explicit input and output paths. Store regenerated
assets under `runs/figures/` or another external result location. Only compact,
reviewed publication figures required by repository documentation may be kept
under `figures/protein/`.

Every generated figure should be accompanied by its plot-data CSV, input
checksums, command line, and software environment record in the corresponding
external artifact release.
