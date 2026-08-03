# Protein Model Architecture

The protein workflow separates reusable model components from executable
reproduction tooling.  The package is deliberately flat, following the same
reader-first convention as `phaseflow/peptide`: each core responsibility is
visible directly under `phaseflow/protein/`.

```text
FASTA / manifest
      │
      ├─ scripts/protein/workflows/features.py
      │   build frozen feature caches and external-model inputs
      ▼
ProteinTokenizer  ──► frozen PLM + biophysical feature arrays
tokenizer.py           features.py
      │                         │
      └──────────────► sparse residue graph
                              structure.py
                                   │
                                   ▼
                          PhaseFlowModel
                            model.py
                         ┌─────────┴──────────┐
                         ▼                    ▼
                     LLPS logit          DPR residue/region logits
                         └─────────┬──────────┘
                                   ▼
                        scores_to_regions and region merging
                              postprocessing.py
```

## Core package

| Module | What an AI worker should look for | Primary input and output |
| --- | --- | --- |
| [`tokenizer.py`](../../phaseflow/protein/tokenizer.py) | `ProteinTokenizer`, the canonical sequence normalization and fixed residue encoding | amino-acid string → `int16[L]`; canonical residues are 1--20 and unknown/gap residues are 0, preserving packed-sidecar compatibility |
| [`contracts.py`](../../phaseflow/protein/contracts.py) | records, cache schema, reproducibility guards, path/config validation | source metadata and HDF5 records ↔ validated contract objects |
| [`data.py`](../../phaseflow/protein/data.py) | datasets, collators, packed sidecars, schedules | feature-cache records → padded batch tensors |
| [`features.py`](../../phaseflow/protein/features.py) | deterministic physicochemical, disorder, biophysical, and frozen-embedding feature transforms | sequence / cached embeddings → per-residue feature matrices |
| [`structure.py`](../../phaseflow/protein/structure.py) | `SparseEdges`, graph building, and model-facing structural tensors | residue and contact evidence → sparse graph arrays |
| [`model.py`](../../phaseflow/protein/model.py) | `PhaseFlowModel`, encoders, fusion, graph transformer, LLPS/DPR heads, checkpoint loaders | collated batch → LLPS and DPR prediction tensors |
| [`objectives.py`](../../phaseflow/protein/objectives.py) | multitask losses and LLPS/DPR metrics | prediction tensors + labels → losses and metrics |
| [`postprocessing.py`](../../phaseflow/protein/postprocessing.py) | smoothing, non-maximum suppression, and score-to-region conversion | DPR probabilities → scored protein regions |

`ProteinTokenizer` is intentionally not a trainable language-model tokenizer:
the model consumes frozen PLM embeddings.  Its role is to make the stable
residue-ID representation used in packed reproduction sidecars explicit and
discoverable rather than hiding it in a release script.

## Reproduction layer

Nothing in `phaseflow/protein` imports a script module or exposes a CLI.  The
scripts layer owns operations that create files, invoke external programs, or
run a complete stage:

| Path | Responsibility |
| --- | --- |
| [`scripts/protein/run.py`](../../scripts/protein/run.py) | canonical LLPS-to-DPR reproduction command |
| [`scripts/protein/{prepare,train,evaluate}.py`](../../scripts/protein/) | compact stage-specific delegates |
| [`scripts/protein/workflows/`](../../scripts/protein/workflows/) | preparation, training, evaluation, release, feature, and external-structure implementations |
| [`scripts/protein/features/`](../../scripts/protein/features/) | shell adapters for AF3 and Protenix |

The canonical commands and artifact layout are in
[the reproduction guide](reproduction.md). Inspect the full stage map before
materializing any artifacts:

```bash
python scripts/protein/run.py reproduce \
  --data-root /path/to/open_release \
  --work-root /path/to/derived_cache \
  --output-root /path/to/runs \
  --dry-run
```
