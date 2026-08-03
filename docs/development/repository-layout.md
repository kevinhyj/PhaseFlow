# Repository Layout

PhaseFlow separates reusable research software from command-line interfaces,
experiment configuration, generated artifacts, and manuscript assets.

## Source Package

`phaseflow/` contains reusable Python implementation for model structure and
core components. Protein model definitions, data contracts, reusable feature
functions, structure parsers, objectives, metrics, and post-processing must be
importable without depending on `scripts/`. Training loops, evaluation CLIs,
release builders, region-target construction, and other reproduction tools live
under `scripts/protein/workflows/`.

The protein package is deliberately flat. `phaseflow/protein/` directly holds
the core modules: `contracts`, `data`, `tokenizer`, `features`, `structure`,
`model`, `objectives`, and `postprocessing`. These modules do not import `scripts/` and
do not expose command-line parsers or training entry points.

| Protein module | Responsibility |
| --- | --- |
| `contracts.py` | Data schemas, feature-cache readers/writers, split resolution, reproducibility guards, and shared configuration helpers. |
| `data.py` | Dataset classes, collators, packed sidecar readers, fixed schedules, and core record materialization used by training/evaluation workflows. |
| `tokenizer.py` | Stable protein sequence normalization and 1--20/0 packed-sidecar residue encoding. |
| `features.py` | Reusable sequence, PLM, physicochemical, disorder, biophysical, Starling, and feature-cache construction functions. |
| `structure.py` | Structure parsers, graph/edge construction, AF3/Protenix input/output parsing, and structure-derived feature helpers. |
| `model.py` | Protein LLPS/DPR neural network modules, bridge components, heads, checkpoint loading helpers, and model-facing tensor transforms. |
| `objectives.py` | Losses, metrics, ranking/calibration objectives, and region/residue scoring utilities. |
| `postprocessing.py` | Prediction smoothing, region extraction, decoder/post-process region merging, non-maximum suppression, and key-residue selection. |

## Configuration

`configs/` contains public YAML experiment specifications. Python code may
parse and validate those specifications, but configuration files are not part
of the importable package. Paths in committed configuration files must resolve
from the repository root and may refer only to documented locations below
`artifacts/` or `runs/`.

## Command-Line Interfaces

`scripts/protein/run.py` is the canonical LLPS-to-DPR reproduction interface:
source validation, local feature construction, input compilation, sidecar and
target construction, the three training commands, PhasePro evaluation, and
the dry-run stage map. `prepare.py`, `train.py`, and `evaluate.py` are small
stage adapters for interactive use; they delegate to `run.py`. The executable
workflow implementations live under `scripts/protein/workflows/` and depend on
`phaseflow.protein` core APIs. Third-party tool wrappers remain under
`scripts/protein/features/` because they adapt external command-line programs.
Protein analysis and inference tools live under `scripts/protein/analysis/`
and `scripts/protein/inference/`, matching the domain-first peptide layout.
Publication figure and manuscript-table tools live with their released protein
results under `artifacts/results/protein/scripts/`; third-party teacher wrappers
remain under `scripts/external/protein/teacher/`.

## Figures

Peptide figure generators live under `scripts/peptide/figures/`; protein
publication generators live under `artifacts/results/protein/scripts/figures/`.
They accept explicit input/output paths where required, write plot-data tables
alongside image assets, and use only repository-contained code. Published image
files and compact analysis outputs may be retained under `artifacts/results/`;
checkpoints and complete generated run directories remain outside Git under the
documented `artifacts/` and `runs/` contracts.

Each final paper panel has its own executable figure script. A script owns its
data filtering, metric transformation, panel ordering, and rendering choices;
repository utilities may provide only generic plotting setup. This keeps every
published figure independently reproducible from its documented input tables.

## Tests And Release Checks

`tests/` mirrors public package and CLI behavior. Tests enforce that package
modules do not import `scripts`, public paths do not contain machine-specific
locations, and figure scripts expose portable input/output interfaces.
