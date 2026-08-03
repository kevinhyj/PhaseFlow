# Results

This directory is for small, curated result tables and figures that are referenced by reports
or documentation.

Do not store large checkpoints or full training output directories here. Use `outputs/` for
active runs and keep heavyweight artifacts outside Git.

Short-peptide analysis outputs are grouped under `analysis/` with the same
study hierarchy as `scripts/peptide/analysis/`. These retained tables, figures,
and logs are separated from executable code so an analysis run cannot overwrite
the repository's script sources.
