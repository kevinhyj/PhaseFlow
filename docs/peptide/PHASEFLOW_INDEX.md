# Peptide Documentation Index

| Document | Purpose |
| --- | --- |
| [README](README.md) | Installation, data contract, training, inference, and document map |
| [Architecture](ARCHITECTURE.md) | Model components, objectives, and configuration fields |
| [Technical overview](phaseflow_technical_overview.md) | Implementation-oriented description and checkpoint compatibility |
| [Data loading](DATALOADER_OPTIMIZATION.md) | Input schema, missing values, split modes, and loader guidance |
| [Script reference](scripts.md) | Shell launcher and Python entry-point usage |
| [Evaluation protocol](ANALYSIS.md) | Reproducible analysis and reporting requirements |
| [Experiment guide](PHASEFLOW_EXPLORATION.md) | Procedure for new experiments and fair comparisons |

## Source Map

```text
phaseflow/
  tokenizer.py                 Amino-acid tokenization
  data.py                      Peptide dataset and dataloaders
  model.py                     Bidirectional peptide model
  transformer.py               Transformer implementation
  utils.py                     Configuration, checkpoints, metrics, and utilities

configs/peptide/              Canonical peptide configurations
research/peptide/experiments/ Training, prediction, generation, and evaluation tools
scripts/peptide/              Convenience launchers
tests/peptide/                Smoke tests
```

Use the repository root README for cross-workflow installation and artifact
policy. Use `docs/full_length/` for protein-level LLPS and DPR documentation.
