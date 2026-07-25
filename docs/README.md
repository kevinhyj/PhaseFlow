# PhaseFlow Documentation

PhaseFlow contains two related research workflows with separate data contracts
and training entry points.

| Workflow | Documentation | Scope |
| --- | --- | --- |
| Short peptides | [Peptide workflow](peptide/README.md) | Bidirectional sequence-to-phase-diagram modeling and peptide generation |
| Full-length proteins | [Full-length workflow](full_length/README.md) | Protein-level LLPS prediction and residue-level DPR prediction |

Start with the repository [README](../README.md) for installation and the
artifact-release policy. The workflow guides explain their inputs, training
commands, evaluation interfaces, and artifacts in detail.

All paths in this documentation are repository-relative. Large datasets,
checkpoints, and generated outputs are intentionally distributed outside Git.
