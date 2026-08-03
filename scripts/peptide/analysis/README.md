# Peptide Analysis Scripts

This directory contains reproducible short-peptide analyses grouped by study:
`de_novo/`, `flow_vs_ddpm_ito/`, `length_kmer_kl/`, `loglikelihood/`,
`scaling/`, `sliding_window_scoring/`, and `visualization/`.

Analysis scripts are command-line entry points, not package modules. Curated
tables, figures, and logs produced by these analyses are stored under
`artifacts/results/peptide/analysis/` with the same study hierarchy. Use
explicit output arguments where available to keep new runs separate from the
retained publication artifacts.
