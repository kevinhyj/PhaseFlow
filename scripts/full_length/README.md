# PhaseFlow Script Index

This directory keeps maintained entry points and reusable builders for the
paper repository. One-off launchers, node-specific wrappers, generated caches,
and retired experiment-maintenance scripts are not included.

## Entry Points

- `training/run_dpr_v6.py`: final DPR v6 training entry point.
- `benchmark/final_overall_benchmark_from_profiles.py`: final LLPS/DPR
  benchmark table builder from frozen profiles.
- `evaluation/analyze_dpr_v6_threshold_curves.py`: threshold and curve
  analysis for the final PhasePro comparison.
- `evaluation/select_dpr_v6_plan_d_composite.py`: final Plan D model
  selection summary.
- `predict_idr_phaseflow.py`: sequence-level PhaseFlow prediction helper.
- `generate_paper_tables.py` and `generate_tables_pdf.py`: manuscript table
  renderers.

## Current Builders

- `data/build_model_ready_dataset.py`
- `data/apply_length_scope_policy.py`
- `data/rebuild_full_external_candidate_pool.py`
- `data/augment_train_external_sources.py`
- `data/build_server_final_dataset.py`
- `data/build_final_region_targets.py`

## Teacher Wrappers

The retained wrappers in `teacher/` expect public tool locations or explicit
environment variables such as `PSPHUNTER_REPO`, `PHASEMOTIF_PACKAGE`, and
`MPLCONFIGDIR`.
