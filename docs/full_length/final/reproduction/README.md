# PhaseFlow DPR v6 rank_p257 Reproduction Package

This directory is the paper-facing reproduction package for the final PhaseFlow DPR v6 `rank_p257` model.

## Files

- `artifact_manifest.tsv`: compact checksum and provenance table for the required code, data, checkpoints, and reports.
- `../../dpr_v6_rankp257_repro_report_20260617.md`: cleaned final DPR reproduction report.
- `../../../configs/full_length/final_dpr.yaml`: paper-facing final DPR configuration.

## Final Artifact

- Final checkpoint: `artifacts/model/checkpoints/update_000050.pt`
- SHA256: `7fb0091e6dd5a85bd3a6be7a0b606501700c4b8f28ff9b6e309267835a2fdff0`
- Final selected run: `d1_flat_seed174_raw_planc_rankp257_p257_lr5e-6_50u_seed202606188`
- Selection protocol: PlanD non-PhasePro validation composite; PhasePro was not used for the final downstream selection.
