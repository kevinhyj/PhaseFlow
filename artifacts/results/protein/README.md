# Protein Results

This directory is a curated, versioned protein publication-results package. It
keeps final benchmark tables, ablation summaries, canonical figures, and the
small scripts used to render manuscript figures and tables in one place. It is
not a source of training inputs or a replacement for the documented
model-reproduction workflow.

The collection is grouped into [LLPS benchmark results](benchmark/llps/),
[DPR benchmark results](benchmark/dpr/), [LLPS ablations](ablation/llps/),
[DPR ablations](ablation/dpr/), [benchmark figures](figures/benchmark/),
[ablation figures](figures/ablation/), and the accompanying
[`scripts/`](scripts/) directory. `scripts/figures/` contains the publication
figure renderers; `scripts/tables/` contains the Markdown and PDF table
generators. These scripts are source code, not release payloads, and are
therefore deliberately excluded from the result manifest.

```text
protein/
├── benchmark/              released LLPS and DPR benchmark tables
├── ablation/               released ablation summaries
├── figures/                rendered publication assets
└── scripts/
    ├── figures/            figure renderers and asset assembler
    └── tables/             manuscript-table renderers
```

Run a generator from the repository root. The LLPS/DPR benchmark renderers,
the LLPS embedding ablation, and the two DPR stream-ablation renderers default
to their adjacent released CSV inputs; always provide an explicit output
directory so a rebuild does not modify the tracked figure snapshot:

```bash
python artifacts/results/protein/scripts/figures/plot_llps_benchmark.py \
  --output-dir /tmp/phaseflow-llps-benchmark
python artifacts/results/protein/scripts/figures/plot_dpr_benchmark.py \
  --output-dir /tmp/phaseflow-dpr-benchmark
python artifacts/results/protein/scripts/figures/plot_llps_embedding_ablation.py \
  --output-dir /tmp/phaseflow-llps-ablation
python artifacts/results/protein/scripts/tables/generate_tables.py --help
python artifacts/results/protein/scripts/tables/generate_tables_pdf.py \
  --task all --output_dir /tmp/phaseflow-paper-tables
```

`plot_dpr_ablation_summary.py`, `plot_phasepro_dpr_top12.py`, and the asset
assembler require explicit inputs because their source metric/profile archives
are intentionally not part of this compact release snapshot. The PDF table
generator writes `llps_tables.pdf` and `dpr_tables.pdf` beside the respective
released benchmark tables unless `--output_dir` is supplied.

The `source_archive_root` field in [manifest.json](manifest.json) identifies an
external release archive; every manifest `source` is relative to that archive,
not to this Git repository. Each published-result entry records its byte count
and SHA256 digest.

Published tables use the `PhaseFlow` model identity and `llps_score` field name.
The benchmark summary JSON and stream-ablation CSV are additionally derived by
removing non-release artifact references. Their `sha256` identifies the
published payload, `source_sha256` identifies the external archive source, and
`transformation` records the fixed normalization rule.

Checkpoints, model sidecars, feature caches, profiles, logs, and other
intermediate training artifacts are deliberately excluded. Rebuild data and
derived artifacts through the public reproduction workflow instead of treating
this package as an input dataset.
