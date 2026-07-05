#!/usr/bin/env python3
"""Generate LaTeX audit tables and rendered previews for the final data package."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "outputs" / "overall" / "final" / "data"
DEFAULT_TECTONIC = Path("tectonic")


class Raw(str):
    """String wrapper for LaTeX that should not be escaped."""


@dataclass(frozen=True)
class TableSpec:
    filename: str
    caption: str
    label: str
    headers: list[str | Raw]
    rows: list[list[str | Raw]]
    align: str
    note: str
    size: str = r"\small"
    wide: bool = False


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_int(value: int | str | float) -> str:
    return f"{int(float(value)):,}"


def fmt_pct(count: int | float, total: int | float, digits: int = 2) -> str:
    if not total:
        return "0"
    return f"{100.0 * float(count) / float(total):.{digits}f}"


def fmt_float(value: int | float | str | None, digits: int = 3) -> str:
    if value is None or value == "":
        return r"\na"
    return f"{float(value):.{digits}f}"


def latex_escape(value: object) -> str:
    if isinstance(value, Raw):
        return str(value)
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def display_source(source: str) -> str:
    names = {
        "BAV_LLPS_bav-llps-curated-ds": "BAV-LLPS curated",
        "BAV_LLPS_bav-llps-homologous-ds": "BAV-LLPS homologous",
        "CD_CODE": "CD-CODE",
        "CD_CODE_v2.2": "CD-CODE v2.2",
        "DisProt_current": "DisProt",
        "DrLLPS": "DrLLPS",
        "LLPSDB_v2": "LLPSDB v2.0",
        "PhaSepDB_3": "PhaSepDB 3.0",
        "RCSB_PDB_SEQRES": "RCSB PDB SEQRES",
        "UniProt_SwissProt_reviewed": "UniProtKB/Swiss-Prot",
        "PhaSepDB_3.0": "PhaSepDB 3.0",
    }
    return names.get(source, source)


def display_group(group: str) -> str:
    names = {
        "associated_context": "Associated context",
        "disordered_negative": "Disordered negative",
        "hard_positive": "Hard positive",
        "pseudo_positive": "Pseudo positive",
        "structured_negative": "Structured negative",
        "unknown_pu": "Unknown PU",
    }
    return names.get(group, group.replace("_", " "))


def strip_cjk_parenthetical(text: str) -> str:
    text = text.strip()
    if text == "其他":
        return "Other"
    return re.sub(r"\s*\([^)]*[\u4e00-\u9fff][^)]*\)", "", text)


def display_phasepro_item(item: str) -> str:
    names = {
        "蛋白数": "Proteins",
        "残基数": "Residues",
        "官方 regions": "Official regions",
        "Protenix graph 成功数": "Protenix graph successes",
        "STARLING graph 成功数": "STARLING graph successes",
        "合法 missing-modality 数": "Legal missing-modality count",
        "非法 fallback 数": "Illegal fallback count",
    }
    return names.get(item, item)


def use_text(value: str) -> str:
    names = {
        "all": "all",
        "weighted": "weighted",
        "sample_per_epoch": "sample/epoch",
        "all_with_oversampling_allowed": "all; oversampling allowed",
        "ignore_or_auxiliary_context": "ignored or auxiliary context",
        "ignore_or_nnPU": "ignored or nnPU",
    }
    return names.get(value, value.replace("_", " "))


def parse_markdown_table_after(path: Path, heading: str) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if heading in line:
            start = idx
            break
    if start is None:
        return []

    table_lines: list[str] = []
    in_table = False
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(stripped)
            in_table = True
        elif in_table:
            break
    if len(table_lines) < 2:
        return []

    rows: list[list[str]] = []
    for line in table_lines[2:]:
        cells = [cell.strip().strip("*") for cell in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def parse_benchmark_integrity_table(path: Path) -> list[list[str]]:
    rows = parse_markdown_table_after(path, "Current cross-model locked-threshold result")
    if not rows:
        rows = parse_markdown_table_after(path, "跨模型比较")
    return rows


def extract_bullet_value(path: Path, label: str) -> str | None:
    pattern = re.compile(rf"^- {re.escape(label)}:\s*(.+)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1).strip()
    return None


def parse_simple_yaml(path: Path) -> dict[str, object]:
    """Parse the small contracts used here without adding a YAML dependency."""
    data: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, data)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("-"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in raw_line:
            continue
        key, value = raw_line.strip().split(":", 1)
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            node: dict[str, object] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = value.strip("'\"")
    return data


def table_to_latex(spec: TableSpec) -> str:
    env = "table*" if spec.wide else "table"
    lines = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\begin{threeparttable}",
        rf"\caption{{{latex_escape(spec.caption)}}}",
        rf"\label{{{latex_escape(spec.label)}}}",
        spec.size,
        rf"\begin{{tabular}}{{@{{}}{spec.align}@{{}}}}",
        r"\toprule",
        " & ".join(latex_escape(h) for h in spec.headers) + r" \\",
        r"\midrule",
    ]
    for row in spec.rows:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    if spec.note:
        lines.extend(
            [
                r"\begin{tablenotes}[flushleft]",
                r"\footnotesize",
                rf"\item \textit{{Note:}} {latex_escape(spec.note)}",
                r"\end{tablenotes}",
            ]
        )
    lines.extend([r"\end{threeparttable}", rf"\end{{{env}}}", ""])
    return "\n".join(lines)


def preamble(title: str, landscape: bool = True) -> str:
    geometry = "margin=0.55in,landscape" if landscape else "margin=0.75in"
    return rf"""\documentclass[10pt]{{article}}
\usepackage[{geometry}]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{threeparttable}}
\usepackage{{array}}
\usepackage{{url}}
\newcommand{{\na}}{{--}}
\makeatletter
\setlength{{\@fptop}}{{0pt}}
\setlength{{\@fpsep}}{{8pt plus 2pt minus 2pt}}
\setlength{{\@fpbot}}{{0pt plus 1fil}}
\makeatother
\pagestyle{{empty}}
\begin{{document}}
"""


def document_for_tables(title: str, specs: Iterable[TableSpec]) -> str:
    body = [preamble(title, landscape=True)]
    for spec in specs:
        body.append(table_to_latex(spec))
        body.append(r"\clearpage")
    body.append(r"\end{document}" + "\n")
    return "\n".join(body)


def wrapper_for_table(spec: TableSpec) -> str:
    return preamble(spec.caption, landscape=True) + table_to_latex(spec) + "\n\\end{document}\n"


def make_llps_tables(data_root: Path) -> list[TableSpec]:
    llps = data_root / "llps"
    summary = read_json(llps / "statistics_summary.json")
    crosstab = read_csv(llps / "source_sampler_group_crosstab.csv")
    registry = read_json(llps / "source_registry.json")
    sampler = read_json(llps / "model_sampler_config.json")
    report = llps / "training_data_audit_report_cn.md"

    total = int(summary["total_proteins"])
    source_rows: list[list[str]] = []
    totals = {
        "associated_context": 0,
        "disordered_negative": 0,
        "hard_positive": 0,
        "pseudo_positive": 0,
        "structured_negative": 0,
        "unknown_pu": 0,
    }
    sorted_sources = sorted(
        crosstab,
        key=lambda row: sum(int(row[key]) for key in totals),
        reverse=True,
    )
    for row in sorted_sources:
        row_total = sum(int(row[key]) for key in totals)
        for key in totals:
            totals[key] += int(row[key])
        source_rows.append(
            [
                display_source(row["source"]),
                fmt_int(row_total),
                fmt_int(row["hard_positive"]),
                fmt_int(row["pseudo_positive"]),
                fmt_int(row["structured_negative"]),
                fmt_int(row["disordered_negative"]),
                fmt_int(row["associated_context"]),
                fmt_int(row["unknown_pu"]),
            ]
        )
    source_rows.append(
        [
            "Total",
            fmt_int(sum(totals.values())),
            fmt_int(totals["hard_positive"]),
            fmt_int(totals["pseudo_positive"]),
            fmt_int(totals["structured_negative"]),
            fmt_int(totals["disordered_negative"]),
            fmt_int(totals["associated_context"]),
            fmt_int(totals["unknown_pu"]),
        ]
    )

    tier_counts = summary["final_label_tier_counts"]
    tier_names = {
        "curated": "Curated",
        "pseudo": "Pseudo",
        "unknown": "Unknown/context",
    }
    tier_notes = {
        "curated": "experimentally curated positives and curated negatives",
        "pseudo": "weakly supervised positives",
        "unknown": "unlabeled proteins and associated context",
    }
    tier_rows = [
        [
            tier_names[key],
            tier_notes[key],
            fmt_int(tier_counts[key]),
            fmt_pct(tier_counts[key], total),
        ]
        for key in ["curated", "pseudo", "unknown"]
    ]

    group_counts = summary["sampler_group_counts"]
    sampler_rows: list[list[str]] = []
    for key in [
        "hard_positive",
        "pseudo_positive",
        "structured_negative",
        "disordered_negative",
        "associated_context",
        "unknown_pu",
    ]:
        cfg = sampler["groups"][key]
        weight = cfg.get("sample_weight", 0)
        use = use_text(str(cfg.get("use", "")))
        if "target_per_epoch" in cfg:
            use = f"{use}; cap {fmt_int(cfg['target_per_epoch'])}"
        sampler_rows.append(
            [
                display_group(key),
                fmt_int(group_counts[key]),
                fmt_pct(group_counts[key], total),
                str(weight),
                use,
            ]
        )

    len_order = [
        ("short_lt30", r"$<30$ aa"),
        ("short_30_100", "30--100 aa"),
        ("normal_100_2048", "100--2,048 aa"),
        ("long_2048_2700", "2,048--2,700 aa"),
        ("very_long_2700_5537", "2,700--5,537 aa"),
        ("ultra_long_gt5537", r"$>5,537$ aa"),
    ]
    len_counts = summary["len_bucket_counts"]
    length_rows = [
        [Raw(label), fmt_int(len_counts[key]), fmt_pct(len_counts[key], total)]
        for key, label in len_order
    ]

    species_rows_raw = parse_markdown_table_after(report, "### 4.1 物种分布")
    hard_total = int(summary["sampler_group_counts"]["hard_positive"])
    species_rows: list[list[str]] = []
    for species, count in species_rows_raw:
        species_rows.append([strip_cjk_parenthetical(species), fmt_int(count), fmt_pct(int(count), hard_total, 1)])

    citation_rows: list[list[str | Raw]] = []
    for src in registry["sources"]:
        citation = src.get("citation_or_doi") or src.get("landing_page") or ""
        citation_rows.append(
            [
                src["name"],
                src["primary_use"],
                src["tier"],
                Raw(rf"\url{{{citation}}}") if citation else r"\na",
            ]
        )

    overview_rows = [
        ["Total proteins", fmt_int(summary["total_proteins"])],
        ["LLPS positives", fmt_int(summary["positive_proteins"])],
        ["LLPS negatives", fmt_int(summary["negative_proteins"])],
        ["Unlabeled or context proteins", fmt_int(summary["unlabeled_proteins"])],
        ["Hard positives", fmt_int(summary["sampler_group_counts"]["hard_positive"])],
        ["Pseudo positives", fmt_int(summary["sampler_group_counts"]["pseudo_positive"])],
        ["DPR silver-candidate proteins", fmt_int(summary["has_dpr_silver_candidate"])],
    ]

    return [
        TableSpec(
            "tab_data_sources.tex",
            "LLPS training data source distribution.",
            "tab:llps-data-sources",
            [
                "Data source",
                "Total",
                "Hard +",
                "Pseudo +",
                "Struct. -",
                "Disord. -",
                "Context",
                "Unknown",
            ],
            source_rows,
            "lrrrrrrr",
            "Counts are unique proteins after source harmonization. Struct. - denotes structured negatives and Disord. - denotes disordered negatives.",
            size=r"\scriptsize",
            wide=True,
        ),
        TableSpec(
            "tab_label_tiers.tex",
            "LLPS label-tier distribution.",
            "tab:llps-label-tiers",
            ["Final label tier", "Definition", "Proteins", r"\%"],
            tier_rows,
            "llrr",
            "Final label tiers summarize the curated, pseudo-labeled, and unknown/context partitions used by the LLPS training pool.",
        ),
        TableSpec(
            "tab_sampler_groups.tex",
            "LLPS sampler-group distribution and training use.",
            "tab:llps-sampler-groups",
            ["Sampler group", "Proteins", r"\%", "Weight", "Training use"],
            sampler_rows,
            "lrrrl",
            "Sampling weights and per-epoch caps are read from model_sampler_config.json.",
            size=r"\footnotesize",
        ),
        TableSpec(
            "tab_length_distribution.tex",
            "Sequence length distribution in the LLPS training pool.",
            "tab:llps-length-distribution",
            ["Length range", "Proteins", r"\%"],
            length_rows,
            "lrr",
            "Length buckets are used for sampling audit only; they are not an exclusion criterion in this audit snapshot.",
        ),
        TableSpec(
            "tab_species_distribution.tex",
            "Species distribution of experimentally validated LLPS drivers.",
            "tab:llps-species-distribution",
            ["Species", "Proteins", r"\% of hard positives"],
            species_rows,
            "lrr",
            "The denominator is 543 hard-positive proteins.",
        ),
        TableSpec(
            "tab_data_citations.tex",
            "Public databases and citations used for LLPS training data construction.",
            "tab:llps-data-citations",
            ["Database", "Primary use", "Tier", "Citation or URL"],
            citation_rows,
            r"p{0.18\textwidth}p{0.42\textwidth}p{0.16\textwidth}p{0.18\textwidth}",
            "PPMC-lab LLPS Datasets are listed for provenance even when benchmark leakage filtering removes overlapping benchmark proteins.",
            size=r"\scriptsize",
            wide=True,
        ),
        TableSpec(
            "tab_dataset_overview.tex",
            "LLPS data audit overview.",
            "tab:llps-dataset-overview",
            ["Audit item", "Value"],
            overview_rows,
            "lr",
            "The audit report was generated on 2026-06-19.",
        ),
    ]


def make_dpr_tables(data_root: Path) -> list[TableSpec]:
    dpr = data_root / "dpr"
    report = dpr / "training_data_audit_report_cn.md"
    label_audit = dpr / "dpr_label_semantics_audit.md"
    benchmark_audit = dpr / "dpr_benchmark_integrity_audit.md"
    pool_rows_raw = read_csv(dpr / "dpr_training_pool_by_tier.csv")
    model_contract = parse_simple_yaml(dpr / "model_contract.yaml")
    feature_contract = parse_simple_yaml(dpr / "feature_contract.yaml")

    pool_order = [
        "gold_high",
        "pseudo_weak",
        "hard_negative",
        "structured_negative",
        "bag_context",
        "ignored",
    ]
    pool_rows_raw = sorted(
        pool_rows_raw,
        key=lambda row: (
            pool_order.index(row["dpr_training_tier"])
            if row["dpr_training_tier"] in pool_order
            else 99,
            row["merged_label_tier"],
        ),
    )
    pool_rows = [
        [
            row["dpr_training_tier"],
            row["merged_label_tier"],
            fmt_int(row["rows"]),
            fmt_int(row["unique_proteins"]),
            fmt_int(row["positive_proteins"]),
            fmt_int(row["valid_residue_count"]),
            fmt_int(row["positive_residue_count"]),
            fmt_int(row["span_count"]),
        ]
        for row in pool_rows_raw
    ]

    silver_sources_raw = parse_markdown_table_after(report, "### 4.1 DPR 银标数据来源分布")
    silver_sources: list[list[str]] = []
    silver_total = 532
    for source, count in silver_sources_raw:
        if "总计" in source:
            continue
        count_i = int(count.replace(",", ""))
        silver_sources.append([display_source(source), fmt_int(count_i), fmt_pct(count_i, silver_total, 1)])

    silver_labels_raw = parse_markdown_table_after(report, "### 4.2 DPR 银标蛋白的 LLPS 标签分布")
    silver_labels: list[list[str]] = []
    for group, count in silver_labels_raw:
        count_i = int(count.replace(",", ""))
        silver_labels.append([display_group(group), fmt_int(count_i), fmt_pct(count_i, silver_total, 1)])

    benchmark_raw = parse_benchmark_integrity_table(benchmark_audit)
    benchmark_rows: list[list[str]] = []
    for row in benchmark_raw:
        if len(row) < 7:
            continue
        model, auprc, spearman, recall, precision, f1, neg_false = row[:7]
        benchmark_rows.append(
            [
                model,
                fmt_float(auprc),
                fmt_float(spearman),
                fmt_float(recall),
                fmt_float(precision),
                fmt_float(f1),
                fmt_float(neg_false),
            ]
        )

    semantics_rows = [
        [
            "Reliable residue spans",
            "Residue-level balanced BCE and Dice",
            "Positive residue supervision",
        ],
        [
            "region_bag_label=1 without span",
            "Presence/MIL only",
            "Not used as residue supervision",
        ],
        [
            "Curated structured/disordered negatives",
            "Negative residue and presence losses",
            "Confirmed residue negatives",
        ],
        [
            "Associated context",
            "Context or presence-only signal",
            "Not confirmed residue negatives",
        ],
        [
            "Unknown PU",
            "Ignored for DPR residue loss",
            "Not confirmed residue negatives",
        ],
        [
            "Pseudo weak spans",
            "Lower tier-weighted residue loss",
            "Weak positive residue supervision",
        ],
    ]

    feature_dims = feature_contract.get("feature_dims", {})
    graph_schema = feature_contract.get("graph_schema", {})
    dtype = feature_contract.get("dtype", {})
    config_rows = [
        ["Model name", str(model_contract.get("name", ""))],
        ["Training updates", fmt_int(str(model_contract.get("updates", "0")))],
        ["Checkpoint SHA256", str(model_contract.get("checkpoint_sha256", ""))[:16] + "..."],
        ["PLM dimension", str(feature_dims.get("plm", "1280"))],
        ["Biophysical dimension", str(feature_dims.get("biophys", "112"))],
        ["PhaseFlow hidden dimension", str(feature_dims.get("full_length_llps_hidden", "256"))],
        ["Edge-attribute dimension", str(graph_schema.get("edge_attr_dim", "32"))],
        ["Maximum neighbors", str(graph_schema.get("max_neighbors", "96"))],
        ["STARLING node embedding", str(model_contract.get("starling_node_embedding", "disabled"))],
        ["STARLING sparse edges", str(model_contract.get("starling_edges", "disabled"))],
        ["Protenix sparse edges", str(model_contract.get("protenix_edges", "optional"))],
        ["PLM dtype", str(dtype.get("plm", "float16"))],
    ]

    phasepro_rows_raw = parse_markdown_table_after(report, "### 7.1 PhasePro Sidecar 信息")
    phasepro_rows = [[display_phasepro_item(name), value] for name, value in phasepro_rows_raw]
    if not phasepro_rows:
        phasepro_rows = [
            ["Sidecar proteins", fmt_int(feature_contract.get("proteins", "121"))],
            ["Sidecar residues", fmt_int(feature_contract.get("total_residues", "86660"))],
        ]

    overview_rows = [
        ["Sample-index rows", extract_bullet_value(label_audit, "total rows") or r"\na"],
        ["Positive-residue proteins", extract_bullet_value(label_audit, "positive residue proteins") or r"\na"],
        ["Confirmed negative proteins", extract_bullet_value(label_audit, "confirmed negative proteins") or r"\na"],
        ["Bag-positive proteins without span", extract_bullet_value(label_audit, "bag-positive-no-span proteins") or r"\na"],
        ["Ignored or unknown rows", extract_bullet_value(label_audit, "ignored/unknown rows") or r"\na"],
        ["Bad span rows", extract_bullet_value(label_audit, "bad span rows") or r"\na"],
    ]

    return [
        TableSpec(
            "tab_dpr_pool_tier.tex",
            "DPR training pool by supervision tier.",
            "tab:dpr-pool-tier",
            [
                "DPR tier",
                "Merged label tier",
                "Rows",
                "Proteins",
                "Pos. proteins",
                "Valid residues",
                "Pos. residues",
                "Spans",
            ],
            pool_rows,
            "llrrrrrr",
            "Rows are tier assignments from dpr_training_pool_by_tier.csv; the sample-index total is audited separately because some proteins appear in more than one supervision role.",
            size=r"\scriptsize",
            wide=True,
        ),
        TableSpec(
            "tab_dpr_silver_sources.tex",
            "Source distribution of DPR silver-candidate proteins.",
            "tab:dpr-silver-sources",
            ["Source database", "Proteins", r"\%"],
            silver_sources,
            "lrr",
            "Percentages use 532 DPR silver-candidate proteins as the denominator.",
        ),
        TableSpec(
            "tab_dpr_silver_llps_labels.tex",
            "LLPS sampler-group distribution among DPR silver-candidate proteins.",
            "tab:dpr-silver-llps-labels",
            ["LLPS sampler group", "Proteins", r"\%"],
            silver_labels,
            "lrr",
            "Most DPR silver-candidate proteins are LLPS-positive at protein level.",
        ),
        TableSpec(
            "tab_dpr_benchmark.tex",
            "PhasePro DPR cross-model benchmark under the audited locked-threshold protocol.",
            "tab:dpr-benchmark",
            [
                "Model",
                "Residue AUPRC",
                "Spearman",
                "Region recall",
                "Region precision",
                r"F1 @ IoU 0.25",
                "Negative false DPR",
            ],
            benchmark_rows,
            "lrrrrrr",
            "Metrics are reproduced from dpr_benchmark_integrity_audit.md using sigmoid scores, smoothing, threshold 0.5, merge gap 4, and minimum region length 8.",
            size=r"\footnotesize",
            wide=True,
        ),
        TableSpec(
            "tab_dpr_label_semantics.tex",
            "DPR training-label semantics and loss routing.",
            "tab:dpr-label-semantics",
            ["Label type", "Loss routing", "Residue-level interpretation"],
            semantics_rows,
            r"p{0.24\textwidth}p{0.34\textwidth}p{0.30\textwidth}",
            "BCE denotes binary cross-entropy and MIL denotes multiple-instance learning.",
            size=r"\footnotesize",
            wide=True,
        ),
        TableSpec(
            "tab_dpr_config.tex",
            "DPR model and feature-contract configuration.",
            "tab:dpr-config",
            ["Configuration item", "Value"],
            config_rows,
            "ll",
            "The DPR v3 portable contract disables STARLING node embeddings and sparse edges; Protenix sparse edges are optional.",
        ),
        TableSpec(
            "tab_phasepro_benchmark.tex",
            "PhaSePro sidecar dataset used for DPR audit.",
            "tab:dpr-phasepro-sidecar",
            ["Audit item", "Value"],
            phasepro_rows,
            "lr",
            "PhaSePro provides experimentally curated LLPS drivers and DPR/segment boundaries for the localization audit.",
        ),
        TableSpec(
            "tab_dataset_overview.tex",
            "DPR data audit overview.",
            "tab:dpr-dataset-overview",
            ["Audit item", "Value"],
            overview_rows,
            "lr",
            "Counts are read from dpr_label_semantics_audit.md.",
        ),
    ]


def write_tables(specs: list[TableSpec], out_dir: Path, title: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in specs:
        path = out_dir / spec.filename
        path.write_text(table_to_latex(spec), encoding="utf-8")
        written.append(path)
    combined = out_dir / "tab_data_statistics.tex"
    combined.write_text(document_for_tables(title, specs), encoding="utf-8")
    written.append(combined)
    return written


def run_checked(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def compile_tex_to_pdf(tex: Path, pdf: Path, tectonic: Path, only_cached: bool) -> bool:
    if not tectonic.exists():
        return False
    cmd = [str(tectonic)]
    if only_cached:
        cmd.append("--only-cached")
    cmd.extend(["-c", "minimal"])
    with tempfile.TemporaryDirectory(prefix="phaseflow_tables_") as tmp_name:
        tmp = Path(tmp_name)
        run_checked(cmd + ["-o", str(tmp), str(tex)])
        built = tmp / (tex.stem + ".pdf")
        if not built.exists():
            return False
        shutil.copy2(built, pdf)
    return True


def compile_snippet_to_pdf(spec: TableSpec, pdf: Path, tectonic: Path, only_cached: bool) -> bool:
    if not tectonic.exists():
        return False
    cmd = [str(tectonic)]
    if only_cached:
        cmd.append("--only-cached")
    cmd.extend(["-c", "minimal"])
    with tempfile.TemporaryDirectory(prefix="phaseflow_table_") as tmp_name:
        tmp = Path(tmp_name)
        wrapper = tmp / (Path(spec.filename).stem + ".tex")
        wrapper.write_text(wrapper_for_table(spec), encoding="utf-8")
        run_checked(cmd + ["-o", str(tmp), str(wrapper)])
        built = tmp / (wrapper.stem + ".pdf")
        if not built.exists():
            return False
        shutil.copy2(built, pdf)
    return True


def render_png(pdf: Path, png_base: Path, single_file: bool) -> bool:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm or not pdf.exists():
        return False
    cmd = [pdftoppm, "-png", "-r", "220"]
    if single_file:
        cmd.append("-singlefile")
    cmd.extend([str(pdf), str(png_base)])
    run_checked(cmd)
    return True


def render_outputs(specs: list[TableSpec], out_dir: Path, tectonic: Path, only_cached: bool) -> None:
    for spec in specs:
        stem = Path(spec.filename).stem
        pdf = out_dir / f"{stem}.pdf"
        if compile_snippet_to_pdf(spec, pdf, tectonic, only_cached):
            render_png(pdf, out_dir / stem, single_file=True)

    combined_tex = out_dir / "tab_data_statistics.tex"
    combined_pdf = out_dir / "tab_data_statistics.pdf"
    if compile_tex_to_pdf(combined_tex, combined_pdf, tectonic, only_cached):
        render_png(combined_pdf, out_dir / "tab_data_statistics", single_file=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["llps", "dpr", "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tectonic", type=Path, default=DEFAULT_TECTONIC)
    parser.add_argument("--no-render", action="store_true", help="Only write LaTeX files.")
    parser.add_argument("--offline-tex", action="store_true", help="Do not let Tectonic fetch missing TeX packages.")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    tasks: list[tuple[str, list[TableSpec], Path, str]] = []
    if args.task in {"llps", "all"}:
        tasks.append(
            (
                "llps",
                make_llps_tables(data_root),
                data_root / "llps" / "figures",
                "PhaseFlow LLPS Data Audit Tables",
            )
        )
    if args.task in {"dpr", "all"}:
        tasks.append(
            (
                "dpr",
                make_dpr_tables(data_root),
                data_root / "dpr" / "figures",
                "PhaseFlow DPR Data Audit Tables",
            )
        )

    for task, specs, out_dir, title in tasks:
        written = write_tables(specs, out_dir, title)
        if not args.no_render:
            render_outputs(specs, out_dir, args.tectonic, args.offline_tex)
        print(f"{task}: wrote {len(written)} LaTeX files to {out_dir}")


if __name__ == "__main__":
    main()
