#!/usr/bin/env python3
"""Write all-length label-tier reports from the current processed manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from build_model_ready_dataset import (
    _scope_skip_reasons,
    collapse_canonical,
    load_benchmark_sets,
    load_deleted_sets,
    normalize_candidate_frame,
    normalize_old_manifest,
    read_csv,
)


DATE = "20260607"

LEN_BUCKET_ORDER = [
    "short_lt30",
    "short_30_100",
    "normal_100_2048",
    "long_2048_2700",
    "very_long_2700_5537",
    "ultra_long_gt5537",
]

SAMPLER_ORDER = [
    "hard_positive",
    "pseudo_positive",
    "structured_negative",
    "disordered_negative",
    "associated_context",
    "unknown_pu",
]

TIER_ORDER = ["gold", "curated", "silver", "pseudo", "unknown"]


def fmt(value: Any) -> str:
    if pd.isna(value):
        return "0"
    return f"{int(value):,}"


def bool_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df:
        return pd.Series(default, index=df.index)
    return df[col].fillna(default).astype(bool)


def first_existing_m8(root: Path) -> tuple[Path | None, set[str]]:
    candidates = sorted((root / "data/interim/model_ready").glob("mmseqs40_benchmark_*/benchmark_vs_model.m8"))
    if not candidates:
        return None, set()
    path = candidates[-1]
    homolog_keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            target = parts[1]
            if target.startswith("model|"):
                homolog_keys.add(target.split("|", 1)[1])
    return path, homolog_keys


def model_rows_as_canonical(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["canonical_key"] = df["canonical_key"].astype(str)
    out["protein_id"] = df.get("protein_id", df["canonical_key"]).astype(str)
    for col in ["uniprot_acc", "gene_name", "organism", "taxonomy_id", "sequence", "sequence_md5"]:
        out[col] = df.get(col, pd.Series("", index=df.index))
    out["length"] = pd.to_numeric(df.get("length", out["sequence"].astype(str).str.len()), errors="coerce").fillna(0).astype(int)
    out["final_llps_label"] = pd.to_numeric(df.get("final_llps_label", df.get("llps_label", -100)), errors="coerce").fillna(-100).astype(int)
    out["final_label_tier"] = df.get("final_label_tier", df.get("label_tier", "unknown")).fillna("unknown").astype(str)
    out["final_role_label"] = df.get("final_role_label", df.get("role_label", "unknown")).fillna("unknown").astype(str)
    out["final_negative_type"] = df.get("final_negative_type", df.get("negative_type", "none")).fillna("none").astype(str)
    out["sample_weight"] = pd.to_numeric(df.get("sample_weight", 0.0), errors="coerce").fillna(0.0)
    out["sampler_group"] = df.get("sampler_group", "unknown_pu").fillna("unknown_pu").astype(str)
    out["source"] = df.get("source", "").fillna("").astype(str)
    out["sources"] = df.get("sources", out["source"]).fillna("").astype(str)
    out["final_leakage_status"] = "clean"
    out["final_leakage_reason"] = ""
    for col in ["seq_valid", "bad_seq", "len_bucket", "train_scope", "teacher_scope", "skip_reason"]:
        out[col] = df.get(col, "")
    return out


def selected_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "source" not in out:
        out["source"] = out.get("sources", pd.Series("", index=out.index)).fillna("").astype(str).str.split(";").str[0]
    if "sources" not in out:
        out["sources"] = out["source"]
    columns = [
        "protein_id",
        "canonical_key",
        "sequence",
        "length",
        "final_llps_label",
        "final_label_tier",
        "final_role_label",
        "final_negative_type",
        "sample_weight",
        "sampler_group",
        "seq_valid",
        "bad_seq",
        "len_bucket",
        "train_scope",
        "teacher_scope",
        "skip_reason",
        "final_leakage_status",
        "final_leakage_reason",
        "source",
        "sources",
        "uniprot_acc",
        "gene_name",
        "organism",
        "taxonomy_id",
        "sequence_md5",
    ]
    for col in columns:
        if col not in out:
            out[col] = ""
    return out[columns]


def count_table(df: pd.DataFrame, col: str, order: list[str] | None = None) -> list[tuple[str, int]]:
    counts = df[col].fillna("").astype(str).value_counts().to_dict()
    rows: list[tuple[str, int]] = []
    used: set[str] = set()
    if order:
        for value in order:
            if value in counts:
                rows.append((value, int(counts[value])))
                used.add(value)
    for value in sorted(set(counts) - used):
        rows.append((value, int(counts[value])))
    return rows


def markdown_count_table(title: str, rows: list[tuple[str, int]]) -> list[str]:
    lines = [f"## {title}", "", "| 层级 | 蛋白数 |", "| --- | ---: |"]
    lines.extend(f"| `{label}` | {fmt(count)} |" for label, count in rows)
    lines.append("")
    return lines


def markdown_crosstab(df: pd.DataFrame, index: str, columns: str, index_order: list[str], column_order: list[str]) -> list[str]:
    tab = pd.crosstab(df[index], df[columns])
    tab = tab.reindex(index=index_order, columns=column_order, fill_value=0)
    lines = ["| 长度桶 | " + " | ".join(f"`{col}`" for col in column_order) + " |", "| --- | " + " | ".join("---:" for _ in column_order) + " |"]
    for idx, row in tab.iterrows():
        lines.append("| `" + idx + "` | " + " | ".join(fmt(row[col]) for col in column_order) + " |")
    return lines


def write_report(
    root: Path,
    all_clean: pd.DataFrame,
    len_oos: pd.DataFrame,
    model_train: pd.DataFrame,
    candidate: pd.DataFrame,
    teacher: pd.DataFrame,
    source_counts: dict[str, int],
    m8_path: Path | None,
    patched_from_model_train: int,
    train_scope_rep_mismatch: int,
) -> Path:
    reports = root / "data/reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"all_length_label_tier_summary_{DATE}.md"

    train_scope = bool_series(all_clean, "train_scope")
    teacher_scope = bool_series(all_clean, "teacher_scope")
    bad_seq = bool_series(all_clean, "bad_seq")

    lines: list[str] = [
        "# 全长度标签层级统计",
        "",
        "统计对象为当前 processed 数据：`full_candidate_pool.csv`、`active_train_manifest.csv`、旧 teacher manifest（如果存在）和已落盘的 `model_train_manifest.csv`。",
        "",
        "## 关键结论",
        "",
        f"- 全长度 clean canonical 蛋白数：{fmt(len(all_clean))}。",
        f"- 当前训练清单以 `model_train_manifest.csv` 为准：{fmt(len(model_train))} 条，全部 `train_scope=true`，不做 valid split。",
        f"- 全长度 canonical 代表序列中 `train_scope=true` 为 {fmt(int(train_scope.sum()))} 条；长度桶只作为审计/采样/模型处理标记，不再决定是否进 train。",
        f"- `train_scope=false` 但 `seq_valid=true` 的蛋白：{fmt(len(len_oos))}；新口径下目标应为 0。",
        f"- `teacher_scope=true`：{fmt(int(teacher_scope.sum()))}；`teacher_scope=false`：{fmt(int((~teacher_scope).sum()))}。",
        f"- `bad_seq`：{fmt(int(bad_seq.sum()))}。",
        "- 所有非 PPMC full / PhasePro exact duplicate 且 seq_valid 的序列都进入 train；监督损失由 sampler_group / merged_label_tier / loss mask 决定。",
        "",
        "## 输入与输出",
        "",
        "| 项目 | 数量 | 说明 |",
        "| --- | ---: | --- |",
        f"| full_candidate_pool clean 来源行 | {fmt(source_counts.get('full_clean_rows', 0))} | 已排除 PPMC full / PhasePro exact duplicate 的来源行；MMseqs40 只审计 |",
        f"| active_train_manifest 来源行 | {fmt(source_counts.get('active_rows', 0))} | 当前 active train 证据 |",
        f"| old teacher clean 来源行 | {fmt(source_counts.get('old_teacher_rows', 0))} | 旧 round0 teacher 证据；不存在则为 0 |",
        f"| canonical collapse 后 clean | {fmt(source_counts.get('canonical_clean_before_patch', 0))} | 完整证据仲裁后 clean canonical |",
        f"| 从 model_train_manifest 补齐缺失 key | {fmt(patched_from_model_train)} | 当前目录缺少旧增强 manifest 时的兜底；当前为 0 表示 key 未缺失 |",
        f"| train-scope 代表序列差异 key | {fmt(train_scope_rep_mismatch)} | 新口径下目标应为 0 |",
        f"| all_length_label_manifest.csv | {fmt(len(all_clean))} | clean 全长度标签 manifest |",
        f"| candidate_manifest.csv | {fmt(len(candidate))} | 合法候选代表行，含 leakage/len_oos 标记 |",
        f"| teacher_manifest.csv | {fmt(len(teacher))} | protein-level teacher 输入 |",
        "",
    ]
    if m8_path is not None:
        lines.append(f"复用已有 model-ready MMseqs40 结果：`{m8_path}`。")
        lines.append("")

    lines.extend(markdown_count_table("全长度 clean 按 sampler_group", count_table(all_clean, "sampler_group", SAMPLER_ORDER)))
    lines.extend(markdown_count_table("全长度 clean 按 final_label_tier", count_table(all_clean, "final_label_tier", TIER_ORDER)))
    lines.extend(markdown_count_table("全长度 clean 按 final_llps_label", count_table(all_clean, "final_llps_label")))
    lines.extend(markdown_count_table("全长度 clean 按长度桶", count_table(all_clean, "len_bucket", LEN_BUCKET_ORDER)))

    lines.extend(
        [
            "## 长度桶 x 标签层级",
            "",
            *markdown_crosstab(all_clean, "len_bucket", "sampler_group", LEN_BUCKET_ORDER, SAMPLER_ORDER),
            "",
            "## `train_scope=false` 残留审计",
            "",
            f"`train_scope=false` 总数为 {fmt(len(len_oos))}。新口径要求所有 clean seq_valid 序列进入 train，因此这里应为 0；长度桶分布保留作审计。",
            "",
            *markdown_crosstab(len_oos, "len_bucket", "sampler_group", LEN_BUCKET_ORDER, SAMPLER_ORDER),
            "",
        ]
    )

    lines.extend(markdown_count_table("当前 model_train_manifest 按 sampler_group", count_table(model_train, "sampler_group", SAMPLER_ORDER)))
    lines.extend(
        [
            "## 训练使用口径",
            "",
            "- `seq_valid=true`：序列本身合法，可以进入候选、audit 或 embedding 流程。",
            "- `train_scope=true`：非 PPMC full / PhasePro exact duplicate 的 clean 合法序列全部进入 train。",
            "- `len_bucket` / `length_oos`：只作为长度处理和采样审计字段，不再排除训练样本。",
            "- `all_length_label_manifest.csv` 是 canonical 代表序列口径，不替代 `model_train_manifest.csv`；需要复现当前主训练时必须读取 `model_train_manifest.csv`。",
            "- `teacher_scope=false`：当前不自动跑 ordinary protein-level teacher，也不自动产生 pseudo positive。",
            "- `final_llps_label=-100`：unknown/associated context，不应进入二分类监督损失，只能作为 PU/background/采样池使用。",
            "",
            "建议下游读取：",
            "",
            "| 文件 | 推荐用途 |",
            "| --- | --- |",
            "| `model_train_manifest.csv` | 当前主模型 full-length 监督训练 |",
            "| `all_length_label_manifest.csv` | 全长度 embedding、audit、chunk/MIL 准备 |",
            "| `teacher_manifest.csv` | protein-level teacher 输入 |",
            "| `short_manifest.csv` | `<100 aa` 特殊序列分析 |",
            "| `long_manifest.csv` | `>2048 aa` 长蛋白分析 |",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    processed = root / "data/processed"

    full_pool = read_csv(processed / "full_candidate_pool.csv")
    active = read_csv(processed / "active_train_manifest.csv")
    old_clean = read_csv(root / "data/pseudo_labels/round0_external/manifest_with_teacher.csv")
    old_aug = read_csv(processed / "augmented_train_manifest.csv")
    model_train = read_csv(processed / "model_train_manifest.csv")
    candidate = read_csv(processed / "candidate_manifest.csv")
    teacher = read_csv(processed / "teacher_manifest.csv")

    if full_pool.empty or model_train.empty:
        raise SystemExit("full_candidate_pool.csv and model_train_manifest.csv are required")

    full_clean = full_pool[
        full_pool["leakage_status"].astype(str).isin(["clean", "removed_mmseqs40_homolog"])
    ].copy()
    homolog_mask = full_clean["leakage_status"].astype(str).eq("removed_mmseqs40_homolog")
    full_clean.loc[homolog_mask, "leakage_status"] = "clean"
    if "leakage_reason" in full_clean:
        full_clean.loc[homolog_mask, "leakage_reason"] = ""
    parts = [normalize_candidate_frame(full_clean, "full_candidate_pool_clean_all_length_20260607")]
    if not active.empty:
        parts.append(normalize_candidate_frame(active, "current_full_active_entry_20260606"))
    if not old_clean.empty:
        parts.append(normalize_old_manifest(old_clean, "old_leakage_clean_train_20260606"))
    if not old_aug.empty:
        parts.append(normalize_old_manifest(old_aug, "previous_augmented_train_20260606"))

    evidence = pd.concat(parts, ignore_index=True, sort=False)
    canonical, _stats = collapse_canonical(evidence, load_benchmark_sets(root), load_deleted_sets(root))
    m8_path, homolog_keys = first_existing_m8(root)
    canonical["skip_reason"] = _scope_skip_reasons(canonical, "candidate")

    clean = canonical[canonical["final_leakage_status"].eq("clean")].copy()
    clean_keys = set(clean["canonical_key"].astype(str))
    missing_model = model_train[~model_train["canonical_key"].astype(str).isin(clean_keys)].copy()
    if not missing_model.empty:
        clean = pd.concat([clean, model_rows_as_canonical(missing_model)], ignore_index=True, sort=False)
        clean = clean.drop_duplicates("canonical_key", keep="first").copy()
        clean["skip_reason"] = _scope_skip_reasons(clean, "candidate")

    if "source" not in clean:
        clean["source"] = clean["sources"].fillna("").astype(str).str.split(";").str[0]

    rep_scope = clean[["canonical_key", "length", "len_bucket", "train_scope"]].copy()
    train_scope_overlap = model_train[["canonical_key", "length", "len_bucket", "train_scope"]].merge(
        rep_scope,
        on="canonical_key",
        suffixes=("_model_train", "_all_length_rep"),
    )
    train_scope_rep_mismatch = int(
        (
            train_scope_overlap["train_scope_model_train"].fillna(False).astype(bool)
            & ~train_scope_overlap["train_scope_all_length_rep"].fillna(False).astype(bool)
        ).sum()
    )

    manifest = selected_manifest_columns(clean)
    manifest_path = processed / "all_length_label_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    len_oos = clean[~bool_series(clean, "train_scope") & bool_series(clean, "seq_valid", True)].copy()
    report_path = write_report(
        root,
        clean,
        len_oos,
        model_train,
        candidate,
        teacher,
        {
            "full_clean_rows": len(full_clean),
            "active_rows": len(active),
            "old_teacher_rows": len(old_clean),
            "old_aug_rows": len(old_aug),
            "canonical_clean_before_patch": int(canonical["final_leakage_status"].eq("clean").sum()),
        },
        m8_path,
        len(missing_model),
        train_scope_rep_mismatch,
    )
    print(f"wrote {manifest_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
