# PhaseFlow Full-length pseudo-label / weak-label teacher weights audit

审计日期：2026-07-02

本文只整理最终用于 full-length 论文结果的 pseudo-label / weak-label 配置。结论以最终 LLPS run 的 `sample_index.parquet`、最终配置、最终 DPR 配置和 teacher 合并产物为准，不把早期 proposal、region/profile teacher 试验或废弃配置并入最终配置。

## 主要结论

最终 full-length LLPS 训练使用了已经物化到最终样本索引中的 protein-level pseudo-positive tier。最终 pseudo-positive 生成是 protein-only Round1 ABC teacher 规则，不是 residue/region/profile teacher 规则。核心 teacher 为 DeePhase、PSPHunter protein-level probability 和 PSPredictor。PSTP protein score 与 catGRANULE2 protein score在该流程中作为 audit auxiliary scores 保留，不能单独触发 `pseudo_positive_high`。FuzDrop、MolPhase、Opt-PredLLPS、PhaseFlow teacher、PSTP scan/profile、PSPHunter region/profile、catGRANULE2 region/profile 不属于最终 reported LLPS pseudo-positive 生成规则。

最终 LLPS 训练配置中 `teacher_llps`、`teacher_dpr`、`teacher_distill`、`final_region_teacher` 和 `final_key_teacher` loss 系数均为 0。因此，最终训练没有使用 teacher score 作为直接 distillation loss。最终进入蛋白级 weighted focal BCE 的训练权重来自 batch 中的 `sample_weight` 列；`llps_sample_weight` 是数据集中保留的标签置信/质量权重，不是最终 loader 直接传入 `sample_weight` 的列。

最终 DPR reported training 没有使用 teacher pseudo regions 或 teacher residue profiles。最终 DPR candidate index 只包含 `S1_CAUSAL_REGION`、`S2_VALIDATED_REGION`、`N2_DISORDERED_NEGATIVE` 和 `N3_STRUCTURED_NEGATIVE`，并且最终 DPR 配置中 `weak_tiers: []`。因此最终 DPR pseudo-label / weak-label teacher regions 应记为：not used in final reported DPR training。

## 1. 最终 full-length LLPS pseudo-positive 生成流程

最终 LLPS checkpoint 对应的样本索引为：

`external_artifacts/final_llps/experiments/s1_rank04_nd15_drop015/sample_index.parquet`

该索引共 107,037 行，其中主要 LLPS label status 为：

| llps_label_status | rows |
|---|---:|
| negative_curated_structured | 47,946 |
| unknown_pu_unlabeled | 45,509 |
| associated_context_unlabeled | 6,051 |
| pseudo_positive_high | 4,137 |
| pseudo_positive_weak_preserved | 2,180 |
| negative_curated_disordered | 891 |
| gold_positive | 323 |

### 1.1 Teacher 候选池

最终 pseudo-positive 生成使用 Round1 ABC protein-only candidate pool：

`data/pseudo_labels/round1_abc_protein_only_corrected_20260607/candidates/candidate_manifest.csv`

candidate policy 来自：

`external_artifacts/pseudo_label/teacher_protein_only_round1_abc.yaml`

该 policy 定义三类候选：

| tier | 作用 |
|---|---|
| A | associated-context 或 unknown PU 中带 LLPS/source/context hints 的候选 |
| B | 已有 pseudo-positive protein，用 teacher 重新校准 |
| C | sampled unknown PU background audit set |

该流程显式禁止 region/profile/residue teacher 输出，source report 中也记录了 “Forbidden / Not Generated Outputs: teacher_region_candidates, residue_profile_h5, teacher_dpr”。

### 1.2 Teacher 输出、归一化和方向

最终 protein-only teacher score 被统一写入：

`data/pseudo_labels/round1_abc_protein_only_corrected_20260607/combined/teacher_scores_ABC.csv`

所有 final-present teacher 在 protein-only consensus 中按高分为阳性方向处理，阈值为 0.5。对于 0--1 范围内的 score，confidence 由代码计算为：

```text
confidence = clip(score * teacher_weight, 0, 1)
positive = score >= 0.5
```

最终核心 teacher 的 `teacher_weight` 均为 1.0，因此核心 teacher 的 confidence 等于其归一化 protein-level score。PSTP protein 和 catGRANULE2 protein 分数被保留为 audit auxiliary scores，其 confidence scaling weight 分别为 0.5 和 0.25，但它们不参与 `pseudo_positive_high` 的核心 teacher 计数。

### 1.3 Teacher 融合规则

核心 teacher 集合为：

```text
C = {DeePhase, PSPHunter_protein, PSPredictor}
```

对 protein `i`，定义核心阳性 teacher 集合：

```text
C_i = {t in C : score_{i,t} >= 0.5}
```

在最终 protein-only consensus 中：

```text
teacher_consensus_score_i = mean(confidence_{i,t} for t in C_i)
```

若满足：

```text
|C_i| >= 2 and teacher_consensus_score_i >= 0.70
```

则写为 `pseudo_positive_high`。该规则在 AB batch 与 C batch 的 `teacher_pseudo_label_report.json` 中均记录为 “>=2 core positives and mean_conf >= 0.70”。

配置中还定义了一个 `pseudo_positive_mid` 规则：

```text
1 core positive + >=1 strong_aux positive and mean_conf >= 0.65
```

但 final-present strong auxiliary teacher 为空，AB 与 C batch 的 `pseudo_positive_mid` 均为 0。因此该规则没有贡献最终 reported LLPS pseudo-positive。

review candidate 规则包括单个高置信核心 teacher 或 auxiliary-majority support；但 review candidates 不写为 final pseudo-positive，并且最终 LLPS loss mask 为 0。

### 1.4 缺失 teacher 的处理

缺失 teacher score 不作为阳性支持计数。`pre_collect_validation.json` 显示 PSPredictor 存在 154 个未打分记录：

| batch | expected records | PSPredictor rows | PSPredictor scored |
|---|---:|---:|---:|
| AB | 11,671 | 11,671 | 11,618 |
| C | 29,868 | 29,868 | 29,767 |

这些缺失分数在 consensus 中不提供 positive support。最终物化后的 sample index 中，非 teacher-scored 行或缺失 teacher score 在 teacher score columns 中常以 0.0 占位；该 0.0 不能解释为该 teacher 明确给出阴性预测。

### 1.5 High-confidence 与 lower-confidence pseudo-positive

Round1 ABC combined summary 显示：

| item | rows |
|---|---:|
| teacher high positive | 4,786 |
| review candidate | 57 |
| teacher none | 36,696 |
| teacher-score rows | 207,541 |
| train recommendation positive | 7,184 |

最终 LLPS run 的 leakage cleanup、training-scope filtering、feature availability 和 final pool building 后，reported run 的 sample index 中保留：

| final tier | rows | rule |
|---|---:|---|
| pseudo_positive_high | 4,137 | 主要由 >=2 core teacher positives 且 mean confidence >=0.70 产生；另有少量非 teacher-core support 的 region/bag positive promoted rows |
| pseudo_positive_weak_preserved | 2,180 | 原 tier B pseudo-positive 未被 teacher high consensus 确认，但不被 teacher none 转为 negative，保留为弱阳性 |

`pseudo_positive_high` 中 teacher support 计数为：

| teacher_core_support_count | rows |
|---|---:|
| 3 | 2,803 |
| 2 | 1,318 |
| 0 | 16 |

16 行 `teacher_core_support_count=0` 的 `pseudo_positive_high` 同时带 `region_bag_label=1` 和 `region_bag_weight=0.75`，对应生成脚本中 explicit DPR span / protein-positive bag evidence 的 promotion 逻辑，而不是核心 teacher ensemble 高置信规则。

### 1.6 Training loss sample weight 与 teacher fusion weight 的区分

最终样本索引中有三个容易混淆的权重概念：

| field | 含义 | 最终 loss 中的作用 |
|---|---|---|
| teacher score/confidence weight | teacher 融合前的 predictor confidence scaling，例如核心 teacher weight=1.0，PSTP protein=0.5，catGRANULE2 protein=0.25 | 只用于生成/记录 teacher consensus，不是训练 sample weight |
| llps_sample_weight | label-confidence / label-quality weight，例如 high tier 0.70/0.85，weak preserved 0.30 | 保留在 final sample index 中；最终 `PhaseFlowOfflineDataset` 未把该列作为 batch `sample_weight` |
| sample_weight | final training pool weight，例如 P1_driver=1.0，P2_client=1.2，P3_C_D=0.95，N1_NP=1.0，N2_ND=1.4 | `PhaseFlowOfflineDataset` 读取为 batch `sample_weight`，`weighted_focal_bce_with_logits` 使用该列 |

最终 LLPS loss 配置中 `weighted_focal_bce=1.0`，`teacher_llps=0.0`。因此 pseudo-positive 训练强度来自 `sample_weight` 与 role/tier loss weights，而不是 teacher distillation loss。

## 2. 最终 DPR pseudo-label / weak-label 生成流程

最终 DPR checkpoint 对应配置为：

`paper/full_length/audit/final_dpr_config.yaml`

最终 candidate index 为：

`data/processed/stage2/dpr_v8r1a/indices/sampler_plans/plan_c_hq_region_candidate_index.parquet`

该 candidate index 共 1,519 行，列名中没有 `teacher`、`pseudo`、`weak`、`profile`、`pstp`、`catgranule`、`psphunter`、`deephase` 或 `pspredictor` 字段。最终 label tier 分布为：

| label_tier | rows |
|---|---:|
| N2_DISORDERED_NEGATIVE | 1,030 |
| N3_STRUCTURED_NEGATIVE | 287 |
| S2_VALIDATED_REGION | 146 |
| S1_CAUSAL_REGION | 56 |

最终 DPR 配置中：

```text
weak_tiers: []
strong_tiers: S1_CAUSAL_REGION, S2_VALIDATED_REGION, REGION_S1_S2
negative_tiers: N2_DISORDERED_NEGATIVE, N3_STRUCTURED_NEGATIVE
```

因此最终 DPR reported training 没有使用 teacher residue profile、teacher region profile 或 teacher pseudo regions。早期 `stage2_frozen_dpr.py`、`stage2_dpr_stack_v1` weak/pilot 日志和 region teacher pretrain 配置不属于最终 reported DPR training。

## Table A. Protein-level pseudo-positive teacher weights

| teacher | task/output | score normalization | ensemble weight | threshold/rule | missing-score handling | rationale | source code/config path |
|---|---|---|---:|---|---|---|---|
| DeePhase | protein-level LLPS score (`deephase_score`) | treated as 0--1 score; high direction | 1.0 | core teacher; positive if score >=0.5; contributes to `pseudo_positive_high` if >=2 core positives and mean core confidence >=0.70 | missing score contributes no core support; final index may show 0.0 placeholder | core teacher in protein-only Round1 ABC consensus | `external_artifacts/pseudo_label/teacher_protein_only_round1_abc.yaml`; `external_artifacts/scripts/pseudo_label/run_protein_only_teacher_batch.py::collect_scores`, `::consensus` |
| PSPHunter_protein | protein-level LLPS probability (`psphunter_probability`) | treated as 0--1 score; high direction | 1.0 | core teacher; positive if score >=0.5; same core consensus rule | missing score contributes no core support | core teacher in protein-only Round1 ABC consensus | same as above; raw shards under `data/pseudo_labels/round1_abc_protein_only_corrected_20260607/batches/*/raw/psphunter/` |
| PSPredictor | protein-level score (`pspredictor_score`) | treated as 0--1 score; high direction | 1.0 | core teacher; positive if score >=0.5; same core consensus rule | 154 scores missing in pre-collect validation; missing rows do not support consensus | required core teacher; manually imported/web batch result | same YAML; `pre_collect_validation.json`; raw file `batches/*/raw/pspredictor/pspredictor_unique.csv` |
| PSTP_protein | protein-level score (`pstp_score` / `protein_score`) | treated as 0--1 score; high direction; confidence scaled by 0.5 | 0.5 for auxiliary confidence only | audit auxiliary; cannot generate `pseudo_positive_high`; may only appear in review/audit logic | missing score contributes no pseudo-positive support | retained as auxiliary audit evidence; scan/profile output disabled | same YAML; `run_protein_only_teacher_batch.py::collect_scores`; raw files under `batches/*/raw/pstp/` |
| catGRANULE2_protein | protein-level score (`catgranule2_score` / `protein_score`) | treated as 0--1 score; high direction; confidence scaled by 0.25 | 0.25 for auxiliary confidence only | audit auxiliary; `catGRANULE2` cannot trigger pseudo-positive alone | missing score contributes no pseudo-positive support | low-weight auxiliary audit evidence; region/profile output disabled | same YAML; `run_protein_only_teacher_batch.py::collect_scores`; raw `catgranule2_protein_scores.csv` |
| FuzDrop_protein | protein-level score if available | not used in final present teacher outputs | 0.8 in disabled strong-aux config | disabled; final `present_strong_aux_teachers=[]`; contributes 0 rows | not present | reserved strong auxiliary, not final reported pseudo-positive evidence | same YAML; AB/C `teacher_pseudo_label_report.json` |
| MolPhase / Opt-PredLLPS / PhaseFlow teacher | protein-level reserved interfaces | not used | not used | disabled / forbidden | not present | not final reported pseudo-positive evidence | same YAML |

## Table B. Pseudo-label training tiers

| tier | label value | sample weight | row count | evidence rule | used loss terms | rationale | source code/config path |
|---|---|---|---:|---|---|---|---|
| LLPS: pseudo_positive_high | 1 | final loss `sample_weight`: P1_driver 1.0 (1,534 rows), P2_client 1.2 (2,500), P3_C_D 0.95 (103); retained `llps_sample_weight`: 0.70 (2,295) or 0.85 (1,842) | 4,137 | mostly >=2 core positives and mean core confidence >=0.70; 16 rows have no core support and are promoted by positive region/bag evidence | weighted focal BCE; positive ranking pools P1/P2/P3; no teacher distillation because `teacher_llps=0.0` | high-confidence weak supervision; final model depends on pseudo-positive supervision but exact threshold choice is policy-based | final sample index; `run_protein_only_teacher_batch.py::consensus`; `build_multitask_round1_abc_dataset.py::assign_final_labels`; `phaseflow/data/offline_dataset.py`; `phaseflow/losses/multitask.py` |
| LLPS: pseudo_positive_weak_preserved | 1 | final loss `sample_weight`: P1_driver 1.0 (1,107), P2_client 1.2 (692), P3_C_D 0.95 (381); retained `llps_sample_weight`: 0.30 (2,180) | 2,180 | original B-tier pseudo-positive retained when teacher consensus did not confirm high positive; teacher none is not converted into a negative | weighted focal BCE; positive ranking pools P1/P2/P3; no teacher distillation | preserves previous positive evidence with weak label status rather than treating teacher non-confirmation as negative evidence | `build_multitask_round1_abc_dataset.py`; `data/processed/merged/reports/03_audits_and_policies/abc_teacher_merge_audit.md` |
| LLPS: associated_context_unlabeled | -1 | final `sample_weight` 0.25; retained `llps_sample_weight` 0.0 | 6,051 | associated/context evidence but not supervised positive; teacher none/review not used as negative | not used by weighted focal BCE because label is -1; may be sampled as context/background pool | keeps ambiguous context from becoming hard negative | final sample index; `paper/full_length/audit/final_llps_config.yaml` |
| LLPS: unknown_pu_unlabeled | -1 | final `sample_weight` 0.25; retained `llps_sample_weight` 0.0 | 45,509 | PU/background unknown; no hard negative assignment | not used by weighted focal BCE because label is -1 | PU/background sampling without negative supervision | final sample index; `abc_teacher_merge_audit.md` |
| LLPS: gold_positive | 1 | final `sample_weight` 1.0; retained `llps_sample_weight` 1.0 | 323 | curated/gold positive | weighted focal BCE; positive ranking pool P1_driver | curated positive evidence | final sample index |
| LLPS: negative_curated_structured | 0 | final `sample_weight`: N1_NP 1.0 (21,596) or N2_ND 1.4 (26,350); retained `llps_sample_weight` 1.0 | 47,946 | curated structured negative | weighted focal BCE; negative ranking pools N1/N2 | curated negative evidence | final sample index; `paper/full_length/audit/final_llps_config.yaml` |
| LLPS: negative_curated_disordered | 0 | final `sample_weight` 1.4; retained `llps_sample_weight` 0.8 | 891 | curated disordered negative | weighted focal BCE; negative ranking pool N2_ND | curated negative evidence with disordered-negative role weighting | final sample index |
| DPR: teacher pseudo regions | not used | not used | 0 | final DPR `weak_tiers: []`; candidate index contains only S1/S2/N2/N3 tiers | no teacher pseudo-region loss in final reported DPR training | final DPR uses curated/validated regions and curated negatives, not teacher pseudo regions | `paper/full_length/audit/final_dpr_config.yaml`; `data/processed/stage2/dpr_v8r1a/indices/sampler_plans/plan_c_hq_region_candidate_index.parquet` |

## 3. 证据路径

### Final LLPS run and training config

- `paper/full_length/audit/final_llps_config.yaml`
- `external_artifacts/final_llps/configs/s1_rank04_nd15_drop015.yaml`
- `external_artifacts/final_llps/experiments/s1_rank04_nd15_drop015/sample_index.parquet`
- `external_artifacts/final_llps/experiments/s1_rank04_nd15_drop015/checkpoints/history.json`

### Protein-only teacher generation and merge

- `external_artifacts/pseudo_label/teacher_protein_only_round1_abc.yaml`
- `data/pseudo_labels/round1_abc_protein_only_corrected_20260607/README.md`
- `data/pseudo_labels/round1_abc_protein_only_corrected_20260607/source_reports/protein_only_dry_run_summary.md`
- `data/pseudo_labels/round1_abc_protein_only_corrected_20260607/pre_collect_validation.json`
- `data/pseudo_labels/round1_abc_protein_only_corrected_20260607/batches/AB/teacher_pseudo_label_report.json`
- `data/pseudo_labels/round1_abc_protein_only_corrected_20260607/batches/C/teacher_pseudo_label_report.json`
- `data/pseudo_labels/round1_abc_protein_only_corrected_20260607/combined/summary.json`
- `data/pseudo_labels/round1_abc_protein_only_corrected_20260607/combined/teacher_scores_ABC.csv`
- `data/pseudo_labels/round1_abc_protein_only_corrected_20260607/combined/teacher_protein_labels_ABC.csv`
- `data/pseudo_labels/round1_abc_protein_only_corrected_20260607/combined/label_view_ABC.csv`
- `data/processed/merged/reports/03_audits_and_policies/abc_teacher_merge_audit.md`
- `data/processed/merged/reports/offline_features/19_teacher样本纳入规则与覆盖报告.md`

### Key implementation functions

- `external_artifacts/scripts/pseudo_label/run_protein_only_teacher_batch.py::collect_scores`
- `external_artifacts/scripts/pseudo_label/run_protein_only_teacher_batch.py::consensus`
- `external_artifacts/scripts/pseudo_label/run_protein_only_teacher_batch.py::write_h5`
- `external_artifacts/scripts/data/build_multitask_round1_abc_dataset.py::merge_teacher_outputs`
- `external_artifacts/scripts/data/build_multitask_round1_abc_dataset.py::assign_final_labels`
- `paper/full_length/audit/final_llps_config.yaml`
- `phaseflow/data/offline_dataset.py::__getitem__`
- `phaseflow/losses/multitask.py::compute_multitask_loss`
- `phaseflow/losses/multitask.py::weighted_focal_bce_with_logits`

### Final DPR evidence

- `paper/full_length/audit/final_dpr_config.yaml`
- `data/processed/stage2/dpr_v8r1a/indices/sampler_plans/plan_c_hq_region_candidate_index.parquet`
- `data/processed/stage2/dpr_v8r1a/indices/sampler_plans/plan_c_hq_region.yaml`

## 4. 无法从代码中确认或不应推测的信息

1. 未找到证据表明 teacher ensemble weights 经过 validation optimization、calibration search 或 literature-based numeric fitting。最终证据支持的是 rule-based consensus。
2. 未找到证据表明 `pseudo_positive_high` 的 0.70 threshold、review 的 0.85 threshold、或 weak preserved 的 0.30 label-confidence weight 来自独立验证集选择。它们应记录为最终 policy values。
3. `llps_sample_weight` 与最终 loss 使用的 `sample_weight` 同时存在，但最终 loader 直接读入的是 `sample_weight`。如果后续论文需要报告训练 loss sample weight，应优先报告 `sample_weight`；若报告 label-confidence weight，应明确写为 retained label-confidence weight。
4. 16 个 final `pseudo_positive_high` 行没有核心 teacher support，证据显示其带 positive region/bag evidence；但最终 sample index 未保留每行 promotion 的完整原始 span 来源，不能把这 16 行写成核心 teacher consensus high。
5. 最终 DPR reported training 未使用 teacher pseudo regions；早期 weak/pilot DPR 目录存在，但不属于最终 reported DPR 配置。
