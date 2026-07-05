# DPR v6 rank_p257 复现说明

生成日期: 2026-06-17
训练环境: `redacted_for_release`

本文只记录可从仓库文件、训练产物和评估产物直接核实的信息。当前执行环境无法读取 git HEAD，因此版本指纹以 checkpoint SHA256、脚本 SHA256 和配置 SHA256 为准。

## 1. 最终模型

推荐用于论文正文和补充材料的最终 checkpoint:

```text
artifacts/model/checkpoints/update_000050.pt
```

- checkpoint SHA256: `7fb0091e6dd5a85bd3a6be7a0b606501700c4b8f28ff9b6e309267835a2fdff0`
- 文件大小: `326M`
- 训练 run: `d1_flat_seed174_raw_planc_rankp257_p257_lr5e-6_50u_seed202606188`
- 评估使用的 variant: `raw`
- 最终选模来源: `PlanD` 非 PhasePro 验证集

本次最终评估与选模产物位于:

```text
artifacts/benchmarks/plan_d_fair_single_matrix_20260617_selection_v2
artifacts/benchmarks/phasepro_fair_single_matrix_20260617_fair_compare_v2
artifacts/benchmarks/final_overall_benchmark_20260617
```

## 2. 从头到尾的训练链

| 阶段 | run / checkpoint | 初始化来源 | 数据集 | 目标 | 更新数 | 关键说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 基座 DPR v6 | `external_artifacts/overall/v6/checkpoints/d1_flat/update_000250.pt` | 上游 v6 训练产物 | v6 体系 | `d1_flat` | 250 | 作为后续一切 v8r1a 微调的起点 |
| 第 1 段微调 | `d1_flat_u0250_ema_pland_seed202606174_s2s_strong_lr3e-5_75u` | 上述基座，`init_variant=ema` | `plan_d_mixed_hq_train` | `strong` | 75 | 非 PhasePro PlanD 训练，seed=202606174 |
| 第 2 段微调 | `d1_flat_seed174_raw_planc_s2s_strong_p257_lr5e-6_50u_seed202606184` | 上一步 75u checkpoint，`init_variant=raw` | `plan_c_hq_region` | `strong_p257` | 50 | 非 PhasePro PlanC 训练，seed=202606184 |
| 第 3 段微调 | `d1_flat_seed174_raw_planc_rankp257_p257_lr5e-6_50u_seed202606188` | 上一步 50u checkpoint，`init_variant=ema` | `plan_c_hq_region` | `rank_p257` | 50 | 最终模型，seed=202606188 |

对应 checkpoint SHA256:

| checkpoint | SHA256 |
| --- | --- |
| `external_artifacts/overall/v6/checkpoints/d1_flat/update_000250.pt` | `aa51aebe7e2620ca222b8d314a68536bb6f0985b5f096da1c8b1cdcb3edb6422` |
| `d1_flat_u0250_ema_pland_seed202606174_s2s_strong_lr3e-5_75u/checkpoints/update_000075.pt` | `6103c008e8f5d923a7f942070ef0a2671338d5f9e0d316683817233d931ade1b` |
| `d1_flat_seed174_raw_planc_s2s_strong_p257_lr5e-6_50u_seed202606184/checkpoints/update_000050.pt` | `d5255e488cb91cecf0a8ff94afa980834fc52ac13534bbed81153f3ad9bdf4bc` |
| `d1_flat_seed174_raw_planc_rankp257_p257_lr5e-6_50u_seed202606188/checkpoints/update_000050.pt` | `7fb0091e6dd5a85bd3a6be7a0b606501700c4b8f28ff9b6e309267835a2fdff0` |

### 基座来源说明

基座 `external_artifacts/overall/v6/checkpoints/d1_flat/update_000250.pt` 来自 DPR v6 上游训练轨迹，训练入口为:

```text
scripts/full_length/training/run_dpr_v6.py
```

可核实的基座产物包括:

- 配置: `external_artifacts/overall/v6/base.yaml`
- 训练摘要: `external_artifacts/overall/v6/reports/d1_flat/train_summary.json`
- 日志: `external_artifacts/overall/v6/logs/d1_flat/global_metrics.csv` 和 `global_metrics.jsonl`
- schedule audit: `external_artifacts/overall/v6/schedules/schedule_next4000_audit.json`
- 决策报告: `external_artifacts/overall/v6/reports/11_next_decision.json`

`11_next_decision.json` 中记录 `d1_flat/update_000250.pt` 是当时 `best_overall`，checkpoint SHA256 为 `aa51aebe7e2620ca222b8d314a68536bb6f0985b5f096da1c8b1cdcb3edb6422`。该上游基座阶段已有 PhasePro 评估报告；本文最终模型的后续 v8r1a 微调、PlanD 选模和阈值选择没有使用 PhasePro，但基座来源在论文补充材料中应透明披露。

## 3. 训练命令和脚本

最终 50u rank_p257 训练由以下脚本触发:

```text
scripts/full_length/training/run_dpr_v6.py
```

对应参数要点:

- `--config configs/full_length/final_dpr.yaml`
- `--arm d1_flat`
- `--init-checkpoint .../d1_flat_seed174_raw_planc_s2s_strong_p257_lr5e-6_50u_seed202606184/checkpoints/update_000050.pt`
- `--init-variant ema`
- `--candidate-index artifacts/data/processed/stage2/dpr_v8r1a/indices/sampler_plans/plan_c_hq_region_candidate_index.parquet`
- `--plan-yaml artifacts/data/processed/stage2/dpr_v8r1a/indices/sampler_plans/plan_c_hq_region.yaml`
- `--run-name d1_flat_seed174_raw_planc_rankp257_p257_lr5e-6_50u_seed202606188`
- `--updates 50`
- `--batch-size 2`
- `--save-every 25`
- `--log-every 5`
- `--lr 5e-6`
- `--loss-objective rank_p257`
- `--seed 202606188`
- `--s2-as-s`

上游 75u PlanD 微调由:

```text
paper/full_length/audit/evidence/dpr/training__finetune_dpr_v6_v8r1a_region.py
```

其关键参数为:

- `--init-checkpoint external_artifacts/overall/v6/checkpoints/d1_flat/update_000250.pt`
- `--init-variant ema`
- `--candidate-index .../plan_d_mixed_hq_train_candidate_index.parquet`
- `--plan-yaml .../plan_d_mixed_hq_train.yaml`
- `--updates 75`
- `--lr 3e-5`
- `--loss-objective strong`
- `--seed 202606174`

第 2 段 PlanC `strong_p257` 微调由以下矩阵脚本中的第 4 个任务产生:

```text
paper/full_length/audit/evidence/dpr/training__finetune_dpr_v6_v8r1a_region.py
```

对应任务参数为:

```text
d1_flat_seed174_raw_planc_s2s_strong_p257_lr5e-6_50u_seed202606184
init = d1_flat_u0250_ema_pland_seed202606174_s2s_strong_lr3e-5_75u/checkpoints/update_000075.pt
init_variant = raw
candidate = plan_c_hq_region_candidate_index.parquet
plan = plan_c_hq_region.yaml
objective = strong_p257
lr = 5e-6
seed = 202606184
updates = 50
```

训练入口代码是:

```text
paper/full_length/audit/evidence/dpr/training__finetune_dpr_v6_v8r1a_region.py
```

## 4. 数据来源和非泄漏审计

### 第 3 段最终训练

最终 rank_p257 训练使用:

```text
artifacts/data/processed/stage2/dpr_v8r1a/indices/sampler_plans/plan_c_hq_region_candidate_index.parquet
artifacts/data/processed/stage2/dpr_v8r1a/indices/sampler_plans/plan_c_hq_region.yaml
```

输入审计结果:

- rows: `1519`
- unique proteins: `1483`
- tier counts:
  - `S1_CAUSAL_REGION`: `56`
  - `S2_VALIDATED_REGION`: `146`
  - `N2_DISORDERED_NEGATIVE`: `1030`
  - `N3_STRUCTURED_NEGATIVE`: `287`
- `phasepro_overlap_rows = 0`
- `packed_sidecar_coverage = 1.0`

### 第 1 段上游训练

上游 75u 训练使用:

```text
artifacts/data/processed/stage2/dpr_v8r1a/indices/sampler_plans/plan_d_mixed_hq_train_candidate_index.parquet
artifacts/data/processed/stage2/dpr_v8r1a/indices/sampler_plans/plan_d_mixed_hq_train.yaml
```

输入审计结果:

- rows: `1397`
- unique proteins: `1337`
- tier counts:
  - `N2_DISORDERED_NEGATIVE`: `831`
  - `N3_STRUCTURED_NEGATIVE`: `231`
  - `S1_CAUSAL_REGION`: `41`
  - `S2_VALIDATED_REGION`: `119`
  - `W1_SELF_DRIVER_BAG`: `61`
  - `W2_CONTEXT_DRIVER_BAG`: `114`
- `phasepro_overlap_rows = 0`
- `packed_sidecar_coverage = 1.0`

### Plan 权重

- `plan_c_hq_region.yaml`
  - `S1_CAUSAL_REGION: 0.22`
  - `S2_VALIDATED_REGION: 0.38`
  - `N2_DISORDERED_NEGATIVE: 0.22`
  - `N3_STRUCTURED_NEGATIVE: 0.18`
- `plan_d_mixed_hq_train.yaml`
  - `N2_DISORDERED_NEGATIVE: 0.27`
  - `N3_STRUCTURED_NEGATIVE: 0.20`
  - `S1_CAUSAL_REGION: 0.10`
  - `S2_VALIDATED_REGION: 0.16`
  - `W1_SELF_DRIVER_BAG: 0.10`
  - `W2_CONTEXT_DRIVER_BAG: 0.17`

三段 v8r1a 微调都显式启用了 `--s2-as-s`，即把 `S2_VALIDATED_REGION` 视作强正例监督。

## 5. 模型结构

最终 run 的 arm 是 `d1_flat`，对应 `configs/full_length/final_dpr.yaml` 中:

- `head_type: big`
- `window_sizes: [33, 129, 257]`
- `topk_fraction: 0.05`
- `dropout: 0.10`
- `adapter_dim: 256`
- `phaseflow_bridge_tokens: 32`
- `phaseflow_bridge_adapter_layers: 2`
- `phaseflow_bridge_gate_init: 0.075`
- `use_base_streams: true`
- `use_full_length_llps_stream: true`
- `use_phaseflow_stream: true`
- `use_pstp650: false`
- `use_pstp_esm8: false`
- `use_pstp_alb: false`

实现位置:

```text
phaseflow/full_length/models/dpr_v6.py
```

关键约束:

- PhaseFlow、PhaseFlow 和 bridge 全部冻结
- 训练仅更新 `v6.` 和 `v6_feature_projectors.` 下参数
- 本次最终 run 没有启用 PSTP 外部特征投影层

PhaseFlow 接入来源:

- final v6 配置中的 `full_length_llps_checkpoint` 指向 `external_artifacts/stage1/phaseflow_llps_final/final_model/best_single_model.pt`
- 该 checkpoint 是 full PPMC AUROC/AUPRC 最优的 Stage1 PhaseFlow/PhaseFlow
- final v6 checkpoint 内保存的 `phaseflow.*` 权重与上述 Stage1 checkpoint 逐张量一致
- `best_single_model.pt` 与 `best_single_model_calibrated.pt` 的模型权重逐张量一致；calibrated 文件额外包含校准元数据

## 6. 损失设计

实现位置:

```text
phaseflow/full_length/models/dpr_v6.py
```

`strong_p257` 的组成:

- `bag_hard`
- `0.25 * bag_topk`
- `p257` 上的 `S` BCE
- `p257` 上的 `S` Dice
- `p257` 上的 `S` rank
- `p257` 上的 `W` BCE
- `p257` 上的 `W` rank

`rank_p257` 的组成:

- `bag_hard`
- `0.25 * bag_topk`
- `p257` 上的 `S` BCE
- `p257` 上的 `S` Dice
- `p257` 上的 `S` pairwise rank
- `p257` 上的 `W` pairwise rank
- `ND/NP` 的 negative top suppression

按当前默认系数展开后，`rank_p257` 的有效权重是:

- `bag_hard: 1.0`
- `bag_topk: 0.25`
- `p257 S BCE: 0.35`
- `p257 S Dice: 0.06`
- `p257 S pairwise rank: 0.75`
- `p257 W pairwise rank: 0.1875`
- `ND/NP top suppression: 0.35`

公共损失超参:

- `boundary_radius = 17`
- `topk_fraction = 0.05`
- `EMA decay = 0.997`
- `gradient clip norm = 1.0`

## 7. 训练环境

Release 训练环境摘要:

```text
compute_node: redacted_for_release
GPU: 8 x NVIDIA A100-SXM4-80GB
driver: 580.105.08
Python: 3.12.11
torch: 2.12.0+cu130
CUDA runtime: 13.0
numpy: 2.4.6
pandas: 3.0.3
```

运行时强制离线:

- `PHASEFLOW_DISABLE_STARLING_READ=1`
- `PHASEFLOW_DISABLE_PROTENIX_READ=1`
- `PHASEFLOW_STRICT_OFFLINE=1`

## 8. 选模与阈值

### PlanD 非 PhasePro 选模

最终 checkpoint 的选择没有使用 PhasePro。选模依据是:

- `PlanD` mixed HQ 非 PhasePro validation
- 选取子集: `PlanD_PSTP_common_by_scale`
- common proteins: `240`
- `phasepro_used_for_selection = false`

选择规则在:

```text
scripts/full_length/evaluation/select_dpr_v6_plan_d_composite.py
```

预先定义的 composite 为:

```text
0.7 * mean_percentile(AUROC, AUPRC, Spearman, median per-protein Spearman)
+ 0.3 * mean_percentile(MCC, F1)
```

最终 rank_p257 选中项:

- model: `d1_flat_seed174_raw_planc_rankp257_p257_lr5e-6_50u_seed202606188_u0050_raw`
- threshold: `0.667148`
- validation profile: `.../p257_profiles.npz`

### 阈值策略

主结果不固定用 `0.5`。`0.5` 只作为诊断阈值。

更公平的主比较使用两种阈值策略:

1. 同一固定阈值，两个模型都用
2. 各自用同一非 PhasePro PlanD validation 上选出的 MCC 阈值

最终推荐主结果是第 2 种。

### DPR v6 vs PSTP 结果

#### Threshold-free

| model | AUROC | AUPRC | Spearman | median pp Spearman | pairwise |
| --- | --- | --- | --- | --- | --- |
| DPR v6 | 0.666112 | 0.712238 | 0.286609 | 0.531198 | 0.623744 |
| PSTP | 0.669670 | 0.715462 | 0.292745 | 0.421235 | 0.584457 |

#### Per-model PlanD MCC threshold

| model | threshold | precision | recall | F1 | MCC | IoU |
| --- | --- | --- | --- | --- | --- | --- |
| DPR v6 | 0.667148 | 0.718444 | 0.416304 | 0.527150 | 0.237796 | 0.357912 |
| PSTP | 0.607544 | 0.753503 | 0.362898 | 0.489868 | 0.250763 | 0.324387 |

#### Fixed 0.5

| model | precision | recall | F1 | MCC | IoU |
| --- | --- | --- | --- | --- | --- |
| DPR v6 | 0.676305 | 0.510662 | 0.581926 | 0.222001 | 0.410364 |
| PSTP | 0.709463 | 0.444609 | 0.546645 | 0.239105 | 0.376126 |

结论:

- 如果看 DPR 相关的 region recovery，`v6` 在 `F1 / IoU / recall` 上超过 PSTP
- 如果看 `AUROC / AUPRC / global Spearman / MCC`，PSTP 仍略高
- 固定 `0.5` 不是主结论，只是诊断值

### Overall requested benchmark

本 final 包另有一个按论文表格需求整理的总 benchmark:

```text
artifacts/benchmarks/final_overall_benchmark_20260617
```

主要输出:

- `reports/final_overall_benchmark_report.md`
- `llps/llps_requested_table.csv`
- `dpr/dpr_requested_table.csv`
- `dpr/dpr_profile_availability.csv`
- `dpr/dpr_ppmc_negative_region_metrics.csv`
- `dpr/dpr_plan_d_negative_region_metrics.csv`

该总表把 LLPS 与 DPR 分开评估:

- LLPS 使用 final v6 中冻结接入的 SOTA PhaseFlow head，PPMC full panel score 表来自 `phaseflow_llps_final/phaseflow_no_starling_embed_calibrated`，重新计算 AUROC、AUPRC、MCC、F1、Precision、Recall、Recall@FPR5%、ECE、Brier。
- LLPS full PPMC 主结果为 PhaseFlow/PhaseFlow AUPRC `0.752183`、AUROC `0.874282`；同表 DeePhase 为 AUPRC `0.720529`、AUROC `0.860489`。
- DPR 使用 final DPR v6 raw p257 profile 计算 PhasePro residue/region 指标；PSTP-Scan 正集对比使用 no-PhasePro selected-family p257 profile。
- DPR negative false-DPR 主表使用 PPMC NP/ND 负蛋白中release artifact 中已有 residue-level profile 的模型。当前 final v6 的 PPMC NP/ND 蛋白不在 v6 offline packed feature 中，因此 PhaseFlow 的 PPMC negative false-DPR 不填；同时单独报告 PlanD N2/N3 非 PhasePro negative audit，PhaseFlow 在该 audit 上 coverage 为 `255/255`。

## 9. 关键产物

训练产物:

```text
artifacts/model/
```

其中重要文件:

- `checkpoints/update_000050.pt`
- `configs/resolved_finetune_config.json`
- `logs/environment.json`
- `logs/raw_metrics.jsonl`
- `logs/global_metrics.jsonl`
- `reports/input_audit.json`
- `reports/train_summary.json`

最终验证/比较产物:

- `artifacts/benchmarks/plan_d_fair_single_matrix_20260617_selection_v2`
- `artifacts/benchmarks/plan_d_external_val_rankp257_single_20260617`
- `artifacts/benchmarks/phasepro_fair_single_matrix_20260617_fair_compare_v2`
- `artifacts/benchmarks/final_overall_benchmark_20260617`

## 10. 当前仓库入口

论文仓库保留当前可维护入口，而不是训练时的 node-specific wrapper。核心文件为：

- `configs/full_length/final_dpr.yaml`
- `artifacts/model/configs/resolved_finetune_config.json`
- `phaseflow/full_length/models/dpr_v6.py`
- `scripts/full_length/training/run_dpr_v6.py`
- `scripts/full_length/evaluation/select_dpr_v6_plan_d_composite.py`
- `scripts/full_length/evaluation/compare_dpr_v6_plan_d_phasepro_final.py`
- `scripts/full_length/evaluation/compare_dpr_v6_fair_threshold_policies.py`
- `scripts/full_length/benchmark/final_overall_benchmark_from_profiles.py`

## 11. 复现入口

从当前仓库复现实验时，应使用上面的通用 Python 入口和 `configs/full_length/final_dpr.yaml` 中的最终参数。原始训练中的多阶段 checkpoint 链、选择策略、阈值策略和最终 SHA 已在本报告前文及 `docs/full_length/final/reproduction/artifact_manifest.tsv` 中记录。

最终论文结果可直接引用：

- `artifacts/model/checkpoints/update_000050.pt`
- `artifacts/benchmarks/plan_d_fair_single_matrix_20260617_selection_v2/validation_selection_summary_all_scales.json`
- `artifacts/benchmarks/phasepro_fair_single_matrix_20260617_fair_compare_v2/phasepro_fair_threshold_policy_report.md`
- `artifacts/benchmarks/final_overall_benchmark_20260617/final_overall_benchmark_summary.json`
