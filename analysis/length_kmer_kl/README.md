# PhaseFlow 序列生成：从 missing_0 相图

使用 PhaseFlow 模型的 phase→seq 生成能力，从 `missing_0.csv` 的完整相图数据生成新的序列数据集。

## 数据来源

- **输入**: `/data/yanjie_huang/LLPS/phase_diagram/by_missing/missing_0.csv`
  - 10,165 条完整相图数据（无缺失值）
  - PSSI 范围: [-2.17, 1.64]
  - 16 维相图: group_11 到 group_44

- **模型**: `outputs_set/output_set_flow1_missing11/best_model.pt`
  - 架构: Set Encoder + Transfusion (Flow Matching + LM)
  - 训练数据: missing ≤ 11 (147,949 条)
  - 配置: flow_weight=1, lm_weight=1
  - 参数量: ~7M

## 生成数据集

### dataset_1x (10,165 条)
- 每个相图生成 1 条序列
- 1:1 对应原始相图
- 用途: 对比分析、验证生成质量

### dataset_5x (50,825 条)
- 每个相图生成 5 条序列
- 5:1 对应原始相图
- 用途: 数据增强、增加训练集多样性

## 使用方法

```bash
conda activate phaseflow
cd /data/yanjie_huang/LLPS/predictor/PhaseFlow/generated_from_missing0

# 生成 1x 数据集
python generate_sequences.py \
    --checkpoint ../outputs_set/output_set_flow1_missing11/best_model.pt \
    --input_csv /data/yanjie_huang/LLPS/phase_diagram/by_missing/missing_0.csv \
    --output_dir outputs/dataset_1x \
    --n_samples 1 \
    --batch_size 64 \
    --temperature 1.0 \
    --max_len 25 \
    --seed 42 \
    --gpu 0

# 生成 5x 数据集
python generate_sequences.py \
    --checkpoint ../outputs_set/output_set_flow1_missing11/best_model.pt \
    --input_csv /data/yanjie_huang/LLPS/phase_diagram/by_missing/missing_0.csv \
    --output_dir outputs/dataset_5x \
    --n_samples 5 \
    --batch_size 64 \
    --temperature 1.0 \
    --max_len 25 \
    --seed 42 \
    --gpu 0
```

## 输出格式

CSV 文件，包含以下列：

| 列名 | 说明 |
|------|------|
| `original_sequence` | 原始序列（来自 missing_0.csv） |
| `generated_sequence` | 生成的序列 |
| `generation_idx` | 同一相图的第几次生成（0-indexed） |
| `original_idx` | 原始数据行号 |
| `length` | 生成序列长度 |
| `pssi_0` ~ `pssi_15` | 16 维 PSSI 值（group_11 到 group_44） |
| `mean_pssi` | PSSI 均值 |
| `std_pssi` | PSSI 标准差 |
| `min_pssi` | PSSI 最小值 |
| `max_pssi` | PSSI 最大值 |

## 生成参数

| 参数 | 值 | 说明 |
|------|-----|------|
| temperature | 1.0 | 标准采样，保持多样性 |
| max_len | 25 | 最大生成长度（含特殊token） |
| top_k | None | 不限制 top-k |
| top_p | None | 不限制 nucleus sampling |
| seed | 42 | 固定随机种子，保证可复现 |
| batch_size | 64 | 批量大小 |

## 目录结构

```
generated_from_missing0/
├── generate_sequences.py    # 主生成脚本
├── config.yaml              # 生成配置
├── README.md                # 本文档
└── outputs/
    ├── dataset_1x/          # 1条/相图 (10,165条)
    │   ├── sequences.csv
    │   └── generation_stats.txt
    └── dataset_5x/          # 5条/相图 (50,825条)
        ├── sequences.csv
        └── generation_stats.txt
```

## 数据验证

生成完成后，可用以下命令验证：

```bash
# 检查行数
wc -l outputs/dataset_1x/sequences.csv   # 应为 10,166 (含header)
wc -l outputs/dataset_5x/sequences.csv   # 应为 50,826 (含header)
```

```python
import pandas as pd

df = pd.read_csv('outputs/dataset_1x/sequences.csv')
print(f"总行数: {len(df)}")
print(f"唯一序列: {df['generated_sequence'].nunique()}")
print(f"序列长度分布:\n{df['length'].value_counts().sort_index()}")
print(f"缺失值: {df.isna().sum().sum()}")
```
