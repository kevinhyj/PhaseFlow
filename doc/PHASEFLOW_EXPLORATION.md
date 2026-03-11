# PhaseFlow 探索总结

## 快速概览

PhaseFlow 是一个 **Transfusion 架构的双向预测模型**，用于蛋白质序列和相图之间的相互转换。

- **路径**: `/data4/huangyanjie/LLPS/predictor/PhaseFlow`
- **核心代码**: 2164 LOC (phaseflow/ 目录)
- **模型大小**: ~7M 参数, 79MB 检查点
- **性能**: Spearman ρ ≈ 0.575 (seq2phase)

---

## 1. 核心架构

### 双向任务

```
seq2phase (Flow Matching)     phase2seq (Language Modeling)
序列 ──────────────────→ 相图    相图 ──────────────────→ 序列
ODE求解 (平滑生成)           自回归采样 (快速生成)
```

### 模型组件

| 组件 | 功能 | 实现 |
|------|------|------|
| **序列编码** | 氨基酸 → 向量 | Token Embedding (32) 或 ESM2 (1280→256) |
| **相图编码** | 16维相图 → 向量 | Linear (简单) 或 Set Transformer (高级) |
| **Transformer** | 特征融合 | 6层, 256维, 8头, RoPE, RMSNorm |
| **Flow Head** | 速度预测 | Linear(256→16) |
| **LM Head** | 下一token预测 | Linear(256→32) |

### Set Transformer 相图编码 (新)

```
相图 (16维) + Mask
    ↓
Fourier特征 (坐标) + 数值特征
    ↓
Set Transformer (2层, 4头)
    ↓
Attention Pooling (可学习查询)
    ↓
全局相图向量 (256维)
```

**优势**: 处理缺失值 (~62.6%), 编码空间结构

---

## 2. 训练配置

### 基础参数

```yaml
model:
  dim: 256, depth: 6, heads: 8
  vocab_size: 32, phase_dim: 16
  max_seq_len: 32, dropout: 0.1

training:
  batch_size: 2048 (512×4梯度累积)
  lr: 0.0008, weight_decay: 0.01
  epochs: 2000, warmup_steps: 1000
  flow_loss_weight: 32, lm_loss_weight: 1
  early_stopping_patience: 200
```

### 实验变体

- **flow1/5/10/32**: 不同Flow权重 (基础编码器)
- **set_encoder_flow***: Set Transformer编码器
- **missing*_flow32**: 缺失值分割实验

---

## 3. 损失函数

### Flow Matching Loss

```python
# CondOT路径: x_t = (1-t)·x_0 + t·x_1
# 目标: v = x_1 - x_0

loss = MSE(v_pred, v_target) * phase_mask

# 二次加权: w = (valid_count/16)²
# 完整(16): w=1.0, 半数(8): w=0.25, 单个(1): w≈0.004
```

### Language Modeling Loss

```python
loss = CrossEntropy(logits, labels, ignore_index=-100)
perplexity = exp(loss)
```

### 总损失

```python
total_loss = 32 * flow_loss + 1 * lm_loss
```

---

## 4. 推理方法

### Phase Generation (seq → phase)

```python
# ODE求解: dx/dt = v(x, t)
1. 初始化: x_0 ~ N(0,I)
2. ODE求解器: torchdiffeq (dopri5/euler/midpoint)
3. 时间范围: t ∈ [0, 1]
4. 返回: x_1 (最终相图)
```

**特点**: 平滑轨迹, 可控生成, 支持条件生成

### Sequence Generation (phase → seq)

```python
# 自回归采样
1. 初始化: tokens = [SOS]
2. 循环:
   - logits = model(tokens, phase)
   - next_token ~ softmax(logits[-1] / temperature)
   - tokens.append(next_token)
3. 停止: EOS 或 max_len
```

**支持**: Temperature, Top-k, Top-p 采样

---

## 5. 数据处理

### 输入格式

```csv
AminoAcidSequence,group_11,group_12,...,group_44
ACDEFGHIKL,1.0,-0.5,0.3,...,NaN,...
```

- 序列长度: 5-20 氨基酸
- 相图: 16维 (4×4网格)
- 缺失值: ~62.6%

### 数据分割

**随机分割** (missing_threshold=-1):
- 训练: 90%, 验证: 5%, 测试: 5%

**缺失值分割** (missing_threshold≥0):
- 训练: missing_0 到 missing_threshold
- 验证/测试: missing_{threshold+1} 到 missing_15

### 预处理

```python
phase_mask = ~isnan(phase)      # 1=有效, 0=缺失
phase = fillna(phase, 0)        # 缺失值填0
phase = normalize(phase)        # 可选归一化
```

---

## 6. 模型输出

### 测试指标

```json
{
  "loss": 84.97,
  "flow_loss": 2.59,
  "lm_loss": 2.24,
  "perplexity": 9.41,
  "mse": 0.759,
  "mae": 0.691,
  "rmse": 0.871,
  "pearson": 0.575,
  "spearman": 0.563
}
```

### 输出文件

```
outputs/output_set_encoder_bs2048_lr0.0008_flow32_20260128/
├── best_model.pt              # 最佳权重 (79MB)
├── config.yaml                # 训练配置
├── test_results.json          # 测试指标
└── checkpoint_epoch_*.pt      # 中间检查点

visual_training/bs2048_lr0.0008_flow32_20260128/
├── loss.png                   # 损失曲线
├── flow_lm_loss.png           # Flow/LM损失
├── perplexity.png             # 困惑度
└── correlation.png            # 相关性
```

---

## 7. PhaseFlow vs Seq2Phase

| 特性 | PhaseFlow | Seq2Phase |
|------|-----------|-----------|
| **架构** | Transfusion (双向) | Cross-Attention (单向) |
| **任务** | seq2phase + phase2seq | seq2phase only |
| **序列编码** | Token/ESM2 | ESM2 only |
| **相图编码** | Linear/Set Transformer | Grid Queries |
| **生成** | Flow Matching (ODE) | 直接回归 |
| **参数** | ~7M | ~2-3M |
| **损失** | Flow + LM | MSE |
| **缺失值** | 掩码 + 二次加权 | 掩码 |
| **推理速度** | 慢 (ODE) | 快 (单次) |
| **性能** | Spearman 0.575 | Spearman 0.57+ |

---

## 8. 关键创新

### 8.1 Set Transformer 相图编码

**问题**: Linear编码器忽视空间结构和缺失值模式

**解决**:
- Fourier特征编码坐标
- Set Transformer让有效点互相交流
- Attention Pooling可学习汇总

### 8.2 Flow Matching 生成

**优势**:
- 平滑生成轨迹
- 可控的生成过程
- 支持条件生成

**实现**: CondOT路径 + torchdiffeq ODE求解

### 8.3 双向预测

**seq2phase**: Flow Matching (平滑)
**phase2seq**: Language Modeling (快速)

**优势**: 统一框架, 相互验证, 数据增强

### 8.4 缺失值处理

**二次加权**: w = (valid_count/16)²

- 完整数据优先学习
- 逐步适应缺失值
- 平衡不同数据质量

---

## 9. 文件清单

### 核心模块 (phaseflow/)

| 文件 | 行数 | 功能 |
|------|------|------|
| model.py | 813 | PhaseFlow + 编码器 |
| transformer.py | 326 | Transformer骨干 |
| data.py | 389 | 数据集/DataLoader |
| tokenizer.py | 191 | 氨基酸分词器 |
| utils.py | 434 | 工具函数 |
| **总计** | **2164** | **核心实现** |

### 训练和推理

- `train/train.py`: 训练脚本 (~500 LOC)
- `infer/example_usage.py`: 推理示例
- `run_phase2seq.py`: Phase→Seq生成
- `predict_random_peptides_phaseflow.py`: 随机肽预测

### 配置和输出

- `config/`: 8个YAML配置文件
- `outputs/`: 12个训练输出目录
- `logs/`: 70+个训练日志
- `visual_training/`: 训练可视化

---

## 10. 使用指南

### 训练

```bash
cd /data4/huangyanjie/LLPS/predictor/PhaseFlow
python train/train.py \
    --config config/set_encoder_bs2048_lr0.0008_flow32_20260128.yaml \
    --data_path /data4/huangyanjie/LLPS/phase_diagram/phase_diagram.csv \
    --output_dir outputs/my_run \
    --batch_size 2048 \
    --epochs 2000
```

### 推理

```bash
# Seq → Phase
python infer/example_usage.py \
    --checkpoint outputs/output_set_encoder_bs2048_lr0.0008_flow32_20260128/best_model.pt \
    --mode seq2phase \
    --input "ACDEFGHIKLMN"

# Phase → Seq
python run_phase2seq.py \
    --checkpoint outputs/output_set_encoder_bs2048_lr0.0008_flow32_20260128/best_model.pt \
    --phase_values "1.0,-0.5,0.3,..."
```

### 滑动窗口评分

```bash
cd sliding_window_scoring
python score_windows.py \
    --checkpoint ../outputs/output_set_encoder_bs2048_lr0.0008_flow32_20260128/best_model.pt \
    --protein_file proteins.fasta \
    --window_size 10
```

---

## 11. 依赖

### 核心依赖

```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
einops>=0.6.0
rotary_embedding_torch>=0.8.4
torchdiffeq
```

### 环境设置

```bash
conda activate llps
cd /data4/huangyanjie/LLPS/predictor/PhaseFlow
pip install -r requirements.txt
```

---

## 12. 关键路径

| 路径 | 说明 |
|------|------|
| `/data4/huangyanjie/LLPS/predictor/PhaseFlow` | PhaseFlow主目录 |
| `phaseflow/model.py` | 主模型实现 |
| `train/train.py` | 训练脚本 |
| `config/set_encoder_bs2048_lr0.0008_flow32_20260128.yaml` | 最佳配置 |
| `outputs/output_set_encoder_bs2048_lr0.0008_flow32_20260128/best_model.pt` | 最佳模型 |
| `/data4/huangyanjie/LLPS/phase_diagram/phase_diagram.csv` | 训练数据 |

---

## 13. 与 seq2phase 的关系

### seq2phase (v1/v2)

- **单向**: 序列 → 相图
- **架构**: Cross-Attention (Grid Queries)
- **生成**: 直接回归 (单次前向)
- **参数**: ~2-3M
- **性能**: Spearman 0.57+

### PhaseFlow

- **双向**: 序列 ↔ 相图
- **架构**: Transfusion (Flow Matching + LM)
- **生成**: ODE求解 (平滑轨迹)
- **参数**: ~7M
- **性能**: Spearman 0.575

### 互补关系

- seq2phase: 快速, 轻量, 生产环境
- PhaseFlow: 灵活, 双向, 研究探索

---

## 14. 下一步方向

### 可能的改进

1. **模型融合**: 结合seq2phase的轻量性和PhaseFlow的灵活性
2. **多任务学习**: 加入其他蛋白质性质预测
3. **条件生成**: 基于特定性质生成序列
4. **蒸馏**: 将PhaseFlow知识蒸馏到seq2phase
5. **集成**: 多模型投票提高鲁棒性

### 实验建议

1. 对比不同Flow权重的效果
2. 评估Set Transformer vs Linear编码器
3. 分析缺失值对性能的影响
4. 测试不同ODE求解器的精度/速度权衡
5. 探索phase2seq的生成质量

---

## 附录: 完整目录树

```
PhaseFlow/
├── phaseflow/                          # 核心模块
│   ├── __init__.py
│   ├── model.py                        # 主模型
│   ├── transformer.py                  # Transformer骨干
│   ├── data.py                         # 数据集
│   ├── tokenizer.py                    # 分词器
│   ├── utils.py                        # 工具函数
│   └── token_mapping.yaml
├── train/
│   └── train.py                        # 训练脚本
├── config/                             # 配置文件 (8个)
├── outputs/                            # 训练输出 (12个)
├── logs/                               # 训练日志 (70+个)
├── visual_training/                    # 可视化 (8个)
├── infer/                              # 推理脚本
│   ├── example_usage.py
│   ├── infer_mean_correlation.py
│   └── comparison/
├── sliding_window_scoring/             # 滑动窗口
│   ├── score_windows.py
│   ├── compute_idr_stats.py
│   ├── protein_window_index.pkl
│   ├── protein_window_scores.jsonl
│   └── sliding_window_scoring.log
├── scripts/                            # 辅助脚本
│   ├── train.sh
│   ├── infer.sh
│   ├── resume.sh
│   ├── kill_training.sh
│   └── README.md
├── predict_random_peptides_phaseflow.py
├── run_phase2seq.py
├── README.md
└── requirements.txt
```

---

**生成时间**: 2026-02-25
**分析深度**: Medium
**总代码行数**: 2164+ LOC (核心模块)
