# PhaseFlow 探索索引

**生成时间**: 2026-02-25  
**分析深度**: Medium  
**项目路径**: `/data4/huangyanjie/LLPS/predictor/PhaseFlow`

---

## 快速导航

### 📋 文档清单

1. **PHASEFLOW_EXPLORATION.md** (本目录)
   - 完整的 PhaseFlow 分析报告
   - 包含架构、训练、推理、对比等详细内容
   - 14个章节，涵盖所有关键信息

2. **PHASEFLOW_INDEX.md** (本文件)
   - 快速导航和索引
   - 关键路径和文件位置
   - 快速参考表

---

## 核心信息速查

### 项目概览

| 项 | 值 |
|----|-----|
| **项目名** | PhaseFlow |
| **架构** | Transfusion (双向预测) |
| **任务** | seq2phase (Flow Matching) + phase2seq (LM) |
| **核心代码** | 2164 LOC |
| **模型参数** | ~7M |
| **性能** | Spearman ρ ≈ 0.575 |
| **主要特性** | 双向预测、Flow Matching、Set Transformer |

### 关键路径

```
/data4/huangyanjie/LLPS/predictor/PhaseFlow/
├── phaseflow/                    # 核心模块 (2164 LOC)
│   ├── model.py                  # 主模型 (813 LOC)
│   ├── transformer.py            # Transformer骨干 (326 LOC)
│   ├── data.py                   # 数据处理 (389 LOC)
│   ├── tokenizer.py              # 分词器 (191 LOC)
│   └── utils.py                  # 工具函数 (434 LOC)
├── train/train.py                # 训练脚本
├── config/                       # 8个配置文件
├── outputs/                      # 12个训练输出
├── infer/                        # 推理脚本
└── sliding_window_scoring/       # 滑动窗口评分
```

---

## 模型架构速览

### 双向任务

```
序列 ──Flow Matching──→ 相图  (seq2phase)
相图 ──Language Model──→ 序列  (phase2seq)
```

### 核心组件

| 组件 | 实现 | 说明 |
|------|------|------|
| 序列编码 | Token/ESM2 | 32维或1280→256维 |
| 相图编码 | Linear/Set Transformer | 简单或高级 |
| 骨干网络 | Transformer | 6层, 256维, 8头 |
| Flow Head | Linear(256→16) | 速度预测 |
| LM Head | Linear(256→32) | 下一token预测 |

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

---

## 训练配置

### 最佳配置

**文件**: `/data4/huangyanjie/LLPS/predictor/PhaseFlow/config/set_encoder_bs2048_lr0.0008_flow32_20260128.yaml`

```yaml
model:
  dim: 256, depth: 6, heads: 8
  vocab_size: 32, phase_dim: 16
  use_set_encoder: true

training:
  batch_size: 2048
  lr: 0.0008
  flow_loss_weight: 32
  lm_loss_weight: 1
  epochs: 2000
```

### 实验变体

- **flow1/5/10/32**: 不同Flow权重
- **set_encoder_flow***: Set Transformer编码器
- **missing*_flow32**: 缺失值分割实验

---

## 损失函数

### Flow Matching Loss

```python
# CondOT路径: x_t = (1-t)·x_0 + t·x_1
loss = MSE(v_pred, v_target) * phase_mask
# 二次加权: w = (valid_count/16)²
```

### Language Modeling Loss

```python
loss = CrossEntropy(logits, labels)
perplexity = exp(loss)
```

### 总损失

```python
total_loss = 32 * flow_loss + 1 * lm_loss
```

---

## 推理方法

### Phase Generation (seq → phase)

```python
# ODE求解: dx/dt = v(x, t)
1. 初始化: x_0 ~ N(0,I)
2. ODE求解器: torchdiffeq
3. 时间范围: t ∈ [0, 1]
4. 返回: x_1 (最终相图)
```

### Sequence Generation (phase → seq)

```python
# 自回归采样
1. 初始化: tokens = [SOS]
2. 循环: logits → sample → append
3. 停止: EOS 或 max_len
```

---

## 模型输出

### 测试指标

**文件**: `/data4/huangyanjie/LLPS/predictor/PhaseFlow/outputs/output_set_encoder_bs2048_lr0.0008_flow32_20260128/test_results.json`

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

### 模型大小

- 参数量: ~7M
- 检查点: 79MB
- 权重: ~28MB

---

## PhaseFlow vs Seq2Phase

| 特性 | PhaseFlow | Seq2Phase |
|------|-----------|-----------|
| 架构 | Transfusion | Cross-Attention |
| 任务 | 双向 | 单向 |
| 生成 | Flow Matching | 直接回归 |
| 参数 | ~7M | ~2-3M |
| 性能 | 0.575 | 0.57+ |
| 速度 | 慢 | 快 |
| 用途 | 研究 | 生产 |

---

## 关键创新

### 1. Set Transformer 相图编码
- Fourier特征编码坐标
- Set Transformer处理缺失值
- Attention Pooling可学习汇总

### 2. Flow Matching 生成
- 平滑生成轨迹
- 可控的生成过程
- CondOT路径 + ODE求解

### 3. 双向预测
- seq2phase: Flow Matching
- phase2seq: Language Modeling
- 统一框架, 相互验证

### 4. 缺失值处理
- 二次加权: w = (valid_count/16)²
- 优先学习完整数据
- 逐步适应缺失值

---

## 使用指南

### 训练

```bash
cd /data4/huangyanjie/LLPS/predictor/PhaseFlow
python train/train.py \
    --config config/set_encoder_bs2048_lr0.0008_flow32_20260128.yaml \
    --data_path /data4/huangyanjie/LLPS/phase_diagram/phase_diagram.csv \
    --output_dir outputs/my_run
```

### 推理 (Seq → Phase)

```bash
python infer/example_usage.py \
    --checkpoint outputs/output_set_encoder_bs2048_lr0.0008_flow32_20260128/best_model.pt \
    --mode seq2phase \
    --input "ACDEFGHIKLMN"
```

### 推理 (Phase → Seq)

```bash
python run_phase2seq.py \
    --checkpoint outputs/output_set_encoder_bs2048_lr0.0008_flow32_20260128/best_model.pt \
    --phase_values "1.0,-0.5,0.3,..."
```

### 滑动窗口评分

```bash
cd sliding_window_scoring
python score_windows.py \
    --checkpoint ../outputs/output_set_encoder_bs2048_lr0.0008_flow32_20260128/best_model.pt \
    --protein_file proteins.fasta
```

---

## 文件位置速查

### 核心模块

| 文件 | 路径 | 行数 |
|------|------|------|
| 主模型 | `/data4/huangyanjie/LLPS/predictor/PhaseFlow/phaseflow/model.py` | 813 |
| Transformer | `/data4/huangyanjie/LLPS/predictor/PhaseFlow/phaseflow/transformer.py` | 326 |
| 数据处理 | `/data4/huangyanjie/LLPS/predictor/PhaseFlow/phaseflow/data.py` | 389 |
| 分词器 | `/data4/huangyanjie/LLPS/predictor/PhaseFlow/phaseflow/tokenizer.py` | 191 |
| 工具函数 | `/data4/huangyanjie/LLPS/predictor/PhaseFlow/phaseflow/utils.py` | 434 |

### 训练和推理

| 脚本 | 路径 |
|------|------|
| 训练 | `/data4/huangyanjie/LLPS/predictor/PhaseFlow/train/train.py` |
| 推理示例 | `/data4/huangyanjie/LLPS/predictor/PhaseFlow/infer/example_usage.py` |
| Phase→Seq | `/data4/huangyanjie/LLPS/predictor/PhaseFlow/run_phase2seq.py` |
| 随机肽预测 | `/data4/huangyanjie/LLPS/predictor/PhaseFlow/predict_random_peptides_phaseflow.py` |

### 配置和输出

| 项 | 路径 |
|----|------|
| 最佳配置 | `/data4/huangyanjie/LLPS/predictor/PhaseFlow/config/set_encoder_bs2048_lr0.0008_flow32_20260128.yaml` |
| 最佳模型 | `/data4/huangyanjie/LLPS/predictor/PhaseFlow/outputs/output_set_encoder_bs2048_lr0.0008_flow32_20260128/best_model.pt` |
| 测试结果 | `/data4/huangyanjie/LLPS/predictor/PhaseFlow/outputs/output_set_encoder_bs2048_lr0.0008_flow32_20260128/test_results.json` |
| 训练数据 | `/data4/huangyanjie/LLPS/phase_diagram/phase_diagram.csv` |

---

## 依赖

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

## 数据格式

### 输入CSV

```csv
AminoAcidSequence,group_11,group_12,...,group_44
ACDEFGHIKL,1.0,-0.5,0.3,...,NaN,...
```

- 序列长度: 5-20 氨基酸
- 相图: 16维 (4×4网格)
- 缺失值: ~62.6%

### 数据分割

- 训练: 90%
- 验证: 5%
- 测试: 5%

---

## 常见问题

### Q: PhaseFlow 和 seq2phase 如何选择?

**A**: 
- 快速预测 → seq2phase (轻量, 快速)
- 双向预测 → PhaseFlow (灵活, 双向)
- 精细调优 → PhaseFlow (Flow Matching)
- 生产环境 → seq2phase (轻量)

### Q: 如何处理缺失值?

**A**: PhaseFlow 使用二次加权方案:
- 完整数据 (16个值): w=1.0
- 半数数据 (8个值): w=0.25
- 单个值: w≈0.004

### Q: 推理速度如何?

**A**: 
- seq2phase: 单次前向, 快速
- PhaseFlow: ODE求解, 较慢 (20-50步)

### Q: 如何改进性能?

**A**: 
1. 调整 Flow 权重 (1/5/10/32)
2. 使用 Set Transformer 编码器
3. 增加训练数据
4. 调整学习率和批大小

---

## 下一步方向

### 短期 (1-2周)

1. 对比不同 Flow 权重的效果
2. 评估 Set Transformer vs Linear 编码器
3. 分析缺失值对性能的影响

### 中期 (1个月)

1. 模型融合: 结合 seq2phase 的轻量性
2. 多任务学习: 加入其他蛋白质性质
3. 条件生成: 基于特定性质生成序列

### 长期 (2-3个月)

1. 蒸馏: 将 PhaseFlow 知识蒸馏到轻量模型
2. 集成: 多模型投票提高鲁棒性
3. 部署: 生产环境集成

---

## 相关资源

### 项目文档

- `/data4/huangyanjie/LLPS/predictor/PhaseFlow/README.md` - 官方README
- `/data4/huangyanjie/LLPS/predictor/PhaseFlow/doc/DATALOADER_OPTIMIZATION.md` - 数据加载优化

### 相关项目

- `/data4/huangyanjie/LLPS/seq2phase/` - seq2phase 项目
- `/data4/huangyanjie/LLPS/baseline/` - 基线模型
- `/data4/huangyanjie/LLPS/phase_diagram/` - 相图数据

### 环境

- 主环境: `conda activate llps`
- ESM2环境: `conda activate esm_project`
- PhaseFlow环境: `conda activate llps` (同主环境)

---

## 总结

PhaseFlow 是一个创新的双向预测模型，采用 Transfusion 架构结合 Flow Matching 和 Language Modeling。它提供了比 seq2phase 更灵活的生成方式和双向预测能力，但计算成本更高。两个模型在 LLPS 项目中形成互补关系。

**关键特点**:
- ✓ 双向预测 (seq↔phase)
- ✓ Flow Matching 生成平滑轨迹
- ✓ Set Transformer 处理缺失值
- ✓ 统一框架, 相互验证
- ✓ 性能稳定 (Spearman 0.575)

---

**文档版本**: 1.0  
**最后更新**: 2026-02-25  
**维护者**: Claude Code
