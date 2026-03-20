# PhaseFlow 项目技术文档

## 1. 项目概述

PhaseFlow 是一个基于 Transfusion 架构的双向生成模型，结合 Flow Matching（序列→相图）和语言建模（相图→序列），用于预测氨基酸序列的液-液相分离（LLPS）特性。

- **输入**：5-20 个氨基酸的短肽序列
- **输出**：16 维 PSSI 向量（4×4 网格：4 个光照强度 × 4 个浓度条件）
- **参数量**：~7M
- **训练数据**：~280K 条序列，每条最多 16 个 PSSI 值（大量缺失，~62.6%）

---

## 2. 目录结构

```
PhaseFlow/
├── phaseflow/                  # 核心模块
│   ├── __init__.py             # 导出 PhaseFlow, AminoAcidTokenizer, PhaseDataset
│   ├── model.py                # 主模型（PhaseCNNEncoder + PhaseFlow）
│   ├── transformer.py          # Transformer 骨干网络（RoPE, SwiGLU, RMSNorm）
│   ├── tokenizer.py            # 氨基酸分词器（32 token 词表）
│   ├── data.py                 # 数据集和 DataLoader
│   └── utils.py                # 工具函数（指标、检查点、调度器等）
├── train/
│   └── train.py                # 训练主循环（~800行）
├── scripts/
│   └── train.sh                # 训练启动脚本（nohup + conda + 参数覆盖）
├── config/                     # YAML 配置文件
│   ├── bs2048_lr0.0008_flow{1,5,10,32}_20260123.yaml
│   └── flow{5,32}_missing{0,3,7,11}_20260310.yaml
├── infer/                      # 推理脚本
│   ├── infer_mean_correlation.py   # 均值相关性评估
│   ├── example_usage.py            # 使用示例
│   └── flow5_length_hist.py        # 长度分布直方图
├── sliding_window_scoring/
│   └── score_windows.py        # 蛋白质滑动窗口打分
├── step1_generate_phaseflow.py # 序列生成（100K 候选）
├── test_phase2seq.py           # 相图→序列测试
├── run_phase2seq.py            # 真实数据生成测试
└── outputs/                    # 训练输出（模型、日志、配置）
```

---

## 3. 模型架构

### 3.1 整体架构

```
                    ┌─────────────────────────────────┐
                    │         PhaseFlow 模型            │
                    │                                   │
  seq→phase 方向    │  [seq tokens] + [phase token]     │    phase→seq 方向
  (Flow Matching)   │         ↓                         │    (Language Modeling)
                    │   6层 Transformer (共享骨干)       │
                    │   - RoPE 位置编码                  │
                    │   - SwiGLU FFN                    │
                    │   - RMSNorm + Pre-norm             │
                    │         ↓                         │
                    │  velocity_head  /  lm_head        │
                    │    (B, 16)     /  (B, S, 32)      │
                    └─────────────────────────────────┘
```

### 3.2 核心组件

#### Token 嵌入

- `token_embed = nn.Embedding(vocab_size=32, dim=256)`
- 词表：20 个氨基酸 + 6 个特殊 token + 6 个形状编码

#### Phase 编码器（PhaseCNNEncoder）

- `nn.Linear(16, 256)` → `(B, 1, 256)`
- 将 16 维 PSSI 向量映射为单个 256 维 token
- 缺失值用 0 填充，mask 单独传递

#### 时间编码（仅 flow matching 方向使用）

```python
time_encoder = SinusoidalPosEmb(dim)           # (B,) → (B, dim)
time_mlp = Linear(dim, 4*dim) → SiLU → Linear(4*dim, dim)  # (B, dim) → (B, dim)
# 时间嵌入直接加到 phase 嵌入上
phase_emb = phase_emb + time_emb.unsqueeze(1)
```

#### Transformer 骨干

| 参数 | 值 | 说明 |
|------|-----|------|
| dim | 256 | 隐藏维度 |
| depth | 6 | Transformer 层数 |
| heads | 8 | 注意力头数 |
| dim_head | 32 | 每头维度 |
| ff_mult | 4 | FFN 扩展倍数 |
| max_seq_len | 32 | 最大序列长度 |

每层结构（Pre-norm + 残差）：
```
x → RMSNorm → MultiHeadAttention(RoPE) → + x → RMSNorm → FeedForward(SwiGLU) → + x
```

**注意力掩码策略**：
- 序列 token：因果注意力（token i 只看 0..i）
- Phase token：双向注意力（可看所有 token 和自身）
- 通过 `phase_start_idx` 参数控制

#### 输出头

- `velocity_head = nn.Linear(256, 16)` — Flow Matching 速度预测
- `lm_head = nn.Linear(256, 32, bias=False)` — 语言建模 next-token 预测

---

## 4. 双向训练

### 4.1 Forward: seq → phase（Flow Matching）

**`forward_flow(input_ids, attention_mask, phase_t, phase_mask, time, seq_len)`**

数据流：
```
1. token_emb = embed_tokens(input_ids)           # (B, S, 256)
2. phase_emb = embed_phase(phase_t, mask, time)  # (B, 1, 256) 含时间编码
3. x = cat([token_emb, phase_emb], dim=1)        # (B, S+1, 256)
4. hidden = transformer(x, mask, phase_start_idx=S)  # (B, S+1, 256)
5. phase_hidden = hidden[:, -1, :]               # (B, 256) 取最后一个 token
6. velocity = velocity_head(phase_hidden)         # (B, 16)
```

**Flow Matching 损失（CondOT 路径）**：
```python
t = rand(B)                            # 采样时间 t ∈ [0, 1]
x_0 = randn_like(phase)               # 噪声
x_t = (1 - t) * x_0 + t * phase       # 线性插值（条件最优传输路径）
v_target = phase - x_0                 # 目标速度
v_pred = forward_flow(...)             # 预测速度

# 带掩码的 MSE 损失 + 二次加权
diff = (v_pred - v_target)^2 * phase_mask      # 仅计算有效位置
valid_count = phase_mask.sum(dim=-1)
weight = (valid_count / 16)^2                   # 完整样本权重=1，8/16→0.25，1/16→0.004
loss = ((diff.sum(dim=-1) / valid_count) * weight).mean()
```

### 4.2 Backward: phase → seq（Language Modeling）

**`forward_lm(input_ids, attention_mask, phase, phase_mask)`**

数据流：
```
1. phase_emb = embed_phase(phase, mask, time=None)  # (B, 1, 256) 无时间编码
2. token_emb = embed_tokens(input_ids)               # (B, S, 256)
3. x = cat([phase_emb, token_emb], dim=1)            # (B, 1+S, 256) phase 在前！
4. hidden = transformer(x, mask, phase_start_idx=None)  # 纯因果注意力
5. token_hidden = hidden[:, 1:, :]                   # (B, S, 256) 跳过 phase token
6. logits = lm_head(token_hidden)                    # (B, S, 32)

loss = cross_entropy(logits, labels, ignore_index=-100)
```

### 4.3 联合训练

```python
total_loss = flow_weight × flow_loss + lm_weight × lm_loss
```

典型权重：`flow_weight=32, lm_weight=1`（强调相图预测任务）

---

## 5. 推理

### 5.1 序列 → 相图（ODE 积分）

```python
def generate_phase(input_ids, attention_mask, seq_len, method='euler'):
    x_init = randn(B, 16)      # t=0: 从噪声开始
    phase_mask = ones(B, 16)    # 预测所有 16 维

    def ode_func(t, x):
        return forward_flow(input_ids, attention_mask, x, phase_mask, t, seq_len)

    trajectory = odeint(ode_func, x_init, [0.0, 1.0], method=method)
    return trajectory[-1]       # t=1: 最终预测
```

支持的 ODE 求解器：`euler`（快）、`midpoint`、`rk4`、`dopri5`（精确，自适应步长）

### 5.2 相图 → 序列（自回归采样）

```python
def generate_sequence(phase, tokenizer, max_len=25, temperature=1.0):
    tokens = [SOS]
    for _ in range(max_len):
        logits = forward_lm(tokens, attention_mask, phase, phase_mask)
        next_token = sample(logits[:, -1, :] / temperature)
        if next_token == EOS: break
        tokens.append(next_token)
    return tokenizer.decode(tokens)
```

---

## 6. 分词器（AminoAcidTokenizer）

### 词表（共 32 个 token）

| ID 范围 | 类型 | 内容 |
|---------|------|------|
| 0-19 | 氨基酸 | A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y |
| 20 | PAD | 填充 |
| 21 | SOS | 序列起始 |
| 22 | EOS | 序列结束 |
| 23 | META | 元数据分隔符 |
| 24 | SOM | 模态起始（phase tokens） |
| 25 | EOM | 模态结束 |
| 26-31 | Shape | 网格维度编码（0-5） |

### 编码格式

```
输入序列: "ACDEF"
编码后:   [SOS=21, A=0, C=1, D=2, E=3, F=4, EOS=22, META=23, shape=30, shape=30, SOM=24]
填充到:   max_seq_len=32
```

---

## 7. 数据管线

### 7.1 数据集（PhaseDataset）

**输入**：`phase_diagram_original_scale.csv`（280K 行，16 列 PSSI + 序列列）

**两种加载模式**：

| 模式 | 条件 | 说明 |
|------|------|------|
| 随机划分 | `missing_threshold=-1` | 所有数据 90/5/5 划分 |
| 缺失值划分 | `missing_threshold=N` | 只用 missing≤N 的数据，再 90/5/5 划分 |

实际训练中使用独立的 val_set.csv (~15K) 和 test_set.csv (500) 作为验证/测试集。

**每条样本返回**：
```python
{
    'input_ids':      (max_seq_len,)  # token ID 序列
    'attention_mask':  (max_seq_len,)  # 1=有效, 0=padding
    'phase_values':   (16,)           # PSSI 值（NaN→0）
    'phase_mask':     (16,)           # 1=有效, 0=缺失
    'seq_len':        scalar           # 原始序列长度
    'sequence':       str              # 原始氨基酸序列
}
```

### 7.2 数据处理流程

```
CSV/NPZ → PhaseDataset
    → NaN → 0, mask = ~isNaN
    → 序列 tokenize + pad 到 32
    → 可选: normalize_phase（按全局均值/标准差归一化）
        ↓
DataLoader (batch_size=2048, num_workers=4, pin_memory=True)
        ↓
训练循环
```

---

## 8. 训练配置

### 8.1 YAML 配置示例

```yaml
model:
  dim: 256              # 模型隐藏维度
  depth: 6              # Transformer 层数
  heads: 8              # 注意力头数
  dim_head: 32          # 每头维度
  vocab_size: 32        # 词表大小
  phase_dim: 16         # 相图维度 (4×4)
  max_seq_len: 32       # 最大序列长度
  dropout: 0.1          # Dropout 率

training:
  batch_size: 2048      # 批大小
  lr: 0.0008            # 学习率
  weight_decay: 0.01    # L2 正则化
  epochs: 200           # 最大训练轮数
  warmup_steps: 1000    # 学习率预热步数
  flow_loss_weight: 32  # Flow Matching 损失权重
  lm_loss_weight: 1     # 语言建模损失权重
  max_grad_norm: 1.0    # 梯度裁剪
  use_amp: true         # 混合精度训练
  early_stopping_patience: 20  # 早停耐心
  save_every: 20        # 检查点保存间隔

sampling:
  method: euler         # ODE 求解器
  temperature: 1.0

data:
  train_ratio: 0.9
  val_ratio: 0.05
  num_workers: 4
  normalize_phase: true
```

### 8.2 训练流程

```
每个 epoch:
  1. 遍历训练 DataLoader
  2. 创建 LM labels（input_ids 右移一位，padding=-100）
  3. AMP autocast 前向传播:
     outputs = model(input_ids, attention_mask, phase, phase_mask, seq_len, labels, flow_weight, lm_weight)
  4. scaler.scale(loss).backward()
  5. clip_grad_norm_(model.parameters(), max_grad_norm=1.0)
  6. scaler.step(optimizer), scaler.update()
  7. 余弦退火调度器 step

验证:
  1. 前向传播计算 loss
  2. generate_phase() 生成相图预测
  3. 计算指标: MSE, MAE, RMSE, Pearson, Spearman
  4. 早停判断（基于 val_loss）
  5. 保存最佳模型 best_model.pt
```

### 8.3 训练命令

```bash
# 标准训练
bash scripts/train.sh -g 0 -c bs2048_lr0.0008_flow32_20260123.yaml

# 层次化训练（只用 missing≤11 的数据）
bash scripts/train.sh -g 0 -c flow32_missing11_20260310.yaml -t 11

# 参数覆盖
bash scripts/train.sh -g 0 -c config.yaml -b 1024 -l 0.0004 -e 100
```

---

## 9. 评估指标

| 指标 | 计算方式 | 说明 |
|------|---------|------|
| MSE | `mean((pred - target)^2)` 仅有效值 | 均方误差 |
| MAE | `mean(|pred - target|)` 仅有效值 | 平均绝对误差 |
| RMSE | `sqrt(MSE)` | 均方根误差 |
| Pearson | `corrcoef(pred, target)` 仅有效值 | 线性相关系数 |
| Spearman | `spearmanr(pred, target)` 仅有效值 | 秩相关系数 |
| Perplexity | `exp(cross_entropy)` 仅有效 token | 语言建模困惑度 |

---

## 10. 缺失值处理策略

1. **数据层**：NaN → 0，同时记录 phase_mask（1=有效，0=缺失）
2. **损失层**：`loss = (pred - target)^2 * phase_mask`，仅在有效位置计算
3. **加权层**：二次权重 `w = (n_valid / 16)^2`，完整样本权重大，稀疏样本权重小
4. **推理层**：生成时 phase_mask 全为 1，预测所有 16 维

---

## 11. 输出目录结构

```
outputs/output_{config_name}/
├── config.yaml            # 保存的配置副本
├── train.log              # 训练日志
├── best_model.pt          # 最佳模型（最低 val_loss）
├── final_model.pt         # 最终模型
├── checkpoint_epoch{N}.pt # 周期性检查点
├── test_results.json      # 测试集评估结果
└── figures/               # 训练曲线图
    ├── loss.png
    ├── flow_lm_loss.png
    ├── correlation.png
    ├── perplexity.png
    └── summary.png
```

---

## 12. 关键 Tensor 形状速查表

| Tensor | 形状 | 说明 |
|--------|------|------|
| input_ids | (B, 32) | token ID |
| attention_mask | (B, 32) | 1=有效, 0=padding |
| phase_values | (B, 16) | PSSI 值 |
| phase_mask | (B, 16) | 1=有效, 0=缺失 |
| seq_len | (B,) | 序列长度 |
| token_emb | (B, 32, 256) | token 嵌入 |
| phase_emb | (B, 1, 256) | phase 嵌入 |
| transformer 输入 | (B, 33, 256) | seq + phase 拼接 |
| transformer 输出 | (B, 33, 256) | 隐藏状态 |
| velocity | (B, 16) | 速度场预测 |
| lm_logits | (B, 32, 32) | next-token 概率 |
| ODE trajectory | (T, B, 16) | ODE 轨迹 |

---

## 13. 实验结果汇总

### Flow32 层次化训练（2026-03-10）

| 模型 | 训练样本数 | Spearman | MSE | Perplexity |
|------|-----------|----------|-----|-----------|
| missing0 | 10,165 | 0.319 | 0.651 | 8.31 |
| missing3 | 33,199 | 0.335 | 0.685 | 8.28 |
| missing7 | 75,509 | 0.377 | 0.653 | 8.39 |
| **missing11** | **147,949** | **0.392** | **0.613** | **8.32** |

### Flow5 层次化训练（2026-03-10）

| 模型 | Spearman | MSE | Perplexity |
|------|----------|-----|-----------|
| missing0 | 0.335 | 0.628 | 8.58 |
| missing3 | 0.300 | 0.677 | 8.35 |
| missing7 | 0.335 | 0.662 | 8.36 |
| **missing11** | **0.366** | **0.611** | **8.34** |

**结论**：Flow32 优于 Flow5 在相图预测上；missing11（~148K 样本）达到最佳 Spearman。

---

## 14. 依赖

```
pytorch >= 2.0
torchdiffeq           # ODE 求解器
einops                # Tensor 变换
rotary_embedding_torch  # RoPE 位置编码
pyyaml                # 配置文件
scipy                 # Spearman 相关系数
matplotlib            # 可视化
pandas, numpy         # 数据处理
```

Conda 环境：`phaseflow`
