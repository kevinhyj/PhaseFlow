# PhaseFlow: 蛋白质序列与相图双向生成模型

## 研究背景与创新点

PhaseFlow 将 **Transfusion** 架构首次迁移至 **液-液相分离 (LLPS) 蛋白质预测与设计** 领域，实现了蛋白质氨基酸序列与相图之间的双向生成。

### 核心创新

1. **领域迁移**: 将 Transfusion 从一般性的序列-模态对齐，迁移至相分离蛋白的序列-相图联合建模

2. **架构改进**:
   - 设计了 **SetPhaseEncoder**，将 4×4 相图网格的每个 PSSI 值独立编码为 token，支持缺失值掩码
   - 采用 Rotary Position Embedding (RoPE) 实现高效的位置编码

3. **扩散模型升级**: 将原版 Transfusion 中的 **DDPM 替换为 Flow Matching**
   - Flow Matching 使用条件最优传输 (CondOT) 路径，训练更稳定
   - 推理时通过 ODE 求解器生成，无需迭代采样，效率大幅提升

---

## 目录
- [模型架构](#模型架构)
- [使用方法](#使用方法)
- [训练指南](#训练指南)
- [推理示例](#推理示例)

---

## 模型架构

### 1. 整体设计

PhaseFlow 基于 **Transfusion** 架构改进而来，实现了 **双向预测**:

| 方向 | 方法 | 说明 |
|------|------|------|
| 序列 → 相图 | **Flow Matching** (改进自 DDPM) | 预测速度场 v(x,t)，通过 ODE 求解生成相图 |
| 相图 → 序列 | 条件语言建模 | 以相图为条件，自回归生成氨基酸序列 |

> **注**: 原版 Transfusion 使用 DDPM 作为扩散模块，我们将其替换为 Flow Matching 后，模型性能大幅提升。

```
输入格式: [SOS] [AA序列] [EOS] [META] [shape] [SOM] [相图16维] [EOM] [EOS]
```

### 2. 核心组件

#### 2.1 相图编码器 (Phase Encoder)

**SetPhaseEncoder** (默认使用):
- 每个有效的 PSSI 值 (16个网格位置) 独立编码为 token
- 使用正弦位置编码 + MLP 学习: `token = MLP(sinusoidal(pssi_i)) + pos_emb[i]`
- 支持缺失值掩码，缺失位置不参与注意力计算

```python
# phaseflow/model.py - SetPhaseEncoder
class SetPhaseEncoder(nn.Module):
    def __init__(self, dim: int = 256, phase_dim: int = 16):
        self.value_sinusoidal = SinusoidalPosEmb(dim)
        self.value_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.pos_emb = nn.Embedding(phase_dim, dim)
```

#### 2.2 Transformer 骨干

- **层数**: 6层 (depth=6)
- **隐藏维度**: 256 (dim=256)
- **注意力头数**: 8 (heads=8)
- **头维度**: 32 (dim_head=32)
- **位置编码**: Rotary Position Embedding (RoPE)
- **前馈网络**: SwiGLU 激活

```python
# phaseflow/transformer.py
class Transformer(nn.Module):
    def __init__(self, dim=256, depth=6, heads=8, dim_head=32, dropout=0.0):
        # Multi-head attention with rotary embeddings
        # Feed-forward network with SwiGLU
```

#### 2.3 扩散模型 (改进: Flow Matching)

> **核心改进**: 将原版 Transfusion 的 DDPM 替换为 Flow Matching，显著提升生成质量和效率

**Flow Matching** (默认采用):
- 使用条件最优传输 (CondOT) 路径: `x_t = (1-t) * x_0 + t * x_1`
- 速度目标: `v = x_1 - x_0`
- 通过 ODE 求解器 (Euler / DOPRI5) 生成，无需迭代采样
- **优势**: 训练更稳定、推理更高效、性能大幅提升

**DDPM** (保留兼容):
- 支持线性/余弦噪声调度
- 支持 DDIM 加速采样
- 1000 步扩散 (默认)

```python
# phaseflow/model.py - PhaseFlow
class PhaseFlow(nn.Module):
    def __init__(self,
        dim=256, depth=6, heads=8, dim_head=32,
        vocab_size=32, phase_dim=16, max_seq_len=32,
        dropout=0.0,
        diffusion_type="flow_matching",  # "flow_matching" | "ddpm"
        num_timesteps=1000,
        beta_schedule="cosine",
    ):
```

### 3. 损失函数

总损失 = Flow 损失 + λ × LM 损失

```python
# 训练时前向传播
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    phase=phase_values,
    phase_mask=phase_mask,
    seq_len=seq_len,
    labels=labels,
    flow_weight=10.0,   # flow_loss_weight
    lm_weight=1.0,      # lm_loss_weight
)
```

---

## 使用方法

### 1. 环境配置

```bash
# 创建 conda 环境
conda env create -f environment.yml
conda activate phaseflow
```

### 2. 训练模型

#### 基本训练命令

```bash
cd /data/yanjie_huang/LLPS/predictor/PhaseFlow
bash scripts/train.sh -c <config_file> -g <gpu_id>
```

#### 示例配置

```bash
# Flow Matching (默认)
bash scripts/train.sh -c set_flow32_missing15.yaml -g 0

# DDPM
bash scripts/train.sh -c set_ddpm_cosine_missing15.yaml -g 0

# 纯语言模型 (flow_weight=0)
bash scripts/train.sh -c set_lm32_missing15.yaml -g 0
```

#### 关键参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `-c, --config` | 配置文件路径 | `set_flow32_missing15.yaml` |
| `-g, --gpu` | GPU ID | `0` |
| `-b, --batch` | 覆盖 batch_size | `2048` |
| `-l, --lr` | 覆盖学习率 | `0.0008` |
| `-e, --epochs` | 覆盖训练轮数 | `200` |
| `-t, --threshold` | 缺失值阈值 | `-1` (使用所有数据) |

#### 配置文件示例

```yaml
# config/set_ddpm_cosine_missing15.yaml
model:
  dim: 256
  depth: 6
  heads: 8
  dim_head: 32
  vocab_size: 32
  phase_dim: 16
  max_seq_len: 32
  dropout: 0.1
  use_set_encoder: true
  diffusion_type: ddpm
  num_timesteps: 1000
  beta_schedule: cosine

training:
  batch_size: 2048
  lr: 0.0008
  weight_decay: 0.01
  epochs: 200
  warmup_steps: 1000
  flow_loss_weight: 32
  lm_loss_weight: 0
  max_grad_norm: 1.0
  use_amp: true
  early_stopping_patience: 20

sampling:
  method: ddim
  sampling_steps: 50
  use_ddim: true
```

### 3. 推理示例

#### 3.1 从序列预测相图

```python
import torch
from phaseflow import PhaseFlow, AminoAcidTokenizer

# 加载模型
checkpoint = torch.load('outputs_set/output_set_flow32_missing15/best_model.pt')
config = checkpoint['config']

model = PhaseFlow(
    dim=config['model']['dim'],
    depth=config['model']['depth'],
    heads=config['model']['heads'],
    dim_head=config['model']['dim_head'],
    vocab_size=config['model']['vocab_size'],
    phase_dim=config['model']['phase_dim'],
    max_seq_len=config['model']['max_seq_len'],
    dropout=0.0,
    use_set_encoder=config['model'].get('use_set_encoder', False),
    diffusion_type=config['model'].get('diffusion_type', 'flow_matching'),
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 编码序列
tokenizer = AminoAcidTokenizer()
sequence = "ACDEFGHIKLMNPQRSTVWY"
tokens = tokenizer.build_input_sequence(sequence)
input_ids = torch.tensor([tokenizer.pad_sequence(tokens, 32)])
attention_mask = (input_ids != tokenizer.PAD_ID).long()
seq_len = torch.tensor([len(sequence)])

# 生成相图 (Flow Matching)
with torch.no_grad():
    pred_phase = model.generate_phase(
        input_ids, attention_mask, seq_len,
        method='euler'  # 或 'dopri5' for ODE solver
    )

# DDPM 采样
with torch.no_grad():
    pred_phase = model.generate_phase(
        input_ids, attention_mask, seq_len,
        num_steps=50,      # DDIM 步数
        use_ddim=True,    # 使用 DDIM
    )

print("Predicted PSSI values:", pred_phase)
```

#### 3.2 从相图生成序列

```python
import torch
from phaseflow import PhaseFlow, AminoAcidTokenizer

# 加载模型 (同上)

# 创建相图输入 (batch_size=1, phase_dim=16)
phase = torch.randn(1, 16)  # 或使用真实 PSSI 值
phase_mask = torch.ones(1, 16)  # 所有位置有效

# 生成序列
with torch.no_grad():
    tokens, sequences = model.generate_sequence(
        phase, tokenizer,
        max_len=32,
        temperature=1.0,
        top_k=40,
        top_p=0.9,
    )

print("Generated sequence:", sequences[0])
```

---

## 训练指南

### 1. 数据格式

输入数据为 CSV 文件，包含:
- `AminoAcidSequence`: 氨基酸序列 (5-20 AA)
- `group_11` ~ `group_44`: 16 个 PSSI 值 (4×4 网格)

```csv
AminoAcidSequence,group_11,group_12,...,group_44
ACDEFGHIKLMNPQRSTVWY,1.23,0.85,...,2.01
```

### 2. 训练脚本参数

```bash
# 标准训练
bash scripts/train.sh -c config/set_flow32_missing15.yaml -g 0

# 从断点恢复
bash scripts/train.sh -c config/set_flow32_missing15.yaml -g 0 --resume /path/to/checkpoint.pt
```

### 3. 超参数搜索

项目中包含多个预定义配置用于超参搜索:

| 配置 | flow_weight | lm_weight | 用途 |
|------|-------------|-----------|------|
| `set_flow32_*` | 32 | 1 | 强 Flow, 标准 LM |
| `set_flow5_*` | 5 | 1 | 中等 Flow |
| `set_flow1_*` | 1 | 1 | 弱 Flow |
| `set_flow0_*` | 0 | 1 | 纯 LM |
| `set_lm32_*` | 1 | 32 | 强 LM |
| `set_lm5_*` | 1 | 5 | 中等 LM |
| `set_lm0_*` | 1 | 0 | 纯 Flow |

### 4. 监控训练

```bash
# 查看训练日志
tail -f logs/train_set_flow32_missing15_*.log

# 查看训练曲线
ls visual_training/set_flow32_missing15/
```

训练会自动保存:
- `best_model.pt`: 验证集最优模型
- `final_model.pt`: 最后一轮模型
- `checkpoint_epoch*.pt`: 定期检查点

---

## 模型规模

默认配置参数量: **~7M 参数**

```
dim=256, depth=6, heads=8, dim_head=32
- Transformer: 6 layers × (8 heads × 32 dim_head) = 6 × 256 × 8 × 32 ≈ 393K
- FFN: 6 × 4×256×256 ≈ 1.5M
- Embeddings: ~3M
- 输出头: ~2M
```

---

## 评估指标

训练过程监控以下指标:

| 指标 | 说明 |
|------|------|
| `flow_loss` | Flow Matching 预测速度的 MSE |
| `lm_loss` | 语言模型交叉熵 |
| `perplexity` | 语言模型困惑度 |
| `mse/mae/rmse` | 相图预测误差 |
| `pearson/spearman` | 相图预测相关性 |

---

## 引用

如需引用 PhaseFlow，请参考:

```bibtex
@article{phaseflow2025,
  title={PhaseFlow: Bidirectional Generation of Protein Sequences and Phase Diagrams for LLPS Prediction},
  author={},
  journal={},
  year={2025}
}
```
