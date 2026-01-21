# PhaseFlow 安装指南

## 依赖对比与说明

### 依赖来源

PhaseFlow 是基于以下两个项目改编的：
- **flow_matching** (Meta): Flow Matching 框架
  - 依赖: `numpy`, `torch`, `torchdiffeq`
- **transfusion-pytorch** (Phil Wang): Transfusion 模型实现
  - 依赖: `torch>=2.0`, `einops>=0.8.0`, `rotary_embedding_torch>=0.8.4`, `torchdiffeq`, 等

### PhaseFlow 的实现策略

为了避免过多外部依赖，PhaseFlow 采用了**最小化依赖**策略：

| 功能 | 原库使用 | PhaseFlow实现 | 依赖 |
|------|---------|------------|------|
| Rotary 位置编码 | `rotary_embedding_torch` | 自己实现 | ✓可选 |
| ODE 求解 | `torchdiffeq.odeint` | Euler 方法（简单） | ✓可选 |
| 类型注解 | `jaxtyping` | Python标准注解 | ✓可选 |
| 配置管理 | YAML | `pyyaml` | ✓必需 |
| Tensor操作 | `einops` | `einops` | ✓必需 |

## 快速安装

### 最小安装（推荐用于快速开发）
```bash
cd /data4/huangyanjie/LLPS/predictor/PhaseFlow
pip install torch numpy pandas pyyaml einops tqdm
```

### 标准安装（包含可选库）
```bash
cd /data4/huangyanjie/LLPS/predictor/PhaseFlow
pip install -r requirements.txt
```

### 完整安装（包含所有可选优化）
```bash
cd /data4/huangyanjie/LLPS/predictor/PhaseFlow
pip install -r requirements.txt rotary_embedding_torch torchdiffeq jaxtyping beartype
```

## 依赖详解

### 必需依赖 (3个)

| 库 | 版本 | 用途 |
|----|------|------|
| **torch** | >=2.0.0 | 深度学习框架 |
| **numpy** | >=1.21.0 | 数值计算 |
| **pandas** | >=1.3.0 | 数据处理 |
| **pyyaml** | >=6.0 | 配置文件解析 |
| **einops** | >=0.6.0 | Tensor 重塑操作 |
| **tqdm** | >=4.64.0 | 进度条 |

### 可选依赖与优化

#### 1. **高级 ODE 求解** (torchdiffeq)
```python
# 默认实现（无需额外依赖）
for t_val in times:
    v = model.forward_flow(...)
    x = x + v * dt  # Euler 方法

# 使用 torchdiffeq 的高级求解器（可选优化）
from torchdiffeq import odeint
x = odeint(velocity_fn, x0, t, method='dopri')  # Runge-Kutta 方法
```

**何时需要**: 需要更高精度的 ODE 求解时
```bash
pip install torchdiffeq
```

#### 2. **优化的 Rotary 位置编码** (rotary_embedding_torch)
```python
# 默认实现（自己写）
class RotaryEmbedding(nn.Module):
    def forward(self, seq_len, device):
        ...

# 使用优化版本（可选）
from rotary_embedding_torch import RotaryEmbedding as OptimizedRoPE
```

**何时需要**: 追求最大训练速度时
```bash
pip install rotary_embedding_torch
```

#### 3. **类型检查与注解** (jaxtyping, beartype)
```python
# 默认 (Python标准注解)
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...

# 使用 jaxtyping 的更详细注解（可选）
from jaxtyping import Float
def forward(self, x: Float[torch.Tensor, "batch seq dim"]) -> Float[torch.Tensor, "batch 16"]:
    ...
```

**何时需要**: IDE 辅助和静态检查时
```bash
pip install jaxtyping beartype
```

#### 4. **可视化** (matplotlib)
```python
from phaseflow.utils import visualize_phase_diagram
visualize_phase_diagram(phase, save_path="output.png")
```

**何时需要**: 需要绘制相图时
```bash
pip install matplotlib
```

## 版本检查

安装后验证环境：

```bash
python -c "
import torch
import numpy as np
import pandas as pd
import einops
import yaml

print(f'torch: {torch.__version__}')
print(f'numpy: {np.__version__}')
print(f'pandas: {pd.__version__}')
print(f'einops: {einops.__version__}')
print('✓ 基础依赖安装成功！')

# 检查可选依赖
try:
    import torchdiffeq
    print(f'torchdiffeq: {torchdiffeq.__version__}')
except ImportError:
    print('torchdiffeq: 未安装（可选）')

try:
    import rotary_embedding_torch
    print('rotary_embedding_torch: 已安装')
except ImportError:
    print('rotary_embedding_torch: 未安装（可选）')

try:
    import jaxtyping
    print('jaxtyping: 已安装')
except ImportError:
    print('jaxtyping: 未安装（可选）')
"
```

## 环境创建推荐

### Conda 环境
```bash
# 创建新环境
conda create -n phaseflow python=3.10

# 激活
conda activate phaseflow

# 安装 PyTorch
conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio -c pytorch
# 或使用 pip (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 PhaseFlow 依赖
cd /data4/huangyanjie/LLPS/predictor/PhaseFlow
pip install -r requirements.txt
```

### 虚拟环境
```bash
# 创建
python3.10 -m venv phaseflow_env

# 激活
source phaseflow_env/bin/activate  # Linux/Mac
phaseflow_env\Scripts\activate  # Windows

# 安装
pip install torch
pip install -r requirements.txt
```

## GPU 支持

### CUDA 12.1 (推荐)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CPU-only
```bash
pip install torch torchvision torchaudio
```

### 验证 GPU
```python
import torch
print(torch.cuda.is_available())      # True if GPU available
print(torch.cuda.current_device())    # Device index
print(torch.cuda.get_device_name(0))  # Device name
```

## 故障排除

### ImportError: No module named 'torch'
```bash
# 重新安装 torch
pip install torch --upgrade
```

### CUDA/GPU 问题
```bash
# 检查 CUDA 版本
nvidia-smi

# 验证 PyTorch CUDA 绑定
python -c "import torch; print(torch.cuda.is_available())"

# 重新安装匹配的 PyTorch 版本
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 内存不足
- 减小 `batch_size` 配置
- 使用 `torch.cuda.empty_cache()` 清理缓存
- 考虑使用 gradient accumulation

### 导入错误
```bash
# 验证包路径
python -c "from phaseflow import PhaseFlow; print('✓')"

# 或者在项目目录运行
cd /data4/huangyanjie/LLPS/predictor/PhaseFlow
python test_installation.py
```

## 生产环境推荐配置

```bash
# Python 3.10+, CUDA 12.1, GPU A100/H100
pip install torch==2.1.0 numpy==1.24.0 pandas==2.0.0
pip install einops==0.7.0 pyyaml==6.0.1 tqdm==4.66.0
pip install torchdiffeq rotary_embedding_torch matplotlib
pip install jaxtyping beartype  # 可选
```

## 进一步阅读

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [Flow Matching 文档](https://github.com/facebookresearch/flow_matching)
- [Transfusion 文档](https://github.com/lucidrains/transfusion-pytorch)
