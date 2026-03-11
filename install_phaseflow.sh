#!/bin/bash
set -e

echo "=========================================="
echo "创建 PhaseFlow 环境"
echo "=========================================="

# 创建环境
conda create -n phaseflow python=3.10 -y

# 激活环境并安装依赖
source $(conda info --base)/etc/profile.d/conda.sh
conda activate phaseflow

echo ""
echo "安装 PyTorch (CUDA 12.1)..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo ""
echo "安装 conda 包..."
conda install numpy pandas pyyaml tqdm scipy matplotlib -c conda-forge -y

echo ""
echo "安装 pip 包..."
pip install einops rotary-embedding-torch torchdiffeq jaxtyping beartype typing-extensions

echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="
python << 'PYEOF'
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU count: {torch.cuda.device_count()}")

import einops, rotary_embedding_torch, torchdiffeq, numpy, pandas
print("✓ einops")
print("✓ rotary_embedding_torch") 
print("✓ torchdiffeq")
print("✓ numpy, pandas")

from phaseflow import PhaseFlow, AminoAcidTokenizer
print("✓ PhaseFlow 导入成功")
PYEOF

echo ""
echo "=========================================="
echo "✓ 环境创建完成！"
echo "使用: conda activate phaseflow"
echo "=========================================="
