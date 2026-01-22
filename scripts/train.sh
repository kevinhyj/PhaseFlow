#!/bin/bash
# PhaseFlow Training Script
# 用法: bash scripts/train.sh [gpu_id]

# 默认使用 GPU 0
GPU_ID=${1:-0}

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
# TORCH_CUDA_ARCH_LIST 仅用于编译，运行时不需要设置

# 路径设置
PROJECT_DIR="/data4/huangyanjie/LLPS/predictor/PhaseFlow"
DATA_DIR="/data4/huangyanjie/LLPS/phase_diagram"
OUTPUT_DIR="${PROJECT_DIR}/outputs"
LOG_DIR="${PROJECT_DIR}/logs"

# 添加PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# CUDA 显存优化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0

# 创建输出和日志目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# 生成日志文件名（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "========================================"
echo "PhaseFlow Training"
echo "========================================"
echo "GPU ID: $GPU_ID"
echo "Data: $DATA_DIR/phase_diagram.csv"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "========================================"

# 激活 conda 环境
source /data4/huangyanjie/miniconda3/bin/activate phaseflow

# 检查 GPU 可用性（输出到日志）
echo "=== GPU Info ===" >> $LOG_FILE
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')" >> $LOG_FILE 2>&1
echo "" >> $LOG_FILE

# 使用 nohup 后台运行训练，输出到日志文件
cd $PROJECT_DIR
nohup python -u train/train.py \
    --config $PROJECT_DIR/config/default.yaml \
    --data_path $DATA_DIR/phase_diagram.csv \
    --output_dir $OUTPUT_DIR \
    --device cuda \
    --seed 42 \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 100 \
    >> $LOG_FILE 2>&1 &

# 获取进程 PID
PID=$!

echo "Training started in background"
echo "PID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor training with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check process status:"
echo "  ps -p $PID"
echo ""
echo "Kill training:"
echo "  kill $PID"
echo "========================================"
