#!/bin/bash
# PhaseFlow Inference Script
# 使用训练好的模型进行推理
# 用法: bash scripts/infer.sh <checkpoint> [gpu_id]

CHECKPOINT=${1:-"outputs/run_xxx/best_model.pt"}
GPU_ID=${2:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

PROJECT_DIR="/data/yanjie_huang/LLPS/predictor/PhaseFlow"

# 添加PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

echo "========================================"
echo "PhaseFlow Inference"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "GPU ID: $GPU_ID"
echo "========================================"

# 1. 从文件预测相图
echo ""
echo ">>> 预测 sequences.txt 中序列的相图..."
cd $PROJECT_DIR
python sample.py \
    --checkpoint $CHECKPOINT \
    --mode seq2phase \
    --input_file sequences.txt \
    --output predicted_phases.csv \
    --method euler \
    --batch_size 32 \
    --device cuda

echo ""
echo "Results saved to predicted_phases.csv"

# 2. 可视化前10个相图
echo ""
echo ">>> 生成相图可视化..."
python sample.py \
    --checkpoint $CHECKPOINT \
    --mode seq2phase \
    --input_file sequences.txt \
    --output /dev/null \
    --method euler \
    --visualize \
    --batch_size 1

echo ""
echo "Visualizations saved to visualizations/"
echo "========================================"
