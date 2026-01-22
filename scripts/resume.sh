#!/bin/bash
# PhaseFlow Resume Training Script
# 从断点恢复训练
# 用法: bash scripts/resume.sh [gpu_id]

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

PROJECT_DIR="/data4/huangyanjie/LLPS/predictor/PhaseFlow"
DATA_DIR="/data4/huangyanjie/LLPS/phase_diagram"
CHECKPOINT="${2:-outputs/run_xxx/best_model.pt}"

# 添加PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

echo "========================================"
echo "PhaseFlow Resume Training"
echo "========================================"
echo "GPU ID: $GPU_ID"
echo "Checkpoint: $CHECKPOINT"
echo "========================================"

cd $PROJECT_DIR
python train/train.py \
    --config $PROJECT_DIR/config/default.yaml \
    --data_path $DATA_DIR/phase_diagram.csv \
    --output_dir $PROJECT_DIR/outputs \
    --resume $CHECKPOINT \
    --device cuda \
    --seed 42 \
    --batch_size 64 \
    --lr 1e-4 \
    --epochs 100

echo "========================================"
echo "Resume training completed!"
echo "========================================"
