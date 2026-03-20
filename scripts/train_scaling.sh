#!/bin/bash
# PhaseFlow Model Size Scaling - 8 GPU 并行训练
# 8 个 scaling 配置分配到 GPU 0-7

PROJECT_DIR="/data/yanjie_huang/LLPS/predictor/PhaseFlow"
cd $PROJECT_DIR

declare -a CONFIGS=(
    "scale_a_dim320_d8.yaml"
    "scale_b_dim384_d8.yaml"
    "scale_c_dim512_d8.yaml"
    "scale_d_dim640_d10.yaml"
    "scale_e_dim768_d10.yaml"
    "scale_f_dim1024_d10.yaml"
    "scale_d_shallow_dim640_d6.yaml"
    "scale_e_shallow_dim768_d6.yaml"
)

GPUS=(0 1 2 3 4 5 6 7)
TOTAL=${#CONFIGS[@]}

echo "=========================================="
echo "PhaseFlow Model Size Scaling"
echo "=========================================="
echo "Total configs: $TOTAL"
echo "GPUs: ${GPUS[@]}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

for ((i=0; i<TOTAL; i++)); do
    config=${CONFIGS[$i]}
    gpu=${GPUS[$i]}
    echo "[$((i+1))/$TOTAL] GPU $gpu: $config"
    bash scripts/train.sh -g $gpu -c $config -t 15 &
    sleep 3
done

echo ""
echo "All $TOTAL jobs launched, monitoring..."
echo ""

while true; do
    running=$(ps aux | grep "train/train.py" | grep -v grep | wc -l)
    if [ $running -eq 0 ]; then
        break
    fi
    echo "  [$(date +%H:%M:%S)] Running: $running jobs"
    sleep 120
done

echo ""
echo "=========================================="
echo "All scaling training completed!"
echo "End time: $(date)"
echo "=========================================="
