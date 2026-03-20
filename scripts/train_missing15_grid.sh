#!/bin/bash
# PhaseFlow Missing15 全权重网格训练
# 15个组合，4个GPU并行，自动分批

PROJECT_DIR="/data/yanjie_huang/LLPS/predictor/PhaseFlow"
cd $PROJECT_DIR

# 定义所有配置（使用实际存在的文件名）
declare -a CONFIGS=(
    "set_flow0_missing15.yaml"          # flow=0, lm=1
    "set_flow0_lm5_missing15.yaml"      # flow=0, lm=5
    "set_flow0_lm32_missing15.yaml"     # flow=0, lm=32
    "set_lm0_missing15.yaml"            # flow=1, lm=0
    "set_flow1_missing15.yaml"          # flow=1, lm=1
    "set_flow1_lm5_missing15.yaml"      # flow=1, lm=5
    "set_flow1_lm32_missing15.yaml"     # flow=1, lm=32
    "set_flow5_lm0_missing15.yaml"      # flow=5, lm=0
    "set_flow5_missing15.yaml"          # flow=5, lm=1
    "set_flow5_lm5_missing15.yaml"      # flow=5, lm=5
    "set_flow5_lm32_missing15.yaml"     # flow=5, lm=32
    "set_flow32_lm0_missing15.yaml"     # flow=32, lm=0
    "set_flow32_missing15.yaml"         # flow=32, lm=1
    "set_flow32_lm5_missing15.yaml"     # flow=32, lm=5
    "set_flow32_lm32_missing15.yaml"    # flow=32, lm=32
)

GPUS=(0 1 2 3)
BATCH_SIZE=4
TOTAL=${#CONFIGS[@]}

echo "=========================================="
echo "PhaseFlow Missing15 Grid Training"
echo "=========================================="
echo "Total configs: $TOTAL"
echo "GPUs: ${GPUS[@]}"
echo "Batch size: $BATCH_SIZE"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# 分批训练
for ((i=0; i<TOTAL; i+=BATCH_SIZE)); do
    batch_num=$((i/BATCH_SIZE + 1))
    echo "=========================================="
    echo "Batch $batch_num - $(date)"
    echo "=========================================="

    # 启动当前批次
    for ((j=0; j<BATCH_SIZE && i+j<TOTAL; j++)); do
        idx=$((i+j))
        config=${CONFIGS[$idx]}
        gpu=${GPUS[$j]}

        echo "[$((idx+1))/$TOTAL] GPU $gpu: $config"

        # 直接调用train.sh，只传文件名
        bash scripts/train.sh -g $gpu -c $config -t 15 &
        sleep 3
    done

    echo ""
    echo "Batch $batch_num launched, waiting for completion..."
    echo ""

    # 等待当前批次完成
    while true; do
        running=$(ps aux | grep "train/train.py" | grep -v grep | wc -l)
        if [ $running -eq 0 ]; then
            break
        fi
        echo "  [$(date +%H:%M:%S)] Running: $running jobs"
        sleep 60
    done

    echo ""
    echo "Batch $batch_num completed at $(date)!"
    echo ""
    sleep 10
done

echo "=========================================="
echo "All training completed!"
echo "End time: $(date)"
echo "=========================================="
