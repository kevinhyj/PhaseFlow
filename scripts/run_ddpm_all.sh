#!/bin/bash
# 批量训练全部 15 个 DDPM cosine m15 配置
# 4 个一批，GPU 2,4,5,6，等一批完成再跑下一批
# 用法: nohup bash scripts/run_ddpm_all.sh > logs/run_ddpm_all.log 2>&1 &

PROJECT_DIR="/data/yanjie_huang/LLPS/predictor/PhaseFlow"
cd "$PROJECT_DIR"

source /home/huangyanjie/miniconda3/etc/profile.d/conda.sh
conda activate phaseflow

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

DATA_PATH="/data/yanjie_huang/LLPS/phase_diagram/phase_diagram_original_scale.csv"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
    set_ddpm_cosine_missing15.yaml
    set_ddpm_flow32_missing15.yaml
    set_ddpm_flow32_lm5_missing15.yaml
    set_ddpm_flow32_lm32_missing15.yaml
    set_ddpm_flow5_lm0_missing15.yaml
    set_ddpm_flow5_missing15.yaml
    set_ddpm_flow5_lm5_missing15.yaml
    set_ddpm_flow5_lm32_missing15.yaml
    set_ddpm_flow1_lm0_missing15.yaml
    set_ddpm_flow1_missing15.yaml
    set_ddpm_flow1_lm5_missing15.yaml
    set_ddpm_flow1_lm32_missing15.yaml
    set_ddpm_flow0_lm1_missing15.yaml
    set_ddpm_flow0_lm5_missing15.yaml
    set_ddpm_flow0_lm32_missing15.yaml
)

GPUS=(2 4 5 6)
BATCH_SIZE=4
TOTAL=${#CONFIGS[@]}

echo "========================================"
echo "DDPM Cosine M15 batch training"
echo "Total: ${TOTAL} configs, ${BATCH_SIZE} per batch"
echo "GPUs: ${GPUS[*]}"
echo "Start: $(date)"
echo "========================================"

batch_idx=0
i=0
while [ $i -lt $TOTAL ]; do
    batch_idx=$((batch_idx + 1))
    end=$((i + BATCH_SIZE))
    if [ $end -gt $TOTAL ]; then
        end=$TOTAL
    fi
    n_this=$((end - i))

    echo ""
    echo "========================================"
    echo "Batch ${batch_idx}: config $((i+1))-${end} / ${TOTAL}"
    echo "Time: $(date)"
    echo "========================================"

    PIDS=()
    j=0
    while [ $j -lt $n_this ]; do
        idx=$((i + j))
        cfg="${CONFIGS[$idx]}"
        gpu="${GPUS[$j]}"
        cfg_name=$(basename "$cfg" .yaml)
        log_file="${LOG_DIR}/train_${cfg_name}.log"

        echo "  Launch: ${cfg_name} -> GPU ${gpu}"

        CUDA_VISIBLE_DEVICES=$gpu python -u train/train.py \
            --config "config/${cfg}" \
            --data_path "$DATA_PATH" \
            --output_dir outputs_ddpm \
            --device cuda \
            --seed 42 \
            --missing_threshold 16 \
            > "$log_file" 2>&1 &

        PIDS+=($!)
        j=$((j + 1))
    done

    echo "  Waiting for Batch ${batch_idx} (PIDs: ${PIDS[*]})..."

    for pid in "${PIDS[@]}"; do
        wait $pid || echo "  [WARN] PID $pid exited with error"
    done

    echo "  Batch ${batch_idx} done! $(date)"
    i=$end
done

echo ""
echo "========================================"
echo "All training complete!"
echo "End: $(date)"
echo "========================================"
