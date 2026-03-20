#!/bin/bash
# PhaseFlow Training Script
# 用法: bash scripts/train.sh [options]
#
# 选项:
#   -g, --gpu       GPU ID (default: 0)
#   -c, --config    配置文件名，位于 config/ 目录下 (default: default.yaml)
#   -b, --batch     覆盖 batch_size
#   -l, --lr        覆盖 learning rate
#   -e, --epochs    覆盖训练轮数
#   -h, --help      显示帮助信息
#
# 示例:
#   bash scripts/train.sh                           # 使用默认配置
#   bash scripts/train.sh -c large_batch.yaml       # 使用 large_batch 配置
#   bash scripts/train.sh -g 1 -c large_batch.yaml  # GPU 1 + large_batch 配置
#   bash scripts/train.sh -c default.yaml -b 128 -l 2e-4  # 覆盖参数

# ========================================
# 配置区域 - 可直接修改这里的默认值
# ========================================
DEFAULT_GPU=4
DEFAULT_CONFIG="/data/yanjie_huang/LLPS/predictor/PhaseFlow/config/bs2048_lr0.0008_flow32_20260123.yaml"
DEFAULT_BATCH_SIZE=""      # 留空则使用 yaml 中的值
DEFAULT_LR=""              # 留空则使用 yaml 中的值
DEFAULT_EPOCHS=""          # 留空则使用 yaml 中的值
DEFAULT_MISSING_THRESHOLD="-1"   # 按缺失值划分阈值，-1表示使用所有数据（N表示只使用missing<=N的数据）

# ========================================
# 解析命令行参数
# ========================================
GPU_ID=$DEFAULT_GPU
CONFIG_FILE=$DEFAULT_CONFIG
BATCH_SIZE=$DEFAULT_BATCH_SIZE
LR=$DEFAULT_LR
EPOCHS=$DEFAULT_EPOCHS
MISSING_THRESHOLD=$DEFAULT_MISSING_THRESHOLD

show_help() {
    echo "PhaseFlow Training Script"
    echo ""
    echo "用法: bash scripts/train.sh [options]"
    echo ""
    echo "选项:"
    echo "  -g, --gpu           GPU ID (default: $DEFAULT_GPU)"
    echo "  -c, --config        配置文件名 (default: $DEFAULT_CONFIG)"
    echo "  -b, --batch         覆盖 batch_size"
    echo "  -l, --lr            覆盖 learning rate"
    echo "  -e, --epochs        覆盖训练轮数"
    echo "  -t, --threshold     缺失值阈值 (default: $DEFAULT_MISSING_THRESHOLD)"
    echo "                       只使用 missing <= threshold 的数据"
    echo "                       全部作为训练集（不再划分）"
    echo "                       -1 表示使用所有数据"
    echo "                       验证集/测试集: val_set.csv / test_set.csv (500条)"
    echo "  -h, --help          显示帮助信息"
    echo ""
    echo "可用配置文件:"
    ls -1 ${PROJECT_DIR}/config/*.yaml 2>/dev/null | xargs -n1 basename
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--lr)
            LR="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -t|--threshold)
            MISSING_THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            # 兼容旧用法: bash train.sh 0
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                GPU_ID="$1"
            fi
            shift
            ;;
    esac
done

# ========================================
# 路径设置
# ========================================
PROJECT_DIR="/data/yanjie_huang/LLPS/predictor/PhaseFlow"
DATA_DIR="/data/yanjie_huang/LLPS/phase_diagram"
LOG_DIR="${PROJECT_DIR}/logs"

# Auto-detect set encoder from config → switch output dir
# Default to "outputs", switch to "outputs_set" if config has use_set_encoder: true

# 支持绝对路径和相对路径
if [[ "$CONFIG_FILE" == /* ]]; then
    # 绝对路径，直接使用
    CONFIG_PATH="$CONFIG_FILE"
else
    # 相对路径，拼接 config 目录
    CONFIG_PATH="${PROJECT_DIR}/config/${CONFIG_FILE}"
fi

# 检查配置文件是否存在
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: 配置文件不存在: $CONFIG_PATH"
    echo ""
    echo "可用配置文件:"
    ls -1 ${PROJECT_DIR}/config/*.yaml 2>/dev/null | xargs -n1 basename
    exit 1
fi

# Detect output dir from config: ddpm > set_encoder > default
if grep -q "diffusion_type.*ddpm" "$CONFIG_PATH" 2>/dev/null; then
    OUTPUT_DIR="${PROJECT_DIR}/outputs_ddpm"
elif grep -q "use_set_encoder.*true" "$CONFIG_PATH" 2>/dev/null; then
    OUTPUT_DIR="${PROJECT_DIR}/outputs_set"
else
    OUTPUT_DIR="${PROJECT_DIR}/outputs"
fi

# ========================================
# 环境设置
# ========================================
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0

# 创建输出和日志目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# 生成日志文件名（带时间戳和配置名）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
LOG_FILE="${LOG_DIR}/train_${CONFIG_NAME}_${TIMESTAMP}.log"

# ========================================
# 构建训练命令参数
# ========================================
TRAIN_ARGS="--config $CONFIG_PATH \
    --data_path $DATA_DIR/phase_diagram_original_scale.csv \
    --output_dir $OUTPUT_DIR \
    --device cuda \
    --seed 42"

# 添加可选的覆盖参数
if [[ -n "$BATCH_SIZE" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --batch_size $BATCH_SIZE"
fi
if [[ -n "$LR" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --lr $LR"
fi
if [[ -n "$EPOCHS" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --epochs $EPOCHS"
fi
if [[ -n "$MISSING_THRESHOLD" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --missing_threshold $MISSING_THRESHOLD"
fi

# ========================================
# 打印配置信息
# ========================================
echo "========================================"
echo "PhaseFlow Training"
echo "========================================"
echo "GPU ID:      $GPU_ID"
echo "Config:      $CONFIG_FILE"
echo "Data:        $DATA_DIR/phase_diagram.csv"
echo "Output:      $OUTPUT_DIR"
echo "Log:         $LOG_FILE"
if [[ -n "$BATCH_SIZE" ]]; then
    echo "Batch size:  $BATCH_SIZE (覆盖)"
fi
if [[ -n "$LR" ]]; then
    echo "LR:          $LR (覆盖)"
fi
if [[ -n "$EPOCHS" ]]; then
    echo "Epochs:      $EPOCHS (覆盖)"
fi
if [[ "$MISSING_THRESHOLD" != "-1" ]]; then
    echo "Missing Threshold: $MISSING_THRESHOLD (使用 missing <= $MISSING_THRESHOLD 的数据，全部作为训练集)"
fi
echo "========================================"

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phaseflow

# 检查 GPU 可用性（输出到日志）
echo "=== Training Config ===" >> $LOG_FILE
echo "Config file: $CONFIG_PATH" >> $LOG_FILE
echo "GPU ID: $GPU_ID" >> $LOG_FILE
echo "Arguments: $TRAIN_ARGS" >> $LOG_FILE
echo "" >> $LOG_FILE
echo "=== GPU Info ===" >> $LOG_FILE
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')" >> $LOG_FILE 2>&1
echo "" >> $LOG_FILE

# 使用 nohup 后台运行训练，输出到日志文件
cd $PROJECT_DIR
nohup python -u train/train.py $TRAIN_ARGS >> $LOG_FILE 2>&1 &

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
