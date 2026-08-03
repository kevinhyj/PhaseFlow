#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DEFAULT_GPU=0
DEFAULT_CONFIG="configs/peptide/peptide.yaml"
DEFAULT_DATA_PATH="${PHASEFLOW_DATA_PATH:-${PROJECT_DIR}/artifacts/data/peptide/phase_diagram_original_scale.csv}"
DEFAULT_OUTPUT_DIR="${PROJECT_DIR}/outputs"
DEFAULT_MISSING_THRESHOLD=""

GPU_ID="$DEFAULT_GPU"
CONFIG_FILE="$DEFAULT_CONFIG"
DATA_PATH="$DEFAULT_DATA_PATH"
VAL_PATH="${PHASEFLOW_VAL_PATH:-}"
TEST_PATH="${PHASEFLOW_TEST_PATH:-}"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
BATCH_SIZE=""
LR=""
EPOCHS=""
MISSING_THRESHOLD="$DEFAULT_MISSING_THRESHOLD"
FOREGROUND=false

show_help() {
    echo "PhaseFlow training launcher"
    echo ""
    echo "Usage: bash scripts/peptide/train.sh [options]"
    echo ""
    echo "Options:"
    echo "  -g, --gpu ID              GPU ID (default: ${DEFAULT_GPU})"
    echo "  -c, --config PATH         Config path or name under configs/peptide/ (default: ${DEFAULT_CONFIG})"
    echo "  -d, --data PATH           Training CSV (default: PHASEFLOW_DATA_PATH or artifacts/data/peptide/phase_diagram_original_scale.csv)"
    echo "      --val PATH            Optional validation CSV"
    echo "      --test PATH           Optional test CSV"
    echo "  -o, --output-dir PATH     Output directory (default: outputs/)"
    echo "  -b, --batch N             Override batch size"
    echo "  -l, --lr FLOAT            Override learning rate"
    echo "  -e, --epochs N            Override epochs"
    echo "  -t, --threshold N         Override data.missing_threshold from the YAML config"
    echo "      --foreground          Run in foreground instead of nohup background"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Available configs:"
    find "${PROJECT_DIR}/configs/peptide" -maxdepth 1 -name "*.yaml" -printf "  %f\n" 2>/dev/null | sort
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--data|--data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --val|--val-path)
            VAL_PATH="$2"
            shift 2
            ;;
        --test|--test-path)
            TEST_PATH="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch|--batch-size)
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
        --foreground)
            FOREGROUND=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                GPU_ID="$1"
            fi
            shift
            ;;
    esac
done

if [[ "$CONFIG_FILE" = /* ]]; then
    CONFIG_PATH="$CONFIG_FILE"
elif [[ -f "${PROJECT_DIR}/${CONFIG_FILE}" ]]; then
    CONFIG_PATH="${PROJECT_DIR}/${CONFIG_FILE}"
else
    CONFIG_PATH="${PROJECT_DIR}/configs/peptide/${CONFIG_FILE}"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: config not found: ${CONFIG_PATH}" >&2
    echo "" >&2
    show_help
    exit 1
fi

if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ -n "${PHASEFLOW_CONDA_ENV:-}" ]]; then
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "$PHASEFLOW_CONDA_ENV"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

CONFIG_NAME="$(basename "$CONFIG_PATH" .yaml)"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/train_${CONFIG_NAME}_${TIMESTAMP}.log"

TRAIN_ARGS=(
    --config "$CONFIG_PATH"
    --data_path "$DATA_PATH"
    --output_dir "$OUTPUT_DIR"
    --device cuda
    --seed 42
)

[[ -n "$MISSING_THRESHOLD" ]] && TRAIN_ARGS+=(--missing_threshold "$MISSING_THRESHOLD")
[[ -n "$VAL_PATH" ]] && TRAIN_ARGS+=(--val_path "$VAL_PATH")
[[ -n "$TEST_PATH" ]] && TRAIN_ARGS+=(--test_path "$TEST_PATH")
[[ -n "$BATCH_SIZE" ]] && TRAIN_ARGS+=(--batch_size "$BATCH_SIZE")
[[ -n "$LR" ]] && TRAIN_ARGS+=(--lr "$LR")
[[ -n "$EPOCHS" ]] && TRAIN_ARGS+=(--epochs "$EPOCHS")

echo "========================================"
echo "PhaseFlow Training"
echo "========================================"
echo "Project:   $PROJECT_DIR"
echo "GPU ID:    $GPU_ID"
echo "Config:    $CONFIG_PATH"
echo "Data:      $DATA_PATH"
[[ -n "$VAL_PATH" ]] && echo "Val:       $VAL_PATH"
[[ -n "$TEST_PATH" ]] && echo "Test:      $TEST_PATH"
echo "Output:    $OUTPUT_DIR"
echo "Log:       $LOG_FILE"
echo "Threshold: ${MISSING_THRESHOLD:-from YAML config}"
echo "========================================"

cd "$PROJECT_DIR"
if "$FOREGROUND"; then
    python -u scripts/peptide/workflows/train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
else
    nohup python -u scripts/peptide/workflows/train.py "${TRAIN_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
    PID=$!
    echo "Training started in background"
    echo "PID: $PID"
    echo "Log file: $LOG_FILE"
    echo "Monitor with: tail -f $LOG_FILE"
fi
