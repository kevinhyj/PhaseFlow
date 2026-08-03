#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GPU_ID="${1:-0}"
CHECKPOINT="${2:-outputs/run_xxx/best_model.pt}"
CONFIG_PATH="${PHASEFLOW_CONFIG:-${PROJECT_DIR}/configs/peptide/peptide.yaml}"
DATA_PATH="${PHASEFLOW_DATA_PATH:-${PROJECT_DIR}/artifacts/data/peptide/phase_diagram_original_scale.csv}"
OUTPUT_DIR="${PHASEFLOW_OUTPUT_DIR:-${PROJECT_DIR}/outputs}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "========================================"
echo "PhaseFlow Resume Training"
echo "========================================"
echo "GPU ID:     $GPU_ID"
echo "Checkpoint: $CHECKPOINT"
echo "Config:     $CONFIG_PATH"
echo "Data:       $DATA_PATH"
echo "Output:     $OUTPUT_DIR"
echo "========================================"

cd "$PROJECT_DIR"
python scripts/peptide/workflows/train.py \
    --config "$CONFIG_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --resume "$CHECKPOINT" \
    --device cuda \
    --seed 42
