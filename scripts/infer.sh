#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT="${1:-outputs/run_xxx/best_model.pt}"
INPUT_FILE="${2:-examples/sequences.txt}"
OUTPUT_FILE="${3:-results/predicted_phases.csv}"
GPU_ID="${4:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ "$CHECKPOINT" != /* ]]; then
    CHECKPOINT="${PROJECT_DIR}/${CHECKPOINT}"
fi
if [[ "$INPUT_FILE" != /* ]]; then
    INPUT_FILE="${PROJECT_DIR}/${INPUT_FILE}"
fi
if [[ "$OUTPUT_FILE" != /* ]]; then
    OUTPUT_FILE="${PROJECT_DIR}/${OUTPUT_FILE}"
fi

echo "========================================"
echo "PhaseFlow Seq2Phase Inference"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Input:      $INPUT_FILE"
echo "Output:     $OUTPUT_FILE"
echo "GPU ID:     $GPU_ID"
echo "========================================"

cd "$PROJECT_DIR"
python experiments/predict_seq2phase.py \
    --checkpoint "$CHECKPOINT" \
    --input_file "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --method euler \
    --batch_size 32 \
    --device cuda
