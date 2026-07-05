#!/usr/bin/env bash
set -euo pipefail

JSON_DIR=""
OUT_DIR=""
MODEL_NAME="protenix_base_default_v1.0.0"
SEEDS="101"
CYCLE=4
STEP=20
SAMPLE=1
DTYPE="bf16"
USE_MSA="false"
USE_TEMPLATE="false"
USE_DEFAULT_PARAMS="false"
NEED_ATOM_CONFIDENCE="true"
TRIMUL_KERNEL="torch"
TRIATT_KERNEL="torch"
PROTENIX_BIN="${PROTENIX_BIN:-protenix}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json-dir) JSON_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --cycle) CYCLE="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --sample) SAMPLE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --use-msa) USE_MSA="$2"; shift 2 ;;
    --use-template) USE_TEMPLATE="$2"; shift 2 ;;
    --use-default-params) USE_DEFAULT_PARAMS="$2"; shift 2 ;;
    --need-atom-confidence) NEED_ATOM_CONFIDENCE="$2"; shift 2 ;;
    --trimul-kernel) TRIMUL_KERNEL="$2"; shift 2 ;;
    --triatt-kernel) TRIATT_KERNEL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$JSON_DIR" || -z "$OUT_DIR" ]]; then
  echo "Usage: $0 --json-dir protenix/input_json --out-dir protenix/output [--model-name protenix_base_default_v1.0.0] [--use-msa false] [--need-atom-confidence true]" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

for json_path in "$JSON_DIR"/*.json; do
  [[ -e "$json_path" ]] || continue
  job_name="$(basename "$json_path" .json)"
  if find "$OUT_DIR" -path "*/${job_name}_sample_*.cif" -print -quit | grep -q .; then
    echo "[skip] $job_name"
    continue
  fi
  echo "[protenix] $job_name"
  "$PROTENIX_BIN" pred \
    --input "$json_path" \
    --out_dir "$OUT_DIR" \
    --seeds "$SEEDS" \
    --cycle "$CYCLE" \
    --step "$STEP" \
    --sample "$SAMPLE" \
    --dtype "$DTYPE" \
    --model_name "$MODEL_NAME" \
    --use_msa "$USE_MSA" \
    --use_template "$USE_TEMPLATE" \
    --use_default_params "$USE_DEFAULT_PARAMS" \
    --need_atom_confidence "$NEED_ATOM_CONFIDENCE" \
    --trimul_kernel "$TRIMUL_KERNEL" \
    --triatt_kernel "$TRIATT_KERNEL"
done
