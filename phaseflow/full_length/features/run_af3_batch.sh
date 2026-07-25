#!/usr/bin/env bash
set -euo pipefail

JSON_DIR=""
OUT_DIR=""
MODEL_DIR=""
DB_DIR=""
DOCKER_IMAGE="alphafold3"
MODE="no_msa"
GPUS="all"

usage() {
  echo "Usage: $0 --json-dir af3/input_json --out-dir af3/output --model-dir artifacts/models/full_length/af3 [--mode no_msa|full_pipeline] [--db-dir artifacts/data/full_length/af3] [--docker-image alphafold3] [--gpus all|none|device=0]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --json-dir) JSON_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --db-dir) DB_DIR="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$JSON_DIR" || -z "$OUT_DIR" || -z "$MODEL_DIR" ]]; then
  usage >&2
  exit 2
fi
if [[ "$MODE" != "no_msa" && "$MODE" != "full_pipeline" ]]; then
  echo "Unsupported --mode: $MODE" >&2
  exit 2
fi
if [[ "$MODE" == "full_pipeline" && -z "$DB_DIR" ]]; then
  echo "--db-dir is required when --mode full_pipeline" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

for json_path in "$JSON_DIR"/*.json; do
  [[ -e "$json_path" ]] || continue
  job_name="$(basename "$json_path" .json)"
  if [[ -d "$OUT_DIR/$job_name" || -d "$OUT_DIR/${job_name,,}" ]]; then
    echo "[skip] $job_name"
    continue
  fi
  echo "[af3] $job_name"
  docker_args=(
    run --rm
    --volume "$JSON_DIR:/root/af_input:ro"
    --volume "$OUT_DIR:/root/af_output"
    --volume "$MODEL_DIR:/root/models:ro"
  )
  if [[ -n "$GPUS" && "$GPUS" != "none" ]]; then
    docker_args+=(--gpus "$GPUS")
  fi
  af3_args=(
    python run_alphafold.py
    --json_path="/root/af_input/$(basename "$json_path")"
    --model_dir=/root/models
    --output_dir=/root/af_output
    --run_inference=true
    --save_embeddings=true
  )
  if [[ "$MODE" == "no_msa" ]]; then
    af3_args+=(--run_data_pipeline=false)
  else
    docker_args+=(--volume "$DB_DIR:/root/public_databases:ro")
    af3_args+=(--db_dir=/root/public_databases --run_data_pipeline=true)
  fi
  docker "${docker_args[@]}" "$DOCKER_IMAGE" "${af3_args[@]}"
done
