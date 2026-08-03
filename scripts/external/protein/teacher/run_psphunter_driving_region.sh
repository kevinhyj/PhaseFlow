#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PSPHUNTER_REPO="${PSPHUNTER_REPO:-external_tools/PSPHunter}"
PSPHUNTER_ENV="${PSPHUNTER_REPO}/.pixi/envs/default"
WORK_DIR="${ROOT_DIR}/external/teachers/psphunter_work"

INPUT_FASTA="$(readlink -f "${1:?input FASTA required}")"
OUTPUT_FILE="$(readlink -m "${2:?output file required}")"

mkdir -p "$WORK_DIR" "$(dirname "$OUTPUT_FILE")"
export PERL5LIB="${ROOT_DIR}/external/teachers/perl5lib:${PERL5LIB:-}"
export PATH="${PSPHUNTER_ENV}/bin:${PATH}"

cd "$ROOT_DIR"
exec "${PSPHUNTER_ENV}/bin/python" \
  scripts/external/protein/teacher/run_psphunter_driving_region_batch.py \
  "$INPUT_FASTA" \
  "$OUTPUT_FILE" \
  --psphunter-repo "$PSPHUNTER_REPO" \
  --model-jobs "${PSPHUNTER_MODEL_JOBS:-8}" \
  --batch-windows "${PSPHUNTER_BATCH_WINDOWS:-200000}"
