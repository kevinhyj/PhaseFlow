#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="protenix/cache"
CHECKPOINT_SOURCE="artifacts/models/protein/protenix/protenix_base.pt"
CCD_SOURCE="artifacts/data/protein/protenix/components.cif"
CCD_RDKIT_SOURCE="artifacts/data/protein/protenix/components.cif.rdkit_mol.pkl"

usage() {
  echo "Usage: $0 [--cache-root PATH] [--checkpoint-source PATH] [--ccd-source PATH] [--ccd-rdkit-source PATH]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --checkpoint-source) CHECKPOINT_SOURCE="$2"; shift 2 ;;
    --ccd-source) CCD_SOURCE="$2"; shift 2 ;;
    --ccd-rdkit-source) CCD_RDKIT_SOURCE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$CACHE_ROOT/checkpoint" "$CACHE_ROOT/common"

if [[ -e "$CHECKPOINT_SOURCE" ]]; then
  ln -sfn "$CHECKPOINT_SOURCE" "$CACHE_ROOT/checkpoint/$(basename "$CHECKPOINT_SOURCE")"
else
  echo "[warn] checkpoint source not found: $CHECKPOINT_SOURCE" >&2
fi

if [[ -e "$CCD_SOURCE" ]]; then
  ln -sfn "$CCD_SOURCE" "$CACHE_ROOT/common/components.cif"
else
  echo "[warn] CCD source not found: $CCD_SOURCE" >&2
fi

if [[ -e "$CCD_RDKIT_SOURCE" ]]; then
  ln -sfn "$CCD_RDKIT_SOURCE" "$CACHE_ROOT/common/components.cif.rdkit_mol.pkl"
else
  echo "[warn] CCD RDKit source not found: $CCD_RDKIT_SOURCE" >&2
fi

for filename in clusters-by-entity-40.txt obsolete_release_date.csv obsolete_to_successor.json release_date_cache.json; do
  if [[ ! -e "$CACHE_ROOT/common/$filename" ]]; then
    echo "[info] Protenix will download missing common cache on first inference: $CACHE_ROOT/common/$filename"
  fi
done

echo "PROTENIX_ROOT_DIR=$CACHE_ROOT"
