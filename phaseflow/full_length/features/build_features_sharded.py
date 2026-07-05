from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from phaseflow.full_length.features.build_features import build_feature_cache_from_manifest
from phaseflow.full_length.features.plm_embedder import ESM2Config


def build_sharded_feature_cache(args: argparse.Namespace) -> list[Path]:
    frame = pd.read_csv(args.manifest)
    workers = max(1, min(int(args.workers), len(frame) if len(frame) else 1))
    if workers == 1:
        return build_feature_cache_from_manifest(
            manifest=args.manifest,
            out_dir=args.out_dir,
            regions=args.regions,
            mil_bags=args.mil_bags,
            candidate_priors=args.candidate_priors,
            teacher_scores=args.teacher_scores,
            mode=args.mode,
            esm2_dir=args.esm2_dir,
            esm2_config=_esm2_config(args),
            structure_dir=args.structure_dir,
            protenix_embedding_dir=args.protenix_embedding_dir,
            protenix_embedding_dim=args.protenix_embedding_dim,
            af3_dir=args.af3_dir,
            starling_dir=args.starling_dir,
            starling_embedding_dir=args.starling_embedding_dir,
            starling_distance_dir=args.starling_distance_dir,
            local_window=args.local_window,
            graph_max_neighbors=args.graph_max_neighbors,
            graph_edge_dim=args.graph_edge_dim,
            starling_distance_topk=args.starling_distance_topk,
            require_structure=args.require_structure,
            require_starling=args.require_starling,
            overwrite=not args.no_overwrite,
        )

    shard_dir = Path(args.shard_dir or Path(args.out_dir) / f".feature_shards_{int(time.time())}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = _balanced_shards(frame, workers)
    shard_specs: list[dict[str, Any]] = []
    for index, row_indices in enumerate(shards):
        shard_path = shard_dir / f"manifest_shard_{index:03d}.csv"
        frame.iloc[row_indices].to_csv(shard_path, index=False)
        shard_specs.append(
            {
                "index": index,
                "manifest": str(shard_path),
                "records": len(row_indices),
                "args": vars(args),
            }
        )
    (shard_dir / "shards.json").write_text(json.dumps(shard_specs, indent=2, sort_keys=True) + "\n")

    written: list[Path] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_build_one_shard, spec) for spec in shard_specs if spec["records"]]
        for future in as_completed(futures):
            paths = future.result()
            written.extend(Path(path) for path in paths)
            print(f"finished_shard_paths={len(paths)} total_finished_paths={len(written)}", flush=True)
    return written


def _balanced_shards(frame: pd.DataFrame, workers: int) -> list[list[int]]:
    costs = [0 for _ in range(workers)]
    shards: list[list[int]] = [[] for _ in range(workers)]
    if "length" in frame.columns:
        items = [(index, int(row.get("length", 1) or 1)) for index, row in frame.iterrows()]
    else:
        items = [(index, len(str(row.get("sequence", "")))) for index, row in frame.iterrows()]
    for index, length in sorted(items, key=lambda item: item[1], reverse=True):
        target = min(range(workers), key=lambda worker: costs[worker])
        shards[target].append(index)
        costs[target] += max(length, 1)
    return shards


def _build_one_shard(spec: dict[str, Any]) -> list[str]:
    raw_args = spec["args"]
    paths = build_feature_cache_from_manifest(
        manifest=spec["manifest"],
        out_dir=raw_args["out_dir"],
        regions=raw_args.get("regions"),
        mil_bags=raw_args.get("mil_bags"),
        candidate_priors=raw_args.get("candidate_priors"),
        teacher_scores=raw_args.get("teacher_scores"),
        mode=raw_args.get("mode", "simple"),
        esm2_dir=raw_args.get("esm2_dir"),
        esm2_config=_esm2_config_from_mapping(raw_args),
        structure_dir=raw_args.get("structure_dir"),
        protenix_embedding_dir=raw_args.get("protenix_embedding_dir"),
        protenix_embedding_dim=int(raw_args.get("protenix_embedding_dim", 512)),
        af3_dir=raw_args.get("af3_dir"),
        starling_dir=raw_args.get("starling_dir"),
        starling_embedding_dir=raw_args.get("starling_embedding_dir"),
        starling_distance_dir=raw_args.get("starling_distance_dir"),
        local_window=int(raw_args.get("local_window", 16)),
        graph_max_neighbors=int(raw_args.get("graph_max_neighbors", 96)),
        graph_edge_dim=int(raw_args.get("graph_edge_dim", 13)),
        starling_distance_topk=int(raw_args.get("starling_distance_topk", 48)),
        require_structure=bool(raw_args.get("require_structure", False)),
        require_starling=bool(raw_args.get("require_starling", False)),
        overwrite=not bool(raw_args.get("no_overwrite", False)),
    )
    return [str(path) for path in paths]


def _esm2_config(args: argparse.Namespace) -> ESM2Config:
    return _esm2_config_from_mapping(vars(args))


def _esm2_config_from_mapping(values: dict[str, Any]) -> ESM2Config:
    return ESM2Config(
        model_name=values.get("esm2_model_name", "facebook/esm2_t33_650M_UR50D"),
        model_dir=values.get("esm2_model_dir"),
        device=values.get("esm2_device", "auto"),
        dtype=values.get("esm2_dtype", "float32"),
        storage_dtype=values.get("esm2_storage_dtype", "float32"),
        local_files_only=bool(values.get("esm2_local_files_only", False)),
        chunk_size=values.get("esm2_chunk_size"),
        overlap=int(values.get("esm2_overlap", 128)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PhaseFlow HDF5 feature caches from a manifest in parallel shards.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--regions")
    parser.add_argument("--mil-bags")
    parser.add_argument("--candidate-priors")
    parser.add_argument("--teacher-scores")
    parser.add_argument("--mode", choices=["simple", "esm2"], default="simple")
    parser.add_argument("--esm2-dir")
    parser.add_argument("--esm2-model-name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--esm2-model-dir")
    parser.add_argument("--esm2-device", default="auto")
    parser.add_argument("--esm2-dtype", default="float32")
    parser.add_argument("--esm2-storage-dtype", default="float32")
    parser.add_argument("--esm2-local-files-only", action="store_true")
    parser.add_argument("--esm2-chunk-size", type=int)
    parser.add_argument("--esm2-overlap", type=int, default=128)
    parser.add_argument("--structure-dir")
    parser.add_argument("--protenix-embedding-dir")
    parser.add_argument("--protenix-embedding-dim", type=int, default=512)
    parser.add_argument("--af3-dir")
    parser.add_argument("--starling-dir", help="Deprecated alias for --starling-embedding-dir.")
    parser.add_argument("--starling-embedding-dir")
    parser.add_argument("--starling-distance-dir")
    parser.add_argument("--local-window", type=int, default=16)
    parser.add_argument("--graph-max-neighbors", type=int, default=96)
    parser.add_argument("--graph-edge-dim", type=int, default=13)
    parser.add_argument("--starling-distance-topk", type=int, default=48)
    parser.add_argument("--require-structure", action="store_true")
    parser.add_argument("--require-starling", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-dir")
    args = parser.parse_args()
    paths = build_sharded_feature_cache(args)
    print(f"Wrote {len(paths)} feature caches to {args.out_dir}")


if __name__ == "__main__":
    main()
