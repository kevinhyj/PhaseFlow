"""Protein feature-generation command entry points."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from phaseflow.protein.features import (
    DEFAULT_GRAPH_EDGE_DIM,
    DEFAULT_PROTENIX_EMBEDDING_DIM,
    DEFAULT_STARLING_DISTANCE_TOPK,
    ESM2Config,
    build_feature_cache,
    build_feature_cache_from_manifest,
    build_sharded_feature_cache,
    download_esm2_model,
    records_from_fasta,
    records_from_manifest,
    run_esm2_embeddings,
    run_starling_features,
)


def run_starling_segment(
    sequence: str,
    segment_name: str,
    segment_dir: Path,
    *,
    starling_binary: str,
    env: dict[str, str],
    conformations: int,
    steps: int,
    batch_size: int,
    device: str | None,
) -> Path:
    """Run one external STARLING segment and return its ensemble file."""
    candidates = sorted(segment_dir.glob(f"{segment_name}.starling*")) or sorted(segment_dir.glob("*.starling*"))
    if candidates:
        return candidates[0]
    command = [
        starling_binary, sequence, "-o", str(segment_dir), "--outname", segment_name,
        "--conformations", str(int(conformations)), "--steps", str(int(steps)),
        "--batch_size", str(int(batch_size)), "--disable_progress_bar",
    ]
    if device:
        command.extend(["--device", str(device)])
    result = subprocess.run(command, check=False, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"STARLING failed for {segment_name}: {result.stderr.strip()}")
    candidates = sorted(segment_dir.glob(f"{segment_name}.starling*")) or sorted(segment_dir.glob("*.starling*"))
    if not candidates:
        raise FileNotFoundError(f"STARLING did not produce a .starling file for {segment_name} under {segment_dir}")
    return candidates[0]


def run_esm2_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract frozen residue-level ESM-2 embeddings.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="CSV with protein_id and sequence columns.")
    source.add_argument("--fasta", help="FASTA file used when a manifest is not available.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--model-dir")
    parser.add_argument("--download", action="store_true", help="Download the Hugging Face snapshot before extraction.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--storage-dtype", default="float32", choices=["float16", "fp16", "float32", "fp32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length-policy", default="chunk", choices=["chunk", "error"])
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    model_dir = args.model_dir
    if args.download:
        model_dir = str(download_esm2_model(args.model_name, args.model_dir))
    records = records_from_manifest(args.manifest) if args.manifest else records_from_fasta(args.fasta)
    config = ESM2Config(
        model_name=args.model_name,
        model_dir=model_dir,
        device=args.device,
        dtype=args.dtype,
        storage_dtype=args.storage_dtype,
        local_files_only=args.local_files_only,
        max_length_policy=args.max_length_policy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    written = run_esm2_embeddings(records, args.out_dir, config, overwrite=args.overwrite)
    print(f"Wrote {len(written)} ESM-2 embedding files to {args.out_dir}")


def run_starling_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate PhaseFlow STARLING intermediate feature npz files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--fasta")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["heuristic", "external", "python_distance_api"], default="heuristic")
    parser.add_argument("--starling-binary", default="starling")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--conformations", type=int, default=400)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--device")
    parser.add_argument("--max-segment-length", type=int, default=384)
    parser.add_argument("--min-segment-length", type=int, default=16)
    parser.add_argument("--segment-score-threshold", type=float, default=0.5)
    parser.add_argument("--contact-threshold", type=float, default=11.0)
    parser.add_argument("--contact-topk", type=int, default=16)
    parser.add_argument("--min-contact-probability", type=float, default=0.05)
    parser.add_argument("--cleanup-raw", action="store_true")
    parser.add_argument("--ionic-strength", type=float, default=150.0)
    parser.add_argument("--api-sequence-batch-size", type=int, default=8)
    parser.add_argument("--skip-completeness-check", action="store_true")
    parser.add_argument("--limit-records", type=int, help="Process only the first N records; intended for smoke tests.")
    args = parser.parse_args(argv)
    records = records_from_manifest(args.manifest) if args.manifest else records_from_fasta(args.fasta)
    if args.limit_records is not None:
        records = records[: max(int(args.limit_records), 0)]
    written = run_starling_features(
        records,
        args.out_dir,
        mode=args.mode,
        starling_binary=args.starling_binary,
        overwrite=args.overwrite,
        conformations=args.conformations,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        max_segment_length=args.max_segment_length,
        min_segment_length=args.min_segment_length,
        segment_score_threshold=args.segment_score_threshold,
        contact_threshold=args.contact_threshold,
        contact_topk=args.contact_topk,
        min_contact_probability=args.min_contact_probability,
        cleanup_raw=args.cleanup_raw,
        ionic_strength=args.ionic_strength,
        api_sequence_batch_size=args.api_sequence_batch_size,
        require_complete=not args.skip_completeness_check,
        external_segment_runner=run_starling_segment,
    )
    print(f"Wrote {len(written)} STARLING feature files to {args.out_dir}")


def build_features_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fasta")
    source.add_argument("--manifest")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--protein-labels")
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
    parser.add_argument("--protenix-embedding-dim", type=int, default=DEFAULT_PROTENIX_EMBEDDING_DIM)
    parser.add_argument("--af3-dir")
    parser.add_argument("--starling-dir", help="Deprecated alias for --starling-embedding-dir.")
    parser.add_argument("--starling-embedding-dir")
    parser.add_argument("--starling-distance-dir")
    parser.add_argument("--local-window", type=int, default=16)
    parser.add_argument("--graph-max-neighbors", type=int, default=96)
    parser.add_argument("--graph-edge-dim", type=int, default=DEFAULT_GRAPH_EDGE_DIM)
    parser.add_argument("--starling-distance-topk", type=int, default=DEFAULT_STARLING_DISTANCE_TOPK)
    parser.add_argument("--require-structure", action="store_true")
    parser.add_argument("--require-starling", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args(argv)
    esm2_config = ESM2Config(
        model_name=args.esm2_model_name,
        model_dir=args.esm2_model_dir,
        device=args.esm2_device,
        dtype=args.esm2_dtype,
        storage_dtype=args.esm2_storage_dtype,
        local_files_only=args.esm2_local_files_only,
        chunk_size=args.esm2_chunk_size,
        overlap=args.esm2_overlap,
    )
    if args.manifest:
        paths = build_feature_cache_from_manifest(
            manifest=args.manifest,
            out_dir=args.out_dir,
            regions=args.regions,
            mil_bags=args.mil_bags,
            candidate_priors=args.candidate_priors,
            teacher_scores=args.teacher_scores,
            mode=args.mode,
            esm2_dir=args.esm2_dir,
            esm2_config=esm2_config,
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
    else:
        paths = build_feature_cache(
            fasta=args.fasta,
            out_dir=args.out_dir,
            protein_labels=args.protein_labels,
            regions=args.regions,
            mil_bags=args.mil_bags,
            candidate_priors=args.candidate_priors,
            teacher_scores=args.teacher_scores,
            mode=args.mode,
            esm2_dir=args.esm2_dir,
            esm2_config=esm2_config,
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
    print(f"Wrote {len(paths)} feature caches to {args.out_dir}")


def build_features_sharded_main(argv: list[str] | None = None) -> None:
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
    args = parser.parse_args(argv)
    paths = build_sharded_feature_cache(args)
    print(f"Wrote {len(paths)} feature caches to {args.out_dir}")
