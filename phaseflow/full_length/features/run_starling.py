from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm

from phaseflow.full_length.features.plm_embedder import clean_protein_sequence
from phaseflow.full_length.features.run_esm2 import records_from_fasta, records_from_manifest
from phaseflow.full_length.features.starling_runner import (
    assemble_starling_segments,
    candidate_starling_segments,
    heuristic_starling_features,
    load_starling_ensemble_file,
    starling_features_from_distance_maps,
)


def run_starling_features(
    records: list[tuple[str, str]],
    out_dir: str | Path,
    mode: str = "heuristic",
    starling_binary: str = "starling",
    overwrite: bool = False,
    conformations: int = 400,
    steps: int = 30,
    batch_size: int = 100,
    device: str | None = None,
    max_segment_length: int = 384,
    min_segment_length: int = 16,
    segment_score_threshold: float = 0.5,
    contact_threshold: float = 11.0,
    contact_topk: int = 16,
    min_contact_probability: float = 0.05,
    cleanup_raw: bool = False,
    ionic_strength: float = 150.0,
    require_complete: bool = True,
    api_sequence_batch_size: int = 8,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "python_distance_api":
        return _run_python_distance_api_starling(
            records,
            out_dir=out_dir,
            overwrite=overwrite,
            conformations=conformations,
            steps=steps,
            batch_size=batch_size,
            device=device,
            max_segment_length=max_segment_length,
            min_segment_length=min_segment_length,
            segment_score_threshold=segment_score_threshold,
            contact_threshold=contact_threshold,
            contact_topk=contact_topk,
            min_contact_probability=min_contact_probability,
            ionic_strength=ionic_strength,
            require_complete=require_complete,
            api_sequence_batch_size=api_sequence_batch_size,
        )
    written: list[Path] = []
    for protein_id, raw_sequence in records:
        sequence = clean_protein_sequence(raw_sequence)
        out_path = out_dir / f"{protein_id}.npz"
        if _complete_starling_npz(out_path, sequence, require_complete=require_complete, expected_node_dim=8) and not overwrite:
            written.append(out_path)
            continue
        if mode == "heuristic":
            node, missing, reliability = heuristic_starling_features(sequence)
            contacts = np.zeros((0, 3), dtype=np.float32)
        elif mode == "external":
            node, missing, reliability, contacts, metadata = _run_external_starling(
                protein_id=protein_id,
                sequence=sequence,
                out_dir=out_dir,
                starling_binary=starling_binary,
                conformations=conformations,
                steps=steps,
                batch_size=batch_size,
                device=device,
                max_segment_length=max_segment_length,
                min_segment_length=min_segment_length,
                segment_score_threshold=segment_score_threshold,
                contact_threshold=contact_threshold,
                contact_topk=contact_topk,
                min_contact_probability=min_contact_probability,
                cleanup_raw=cleanup_raw,
            )
        else:
            raise ValueError(f"Unsupported STARLING mode: {mode}")
        payload: dict[str, object] = {
            "protein_id": np.asarray(protein_id),
            "sequence": np.asarray(sequence),
            "node": node,
            "missing_mask": missing,
            "reliability": reliability,
            "contacts": contacts,
            "starling_mode": np.asarray(mode),
        }
        if mode == "external":
            payload.update({key: np.asarray(str(value)) for key, value in metadata.items()})
        np.savez_compressed(out_path, **payload)
        if mode == "external" and cleanup_raw:
            shutil.rmtree(out_dir / "_raw" / protein_id, ignore_errors=True)
        written.append(out_path)
    return written


def _run_python_distance_api_starling(
    records: list[tuple[str, str]],
    *,
    out_dir: Path,
    overwrite: bool,
    conformations: int,
    steps: int,
    batch_size: int,
    device: str | None,
    max_segment_length: int,
    min_segment_length: int,
    segment_score_threshold: float,
    contact_threshold: float,
    contact_topk: int,
    min_contact_probability: float,
    ionic_strength: float,
    require_complete: bool,
    api_sequence_batch_size: int,
) -> list[Path]:
    mpl_cache = out_dir / "_matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    try:
        from starling import configs as starling_configs  # type: ignore
        from starling.frontend.ensemble_generation import generate  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "STARLING python_distance_api mode requires `from starling.frontend.ensemble_generation import generate`"
        ) from exc

    device = device or "cuda:0"
    effective_max_segment_length = min(int(max_segment_length), int(starling_configs.MAX_SEQUENCE_LENGTH))
    pending_segments: dict[str, str] = {}
    segment_to_parent: dict[str, tuple[str, object]] = {}
    parent_segments: dict[str, list[object]] = {}
    record_sequences: dict[str, str] = {}
    segment_results_by_parent: dict[str, list[tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    written: list[Path] = []

    for protein_id, raw_sequence in records:
        sequence = clean_protein_sequence(raw_sequence)
        record_sequences[protein_id] = sequence
        out_path = out_dir / f"{protein_id}.npz"
        if _complete_starling_npz(out_path, sequence, require_complete=require_complete, expected_node_dim=8) and not overwrite:
            written.append(out_path)
            continue
        segments = candidate_starling_segments(
            protein_id,
            sequence,
            max_segment_length=effective_max_segment_length,
            min_segment_length=min_segment_length,
            score_threshold=segment_score_threshold,
        )
        parent_segments[protein_id] = segments
        segment_results_by_parent[protein_id] = []
        for segment in segments:
            if len(segment.sequence) > int(starling_configs.MAX_SEQUENCE_LENGTH):
                raise RuntimeError(
                    f"STARLING segment {segment.name} length {len(segment.sequence)} exceeds "
                    f"MAX_SEQUENCE_LENGTH={starling_configs.MAX_SEQUENCE_LENGTH}"
                )
            if segment.name in pending_segments:
                raise RuntimeError(f"Duplicate STARLING segment name in python_distance_api mode: {segment.name}")
            pending_segments[segment.name] = segment.sequence
            segment_to_parent[segment.name] = (protein_id, segment)

    segment_names = list(pending_segments)
    chunk_size = max(int(api_sequence_batch_size), 1)
    for start in tqdm(range(0, len(segment_names), chunk_size), desc="STARLING python_distance_api batches"):
        chunk_names = segment_names[start : start + chunk_size]
        sequence_dict = {name: pending_segments[name] for name in chunk_names}
        ensembles = generate(
            sequence_dict,
            conformations=int(conformations),
            ionic_strength=float(ionic_strength),
            device=device,
            steps=int(steps),
            return_structures=False,
            batch_size=int(batch_size),
            output_directory=None,
            return_data=True,
            verbose=False,
            show_progress_bar=False,
            show_per_step_progress_bar=False,
        )
        if ensembles is None:
            raise RuntimeError("STARLING generate returned None in python_distance_api mode")
        missing = sorted(set(sequence_dict).difference(ensembles))
        if missing:
            raise RuntimeError(f"STARLING generate missed {len(missing)} segments: {missing[:5]}")
        for segment_name in chunk_names:
            protein_id, segment = segment_to_parent[segment_name]
            ensemble = ensembles[segment_name]
            segment_node, segment_missing, segment_reliability, segment_contacts = starling_features_from_distance_maps(
                ensemble.distance_maps(return_mean=False),
                segment.sequence,
                contact_threshold=contact_threshold,
                contact_topk=contact_topk,
                min_contact_probability=min_contact_probability,
            )
            if segment_node.shape[1] != 8:
                raise RuntimeError(f"STARLING distance-map features for {segment.name} must be 8D, got {segment_node.shape}")
            segment_results_by_parent[protein_id].append(
                (segment, segment_node, segment_missing, segment_reliability, segment_contacts)
            )

    for protein_id, sequence in tqdm(record_sequences.items(), desc="STARLING python_distance_api features"):
        out_path = out_dir / f"{protein_id}.npz"
        if _complete_starling_npz(out_path, sequence, require_complete=require_complete, expected_node_dim=8) and not overwrite:
            if out_path not in written:
                written.append(out_path)
            continue
        node, missing, reliability, contacts = assemble_starling_segments(
            len(sequence),
            segment_results_by_parent.get(protein_id, []),
        )
        if node.shape != (len(sequence), 8):
            raise RuntimeError(
                f"STARLING python_distance_api output for {protein_id} must have shape {(len(sequence), 8)}, got {node.shape}"
            )
        payload: dict[str, object] = {
            "protein_id": np.asarray(protein_id),
            "sequence": np.asarray(sequence),
            "node": node,
            "missing_mask": missing,
            "reliability": reliability,
            "contacts": contacts,
            "starling_mode": np.asarray("python_distance_api"),
            "starling_api": np.asarray("starling.frontend.ensemble_generation.generate"),
            "starling_node_dim": np.asarray(node.shape[1], dtype=np.int64),
            "starling_segments": np.asarray(len(parent_segments.get(protein_id, [])), dtype=np.int64),
            "starling_conformations": np.asarray(int(conformations), dtype=np.int64),
            "starling_steps": np.asarray(int(steps), dtype=np.int64),
            "starling_batch_size": np.asarray(int(batch_size), dtype=np.int64),
            "starling_max_segment_length": np.asarray(int(effective_max_segment_length), dtype=np.int64),
            "starling_ionic_strength": np.asarray(float(ionic_strength), dtype=np.float32),
            "starling_contact_threshold": np.asarray(float(contact_threshold), dtype=np.float32),
            "starling_contact_topk": np.asarray(int(contact_topk), dtype=np.int64),
        }
        np.savez_compressed(out_path, **payload)
        written.append(out_path)
    return written


def _complete_starling_npz(
    path: Path,
    sequence: str,
    *,
    require_complete: bool = True,
    expected_node_dim: int | None = None,
) -> bool:
    if not path.exists():
        return False
    if not require_complete:
        return True
    try:
        with np.load(path, allow_pickle=False) as data:
            if "sequence" in data and str(data["sequence"].item()) != sequence:
                return False
            required = {"node", "missing_mask", "reliability", "contacts", "starling_mode"}
            if not required.issubset(data.files):
                return False
            node = np.asarray(data["node"])
            missing = np.asarray(data["missing_mask"])
            reliability = np.asarray(data["reliability"])
            contacts = np.asarray(data["contacts"])
            length = len(sequence)
            if node.ndim != 2 or node.shape[0] != length or node.shape[1] == 0:
                return False
            if expected_node_dim is not None and node.shape[1] != int(expected_node_dim):
                return False
            if missing.shape != (length,) or reliability.shape != (length,):
                return False
            if contacts.ndim != 2 or contacts.shape[1] < 3:
                return False
            if not np.isfinite(node).all() or not np.isfinite(missing).all() or not np.isfinite(reliability).all():
                return False
    except Exception:
        return False
    return True


def _run_external_starling(
    protein_id: str,
    sequence: str,
    out_dir: Path,
    starling_binary: str,
    *,
    conformations: int,
    steps: int,
    batch_size: int,
    device: str | None,
    max_segment_length: int,
    min_segment_length: int,
    segment_score_threshold: float,
    contact_threshold: float,
    contact_topk: int,
    min_contact_probability: float,
    cleanup_raw: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    raw_dir = out_dir / "_raw" / protein_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache = out_dir / "_matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    segments = candidate_starling_segments(
        protein_id,
        sequence,
        max_segment_length=max_segment_length,
        min_segment_length=min_segment_length,
        score_threshold=segment_score_threshold,
    )
    segment_results = []
    for segment in segments:
        segment_dir = raw_dir / segment.name
        segment_dir.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(2):
            attempt_dir = segment_dir if attempt == 0 else raw_dir / f"{segment.name}_retry_{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            ensemble_path = _run_starling_segment(
                segment.sequence,
                segment.name,
                attempt_dir,
                starling_binary=starling_binary,
                env=env,
                conformations=conformations,
                steps=steps,
                batch_size=batch_size,
                device=device,
            )
            try:
                ensemble = load_starling_ensemble_file(ensemble_path)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Could not load STARLING ensemble for {segment.name}") from last_error
        segment_node, segment_missing, segment_reliability, segment_contacts = starling_features_from_distance_maps(
            ensemble.distance_maps(return_mean=False),
            segment.sequence,
            contact_threshold=contact_threshold,
            contact_topk=contact_topk,
            min_contact_probability=min_contact_probability,
        )
        segment_results.append((segment, segment_node, segment_missing, segment_reliability, segment_contacts))
        if cleanup_raw:
            shutil.rmtree(segment_dir, ignore_errors=True)

    node, missing, reliability, contacts = assemble_starling_segments(len(sequence), segment_results)
    metadata = {
        "starling_raw_dir": str(raw_dir),
        "starling_segments": len(segments),
        "starling_conformations": conformations,
        "starling_steps": steps,
        "starling_batch_size": batch_size,
        "starling_max_segment_length": max_segment_length,
        "starling_contact_threshold": contact_threshold,
        "starling_contact_topk": contact_topk,
    }
    return node, missing, reliability, contacts, metadata


def _run_starling_segment(
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
    existing = _find_starling_file(segment_dir, segment_name)
    if existing is not None:
        return existing
    command = [
        starling_binary,
        sequence,
        "-o",
        str(segment_dir),
        "--outname",
        segment_name,
        "--conformations",
        str(int(conformations)),
        "--steps",
        str(int(steps)),
        "--batch_size",
        str(int(batch_size)),
        "--disable_progress_bar",
    ]
    if device:
        command.extend(["--device", str(device)])
    result = subprocess.run(
        command,
        check=False,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"STARLING failed for {segment_name}: {result.stderr.strip()}")
    ensemble_path = _find_starling_file(segment_dir, segment_name)
    if ensemble_path is None:
        raise FileNotFoundError(f"STARLING did not produce a .starling file for {segment_name} under {segment_dir}")
    return ensemble_path


def _find_starling_file(directory: Path, segment_name: str) -> Path | None:
    candidates = sorted(directory.glob(f"{segment_name}.starling*"))
    if candidates:
        return candidates[0]
    candidates = sorted(directory.glob("*.starling*"))
    return candidates[0] if candidates else None


def main() -> None:
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
    parser.add_argument(
        "--api-sequence-batch-size",
        type=int,
        default=8,
        help="Number of STARLING sequence segments per Python API generate() call.",
    )
    parser.add_argument("--skip-completeness-check", action="store_true")
    parser.add_argument("--limit-records", type=int, help="Process only the first N records; intended for smoke tests.")
    args = parser.parse_args()
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
    )
    print(f"Wrote {len(written)} STARLING feature files to {args.out_dir}")


if __name__ == "__main__":
    main()
