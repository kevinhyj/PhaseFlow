from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from phaseflow.full_length.features.graph_cache import GRAPH_CACHE_VERSION, edge_list_to_precomputed_graph


def precompute_graph_cache(
    path: Path,
    *,
    max_neighbors: int,
    edge_dim: int,
    overwrite: bool,
    compression: str | None,
) -> str:
    with h5py.File(path, "r+") as handle:
        length = _length(handle)
        if (
            "graph" in handle
            and not overwrite
            and _usable_existing_graph(handle["graph"], length, max_neighbors, edge_dim)
        ):
            return "skipped"
        if not {"edge_src", "edge_dst", "edge_type", "edge_attr"}.issubset(handle.keys()):
            return "missing_edges"

        graph = edge_list_to_precomputed_graph(
            length=length,
            edge_src=np.asarray(handle["edge_src"], dtype=np.int64),
            edge_dst=np.asarray(handle["edge_dst"], dtype=np.int64),
            edge_type=np.asarray(handle["edge_type"], dtype=np.int64),
            edge_attr=np.asarray(handle["edge_attr"], dtype=np.float32),
            max_neighbors=max_neighbors,
            edge_dim=edge_dim,
        )

        if "graph" in handle:
            del handle["graph"]
        group = handle.create_group("graph")
        group.attrs["version"] = GRAPH_CACHE_VERSION
        group.attrs["max_neighbors"] = int(max_neighbors)
        group.attrs["edge_dim"] = int(edge_dim)
        group.attrs["source_edge_count"] = int(handle["edge_src"].shape[0])
        group.create_dataset("neighbors", data=graph.neighbors, compression=compression)
        group.create_dataset("edge_attr", data=graph.edge_attr, compression=compression)
        group.create_dataset("neighbor_mask", data=graph.neighbor_mask, compression=compression)
    return "written"


def _length(handle: h5py.File) -> int:
    if "length" in handle.attrs:
        return int(handle.attrs["length"])
    if "sequence" in handle.attrs:
        return len(str(handle.attrs["sequence"]))
    if "plm" in handle:
        return int(handle["plm"].shape[0])
    raise ValueError("Cannot infer sequence length")


def _usable_existing_graph(group: h5py.Group, length: int, max_neighbors: int, edge_dim: int) -> bool:
    required = {"neighbors", "edge_attr", "neighbor_mask"}
    if not required.issubset(group.keys()):
        return False
    neighbors = group["neighbors"]
    edge_attr = group["edge_attr"]
    neighbor_mask = group["neighbor_mask"]
    return (
        neighbors.ndim == 2
        and edge_attr.ndim == 3
        and neighbor_mask.ndim == 2
        and neighbors.shape[0] == length
        and edge_attr.shape[0] == length
        and neighbor_mask.shape[0] == length
        and neighbors.shape[1] >= max_neighbors
        and edge_attr.shape[1] >= max_neighbors
        and neighbor_mask.shape[1] >= max_neighbors
        and edge_attr.shape[2] >= edge_dim
    )


def _iter_h5(feature_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for feature_dir in feature_dirs:
        for path in sorted(feature_dir.glob("*.h5")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            paths.append(path)
            seen.add(resolved)
    return paths


def _iter_h5_for_ids(feature_dirs: list[Path], ids_files: list[Path]) -> list[Path]:
    wanted: list[str] = []
    seen_ids: set[str] = set()
    for ids_file in ids_files:
        for line in ids_file.read_text().splitlines():
            protein_id = line.strip()
            if not protein_id or protein_id in seen_ids:
                continue
            wanted.append(protein_id)
            seen_ids.add(protein_id)

    paths: list[Path] = []
    for protein_id in wanted:
        for feature_dir in feature_dirs:
            path = feature_dir / f"{protein_id}.h5"
            if path.exists():
                paths.append(path)
                break
    return paths


def _worker(args: tuple[str, int, int, bool, str | None]) -> tuple[str, str, str]:
    path, max_neighbors, edge_dim, overwrite, compression = args
    try:
        status = precompute_graph_cache(
            Path(path),
            max_neighbors=max_neighbors,
            edge_dim=edge_dim,
            overwrite=overwrite,
            compression=compression,
        )
    except Exception as exc:  # pragma: no cover - CLI reporting path
        return path, "failed", str(exc)
    return path, status, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", action="append", required=True)
    parser.add_argument("--ids-file", action="append")
    parser.add_argument("--max-neighbors", type=int, default=96)
    parser.add_argument("--edge-dim", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    feature_dirs = [Path(path) for path in args.feature_dir]
    if args.ids_file:
        paths = _iter_h5_for_ids(feature_dirs, [Path(path) for path in args.ids_file])
    else:
        paths = _iter_h5(feature_dirs)
    if args.limit > 0:
        paths = paths[: args.limit]
    counts = {"written": 0, "skipped": 0, "missing_edges": 0, "failed": 0}
    compression = None if args.no_compression else "gzip"
    if args.workers <= 1:
        for index, path in enumerate(paths, start=1):
            path_text, status, error = _worker(
                (str(path), args.max_neighbors, args.edge_dim, args.overwrite, compression)
            )
            counts[status] = counts.get(status, 0) + 1
            if error:
                print(f"{path_text}\tfailed\t{error}", flush=True)
            if args.progress_every > 0 and index % args.progress_every == 0:
                print(f"processed={index}/{len(paths)} counts={counts}", flush=True)
    else:
        jobs = [
            (str(path), args.max_neighbors, args.edge_dim, args.overwrite, compression)
            for path in paths
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, (path_text, status, error) in enumerate(executor.map(_worker, jobs, chunksize=64), start=1):
                counts[status] = counts.get(status, 0) + 1
                if error:
                    print(f"{path_text}\tfailed\t{error}", flush=True)
                if args.progress_every > 0 and index % args.progress_every == 0:
                    print(f"processed={index}/{len(paths)} counts={counts}", flush=True)
    print(f"processed={len(paths)} counts={counts}", flush=True)


if __name__ == "__main__":
    main()
