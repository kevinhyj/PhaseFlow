"""Protein structure feature command helpers."""

# Source: features/make_af3_json.py


import argparse
from pathlib import Path

from phaseflow.protein.structure import write_af3_input_json
from phaseflow.protein.features import clean_protein_sequence
from phaseflow.protein.features import records_from_fasta, records_from_manifest


def make_af3_jsons(
    records: list[tuple[str, str]],
    out_dir: str | Path,
    model_seeds: list[int],
    msa_mode: str = "no_msa",
) -> list[Path]:
    paths: list[Path] = []
    for protein_id, sequence in records:
        paths.append(
            write_af3_input_json(
                protein_id,
                clean_protein_sequence(sequence),
                out_dir,
                model_seeds=model_seeds,
                msa_mode=msa_mode,
            )
        )
    return paths


def make_af3_json_main() -> None:
    parser = argparse.ArgumentParser(description="Create AlphaFold 3 input JSON files from manifest or FASTA.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--fasta")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--msa-mode",
        choices=["no_msa", "full_pipeline"],
        default="no_msa",
        help="no_msa writes empty MSA/template fields so AF3 can run with --run_data_pipeline=false.",
    )
    args = parser.parse_args()
    records = records_from_manifest(args.manifest) if args.manifest else records_from_fasta(args.fasta)
    paths = make_af3_jsons(records, args.out_dir, args.model_seeds, msa_mode=args.msa_mode)
    print(f"Wrote {len(paths)} AF3 input JSON files to {args.out_dir}")


if __name__ == "__main__":
    make_af3_json_main()


# Source: features/make_protenix_json.py


import argparse
import json
from pathlib import Path

from phaseflow.protein.features import clean_protein_sequence
from phaseflow.protein.features import records_from_fasta, records_from_manifest


def write_protenix_input_json(
    protein_id: str,
    sequence: str,
    out_dir: str | Path,
    model_seeds: list[int] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": protein_id,
            "modelSeeds": [int(seed) for seed in (model_seeds or [101])],
            "covalent_bonds": [],
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": clean_protein_sequence(sequence),
                        "count": 1,
                        "modifications": [],
                    }
                }
            ],
        }
    ]
    path = out_dir / f"{protein_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def make_protenix_jsons(
    records: list[tuple[str, str]],
    out_dir: str | Path,
    model_seeds: list[int],
) -> list[Path]:
    return [write_protenix_input_json(protein_id, sequence, out_dir, model_seeds) for protein_id, sequence in records]


def make_protenix_json_main() -> None:
    parser = argparse.ArgumentParser(description="Create Protenix input JSON files from a PhaseFlow manifest or FASTA.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--fasta")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[101])
    args = parser.parse_args()
    records = records_from_manifest(args.manifest) if args.manifest else records_from_fasta(args.fasta)
    paths = make_protenix_jsons(records, args.out_dir, args.model_seeds)
    print(f"Wrote {len(paths)} Protenix input JSON files to {args.out_dir}")


if __name__ == "__main__":
    make_protenix_json_main()


# Source: features/parse_af3_output.py


import argparse
import json
from pathlib import Path

import numpy as np

from phaseflow.protein.features import records_from_fasta, records_from_manifest


def parse_af3_outputs(
    records: list[tuple[str, str]],
    af3_output: str | Path,
    out_dir: str | Path,
    contact_topk: int = 32,
) -> list[Path]:
    af3_output = Path(af3_output)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for protein_id, sequence in records:
        job_dir = _find_job_dir(af3_output, protein_id)
        if job_dir is None:
            continue
        single = _find_single_embedding(job_dir, len(sequence))
        node = _find_confidence_node(job_dir, len(sequence))
        if single is None and node is None:
            continue
        contacts = _contacts_from_pair_embedding(job_dir, len(sequence), contact_topk)
        reliability = _reliability_from_node(node, len(sequence))
        payload = {
            "protein_id": np.asarray(protein_id),
            "sequence": np.asarray(sequence),
            "reliability": reliability,
        }
        if single is not None:
            payload["single_embedding"] = single.astype(np.float32, copy=False)
        if node is not None:
            payload["node"] = node.astype(np.float32, copy=False)
        if contacts is not None:
            payload["contacts"] = contacts.astype(np.float32, copy=False)
        path = out_dir / f"{protein_id}.npz"
        np.savez_compressed(path, **payload)
        written.append(path)
    return written


def _find_job_dir(root: Path, protein_id: str) -> Path | None:
    candidates = [root / protein_id, root / protein_id.lower(), root / protein_id.upper()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(path for path in root.glob("**/*") if path.is_dir() and path.name.lower() == protein_id.lower())
    return matches[0] if matches else None


def _find_single_embedding(job_dir: Path, length: int) -> np.ndarray | None:
    for path in sorted(job_dir.glob("**/embeddings.npz")):
        with np.load(path, allow_pickle=False) as data:
            for key in ("single_embeddings", "single_embedding"):
                if key in data:
                    value = np.asarray(data[key], dtype=np.float32)
                    if value.ndim == 2 and value.shape[0] >= length:
                        return value[:length]
    return None


def _find_confidence_node(job_dir: Path, length: int) -> np.ndarray | None:
    json_paths = sorted(job_dir.glob("*confidence*.json")) + sorted(job_dir.glob("**/*confidence*.json"))
    for path in json_paths:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        plddt = _first_numeric_vector(data, ("atom_plddts", "plddt", "pae_plddt"))
        if plddt is None:
            continue
        plddt = plddt[:length]
        if plddt.shape[0] != length:
            continue
        return np.stack([plddt / 100.0, np.ones(length, dtype=np.float32)], axis=1)
    return None


def _contacts_from_pair_embedding(job_dir: Path, length: int, topk: int) -> np.ndarray | None:
    for path in sorted(job_dir.glob("**/embeddings.npz")):
        with np.load(path, allow_pickle=False) as data:
            for key in ("pair_embeddings", "pair_embedding"):
                if key not in data:
                    continue
                pair = np.asarray(data[key])
                if pair.ndim != 3 or pair.shape[0] < length or pair.shape[1] < length:
                    continue
                score = np.linalg.norm(pair[:length, :length], axis=-1).astype(np.float32)
                return _topk_contacts(score, topk)
    return None


def _topk_contacts(score: np.ndarray, topk: int) -> np.ndarray:
    rows: list[tuple[int, int, float]] = []
    length = score.shape[0]
    for src in range(length):
        values = score[src].copy()
        values[src] = -np.inf
        if topk < length:
            idx = np.argpartition(-values, topk)[:topk]
        else:
            idx = np.arange(length)
        idx = idx[np.isfinite(values[idx])]
        idx = idx[np.argsort(-values[idx])]
        for dst in idx[:topk]:
            rows.append((src, int(dst), float(values[dst])))
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)


def _reliability_from_node(node: np.ndarray | None, length: int) -> np.ndarray:
    if node is None or node.shape[0] != length:
        return np.ones(length, dtype=np.float32)
    return np.clip(node[:, 0], 0.0, 1.0).astype(np.float32)


def _first_numeric_vector(data: object, names: tuple[str, ...]) -> np.ndarray | None:
    if isinstance(data, dict):
        for name in names:
            if name in data:
                value = np.asarray(data[name], dtype=np.float32).reshape(-1)
                return value
        for value in data.values():
            found = _first_numeric_vector(value, names)
            if found is not None:
                return found
    return None


def parse_af3_output_main() -> None:
    parser = argparse.ArgumentParser(description="Parse AF3 outputs into PhaseFlow intermediate npz files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--fasta")
    parser.add_argument("--af3-output", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--contact-topk", type=int, default=32)
    args = parser.parse_args()
    records = records_from_manifest(args.manifest) if args.manifest else records_from_fasta(args.fasta)
    written = parse_af3_outputs(records, args.af3_output, args.out_dir, args.contact_topk)
    print(f"Wrote {len(written)} parsed AF3 feature files to {args.out_dir}")


if __name__ == "__main__":
    parse_af3_output_main()


# Source: features/parse_protenix_output.py


import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from phaseflow.protein.features import records_from_fasta, records_from_manifest
from phaseflow.protein.structure import parse_single_protenix_output


def parse_protenix_outputs_parallel(
    records: list[tuple[str, str]],
    protenix_output: str | Path,
    out_dir: str | Path,
    contact_topk: int = 32,
    contact_cutoff: float = 8.0,
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), len(records) if records else 1))
    stats = {"written": 0, "skipped": 0, "missing": 0, "failed": 0}
    jobs = [
        {
            "protein_id": protein_id,
            "sequence": sequence,
            "protenix_output": str(protenix_output),
            "out_dir": str(out_dir),
            "contact_topk": int(contact_topk),
            "contact_cutoff": float(contact_cutoff),
            "overwrite": bool(overwrite),
        }
        for protein_id, sequence in records
    ]
    if workers == 1:
        for job in jobs:
            status = _parse_one(job)
            stats[status] = stats.get(status, 0) + 1
        return stats
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_parse_one, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                status = future.result()
            except Exception:
                status = "failed"
            stats[status] = stats.get(status, 0) + 1
            if index == 1 or index % 500 == 0 or index == len(futures):
                print(
                    "parse_progress "
                    f"done={index}/{len(futures)} "
                    f"written={stats.get('written', 0)} "
                    f"skipped={stats.get('skipped', 0)} "
                    f"missing={stats.get('missing', 0)} "
                    f"failed={stats.get('failed', 0)}",
                    flush=True,
                )
    return stats


def _parse_one(job: dict[str, Any]) -> str:
    protein_id = str(job["protein_id"])
    sequence = str(job["sequence"])
    out_path = Path(str(job["out_dir"])) / f"{protein_id}.npz"
    if not bool(job["overwrite"]) and _existing_npz_is_valid(out_path, protein_id, sequence):
        return "skipped"
    parsed = parse_single_protenix_output(
        protein_id=protein_id,
        sequence=sequence,
        protenix_output=Path(str(job["protenix_output"])),
        contact_topk=int(job["contact_topk"]),
        contact_cutoff=float(job["contact_cutoff"]),
    )
    if parsed is None:
        return "missing"
    payload: dict[str, Any] = {
        "protein_id": np.asarray(protein_id),
        "sequence": np.asarray(sequence),
        "node": parsed.node.astype(np.float32, copy=False),
        "missing_mask": parsed.missing_mask.astype(np.float32, copy=False),
        "reliability": parsed.reliability.astype(np.float32, copy=False),
    }
    if parsed.contacts is not None:
        payload["contacts"] = parsed.contacts.astype(np.float32, copy=False)
    for key, value in parsed.metadata.items():
        if value is not None:
            payload[key] = np.asarray(str(value))
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp_path.replace(out_path)
    return "written"


def _existing_npz_is_valid(path: Path, protein_id: str, sequence: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            cached_id = str(data["protein_id"].item()) if "protein_id" in data else protein_id
            cached_sequence = str(data["sequence"].item()) if "sequence" in data else sequence
            node = np.asarray(data["node"], dtype=np.float32) if "node" in data else None
            missing = np.asarray(data["missing_mask"], dtype=np.float32) if "missing_mask" in data else None
            reliability = np.asarray(data["reliability"], dtype=np.float32) if "reliability" in data else None
        return (
            cached_id == protein_id
            and cached_sequence == sequence
            and node is not None
            and missing is not None
            and reliability is not None
            and node.ndim == 2
            and node.shape[0] == len(sequence)
            and missing.shape == (len(sequence),)
            and reliability.shape == (len(sequence),)
        )
    except Exception:
        return False


def parse_protenix_output_main() -> None:
    parser = argparse.ArgumentParser(description="Parse Protenix outputs into PhaseFlow structure intermediate npz files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--fasta")
    parser.add_argument("--protenix-output", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--contact-topk", type=int, default=32)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    records = records_from_manifest(args.manifest) if args.manifest else records_from_fasta(args.fasta)
    stats = parse_protenix_outputs_parallel(
        records=records,
        protenix_output=args.protenix_output,
        out_dir=args.out_dir,
        contact_topk=args.contact_topk,
        contact_cutoff=args.contact_cutoff,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(
        "Parsed Protenix structure features "
        f"to {args.out_dir}: "
        f"written={stats.get('written', 0)} "
        f"skipped={stats.get('skipped', 0)} "
        f"missing={stats.get('missing', 0)} "
        f"failed={stats.get('failed', 0)}"
    )


if __name__ == "__main__":
    parse_protenix_output_main()


# Source: features/run_protenix_dir_multigpu.py


import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunProtenixDirMultigpuJob:
    path: Path
    name: str
    length: int

    @property
    def cost(self) -> int:
        return max(self.length, 1) ** 2


def run_protenix_dir_multigpu_main() -> None:
    parser = argparse.ArgumentParser(description="Run Protenix over per-GPU JSON directories to avoid per-protein model reloads.")
    parser.add_argument("--json-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--protenix-bin", default=os.environ.get("PROTENIX_BIN", "protenix"))
    parser.add_argument("--model-name", default="protenix_base_20250630_v1.0.0")
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--cycle", default="4")
    parser.add_argument("--step", default="20")
    parser.add_argument("--sample", default="1")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--use-msa", default="false")
    parser.add_argument("--use-template", default="false")
    parser.add_argument("--use-default-params", default="false")
    parser.add_argument("--need-atom-confidence", default="true")
    parser.add_argument("--trimul-kernel", default="torch")
    parser.add_argument("--triatt-kernel", default="torch")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--order", choices=["shortest", "longest"], default="shortest")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    failures = run(args)
    raise SystemExit(0 if failures == 0 or args.allow_failures else 1)


def run(args: argparse.Namespace) -> int:
    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    shard_root = log_dir / "protenix_dir_shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids were provided")

    completed = run_protenix_dir_multigpu_completed_jobs(out_dir)
    all_jobs = [run_protenix_dir_multigpu_load_job(path) for path in sorted(json_dir.glob("*.json")) if path.stem not in completed]
    if args.max_length > 0:
        jobs = [job for job in all_jobs if job.length <= args.max_length]
        skipped = [job for job in all_jobs if job.length > args.max_length]
    else:
        jobs = all_jobs
        skipped = []
    if skipped:
        with (log_dir / "protenix_dir_skipped_length.tsv").open("w") as handle:
            handle.write("protein_id\tlength\tjson_path\n")
            for job in sorted(skipped, key=lambda item: item.length, reverse=True):
                handle.write(f"{job.name}\t{job.length}\t{job.path}\n")

    bins: list[list[RunProtenixDirMultigpuJob]] = [[] for _ in gpu_ids]
    bin_costs = [0 for _ in gpu_ids]
    reverse = args.order == "longest"
    for job in sorted(jobs, key=lambda item: item.cost, reverse=reverse):
        target = min(range(len(gpu_ids)), key=lambda index: bin_costs[index])
        bins[target].append(job)
        bin_costs[target] += job.cost

    status = {
        "json_dir": str(json_dir),
        "out_dir": str(out_dir),
        "gpus": gpu_ids,
        "completed_before_start": len(completed),
        "pending_jobs": len(jobs),
        "skipped_length_jobs": len(skipped),
        "max_length": int(args.max_length),
        "order": str(args.order),
        "shards": [
            {"gpu": gpu, "jobs": len(shard), "estimated_cost": int(cost)}
            for gpu, shard, cost in zip(gpu_ids, bins, bin_costs, strict=False)
        ],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (log_dir / "protenix_dir_multigpu_status_start.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    processes: list[tuple[str, subprocess.Popen[bytes]]] = []
    for gpu, shard in zip(gpu_ids, bins, strict=False):
        shard_dir = shard_root / f"gpu_{gpu}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = log_dir / f"protenix_dir_gpu_{gpu}.manifest.tsv"
        with manifest_path.open("w") as manifest:
            manifest.write("protein_id\tlength\tjson_path\tshard_json\n")
            for job in shard:
                shard_json = shard_dir / job.path.name
                if not shard_json.exists():
                    try:
                        shard_json.symlink_to(job.path.resolve())
                    except FileExistsError:
                        pass
                manifest.write(f"{job.name}\t{job.length}\t{job.path}\t{shard_json}\n")
        if not shard:
            continue
        log_path = log_dir / f"protenix_dir_gpu_{gpu}.log"
        handle = log_path.open("ab")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("MPLCONFIGDIR", str(Path(".cache") / f"phaseflow_mpl_cache_gpu_{gpu}"))
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        command = run_protenix_dir_multigpu_protenix_command(args, shard_dir, out_dir)
        processes.append((gpu, subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)))

    failures = 0
    worker_status = []
    for gpu, process in processes:
        code = process.wait()
        worker_status.append({"gpu": gpu, "returncode": code})
        if code != 0:
            failures += 1
    final = dict(status)
    final.update(
        {
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "worker_status": worker_status,
            "completed_after_finish": len(run_protenix_dir_multigpu_completed_jobs(out_dir)),
            "failed_workers": failures,
        }
    )
    (log_dir / "protenix_dir_multigpu_status_finish.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return failures


def run_protenix_dir_multigpu_protenix_command(args: argparse.Namespace, input_dir: Path, out_dir: Path) -> list[str]:
    return [
        str(args.protenix_bin),
        "pred",
        "--input",
        str(input_dir),
        "--out_dir",
        str(out_dir),
        "--seeds",
        str(args.seeds),
        "--cycle",
        str(args.cycle),
        "--step",
        str(args.step),
        "--sample",
        str(args.sample),
        "--dtype",
        str(args.dtype),
        "--model_name",
        str(args.model_name),
        "--use_msa",
        str(args.use_msa),
        "--use_template",
        str(args.use_template),
        "--use_default_params",
        str(args.use_default_params),
        "--need_atom_confidence",
        str(args.need_atom_confidence),
        "--trimul_kernel",
        str(args.trimul_kernel),
        "--triatt_kernel",
        str(args.triatt_kernel),
    ]


def run_protenix_dir_multigpu_completed_jobs(out_dir: Path) -> set[str]:
    done: set[str] = set()
    if not out_dir.exists():
        return done
    for path in out_dir.rglob("*_sample_*.cif"):
        name = path.name.rsplit("_sample_", 1)[0]
        if name:
            done.add(name)
    return done


def run_protenix_dir_multigpu_load_job(path: Path) -> RunProtenixDirMultigpuJob:
    length = 1
    try:
        payload = json.loads(path.read_text())
        first = payload[0] if isinstance(payload, list) and payload else payload
        sequences = first.get("sequences", []) if isinstance(first, dict) else []
        length = 0
        for item in sequences:
            chain = item.get("proteinChain", {}) if isinstance(item, dict) else {}
            length += len(str(chain.get("sequence", "")))
    except Exception:
        length = 1
    return RunProtenixDirMultigpuJob(path=path, name=path.stem, length=max(length, 1))


if __name__ == "__main__":
    run_protenix_dir_multigpu_main()
