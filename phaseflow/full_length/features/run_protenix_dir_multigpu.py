from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Job:
    path: Path
    name: str
    length: int

    @property
    def cost(self) -> int:
        return max(self.length, 1) ** 2


def main() -> None:
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

    completed = completed_jobs(out_dir)
    all_jobs = [load_job(path) for path in sorted(json_dir.glob("*.json")) if path.stem not in completed]
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

    bins: list[list[Job]] = [[] for _ in gpu_ids]
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
        command = protenix_command(args, shard_dir, out_dir)
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
            "completed_after_finish": len(completed_jobs(out_dir)),
            "failed_workers": failures,
        }
    )
    (log_dir / "protenix_dir_multigpu_status_finish.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return failures


def protenix_command(args: argparse.Namespace, input_dir: Path, out_dir: Path) -> list[str]:
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


def completed_jobs(out_dir: Path) -> set[str]:
    done: set[str] = set()
    if not out_dir.exists():
        return done
    for path in out_dir.rglob("*_sample_*.cif"):
        name = path.name.rsplit("_sample_", 1)[0]
        if name:
            done.add(name)
    return done


def load_job(path: Path) -> Job:
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
    return Job(path=path, name=path.stem, length=max(length, 1))


if __name__ == "__main__":
    main()
