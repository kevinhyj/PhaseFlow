from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
    parser = argparse.ArgumentParser(description="Run Protenix prediction over JSON inputs with one worker per GPU.")
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
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard-file")
    parser.add_argument("--gpu")
    parser.add_argument("--max-length", type=int, default=0, help="Skip JSON jobs longer than this residue length. 0 disables the limit.")
    parser.add_argument("--order", choices=["shortest", "longest"], default="longest")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    if args.worker:
        if not args.shard_file or args.gpu is None:
            raise SystemExit("--worker requires --shard-file and --gpu")
        failures = run_worker(args)
        raise SystemExit(0 if failures == 0 or args.allow_failures else 1)

    failures = run_master(args)
    raise SystemExit(0 if failures == 0 or args.allow_failures else 1)


def run_master(args: argparse.Namespace) -> int:
    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    shard_dir = log_dir / "protenix_shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids were provided")

    completed = completed_jobs(out_dir)
    all_jobs = [load_job(path) for path in sorted(json_dir.glob("*.json")) if path.stem not in completed]
    if args.max_length and int(args.max_length) > 0:
        jobs = [job for job in all_jobs if job.length <= int(args.max_length)]
        skipped_length = [job for job in all_jobs if job.length > int(args.max_length)]
    else:
        jobs = all_jobs
        skipped_length = []
    if skipped_length:
        skip_path = log_dir / "protenix_skipped_length.tsv"
        with skip_path.open("w") as handle:
            handle.write("protein_id\tlength\tjson_path\n")
            for job in sorted(skipped_length, key=lambda item: item.length, reverse=True):
                handle.write(f"{job.name}\t{job.length}\t{job.path}\n")
    bins: list[list[Job]] = [[] for _ in gpu_ids]
    bin_costs = [0 for _ in gpu_ids]
    reverse = args.order == "longest"
    for job in sorted(jobs, key=lambda item: item.cost, reverse=reverse):
        index = min(range(len(gpu_ids)), key=lambda idx: bin_costs[idx])
        bins[index].append(job)
        bin_costs[index] += job.cost

    status = {
        "json_dir": str(json_dir),
        "out_dir": str(out_dir),
        "gpus": gpu_ids,
        "completed_before_start": len(completed),
        "pending_jobs": len(jobs),
        "skipped_length_jobs": len(skipped_length),
        "max_length": int(args.max_length),
        "order": str(args.order),
        "shards": [
            {"gpu": gpu, "jobs": len(shard), "estimated_cost": int(cost)}
            for gpu, shard, cost in zip(gpu_ids, bins, bin_costs, strict=False)
        ],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (log_dir / "protenix_multigpu_status_start.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    processes: list[tuple[str, subprocess.Popen[bytes]]] = []
    for gpu, shard in zip(gpu_ids, bins, strict=False):
        shard_file = shard_dir / f"gpu_{gpu}.txt"
        shard_file.write_text("".join(f"{job.path}\n" for job in shard))
        if not shard:
            continue
        command = [
            sys.executable,
            "-m",
            "phaseflow.features.run_protenix_multigpu",
            "--worker",
            "--shard-file",
            str(shard_file),
            "--gpu",
            str(gpu),
            "--json-dir",
            str(json_dir),
            "--out-dir",
            str(out_dir),
            "--log-dir",
            str(log_dir),
            "--gpus",
            str(gpu),
            "--protenix-bin",
            str(args.protenix_bin),
            "--model-name",
            str(args.model_name),
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
            "--use-msa",
            str(args.use_msa),
            "--use-template",
            str(args.use_template),
            "--use-default-params",
            str(args.use_default_params),
            "--need-atom-confidence",
            str(args.need_atom_confidence),
            "--trimul-kernel",
            str(args.trimul_kernel),
            "--triatt-kernel",
            str(args.triatt_kernel),
            "--max-length",
            str(args.max_length),
            "--order",
            str(args.order),
        ]
        if args.allow_failures:
            command.append("--allow-failures")
        log_path = log_dir / f"protenix_gpu_{gpu}.launcher.log"
        handle = log_path.open("ab")
        processes.append((gpu, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)))

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
    (log_dir / "protenix_multigpu_status_finish.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return failures


def run_worker(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    gpu = str(args.gpu)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("MPLCONFIGDIR", str(Path(".cache") / f"phaseflow_mpl_cache_gpu_{gpu}"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    shard_paths = [Path(line.strip()) for line in Path(args.shard_file).read_text().splitlines() if line.strip()]
    status_path = log_dir / f"protenix_gpu_{gpu}.tsv"
    run_log_path = log_dir / f"protenix_gpu_{gpu}.run.log"
    failures = 0
    with status_path.open("a") as status, run_log_path.open("ab") as run_log:
        if status.tell() == 0:
            status.write("protein_id\tstatus\tseconds\treturncode\tjson_path\n")
        for json_path in shard_paths:
            job_name = json_path.stem
            if is_done(out_dir, job_name):
                status.write(f"{job_name}\tskip\t0\t0\t{json_path}\n")
                status.flush()
                continue
            started = time.time()
            command = protenix_command(args, json_path, out_dir)
            run_log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPU {gpu} {job_name}\n".encode())
            run_log.flush()
            result = subprocess.run(command, env=env, stdout=run_log, stderr=subprocess.STDOUT, check=False)
            seconds = time.time() - started
            if result.returncode == 0 and is_done(out_dir, job_name):
                state = "ok"
            else:
                state = "failed"
                failures += 1
            status.write(f"{job_name}\t{state}\t{seconds:.3f}\t{result.returncode}\t{json_path}\n")
            status.flush()
    return failures


def protenix_command(args: argparse.Namespace, json_path: Path, out_dir: Path) -> list[str]:
    return [
        str(args.protenix_bin),
        "pred",
        "--input",
        str(json_path),
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


def is_done(out_dir: Path, job_name: str) -> bool:
    return any(out_dir.rglob(f"{job_name}_sample_*.cif"))


def load_job(path: Path) -> Job:
    length = 1
    try:
        payload = json.loads(path.read_text())
        chain = payload[0]["sequences"][0]["proteinChain"]
        length = len(str(chain["sequence"]))
    except Exception:
        length = 1
    return Job(path=path, name=path.stem, length=length)


if __name__ == "__main__":
    main()
