
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FastaRecord:
    name: str
    sequence: str


def read_fasta(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    current_name: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    records.append(FastaRecord(current_name, "".join(chunks)))
                current_name = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if current_name is not None:
        records.append(FastaRecord(current_name, "".join(chunks)))
    return records


def write_fasta(path: Path, records: Iterable[FastaRecord]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            count += 1
            handle.write(f">{record.name}\n")
            sequence = record.sequence
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
    return count


def split_balanced(records: list[FastaRecord], shards: int) -> list[list[FastaRecord]]:
    out: list[list[FastaRecord]] = [[] for _ in range(shards)]
    loads = [0 for _ in range(shards)]
    for record in sorted(records, key=lambda item: len(item.sequence), reverse=True):
        index = min(range(shards), key=lambda idx: loads[idx])
        out[index].append(record)
        loads[index] += len(record.sequence)
    return out


def format_output(pattern: str, shard_index: int, device: str) -> Path:
    token = f"{shard_index:03d}"
    if "{shard}" in pattern or "{device}" in pattern:
        return Path(pattern.format(shard=token, shard_index=shard_index, device=device))
    path = Path(pattern)
    return path.with_name(f"{path.stem}_{token}{path.suffix}")


def format_command(command: list[str], input_path: Path, output_path: Path, shard_index: int, device: str) -> list[str]:
    token = f"{shard_index:03d}"
    context = {
        "input": str(input_path),
        "output": str(output_path),
        "shard": token,
        "shard_index": str(shard_index),
        "device": device,
    }
    return [part.format(**context) for part in command]


def tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(errors="replace")
    return data[-max_chars:]


def run_shards(args: argparse.Namespace) -> dict[str, object]:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("A child command is required after --")

    devices = [item.strip() for item in str(args.devices).split(",") if item.strip()]
    if not devices:
        raise SystemExit("--devices must contain at least one CUDA device id")

    records = read_fasta(args.input)
    if not records:
        raise SystemExit(f"No FASTA records found in {args.input}")

    args.shard_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    shards = split_balanced(records, len(devices))
    processes: list[tuple[int, str, Path, Path, Path, subprocess.Popen[str]]] = []
    status_rows: list[dict[str, object]] = []

    for shard_index, (device, shard_records) in enumerate(zip(devices, shards)):
        shard_input = args.shard_dir / f"shard_{shard_index:03d}.fasta"
        output_path = format_output(str(args.output_pattern), shard_index, device)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        record_count = write_fasta(shard_input, shard_records)
        stdout_path = args.log_dir / f"shard_{shard_index:03d}.stdout.log"
        stderr_path = args.log_dir / f"shard_{shard_index:03d}.stderr.log"
        child_command = format_command(command, shard_input, output_path, shard_index, device)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        env["PYTHONUNBUFFERED"] = "1"
        env["PHASEFLOW_TEACHER_SHARD_INDEX"] = str(shard_index)
        env["PHASEFLOW_TEACHER_SHARD_DEVICE"] = device
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(
                child_command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                env=env,
            )
        processes.append((shard_index, device, output_path, stdout_path, stderr_path, process))
        status_rows.append(
            {
                "shard": shard_index,
                "device": device,
                "records": record_count,
                "input": str(shard_input),
                "output": str(output_path),
                "command": child_command,
            }
        )

    failures: list[dict[str, object]] = []
    for shard_index, device, output_path, stdout_path, stderr_path, process in processes:
        returncode = process.wait()
        status_rows[shard_index]["returncode"] = returncode
        status_rows[shard_index]["output_exists"] = output_path.exists()
        status_rows[shard_index]["output_size"] = output_path.stat().st_size if output_path.exists() else 0
        if returncode != 0 or not output_path.exists():
            failures.append(
                {
                    "shard": shard_index,
                    "device": device,
                    "returncode": returncode,
                    "stdout_tail": tail_text(stdout_path),
                    "stderr_tail": tail_text(stderr_path),
                    "output": str(output_path),
                }
            )

    summary = {
        "input": str(args.input),
        "records": len(records),
        "devices": devices,
        "output_pattern": str(args.output_pattern),
        "shards": status_rows,
        "failures": failures,
    }
    args.status_json.parent.mkdir(parents=True, exist_ok=True)
    args.status_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("one or more FASTA shards failed")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split a FASTA file and run one child command per CUDA device.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-pattern", required=True, type=Path)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.status_json is None:
        args.status_json = args.log_dir / "shard_status.json"
    run_shards(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
