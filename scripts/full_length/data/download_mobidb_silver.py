#!/usr/bin/env python3
"""Download MobiDB TSV batches for high-confidence disorder silver negatives."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from augment_train_external_sources import (
    base_acc,
    load_active_manifest,
    load_benchmark_sets,
    parse_swissprot_fasta,
)


def high_conf_accessions(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if df.empty or not {"acc", "feature", "content_fraction", "content_count"}.issubset(df.columns):
        return set()
    df["content_fraction"] = pd.to_numeric(df["content_fraction"], errors="coerce")
    df["content_count"] = pd.to_numeric(df["content_count"], errors="coerce")
    features = df["feature"].astype(str)
    mask = (
        features.str.contains(r"^(curated|prediction)-disorder-(priority|merge|th_50|mobidb_lite|disprot)$", case=False, regex=True, na=False)
        & (df["content_fraction"] >= 0.50)
        & (df["content_count"] >= 50)
    )
    bad = features.str.contains(r"phase[_\s-]?separation|llps|condensate|membrane[_\s-]?less|\bmlo\b", case=False, regex=True, na=False)
    good = set(df.loc[mask, "acc"].dropna().astype(str))
    bad_acc = set(df.loc[bad, "acc"].dropna().astype(str))
    return {base_acc(x) for x in good - bad_acc if base_acc(x)}


def fetch_batch(accs: list[str], timeout: int) -> str:
    query = urllib.parse.urlencode({"format": "tsv", "acc": ",".join(accs)})
    url = f"https://mobidb.org/api/download?{query}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--raw-external-dir", default="artifacts/data/raw_external")
    parser.add_argument("--target-highconf", type=int, default=3000)
    parser.add_argument("--max-accessions", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    raw_dir = (root / args.raw_external_dir).resolve()
    mobidb_dir = raw_dir / "mobidb"
    mobidb_dir.mkdir(parents=True, exist_ok=True)
    out_path = mobidb_dir / "mobidb_swissprot_disorder_batches.tsv"
    progress_path = mobidb_dir / "mobidb_swissprot_disorder_batches.progress.json"

    swiss = parse_swissprot_fasta(raw_dir / "uniprot_swissprot/uniprot_sprot.fasta.gz")
    benchmark = load_benchmark_sets(root)
    active = load_active_manifest(root)
    active_positive = set(active.loc[active["llps_label"] == 1, "uniprot_acc"].dropna().astype(str))
    blocked = {base_acc(x) for x in benchmark["accs"] | benchmark["ids"] | active_positive if base_acc(x)}
    candidates = [
        acc
        for acc, meta in sorted(swiss.items())
        if acc not in blocked and 50 <= len(meta.get("sequence", "")) <= 5000
    ][: args.max_accessions]

    done: set[str] = set()
    if progress_path.exists():
        try:
            done = set(json.loads(progress_path.read_text()).get("done_accessions", []))
        except json.JSONDecodeError:
            done = set()
    high_conf = high_conf_accessions(out_path)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", encoding="utf-8", newline="") as out:
        for start in range(0, len(candidates), args.batch_size):
            batch = [x for x in candidates[start : start + args.batch_size] if x not in done]
            if not batch:
                continue
            if len(high_conf) >= args.target_highconf:
                break
            text = fetch_batch(batch, args.timeout)
            lines = [line for line in text.splitlines() if line.strip()]
            if not lines:
                done.update(batch)
                continue
            if write_header:
                out.write(lines[0] + "\n")
                write_header = False
            for line in lines[1:]:
                out.write(line + "\n")
            out.flush()
            done.update(batch)
            high_conf = high_conf_accessions(out_path)
            progress = {
                "done_accessions": sorted(done),
                "done_count": len(done),
                "target_highconf": args.target_highconf,
                "current_highconf": len(high_conf),
                "output": str(out_path),
                "last_batch_start": start,
            }
            progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({k: progress[k] for k in ["done_count", "current_highconf", "target_highconf"]}, ensure_ascii=False))
            time.sleep(args.sleep)

    meta_path = mobidb_dir / "mobidb_swissprot_disorder_batches.columns.json"
    with out_path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, [])
    meta_path.write_text(
        json.dumps(
            {
                "source_name": "MobiDB high-confidence disorder TSV batches",
                "url": "https://mobidb.org/api/download?format=tsv&acc=<comma-separated-accessions>",
                "field_notes": header,
                "target_highconf": args.target_highconf,
                "downloaded_accessions": len(done),
                "highconf_accessions": len(high_conf),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
