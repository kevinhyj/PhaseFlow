from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

PHASEMOTIF_PACKAGE = Path(
    os.environ.get(
        "PHASEMOTIF_PACKAGE",
        "external_tools/PhaseMotif/PhaseMotif",
    )
)
MODEL_PATH = Path(os.environ.get("PHASEMOTIF_MODEL_PATH", str(PHASEMOTIF_PACKAGE / "model_save" / "8.pth")))
AMINO = set("ACDEFGHIKLMNPQRSTVWYU")


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    names: list[str] = []
    sequences: list[str] = []
    current_name: str | None = None
    current_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    names.append(current_name)
                    sequences.append("".join(current_parts).upper())
                current_name = line[1:].split()[0]
                current_parts = []
            else:
                current_parts.append(line)
    if current_name is not None:
        names.append(current_name)
        sequences.append("".join(current_parts).upper())
    if not names:
        raise ValueError(f"No FASTA records found in {path}")
    return sequences, names


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PhaseMotif prediction over IDR FASTA records.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache") / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".cache") / "numba"))
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    print(
        "PhaseMotif runtime:"
        f" CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
        f" cuda_available={torch.cuda.is_available()}"
        f" device_count={torch.cuda.device_count()}",
        flush=True,
    )

    sys.path.insert(0, str(PHASEMOTIF_PACKAGE))
    from src.model import PredictMain
    from utils.seqTrans import seq2Matrix

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = PredictMain(
        cnn1out_channel=8,
        cnn1kernel=15,
        cnn1stride=1,
        cnn1padding=(0, 1),
        num_head=8,
        head_size=8,
        value_size=1,
        num_level=12,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    idr_list, idr_names = read_fasta(args.input)
    records: list[list[object]] = []
    pending: list[tuple[str, str]] = []
    for idr_name, idr in zip(idr_names, idr_list):
        if len(idr) < 50:
            records.append([idr_name, idr, f"Error: The length of {idr} is less than 50."])
        elif not set(idr).issubset(AMINO):
            records.append([idr_name, idr, f"Error: {idr} contains characters not in AMINO."])
        else:
            pending.append((idr_name, idr))

    batch_size = max(1, int(args.batch_size))
    with torch.no_grad():
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            names = [item[0] for item in batch]
            seqs = [item[1] for item in batch]
            one_hot = [torch.tensor(seq2Matrix(seq, "onehot")).unsqueeze(0).float() for seq in seqs]
            alphabet = [torch.tensor(seq2Matrix(seq, "alphabet")).unsqueeze(0).float() for seq in seqs]
            logits = model(one_hot, alphabet, device)
            scores = torch.sigmoid(logits.reshape(-1)).detach().cpu().tolist()
            for name, seq, score in zip(names, seqs, scores):
                records.append([name, seq, float(score)])

    result = pd.DataFrame(records, columns=["IDR Name", "IDR", "Predict Score"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"PhaseMotif predictions written to {args.output}; records={len(result)} valid={len(pending)} device={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
