
import argparse
import math
import warnings
from pathlib import Path

import joblib
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_PSPHUNTER_REPO = Path("external_tools/PSPHunter")
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch PSPHunter driving-region inference.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--psphunter-repo", type=Path, default=DEFAULT_PSPHUNTER_REPO)
    parser.add_argument("--model-jobs", type=int, default=8)
    parser.add_argument("--batch-windows", type=int, default=200_000)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id = ""
    chunks: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    records.append((current_id, "".join(chunks).upper().replace("-", "").replace("*", "")))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id:
        records.append((current_id, "".join(chunks).upper().replace("-", "").replace("*", "")))
    return records


def load_word_vectors(path: Path) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 61 or parts[0] == "Uniprot":
                continue
            vectors[parts[0]] = np.asarray(parts[1:61], dtype=np.float32)
    return vectors


def trimer_vector(trimer: str, vectors: dict[str, np.ndarray], size: int = 60) -> np.ndarray:
    value = vectors.get(trimer)
    if value is None:
        return np.zeros(size, dtype=np.float32)
    return value


def deletion_window_features(sequence: str, vectors: dict[str, np.ndarray], bin_size: int = 20) -> np.ndarray:
    length = len(sequence)
    n_windows = length - bin_size + 1
    if n_windows <= 0:
        return np.zeros((0, 60), dtype=np.float32)

    n_trimers = max(0, length - 2)
    trimer_values = np.zeros((n_trimers, 60), dtype=np.float32)
    for index in range(n_trimers):
        trimer_values[index] = trimer_vector(sequence[index : index + 3], vectors)
    prefix = np.zeros((n_trimers + 1, 60), dtype=np.float32)
    if n_trimers:
        prefix[1:] = np.cumsum(trimer_values, axis=0)

    features = np.zeros((n_windows, 60), dtype=np.float32)
    total = prefix[n_trimers]
    for start in range(n_windows):
        left_end = max(start - 2, 0)
        right_start = min(start + bin_size, n_trimers)
        features[start] = prefix[left_end] + (total - prefix[right_start])
        for joined_start in (start - 2, start - 1):
            if joined_start < 0:
                continue
            chars: list[str] = []
            uses_left = False
            uses_right = False
            valid = True
            for deleted_pos in range(joined_start, joined_start + 3):
                if deleted_pos < start:
                    if deleted_pos >= length:
                        valid = False
                        break
                    chars.append(sequence[deleted_pos])
                    uses_left = True
                else:
                    original_pos = deleted_pos + bin_size
                    if original_pos >= length:
                        valid = False
                        break
                    chars.append(sequence[original_pos])
                    uses_right = True
            if valid and uses_left and uses_right:
                features[start] += trimer_vector("".join(chars), vectors)
    return features


def load_models(psphunter_repo: Path, model_jobs: int) -> list[object]:
    models = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for index in range(1, 101):
            model_path = psphunter_repo / "Trained_model" / str(index) / "word2vec70_60" / "train_model.m"
            model = joblib.load(model_path)
            if hasattr(model, "n_jobs"):
                model.n_jobs = model_jobs
            models.append(model)
    return models


def predict_average_probabilities(models: list[object], features: np.ndarray) -> np.ndarray:
    if features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    summed = np.zeros(features.shape[0], dtype=np.float64)
    for model in models:
        probabilities = model.predict_proba(features)[:, 1]
        summed += np.round(probabilities.astype(np.float64), 3)
    return np.round(summed / max(len(models), 1), 3).astype(np.float32)


def flush_feature_batch(
    handle,
    models: list[object],
    items: list[tuple[str, str, np.ndarray]],
) -> int:
    if not items:
        return 0
    lengths = [features.shape[0] for _, _, features in items]
    if sum(lengths) == 0:
        for protein_id, sequence, _ in items:
            write_record(handle, protein_id, sequence, np.zeros((0,), dtype=np.float32))
        return len(items)

    batch_features = np.concatenate([features for _, _, features in items if features.shape[0] > 0], axis=0)
    batch_probabilities = predict_average_probabilities(models, batch_features)
    offset = 0
    for protein_id, sequence, features in items:
        size = features.shape[0]
        probabilities = batch_probabilities[offset : offset + size] if size else np.zeros((0,), dtype=np.float32)
        write_record(handle, protein_id, sequence, probabilities)
        offset += size
    handle.flush()
    return len(items)


def selected_window_intervals(probabilities: np.ndarray, bin_size: int = 20) -> list[tuple[int, int]]:
    length = int(probabilities.shape[0])
    if length == 0:
        return []
    average = float(np.mean(probabilities))
    ranks = [(index + 1, average - float(probabilities[index])) for index in range(length)]
    ranks.sort(key=lambda item: item[1], reverse=True)
    if length > 2000:
        cutoff = int(length * 0.01)
    elif length > 1000:
        cutoff = int(length * 0.02)
    elif length > 500:
        cutoff = int(length * 0.04)
    else:
        cutoff = int(length * 0.05)
    selected = [key for offset, (key, _) in enumerate(ranks, start=1) if offset < cutoff]
    intervals = sorted((key, key + bin_size) for key in selected)
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def driving_positions(probabilities: np.ndarray, intervals: list[tuple[int, int]], limit: int = 3) -> set[int]:
    scores: list[tuple[float, int, int]] = []
    for start, end in intervals:
        values = []
        for key in range(start, end + 1):
            index = key - 1
            if 0 <= index < probabilities.shape[0]:
                values.append(float(probabilities[index]))
        if values:
            scores.append((float(np.mean(values)), start, end))
    scores.sort(key=lambda item: item[0])
    positions: set[int] = set()
    for _, start, end in scores[:limit]:
        positions.update(range(start, end + 1))
    return positions


def write_record(handle, protein_id: str, sequence: str, probabilities: np.ndarray) -> None:
    intervals = selected_window_intervals(probabilities)
    driving = driving_positions(probabilities, intervals)
    probability_by_key = {index + 1: float(value) for index, value in enumerate(probabilities)}

    handle.write(f"#Sequecing ID:{protein_id}\n")
    handle.write("#Residue in Purple denoted driving residues\n")
    handle.write("Pos\tAA\tProb\tDRegion\n")
    for zero_index, aa in enumerate(sequence):
        position = zero_index + 1
        key = zero_index - 9
        probability = probability_by_key.get(key)
        probability_text = "-" if probability is None or math.isnan(probability) else f"{probability:.3f}"
        flag = 1 if position in driving else 0
        handle.write(f"{position}\t{aa}\t{probability_text}\t{flag}\n")


def main() -> None:
    args = parse_args()
    records = read_fasta(args.input)
    if args.limit > 0:
        records = records[: args.limit]
    wordvec_path = args.psphunter_repo / "datasets" / "wordvec" / "uniprot_sprot70_size60.txt"
    vectors = load_word_vectors(wordvec_path)
    models = load_models(args.psphunter_repo, max(1, args.model_jobs))
    print(f"loaded_models={len(models)} records={len(records)} batch_windows={args.batch_windows}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        pending: list[tuple[str, str, np.ndarray]] = []
        pending_windows = 0
        processed = 0
        for offset, (protein_id, sequence) in enumerate(records, start=1):
            if not sequence or not set(sequence).issubset(VALID_AA):
                continue
            features = deletion_window_features(sequence, vectors)
            if pending and pending_windows + features.shape[0] > max(1, args.batch_windows):
                processed += flush_feature_batch(handle, models, pending)
                pending, pending_windows = [], 0
                print(f"processed={processed}", flush=True)
            pending.append((protein_id, sequence, features))
            pending_windows += int(features.shape[0])
        processed += flush_feature_batch(handle, models, pending)
        print(f"processed={processed}", flush=True)
    print(f"PSPHunter driving-region profiles written to {args.output}; records={len(records)}")


if __name__ == "__main__":
    main()
