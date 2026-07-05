from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import types
from glob import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
CAT_ROOT = ROOT_DIR / "external" / "teachers" / "catGRANULE2.0-v1.0.0" / "tartaglialabIIT-catGRANULE2.0-7420665"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(chunks).upper()))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        records.append((current_id, "".join(chunks).upper()))
    return [(protein_id, clean_sequence(sequence)) for protein_id, sequence in records if clean_sequence(sequence)]


def clean_sequence(sequence: str) -> str:
    return "".join(residue if residue in VALID_AA else "A" for residue in sequence.upper())


def _compute_many(function, args: list[tuple[object, ...]], workers: int) -> list[object]:
    if workers <= 1:
        return [function(*item) for item in args]
    with mp.Pool(workers) as pool:
        return pool.starmap(function, args)


def _fill_missing_feature_columns(frame: pd.DataFrame, columns: list[str], medians: pd.Series) -> pd.DataFrame:
    out = frame.reindex(columns=columns)
    for column in columns:
        fill_value = float(medians.get(column, 0.0))
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(fill_value)
    return out


def _fill_missing_profile_rows(matrix: np.ndarray, names: list[str], columns: list[str], medians: pd.Series, length: int) -> pd.DataFrame:
    frame = pd.DataFrame(matrix, index=names).reindex(index=columns)
    for column in columns:
        if column not in frame.index:
            continue
        fill_value = float(medians.get(column, 0.0))
        values = pd.to_numeric(pd.Series(frame.loc[column]), errors="coerce").fillna(fill_value).to_numpy(dtype=np.float32)
        if values.shape[0] != length:
            resized = np.full(length, fill_value, dtype=np.float32)
            copy_len = min(length, values.shape[0])
            resized[:copy_len] = values[:copy_len]
            values = resized
        frame.loc[column] = values
    return frame.fillna(0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run catGRANULE2.0 ROBOT profiles over FASTA records.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache") / "matplotlib"))
    scales_link = CAT_ROOT / "src" / "ChemicalPhysicalScales_Py_dictionary"
    if not scales_link.exists():
        scales_link.symlink_to(Path("..") / "ChemicalPhysicalScales_Py_dictionary")

    os.chdir(CAT_ROOT)
    sys.path.insert(0, str(CAT_ROOT))
    sys.modules.setdefault("requests", types.ModuleType("requests"))
    sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))
    from catgranuleFunctions import (
        ComputeProfile_fromMatrix2,
        compute_chemphysProfiles,
        compute_chemphysProperties,
        predict,
        smooth,
    )
    from compute_profiles_and_predictions import correct_order_columns

    records = read_fasta(input_path)
    ids = [item[0] for item in records]
    sequences = [item[1] for item in records]
    scales_dir = "./src/ChemicalPhysicalScales_Py_dictionary"
    classifiers_dir = "./src/TRAINED_MODELS/"
    scale_files = glob(f"{scales_dir}/*")
    scale_names = [Path(path).name.replace(".json", "") for path in scale_files]
    columns = list(correct_order_columns[:82])
    train_data = pd.read_csv(CAT_ROOT / "DATASETS" / "TrainSet_data.csv")
    medians = train_data.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").median(numeric_only=True)

    workers = max(1, int(args.workers))
    chunk_size = max(1, int(args.chunk_size))
    rf_classifier = joblib.load(classifiers_dir + "ONLY_PHYSCHEM/RandomForest/gridsearchCV_Object.pkl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if output_path.suffix == ".gz" else open
    with opener(output_path, "wt", encoding="utf-8") as handle:
        for chunk_start in range(0, len(records), chunk_size):
            chunk_ids = ids[chunk_start : chunk_start + chunk_size]
            chunk_sequences = sequences[chunk_start : chunk_start + chunk_size]
            property_args = [(sequence, scale_files, 7, False) for sequence in chunk_sequences]
            property_rows = _compute_many(compute_chemphysProperties, property_args, workers)
            property_frame = pd.DataFrame(data=property_rows, columns=scale_names, index=chunk_ids)
            property_frame = _fill_missing_feature_columns(property_frame, columns, medians)
            protein_scores = predict(property_frame, classifiers_dir, only_pc=True)["RandomForest"]

            profile_args = [(sequence, scale_files, 1, False) for sequence in chunk_sequences]
            profile_matrices = _compute_many(compute_chemphysProfiles, profile_args, workers)
            for protein_id, sequence, matrix, protein_score in zip(chunk_ids, chunk_sequences, profile_matrices, protein_scores):
                profile_frame = _fill_missing_profile_rows(np.asarray(matrix), scale_names, columns, medians, len(sequence))
                raw_profile = np.asarray(ComputeProfile_fromMatrix2(profile_frame, rf_classifier), dtype=np.float32)
                smoothed = smooth(raw_profile, 21)
                max_value = float(np.nanmax(smoothed)) if np.isfinite(smoothed).any() else 0.0
                if max_value > 0:
                    profile = np.asarray(smoothed / max_value, dtype=np.float32) * float(protein_score)
                else:
                    profile = np.zeros(len(sequence), dtype=np.float32)
                handle.write(
                    json.dumps(
                        {
                            "protein_id": str(protein_id),
                            "protein_score": float(protein_score),
                            "sequence_length": len(sequence),
                            "score": [round(float(value), 6) for value in np.clip(profile, 0.0, 1.0)],
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            print(f"catGRANULE2.0 processed {min(chunk_start + chunk_size, len(records))}/{len(records)}", flush=True)
    print(f"catGRANULE2.0 profiles written to {output_path}; records={len(records)} workers={workers} chunk_size={chunk_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
