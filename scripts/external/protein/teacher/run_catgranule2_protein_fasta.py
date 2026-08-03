
import argparse
import csv
import multiprocessing as mp
import os
import sys
import types
from glob import glob
from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[4]
CAT_ROOT = ROOT_DIR / "external" / "teachers" / "catGRANULE2.0-v1.0.0" / "tartaglialabIIT-catGRANULE2.0-7420665"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(sequence: str) -> str:
    return "".join(residue if residue in VALID_AA else "A" for residue in sequence.upper())


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
                    sequence = clean_sequence("".join(chunks))
                    if sequence:
                        records.append((current_id, sequence))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        sequence = clean_sequence("".join(chunks))
        if sequence:
            records.append((current_id, sequence))
    return records


def compute_many(function, args: list[tuple[object, ...]], workers: int) -> list[object]:
    if workers <= 1:
        return [function(*item) for item in args]
    with mp.Pool(workers) as pool:
        return pool.starmap(function, args)


def fill_missing_feature_columns(frame: pd.DataFrame, columns: list[str], medians: pd.Series) -> pd.DataFrame:
    out = frame.reindex(columns=columns)
    for column in columns:
        fill_value = float(medians.get(column, 0.0))
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(fill_value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run catGRANULE2 as a protein-level teacher; writes no region/profile output.")
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
    from catgranuleFunctions import compute_chemphysProperties, predict
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
    joblib.load(classifiers_dir + "ONLY_PHYSCHEM/RandomForest/gridsearchCV_Object.pkl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["protein_id", "catgranule2_score", "protein_score", "sequence_length"]
    processed = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for chunk_start in range(0, len(records), chunk_size):
            chunk_ids = ids[chunk_start : chunk_start + chunk_size]
            chunk_sequences = sequences[chunk_start : chunk_start + chunk_size]
            property_args = [(sequence, scale_files, 7, False) for sequence in chunk_sequences]
            property_rows = compute_many(compute_chemphysProperties, property_args, workers)
            property_frame = pd.DataFrame(data=property_rows, columns=scale_names, index=chunk_ids)
            property_frame = fill_missing_feature_columns(property_frame, columns, medians)
            protein_scores = predict(property_frame, classifiers_dir, only_pc=True)["RandomForest"]
            for protein_id, sequence, protein_score in zip(chunk_ids, chunk_sequences, protein_scores):
                writer.writerow(
                    {
                        "protein_id": protein_id,
                        "catgranule2_score": float(protein_score),
                        "protein_score": float(protein_score),
                        "sequence_length": len(sequence),
                    }
                )
                processed += 1
            handle.flush()
            print(f"catGRANULE2 protein processed {processed}/{len(records)}", flush=True)
    print(f"catGRANULE2 protein scores written to {output_path}; records={processed} workers={workers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
