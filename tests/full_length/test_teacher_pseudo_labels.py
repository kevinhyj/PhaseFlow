import json

import h5py
import pandas as pd

from phaseflow.full_length.teachers.pseudo_label import build_teacher_pseudo_labels


def test_teacher_pseudo_labels_are_train_only_and_write_soft_profiles(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "protein_id": "P_TRAIN",
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "llps_label": -100,
                "sample_weight": 0.0,
            },
            {
                "protein_id": "P_VALID",
                "sequence": "YYYYYYYYYYACDEFGHIKLM",
                "llps_label": -100,
                "sample_weight": 0.0,
            },
        ]
    ).to_csv(manifest, index=False)

    regions = tmp_path / "regions.jsonl"
    regions.write_text(
        json.dumps(
            {
                "protein_id": "P_VALID",
                "regions": [
                    {
                        "start": 1,
                        "end": 4,
                        "region_type": "DPR_gold",
                        "region_label": "gold",
                        "confidence": 1.0,
                    }
                ],
            }
        )
        + "\n"
    )
    train_ids = tmp_path / "train_ids.txt"
    train_ids.write_text("P_TRAIN\n")

    out_dir = tmp_path / "teacher"
    raw_dir = out_dir / "raw"
    (raw_dir / "deephase").mkdir(parents=True)
    (raw_dir / "pscore").mkdir(parents=True)
    (raw_dir / "phasemotif").mkdir(parents=True)
    (raw_dir / "phaseflow").mkdir(parents=True)

    pd.DataFrame(
        [
            {"name": "P_TRAIN", "deephase_score": 0.95},
            {"name": "P_VALID", "deephase_score": 0.99},
        ]
    ).to_csv(raw_dir / "deephase" / "deephase.tsv", sep="\t", index=False)
    (raw_dir / "pscore" / "pscore.txt").write_text("PScore: 5.2 >P_TRAIN\nPScore: 5.5 >P_VALID\n")
    pd.DataFrame(
        [
            {"IDR Name": "P_TRAIN|start=2|end=12", "IDR": "CDEFGHIKLMN", "Predict Score": 0.9},
            {"IDR Name": "P_VALID|start=3|end=14", "IDR": "YYYYYYYYYYAC", "Predict Score": 0.9},
        ]
    ).to_csv(raw_dir / "phasemotif" / "phasemotif.csv", index=False)
    phaseflow_scores = [0.1] * 20
    phaseflow_scores[1:12] = [0.85] * 11
    (raw_dir / "phaseflow" / "phaseflow.jsonl").write_text(
        json.dumps({"record_id": "P_TRAIN", "length": 20, "score": phaseflow_scores}) + "\n"
        + json.dumps({"record_id": "P_VALID", "length": 20, "score": [0.9] * 20}) + "\n"
    )

    config = {
        "paths": {
            "manifest": str(manifest),
            "regions": str(regions),
            "train_ids_file": str(train_ids),
            "out_dir": str(out_dir),
        },
        "consensus": {
            "min_protein_teachers": 2,
            "min_protein_confidence": 0.6,
            "min_region_teachers": 1,
            "min_region_confidence": 0.6,
            "min_region_len": 8,
        },
        "predictors": {
            "deephase": {
                "enabled": True,
                "output": "{raw_dir}/deephase/deephase.tsv",
                "parser": "deephase_tsv",
                "threshold": 0.5,
                "direction": "high",
                "weight": 1.0,
            },
            "pscore": {
                "enabled": True,
                "output": "{raw_dir}/pscore/pscore.txt",
                "parser": "pscore_text",
                "threshold": 4.0,
                "direction": "high",
                "weight": 1.0,
            },
            "phasemotif": {
                "enabled": True,
                "output": "{raw_dir}/phasemotif/phasemotif.csv",
                "parser": "phasemotif_csv",
                "threshold": 0.5,
                "direction": "high",
                "weight": 1.0,
            },
            "phaseflow": {
                "enabled": True,
                "output": "{raw_dir}/phaseflow/phaseflow.jsonl",
                "parser": "phaseflow_window_jsonl",
                "threshold": 0.7,
                "direction": "high",
                "weight": 0.45,
                "min_region_len": 8,
                "merge_gap": 0,
            },
        },
    }

    report = build_teacher_pseudo_labels(config, run_predictors=False)

    assert report["protein_pseudo_labels"] == 1
    assert report["pseudo_regions"] == 1
    updated = pd.read_csv(out_dir / "manifest_with_teacher.csv")
    assert int(updated.loc[updated["protein_id"] == "P_TRAIN", "llps_label"].iloc[0]) == 1
    assert int(updated.loc[updated["protein_id"] == "P_VALID", "llps_label"].iloc[0]) == -100

    assert not (out_dir / "merged_regions.jsonl").exists()
    candidates = [json.loads(line) for line in (out_dir / "teacher_region_candidates.jsonl").read_text().splitlines()]
    assert {row["protein_id"] for row in candidates} == {"P_TRAIN"}
    with h5py.File(out_dir / "teacher_scores.h5", "r") as handle:
        assert "P_TRAIN" in handle
        assert "teacher_consensus" in handle["P_TRAIN"]
        assert "phaseflow_score" in handle["P_TRAIN"]
        assert handle["P_TRAIN"]["phaseflow_score"].shape == (20,)
        assert "P_VALID" not in handle
