import json

from phaseflow.full_length.features.af_parser import write_af3_input_json


def test_write_af3_input_json_no_msa_defaults(tmp_path) -> None:
    path = write_af3_input_json("p1", "ACDEFG", tmp_path)
    payload = json.loads(path.read_text())

    protein = payload["sequences"][0]["protein"]
    assert protein["sequence"] == "ACDEFG"
    assert protein["unpairedMsa"] == ""
    assert protein["pairedMsa"] == ""
    assert protein["templates"] == []


def test_write_af3_input_json_full_pipeline_omits_manual_msa(tmp_path) -> None:
    path = write_af3_input_json("p1", "ACDEFG", tmp_path, msa_mode="full_pipeline")
    protein = json.loads(path.read_text())["sequences"][0]["protein"]

    assert "unpairedMsa" not in protein
    assert "pairedMsa" not in protein
    assert "templates" not in protein
