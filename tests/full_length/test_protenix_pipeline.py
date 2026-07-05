import json
from pathlib import Path

import numpy as np

from phaseflow.full_length.features.make_protenix_json import write_protenix_input_json
from phaseflow.full_length.features.structure_parser import parse_protenix_outputs


def test_write_protenix_input_json(tmp_path) -> None:
    path = write_protenix_input_json("p1", "ACDEFG", tmp_path, model_seeds=[101, 102])
    payload = json.loads(path.read_text())
    assert isinstance(payload, list)
    assert payload[0]["name"] == "p1"
    assert payload[0]["modelSeeds"] == [101, 102]
    assert payload[0]["sequences"][0]["proteinChain"]["sequence"] == "ACDEFG"


def test_parse_protenix_outputs_to_structure_npz(tmp_path) -> None:
    pred_dir = tmp_path / "output" / "p1" / "seed_101" / "predictions"
    pred_dir.mkdir(parents=True)
    _write_minimal_cif(pred_dir / "p1_sample_0.cif", "p1", "ACDEFG")
    (pred_dir / "p1_summary_confidence_sample_0.json").write_text(
        json.dumps({"plddt": 85.0, "gpde": 2.0, "ptm": 0.7, "ranking_score": 0.6, "has_clash": False})
    )

    written = parse_protenix_outputs(
        records=[("p1", "ACDEFG")],
        protenix_output=tmp_path / "output",
        out_dir=tmp_path / "features",
        contact_topk=2,
        contact_cutoff=8.0,
    )

    assert written == [tmp_path / "features" / "p1.npz"]
    with np.load(written[0], allow_pickle=False) as data:
        assert data["node"].shape == (6, 12)
        assert data["reliability"].shape == (6,)
        assert str(data["structure_provider"].item()) == "protenix"
        assert data["contacts"].shape[1] == 4


def _write_minimal_cif(path: Path, protein_id: str, sequence: str) -> None:
    three = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
    }
    lines = [
        f"data_{protein_id}",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_PDB_model_num",
    ]
    for index, aa in enumerate(sequence, start=1):
        lines.append(
            f"ATOM {index} C CA {three[aa]} A {index} {float(index * 3):.3f} 0.000 0.000 1.00 85.00 1"
        )
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")
