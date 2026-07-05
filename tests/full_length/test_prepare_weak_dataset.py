import json

import pandas as pd

from phaseflow.full_length.data.prepare_weak_dataset import prepare_weak_dataset


def test_prepare_weak_dataset_writes_manifest_regions_and_splits(tmp_path) -> None:
    ppmc = tmp_path / "datasets.tsv"
    pd.DataFrame(
        [
            {
                "UniProt.Acc": "P_POS",
                "Gene.Name": "POS",
                "Datasets": "D-;DE",
                "GO": "C",
                "Frac.Order": 0.1,
                "Frac.Disorder": 0.9,
                "Full.seq": "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
            },
            {
                "UniProt.Acc": "P_NEG",
                "Gene.Name": "NEG",
                "Datasets": "ND",
                "GO": "C",
                "Frac.Order": 0.1,
                "Frac.Disorder": 0.9,
                "Full.seq": "YYYYYYYYYYACDEFGHIKLMNPQRSTV",
            },
        ]
    ).to_csv(ppmc, sep="\t", index=False)

    phasepro = tmp_path / "phasepro.json"
    phasepro.write_text(
        json.dumps(
            {
                "P_POS": {
                    "accession": "P_POS",
                    "sequence": "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
                    "boundaries": "2-5",
                    "segment": "test segment",
                    "gene": "POS",
                }
            }
        )
    )

    out_dir = tmp_path / "processed"
    report = prepare_weak_dataset(ppmc, phasepro, out_dir, max_records=None, seed=1)

    manifest = pd.read_csv(out_dir / "manifest.csv")
    assert set(manifest["protein_id"]) == {"P_POS", "P_NEG"}
    assert int(manifest.loc[manifest["protein_id"] == "P_POS", "llps_label"].iloc[0]) == 1
    assert int(manifest.loc[manifest["protein_id"] == "P_NEG", "llps_label"].iloc[0]) == 0
    assert (out_dir / "proteins.csv").exists()
    assert (out_dir / "protein_labels.csv").exists()
    assert (out_dir / "regions.csv").exists()
    assert (out_dir / "evidence.csv").exists()
    assert (out_dir / "source_map.csv").exists()

    region_rows = [json.loads(line) for line in (out_dir / "regions.jsonl").read_text().splitlines()]
    assert region_rows[0]["regions"][0]["start"] == 1
    assert region_rows[0]["regions"][0]["end"] == 4
    assert (out_dir / "splits" / "train_ids.txt").exists()
    assert report["total_records"] == 2
    assert report["phase1_tables"]["proteins"] == 2
    assert report["phase1_tables"]["protein_labels"] == 2


def test_prepare_weak_dataset_accepts_additional_sources_and_cd_code_csv_links(tmp_path) -> None:
    ppmc = tmp_path / "datasets.tsv"
    pd.DataFrame(
        [
            {
                "UniProt.Acc": "P_POS",
                "Gene.Name": "POS",
                "Datasets": "D-;DE",
                "GO": "C",
                "Frac.Order": 0.1,
                "Frac.Disorder": 0.9,
                "Full.seq": "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
            },
            {
                "UniProt.Acc": "P_NEG",
                "Gene.Name": "NEG",
                "Datasets": "ND",
                "GO": "C",
                "Frac.Order": 0.1,
                "Frac.Disorder": 0.9,
                "Full.seq": "YYYYYYYYYYACDEFGHIKLMNPQRSTV",
            },
        ]
    ).to_csv(ppmc, sep="\t", index=False)

    phasepro = tmp_path / "phasepro.json"
    phasepro.write_text(
        json.dumps(
            {
                "P_POS": {
                    "accession": "P_POS",
                    "sequence": "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
                    "boundaries": "2-5",
                    "segment": "test segment",
                    "gene": "POS",
                }
            }
        )
    )

    phasepdb = tmp_path / "phasepdb.csv"
    pd.DataFrame(
        [
            {
                "uniprot_id": "QPHA",
                "class_": "PS-self",
                "primary_name": "PHA",
            }
        ]
    ).to_csv(phasepdb, index=False)

    llpsdb = tmp_path / "llpsdb_positive.csv"
    pd.DataFrame(
        [
            {
                "uniprot_id": "QLLP",
                "sequence_clean": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN",
                "source_subset": "Phase_separation_Unambiguous",
                "protein_level_use": "silver_positive_candidate",
                "gene_name": "LLP",
            }
        ]
    ).to_csv(llpsdb, index=False)

    cd_code_proteins = tmp_path / "cd_code_proteins.csv"
    pd.DataFrame(
        [
            {
                "uniprot_id": "QCD01",
                "gene_name": "CD",
                "protein_level_use": "bronze_silver_condensate_member",
            }
        ]
    ).to_csv(cd_code_proteins, index=False)

    cd_code_links = tmp_path / "cd_code_links.csv"
    pd.DataFrame(
        [
            {
                "uniprotkb_ac": "QCD01",
                "condensate_id": "C1",
                "condensate_name": "Test condensate",
            }
        ]
    ).to_csv(cd_code_links, index=False)

    uniprot_fasta = tmp_path / "uniprot.fasta"
    uniprot_fasta.write_text(
        ">sp|P_POS|POS_HUMAN\n"
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIK\n"
        ">sp|P_NEG|NEG_HUMAN\n"
        "YYYYYYYYYYACDEFGHIKLMNPQRSTV\n"
        ">sp|QPHA|PHA_HUMAN\n"
        "MSTNPKPQRITAYYQQQGGGGGGGGGGGG\n"
        ">sp|QLLP|LLP_HUMAN\n"
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN\n"
        ">sp|QCD01|CD_HUMAN\n"
        "MSTNPKPQRITAYYQQQGGGGGGGGGGAA\n"
    )

    out_dir = tmp_path / "expanded"
    prepare_weak_dataset(
        ppmc,
        phasepro,
        out_dir,
        llpsdb_positive_csv=llpsdb,
        phasepdb_csv=phasepdb,
        cd_code_proteins_csv=cd_code_proteins,
        cd_code_links_csv=cd_code_links,
        uniprot_fasta=uniprot_fasta,
        max_records=None,
        seed=1,
    )

    manifest = pd.read_csv(out_dir / "manifest.csv")
    assert set(manifest["protein_id"]) == {"P_POS", "P_NEG", "QPHA", "QLLP", "QCD01"}
    assert manifest.loc[manifest["protein_id"] == "QPHA", "source"].iloc[0] == "PhaSepDB-3"
    assert manifest.loc[manifest["protein_id"] == "QLLP", "source"].iloc[0] == "LLPSDB-v2"
    assert manifest.loc[manifest["protein_id"] == "QCD01", "source"].iloc[0] == "CD-CODE"
    assert float(manifest.loc[manifest["protein_id"] == "QCD01", "sample_weight"].iloc[0]) == 0.22
    assert (out_dir / "proteins.csv").exists()
    assert (out_dir / "protein_labels.csv").exists()
    assert (out_dir / "regions.csv").exists()
    assert (out_dir / "evidence.csv").exists()
    assert (out_dir / "source_map.csv").exists()
