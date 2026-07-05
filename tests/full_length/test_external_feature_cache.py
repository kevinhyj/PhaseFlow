import numpy as np
import h5py

from phaseflow.full_length.data.feature_cache import FeatureCacheReader
from phaseflow.full_length.features.build_features import build_feature_cache


def test_build_feature_cache_reads_starling512_and_distance_edges(tmp_path) -> None:
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">p1\nACDEFG\n")

    starling_embedding_dir = tmp_path / "starling_embedding"
    starling_embedding_dir.mkdir()
    np.savez_compressed(
        starling_embedding_dir / "p1.npz",
        protein_id=np.asarray("p1"),
        sequence=np.asarray("ACDEFG"),
        embedding=np.full((6, 512), 3.0, dtype=np.float32),
    )
    starling_distance_dir = tmp_path / "starling_distance"
    starling_distance_dir.mkdir()
    maps = np.full((4, 6, 6), 30.0, dtype=np.float32)
    maps[:, 1, 4] = 5.0
    maps[:, 4, 1] = 5.0
    with h5py.File(starling_distance_dir / "p1.h5", "w") as handle:
        handle.attrs["sequence"] = "ACDEFG"
        handle.create_dataset("distance_maps", data=maps)

    build_feature_cache(
        fasta=fasta,
        out_dir=tmp_path / "cache",
        starling_embedding_dir=starling_embedding_dir,
        starling_distance_dir=starling_distance_dir,
        graph_edge_dim=13,
        require_starling=True,
    )

    record = FeatureCacheReader.read_h5(tmp_path / "cache" / "p1.h5")
    assert record.protenix_embed.shape == (6, 512)
    assert record.starling_embed.shape == (6, 512)
    np.testing.assert_allclose(record.starling_embed, 3.0)
    assert record.modality_mask[:, 3].sum() == 6.0
    assert record.modality_mask[:, 4].sum() == 0.0
    assert 2 in set(record.edge_type.tolist())
    star_edges = record.edge_attr[record.edge_type == 2]
    assert star_edges.shape[1] == 13
    assert star_edges[:, 11].max() > 0.0


def test_build_feature_cache_masks_missing_embeddings(tmp_path) -> None:
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">p1\nACDEFG\n")

    build_feature_cache(fasta=fasta, out_dir=tmp_path / "cache")

    record = FeatureCacheReader.read_h5(tmp_path / "cache" / "p1.h5")
    assert record.protenix_embed.shape == (6, 512)
    assert record.starling_embed.shape == (6, 512)
    assert record.modality_mask[:, 3].sum() == 6.0
    assert record.modality_mask[:, 4].sum() == 6.0


def test_build_feature_cache_concats_protenix_embedding(tmp_path) -> None:
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">p1\nACDEFG\n")

    embedding_dir = tmp_path / "protenix_embedding"
    embedding_dir.mkdir()
    np.savez_compressed(
        embedding_dir / "p1.npz",
        s=np.ones((6, 3), dtype=np.float32),
        z=np.full((6, 2), 2.0, dtype=np.float32),
        single_mask=np.ones(6, dtype=np.float32),
        is_ligand=np.zeros(6, dtype=np.float32),
    )

    build_feature_cache(
        fasta=fasta,
        out_dir=tmp_path / "cache",
        protenix_embedding_dir=embedding_dir,
    )

    record = FeatureCacheReader.read_h5(tmp_path / "cache" / "p1.h5")
    assert record.protenix_embed.shape == (6, 5)
    np.testing.assert_allclose(record.protenix_embed[:, :3], 1.0)
    np.testing.assert_allclose(record.protenix_embed[:, 3:], 2.0)
    assert record.modality_mask[:, 3].sum() == 0.0
    assert record.reliability[:, 3].min() == 1.0
    assert record.structure_metadata["protenix_embedding_success"] == "1"
    assert record.structure_metadata["protenix_embedding_dim"] == "5"
