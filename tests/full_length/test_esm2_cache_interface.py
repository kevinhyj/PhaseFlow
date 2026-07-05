import numpy as np

from phaseflow.full_length.features.build_features import build_feature_cache
from phaseflow.full_length.data.feature_cache import FeatureCacheReader


def test_build_feature_cache_from_precomputed_esm2(tmp_path) -> None:
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">p1\nACDEFG\n")
    esm2_dir = tmp_path / "esm2"
    esm2_dir.mkdir()
    embedding = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    np.savez_compressed(
        esm2_dir / "p1.npz",
        protein_id=np.asarray("p1"),
        sequence=np.asarray("ACDEFG"),
        length=np.asarray(6, dtype=np.int64),
        embedding_last_hidden_state=embedding,
        model_name=np.asarray("test-esm2"),
    )

    build_feature_cache(fasta=fasta, out_dir=tmp_path / "cache", mode="esm2", esm2_dir=esm2_dir)

    record = FeatureCacheReader.read_h5(tmp_path / "cache" / "p1.h5")
    assert record.plm.shape == (6, 5)
    np.testing.assert_allclose(record.plm, embedding)
    assert record.modality_mask[:, 0].sum() == 0.0
