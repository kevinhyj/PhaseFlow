import numpy as np

from phaseflow.full_length.features.physchem import compute_physchem_features


def test_physchem_shape_and_no_nan() -> None:
    features, names = compute_physchem_features("ACDX")
    assert features.shape == (4, len(names))
    assert features.shape[1] == 90
    assert not np.isnan(features).any()


def test_physchem_short_sequence() -> None:
    features, _ = compute_physchem_features("X")
    assert features.shape[0] == 1
    assert not np.isnan(features).any()
