import numpy as np

from phaseflow.full_length.data.dpr_offline_labels import HalfOpenSpan, build_dpr_label_arrays


def test_half_open_span_end_boundary_is_end_minus_one() -> None:
    labels = build_dpr_label_arrays(
        length=32,
        spans=[HalfOpenSpan(start=10, end=20, confidence=1.0)],
        boundary_radius=0,
    )

    assert labels["residue_target"][9] == 0.0
    assert labels["residue_target"][10:20].tolist() == [1.0] * 10
    assert labels["residue_target"][20] == 0.0
    assert labels["residue_mask"][10:20].tolist() == [1.0] * 10
    assert labels["residue_mask"][20] == 0.0
    assert labels["start_target"][10] == 1.0
    assert labels["end_target"][19] == 1.0
    assert not np.isfinite(labels["end_target"][20])
    assert labels["boundary_weight"][10] == 1.0
    assert labels["boundary_weight"][19] == 1.0
    assert labels["boundary_weight"][20] == 0.0


def test_soft_boundary_does_not_shift_half_open_end() -> None:
    labels = build_dpr_label_arrays(
        length=32,
        spans=[HalfOpenSpan(start=10, end=20, confidence=0.75)],
        boundary_radius=2,
    )

    assert labels["end_target"][19] == 1.0
    assert labels["end_target"][20] < 1.0
    assert labels["end_target"][18] < 1.0
    assert labels["boundary_weight"][19] == 0.75
    assert labels["boundary_weight"][20] == 0.75
