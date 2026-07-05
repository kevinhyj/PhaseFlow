import numpy as np

from phaseflow.full_length.infer import _evidence_for_sample, _public_regions


def test_public_regions_are_one_based_inclusive() -> None:
    regions = [{"start": 0, "end": 4, "score": 0.9, "source": "postprocess"}]
    assert _public_regions(regions) == [{"start": 1, "end": 5, "score": 0.9, "source": "postprocess"}]


def test_evidence_reports_available_modalities() -> None:
    weights = np.asarray(
        [[0.1, 0.2, 0.1, 0.35, 0.05, 0.2], [0.1, 0.2, 0.1, 0.35, 0.05, 0.2]],
        dtype=np.float32,
    )
    mask = np.zeros_like(weights)
    evidence = _evidence_for_sample(weights, mask, {"structure_provider": "protenix", "structure_success": "1"})
    assert evidence["important_modalities"][0] == "protenix_embed"
    assert evidence["structure_provider"] == "protenix"
