import numpy as np

from phaseflow.full_length.phaseflow_fusion import (
    PSSI_MAX,
    PSSI_MIN,
    PhaseFlowFusionConfig,
    fuse_llps_probability,
    fuse_phaseflow_with_phaseflow,
    gated_lift,
    parse_window_sizes,
    pssi_to_score,
    rank_blend,
    usable_window_sizes,
)


def test_pssi_to_score_inverts_phaseflow_direction() -> None:
    scores = pssi_to_score(np.asarray([PSSI_MIN, PSSI_MAX], dtype=np.float32))
    assert scores[0] > scores[1]
    assert np.allclose(scores, [1.0, 0.0])


def test_gated_lift_only_changes_borderline_phaseflow_residues() -> None:
    phaseflow = np.asarray([0.40, 0.62, 0.66, 0.72], dtype=np.float32)
    phaseflow_rank = np.asarray([1.00, 0.69, 0.80, 0.95], dtype=np.float32)
    fused = gated_lift(
        phaseflow_scores=phaseflow,
        phaseflow_rank=phaseflow_rank,
        phaseflow_low=0.60,
        phaseflow_high=0.68,
        phaseflow_rank_gate=0.70,
        lift=0.70,
        lift_span=0.05,
    )
    assert np.isclose(fused[0], phaseflow[0])
    assert np.isclose(fused[1], phaseflow[1])
    assert fused[2] > phaseflow[2]
    assert np.isclose(fused[3], phaseflow[3])


def test_llps_fusion_is_non_decreasing_and_respects_gate() -> None:
    config = PhaseFlowFusionConfig(llps_gate=0.72, llps_boost_scale=0.70, llps_max_phaseflow=0.80)
    unchanged = fuse_llps_probability(phaseflow_probability=0.45, phaseflow_proxy=0.60, config=config)
    boosted = fuse_llps_probability(phaseflow_probability=0.45, phaseflow_proxy=0.90, config=config)
    high_confidence = fuse_llps_probability(phaseflow_probability=0.85, phaseflow_proxy=0.90, config=config)
    assert np.isclose(unchanged, 0.45)
    assert boosted > 0.45
    assert np.isclose(high_confidence, 0.85)


def test_rank_blend_uses_phaseflow_local_rank_bidirectionally() -> None:
    phaseflow = np.asarray([0.80, 0.20], dtype=np.float32)
    phaseflow_rank = np.asarray([0.00, 1.00], dtype=np.float32)
    fused = rank_blend(phaseflow_scores=phaseflow, phaseflow_rank=phaseflow_rank, alpha=0.15)
    assert fused[0] < phaseflow[0]
    assert fused[1] > phaseflow[1]


def test_full_fusion_exports_phaseflow_profiles_and_metadata() -> None:
    config = PhaseFlowFusionConfig()
    phaseflow = np.asarray([0.61, 0.64, 0.66, 0.50], dtype=np.float32)
    phaseflow = np.asarray([0.10, 0.90, 0.95, 0.20], dtype=np.float32)
    result = fuse_phaseflow_with_phaseflow(
        phaseflow_dpr=phaseflow,
        phaseflow_llps_probability=0.40,
        phaseflow_scores=phaseflow,
        config=config,
        window_sizes=(10, 20),
    )
    assert result.dpr_scores.shape == phaseflow.shape
    assert result.phaseflow_scores.shape == phaseflow.shape
    assert result.phaseflow_rank.shape == phaseflow.shape
    assert result.window_sizes == (10, 20)
    assert result.lifted_residues >= 1
    assert result.suppressed_residues >= 1
    assert result.llps_probability >= 0.40


def test_window_size_parsing_and_short_sequence_fallback() -> None:
    assert parse_window_sizes("20,10,20") == (10, 20)
    assert usable_window_sizes("ACDEFGHIK", (10, 20), min_sequence_len=5) == (9,)
