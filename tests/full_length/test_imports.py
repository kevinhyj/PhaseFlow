def test_imports() -> None:
    import phaseflow
    from phaseflow.full_length.models.phaseflow import PhaseFlowModel

    assert phaseflow.__version__
    assert PhaseFlowModel is not None
