"""Short-peptide sequence-to-phase modeling components."""

__all__ = ["AminoAcidTokenizer", "PhaseDataset", "PhaseFlow"]


def __getattr__(name: str):
    if name == "AminoAcidTokenizer":
        from .tokenizer import AminoAcidTokenizer

        return AminoAcidTokenizer
    if name == "PhaseDataset":
        from .data import PhaseDataset

        return PhaseDataset
    if name == "PhaseFlow":
        from .model import PhaseFlow

        return PhaseFlow
    raise AttributeError(f"module 'phaseflow.peptide' has no attribute {name!r}")
