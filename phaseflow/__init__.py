"""PhaseFlow package."""

__version__ = "0.1.0"
__all__ = ["PhaseFlow", "AminoAcidTokenizer", "PhaseDataset"]


def __getattr__(name):
    if name == "AminoAcidTokenizer":
        from .peptide.tokenizer import AminoAcidTokenizer
        return AminoAcidTokenizer
    if name == "PhaseFlow":
        from .peptide.model import PhaseFlow
        return PhaseFlow
    if name == "PhaseDataset":
        from .peptide.data import PhaseDataset
        return PhaseDataset
    raise AttributeError(f"module 'phaseflow' has no attribute {name!r}")
