
import importlib.util


def test_peptide_implementation_has_a_dedicated_package() -> None:
    assert importlib.util.find_spec("phaseflow.peptide") is not None


def test_peptide_tokenizer_remains_available_from_the_public_package() -> None:
    from phaseflow import AminoAcidTokenizer
    from phaseflow.peptide.tokenizer import AminoAcidTokenizer as PackageTokenizer

    assert AminoAcidTokenizer is PackageTokenizer
