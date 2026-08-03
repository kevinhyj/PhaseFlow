
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_peptide_analysis_code_and_retained_results_are_separated() -> None:
    assert (ROOT / "scripts/peptide/analysis").is_dir()
    assert (ROOT / "artifacts/results/peptide/analysis").is_dir()
    assert not (ROOT / "research/peptide/analyses").exists()


def test_peptide_sources_do_not_reference_retired_layout_or_modules() -> None:
    retired_references = (
        "research/peptide/experiments",
        "research/peptide/analyses",
        "from phaseflow.data import",
        "from phaseflow.model import",
        "from phaseflow.tokenizer import",
        "from phaseflow.transformer import",
        "from phaseflow.utils import",
    )
    source_roots = (ROOT / "phaseflow", ROOT / "scripts/peptide", ROOT / "docs/peptide")

    for source_root in source_roots:
        for path in source_root.rglob("*"):
            if path.suffix not in {".md", ".py", ".sh", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            for reference in retired_references:
                assert reference not in text, f"{path} retains {reference!r}"
