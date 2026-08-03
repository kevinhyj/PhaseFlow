"""Protein sequence normalization and stable residue tokenization."""

from __future__ import annotations

import numpy as np


AA20 = "ACDEFGHIKLMNPQRSTVWY"
UNKNOWN_TOKEN = "X"
GAP_TOKEN = "-"

# Packed reproduction sidecars use one-based canonical residue ids.  Zero is
# reserved for non-canonical residues and gaps, matching the released arrays.
AA_TO_ID = {amino_acid: index + 1 for index, amino_acid in enumerate(AA20)}


class ProteinTokenizer:
    """Normalize protein sequences and encode the stable packed-sidecar ids."""

    def normalize(self, sequence: str) -> str:
        """Uppercase a sequence, remove whitespace, and mark unknown residues."""
        normalized = "".join(sequence.split()).upper()
        return "".join(
            residue if residue in AA_TO_ID or residue == GAP_TOKEN else UNKNOWN_TOKEN
            for residue in normalized
        )

    def encode(self, sequence: str) -> np.ndarray:
        """Encode canonical residues as 1--20 and unknown/gap residues as zero."""
        normalized = self.normalize(sequence)
        return np.asarray([AA_TO_ID.get(residue, 0) for residue in normalized], dtype=np.int16)
