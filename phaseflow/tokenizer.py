"""
Amino acid tokenizer for PhaseFlow.
"""

from typing import List, Union
import torch


class AminoAcidTokenizer:
    """Tokenizer for amino acid sequences with special tokens."""

    # 20 standard amino acids
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

    # Special token IDs
    PAD_ID = 20
    SOS_ID = 21
    EOS_ID = 22
    META_ID = 23   # Separates sequence from metadata
    SOM_ID = 24    # Start of modality (phase diagram)
    EOM_ID = 25    # End of modality

    # Vocabulary size (20 AA + 6 special + extra for shape encoding)
    VOCAB_SIZE = 64  # Extra space for shape info characters

    def __init__(self):
        # Build amino acid to ID mapping
        self.aa_to_id = {aa: i for i, aa in enumerate(self.AMINO_ACIDS)}
        self.id_to_aa = {i: aa for i, aa in enumerate(self.AMINO_ACIDS)}

        # Add special tokens to mapping
        self.special_tokens = {
            '<pad>': self.PAD_ID,
            '<sos>': self.SOS_ID,
            '<eos>': self.EOS_ID,
            '<meta>': self.META_ID,
            '<som>': self.SOM_ID,
            '<eom>': self.EOM_ID,
        }

        # Shape info uses ASCII offset (characters start at index 26)
        self.shape_offset = 26

    def encode_sequence(self, sequence: str) -> List[int]:
        """Encode an amino acid sequence to token IDs.

        Args:
            sequence: Amino acid sequence string (e.g., "ACDEF")

        Returns:
            List of token IDs
        """
        tokens = []
        for aa in sequence.upper():
            if aa in self.aa_to_id:
                tokens.append(self.aa_to_id[aa])
            else:
                raise ValueError(f"Unknown amino acid: {aa}")
        return tokens

    def decode_sequence(self, token_ids: List[int]) -> str:
        """Decode token IDs back to amino acid sequence.

        Args:
            token_ids: List of token IDs

        Returns:
            Amino acid sequence string
        """
        sequence = []
        for tid in token_ids:
            if tid in self.id_to_aa:
                sequence.append(self.id_to_aa[tid])
            elif tid in [self.PAD_ID, self.SOS_ID, self.EOS_ID,
                        self.META_ID, self.SOM_ID, self.EOM_ID]:
                continue  # Skip special tokens
            else:
                # Could be shape info, skip
                continue
        return ''.join(sequence)

    def encode_shape_info(self, shape: str) -> List[int]:
        """Encode shape information (e.g., '4x4') to token IDs.

        Args:
            shape: Shape string like '4x4'

        Returns:
            List of token IDs
        """
        return [ord(c) - ord('0') + self.shape_offset for c in shape]

    def build_input_sequence(self,
                             amino_seq: str,
                             shape: str = "4x4",
                             add_special_tokens: bool = True) -> List[int]:
        """Build full input sequence with special tokens.

        Format: [SOS] [amino tokens] [META] [shape] [SOM] ... [EOM] [EOS]

        Args:
            amino_seq: Amino acid sequence
            shape: Shape info string (default "4x4")
            add_special_tokens: Whether to add SOS/EOS

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.SOS_ID)

        # Add amino acid tokens
        tokens.extend(self.encode_sequence(amino_seq))

        # Add metadata separator and shape info
        tokens.append(self.META_ID)
        tokens.extend(self.encode_shape_info(shape))

        # Add SOM (phase diagram tokens will be added separately)
        tokens.append(self.SOM_ID)

        return tokens

    def pad_sequence(self,
                     tokens: List[int],
                     max_len: int,
                     padding_side: str = 'right') -> List[int]:
        """Pad sequence to max_len.

        Args:
            tokens: Input token IDs
            max_len: Maximum sequence length
            padding_side: 'left' or 'right'

        Returns:
            Padded token sequence
        """
        if len(tokens) >= max_len:
            return tokens[:max_len]

        pad_len = max_len - len(tokens)
        pad_tokens = [self.PAD_ID] * pad_len

        if padding_side == 'right':
            return tokens + pad_tokens
        else:
            return pad_tokens + tokens

    def batch_encode(self,
                     sequences: List[str],
                     max_len: int = None,
                     return_tensors: bool = True) -> Union[List[List[int]], torch.Tensor]:
        """Batch encode multiple sequences.

        Args:
            sequences: List of amino acid sequences
            max_len: Maximum length (if None, uses longest sequence)
            return_tensors: Whether to return PyTorch tensor

        Returns:
            Encoded sequences as list or tensor
        """
        encoded = [self.build_input_sequence(seq) for seq in sequences]

        if max_len is None:
            max_len = max(len(e) for e in encoded)

        padded = [self.pad_sequence(e, max_len) for e in encoded]

        if return_tensors:
            return torch.tensor(padded, dtype=torch.long)
        return padded

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.VOCAB_SIZE

    def __repr__(self) -> str:
        return f"AminoAcidTokenizer(vocab_size={self.vocab_size})"
