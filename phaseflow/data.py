"""
Dataset and DataLoader for PhaseFlow.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
from pathlib import Path

from .tokenizer import AminoAcidTokenizer


class PhaseDataset(Dataset):
    """Dataset for amino acid sequences and phase diagrams."""

    # Column names for 4x4 phase diagram grid
    PHASE_COLUMNS = [
        'group_11', 'group_12', 'group_13', 'group_14',
        'group_21', 'group_22', 'group_23', 'group_24',
        'group_31', 'group_32', 'group_33', 'group_34',
        'group_41', 'group_42', 'group_43', 'group_44'
    ]

    def __init__(
        self,
        csv_path: str,
        tokenizer: Optional[AminoAcidTokenizer] = None,
        max_seq_len: int = 30,
        split: str = 'train',
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        seed: int = 42,
        normalize_phase: bool = True
    ):
        """Initialize the dataset.

        Args:
            csv_path: Path to phase_diagram.csv
            tokenizer: AminoAcidTokenizer instance (creates one if None)
            max_seq_len: Maximum sequence length
            split: One of 'train', 'val', 'test'
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            seed: Random seed for splitting
            normalize_phase: Whether to normalize phase values to [-1, 1]
        """
        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer or AminoAcidTokenizer()
        self.max_seq_len = max_seq_len
        self.normalize_phase = normalize_phase

        # Load data
        self.df = pd.read_csv(csv_path)

        # Split data
        np.random.seed(seed)
        n = len(self.df)
        indices = np.random.permutation(n)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == 'train':
            self.indices = indices[:train_end]
        elif split == 'val':
            self.indices = indices[train_end:val_end]
        elif split == 'test':
            self.indices = indices[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Precompute statistics for normalization
        if normalize_phase:
            phase_data = self.df[self.PHASE_COLUMNS].values
            self.phase_mean = np.nanmean(phase_data)
            self.phase_std = np.nanstd(phase_data)
            # Clamp std to avoid division by zero
            if self.phase_std < 1e-6:
                self.phase_std = 1.0
        else:
            self.phase_mean = 0.0
            self.phase_std = 1.0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns dict with:
            - input_ids: Token IDs for amino sequence
            - phase_values: 16-dim phase diagram vector
            - phase_mask: Binary mask (1 = valid, 0 = missing)
            - attention_mask: Attention mask for sequence
            - seq_len: Original sequence length
        """
        row_idx = self.indices[idx]
        row = self.df.iloc[row_idx]

        # Get amino acid sequence
        sequence = row['AminoAcidSequence']

        # Encode sequence
        tokens = self.tokenizer.build_input_sequence(sequence)
        seq_len = len(tokens)

        # Pad sequence
        tokens = self.tokenizer.pad_sequence(tokens, self.max_seq_len)
        input_ids = torch.tensor(tokens, dtype=torch.long)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[:seq_len] = 1

        # Get phase diagram values
        phase_values = []
        phase_mask = []

        for col in self.PHASE_COLUMNS:
            val = row[col]
            if pd.isna(val) or val == '':
                phase_values.append(0.0)  # Placeholder
                phase_mask.append(0)
            else:
                phase_values.append(float(val))
                phase_mask.append(1)

        phase_values = np.array(phase_values, dtype=np.float32)
        phase_mask = np.array(phase_mask, dtype=np.float32)

        # Normalize phase values
        if self.normalize_phase:
            # Only normalize valid values
            valid_idx = phase_mask == 1
            phase_values[valid_idx] = (phase_values[valid_idx] - self.phase_mean) / self.phase_std

        return {
            'input_ids': input_ids,
            'phase_values': torch.tensor(phase_values, dtype=torch.float32),
            'phase_mask': torch.tensor(phase_mask, dtype=torch.float32),
            'attention_mask': attention_mask,
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'sequence': sequence  # Keep original for debugging
        }

    def get_phase_stats(self) -> Tuple[float, float]:
        """Return phase diagram mean and std."""
        return self.phase_mean, self.phase_std

    def denormalize_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """Denormalize phase values back to original scale."""
        return phase * self.phase_std + self.phase_mean


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader.

    Handles variable-length sequences and string fields.
    """
    # Stack tensor fields
    result = {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'phase_values': torch.stack([b['phase_values'] for b in batch]),
        'phase_mask': torch.stack([b['phase_mask'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'seq_len': torch.stack([b['seq_len'] for b in batch]),
    }

    # Keep sequences as list
    result['sequences'] = [b['sequence'] for b in batch]

    return result


def create_dataloader(
    csv_path: str,
    batch_size: int = 64,
    split: str = 'train',
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    **dataset_kwargs
) -> DataLoader:
    """Create a DataLoader for the phase dataset.

    Args:
        csv_path: Path to phase_diagram.csv
        batch_size: Batch size
        split: One of 'train', 'val', 'test'
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (default: True for train)
        **dataset_kwargs: Additional arguments for PhaseDataset

    Returns:
        DataLoader instance
    """
    dataset = PhaseDataset(csv_path, split=split, **dataset_kwargs)

    if shuffle is None:
        shuffle = (split == 'train')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return loader


class BidirectionalSampler:
    """Sampler that alternates between forward and backward tasks.

    Forward task: sequence → phase diagram (flow matching)
    Backward task: phase diagram → sequence (language modeling)
    """

    def __init__(
        self,
        batch: Dict[str, torch.Tensor],
        forward_prob: float = 0.5
    ):
        """Initialize sampler.

        Args:
            batch: Batch from dataloader
            forward_prob: Probability of forward task
        """
        self.batch = batch
        self.forward_prob = forward_prob

    def sample_task(self) -> Tuple[str, Dict[str, torch.Tensor]]:
        """Sample a task direction.

        Returns:
            Tuple of (task_type, batch) where task_type is 'forward' or 'backward'
        """
        if torch.rand(1).item() < self.forward_prob:
            return 'forward', self.batch
        else:
            return 'backward', self.batch
