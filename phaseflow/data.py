"""
Dataset and DataLoader for PhaseFlow.

数据格式说明:
- CSV: AminoAcidSequence + 16列 (group_11 到 group_44)
- NPZ: 预处理的 (N, 16) float32 数组
- 序列长度: 5-20 氨基酸
- 缺失值: ~62.6%, 用 NaN 表示
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
        data_path: str,
        tokenizer: Optional[AminoAcidTokenizer] = None,
        max_seq_len: int = 32,  # 序列长度5-20，加上特殊token后约25，留余量
        split: str = 'train',
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        seed: int = 42,
        normalize_phase: bool = False,  # 数据已经在[-1, 1]范围，通常不需要再归一化
        use_npz: bool = True,  # 优先使用预处理的NPZ文件
        missing_threshold: int = -1,  # 按缺失值划分阈值，-1表示不使用
    ):
        """Initialize the dataset.

        Args:
            data_path: Path to phase_diagram.csv or phase_diagram.npz
            tokenizer: AminoAcidTokenizer instance (creates one if None)
            max_seq_len: Maximum sequence length (default 32 for seq 5-20 + tokens)
            split: One of 'train', 'val', 'test'
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            seed: Random seed for splitting
            normalize_phase: Whether to normalize phase values
            use_npz: Whether to use NPZ file for faster loading
            missing_threshold: Threshold for missing value split
                - If >= 0: train uses missing_0 to missing_{threshold}, val/test uses missing_{threshold+1} to missing_15
                - If -1: use random split from single file
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer or AminoAcidTokenizer()
        self.max_seq_len = max_seq_len
        self.normalize_phase = normalize_phase
        self.missing_threshold = missing_threshold

        # 按缺失值划分模式
        if missing_threshold >= 0:
            self._init_by_missing(split, seed, missing_threshold)
        else:
            self._init_by_random_split(seed, train_ratio, val_ratio, use_npz, split)

        # 计算归一化统计量（如果需要）
        if normalize_phase:
            valid_values = self.phase_data[self.phase_mask]
            self.phase_mean = valid_values.mean()
            self.phase_std = valid_values.std()
            if self.phase_std < 1e-6:
                self.phase_std = 1.0
        else:
            self.phase_mean = 0.0
            self.phase_std = 1.0

        print(f"Dataset loaded: {len(self.indices)} samples ({split})")
        print(f"  Sequence length range: {min(len(s) for s in self.sequences)}-{max(len(s) for s in self.sequences)}")
        print(f"  Missing value ratio: {(~self.phase_mask).sum() / self.phase_mask.size:.1%}")

    def _init_by_missing(self, split: str, seed: int, threshold: int):
        """Initialize by missing value threshold split.

        Only use data with missing count <= threshold.
        Then split all remaining data into train/val/test (0.9/0.05/0.05).

        Args:
            split: One of 'train', 'val', 'test'
            seed: Random seed for shuffling
            threshold: Only use data with missing count <= threshold
        """
        # If data_path is a file (phase_diagram.csv), use its parent directory
        if self.data_path.is_file():
            base_dir = self.data_path.parent
        else:
            base_dir = self.data_path
        by_missing_dir = base_dir / 'by_missing'

        # Load all data with missing count <= threshold
        print(f"Loading data with missing <= {threshold}:")
        all_sequences = []
        all_phase_data = []
        all_phase_mask = []

        for i in range(0, threshold + 1):
            csv_path = by_missing_dir / f'missing_{i}.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_sequences.append(df['AminoAcidSequence'].values)
                phase_data = df[self.PHASE_COLUMNS].values.astype(np.float32)
                phase_mask = ~np.isnan(phase_data)
                phase_data = np.nan_to_num(phase_data, nan=0.0)
                all_phase_data.append(phase_data)
                all_phase_mask.append(phase_mask)
                print(f"  loaded missing_{i}.csv: {len(df)} samples")

        if not all_sequences:
            raise ValueError(f"No data found for missing 0 to {threshold}")

        # Merge all data
        self.sequences = np.concatenate(all_sequences)
        self.phase_data = np.concatenate(all_phase_data)
        self.phase_mask = np.concatenate(all_phase_mask)

        # Random shuffle
        np.random.seed(seed)
        n = len(self.sequences)
        indices = np.random.permutation(n)

        # Split into train/val/test (0.9/0.05/0.05)
        train_end = int(n * 0.9)
        val_end = int(n * 0.95)

        if split == 'train':
            self.indices = indices[:train_end]
        elif split == 'val':
            self.indices = indices[train_end:val_end]
        elif split == 'test':
            self.indices = indices[val_end:]
        elif split == 'all':
            self.indices = indices  # Use all data
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"  Total available: {n}")
        print(f"  {split.capitalize()} samples: {len(self.indices)}")

    def _init_by_random_split(self, seed: int, train_ratio: float, val_ratio: float, use_npz: bool, split: str):
        """Initialize by random split (original behavior)."""
        self.use_npz = use_npz  # Store for potential later use

        # 尝试加载 NPZ 文件（更快）
        npz_path = self.data_path.parent / 'phase_diagram.npz'
        csv_path = self.data_path  # 使用传入的 data_path

        if use_npz and npz_path.exists():
            print(f"Loading from NPZ: {npz_path}")
            npz_data = np.load(npz_path)
            self.phase_data = npz_data['data']  # (N, 16)
            # NPZ 中 NaN 表示缺失值
            self.phase_mask = ~np.isnan(self.phase_data)  # True = valid
            # 将 NaN 替换为 0（占位符）
            self.phase_data = np.nan_to_num(self.phase_data, nan=0.0)
            # 仍需从 CSV 读取序列
            df = pd.read_csv(csv_path)
            self.sequences = df['AminoAcidSequence'].values
        else:
            print(f"Loading from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            self.sequences = df['AminoAcidSequence'].values
            # 从 CSV 提取相图数据
            self.phase_data = df[self.PHASE_COLUMNS].values.astype(np.float32)
            self.phase_mask = ~np.isnan(self.phase_data)
            self.phase_data = np.nan_to_num(self.phase_data, nan=0.0)

        # 数据集划分
        np.random.seed(seed)
        n = len(self.sequences)
        indices = np.random.permutation(n)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == 'train':
            self.indices = indices[:train_end]
        elif split == 'val':
            self.indices = indices[train_end:val_end]
        elif split == 'test':
            self.indices = indices[val_end:]
        elif split == 'all':
            self.indices = indices  # Use all data
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns dict with:
            - input_ids: Token IDs for amino sequence
            - phase_values: 16-dim phase diagram vector
            - phase_mask: Binary mask (1 = valid, 0 = missing)
            - seq_len: Original sequence length
            - sequence: Original sequence string (for debugging)
        """
        row_idx = self.indices[idx]

        # Get amino acid sequence
        sequence = self.sequences[row_idx]

        # Encode sequence
        tokens = self.tokenizer.build_input_sequence(sequence)
        seq_len = len(tokens)

        # Pad sequence
        tokens = self.tokenizer.pad_sequence(tokens, self.max_seq_len)
        input_ids = torch.tensor(tokens, dtype=torch.long)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[:seq_len] = 1

        # Get phase diagram values (already preprocessed)
        phase_values = self.phase_data[row_idx].copy()  # (16,)
        phase_mask = self.phase_mask[row_idx].astype(np.float32)  # (16,)

        # Normalize phase values if needed
        if self.normalize_phase:
            # Only normalize valid values
            valid_idx = phase_mask == 1
            phase_values[valid_idx] = (phase_values[valid_idx] - self.phase_mean) / self.phase_std

        return {
            'input_ids': input_ids,
            'phase_values': torch.from_numpy(phase_values),
            'phase_mask': torch.from_numpy(phase_mask),
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
    data_path: str,
    batch_size: int = 64,
    split: str = 'train',
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    **dataset_kwargs
) -> DataLoader:
    """Create a DataLoader for the phase dataset.

    Args:
        data_path: Path to phase_diagram.csv or phase_diagram.npz
        batch_size: Batch size
        split: One of 'train', 'val', 'test'
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (default: True for train)
        **dataset_kwargs: Additional arguments for PhaseDataset
            - max_seq_len: Maximum sequence length (default 32)
            - normalize_phase: Whether to normalize (default False)
            - use_npz: Use NPZ for faster loading (default True)
            - use_missing_split: Split by missing values (default False)
                - True: train from missing_0.csv, val/test from missing_1-15.csv

    Returns:
        DataLoader instance
    """
    dataset = PhaseDataset(data_path, split=split, **dataset_kwargs)

    if shuffle is None:
        shuffle = (split == 'train')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # Enable for faster CPU->GPU transfer
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
