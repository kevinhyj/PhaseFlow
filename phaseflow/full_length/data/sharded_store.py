from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phaseflow.full_length.data.runtime_guard import assert_offline_path_allowed


@dataclass
class Esm2ShardedStore:
    """Indexable ESM2 store backed by large `.npy` shards.

    Metadata columns are carried on the sample-index rows:
    `esm2_store_path`, `esm2_mask_store_path`, `esm2_store_offset`, `length`.
    The shard arrays stay fp16 on disk and are converted to float32 only for
    the existing model/collator contract.
    """

    required: bool = True
    _node_cache: dict[str, np.memmap] = field(default_factory=dict, init=False)
    _mask_cache: dict[str, np.memmap] = field(default_factory=dict, init=False)

    def read(self, row: Any, length: int) -> dict[str, np.ndarray] | None:
        node_path_value = row.get("esm2_store_path")
        mask_path_value = row.get("esm2_mask_store_path")
        if node_path_value is None or str(node_path_value) in {"", "nan", "None"}:
            if self.required:
                protein_id = str(row.get("protein_id", "<unknown>"))
                raise FileNotFoundError(f"Missing ESM2 sharded-store metadata for {protein_id}")
            return None
        node_path = Path(str(node_path_value))
        mask_path = Path(str(mask_path_value)) if mask_path_value is not None else None
        offset = int(row.get("esm2_store_offset", -1))
        if offset < 0:
            protein_id = str(row.get("protein_id", "<unknown>"))
            raise ValueError(f"Invalid ESM2 sharded-store offset for {protein_id}: {offset}")
        node_shard = self._open_node(node_path)
        node = np.asarray(node_shard[offset : offset + int(length)], dtype=np.float32)
        if node.shape != (int(length), 1280):
            protein_id = str(row.get("protein_id", "<unknown>"))
            raise ValueError(f"ESM2 sharded node shape mismatch for {protein_id}: {node.shape}")
        if mask_path is not None and str(mask_path) not in {"", "nan", "None"}:
            mask_shard = self._open_mask(mask_path)
            mask = np.asarray(mask_shard[offset : offset + int(length)], dtype=np.float32)
        else:
            mask = np.ones(int(length), dtype=np.float32)
        if mask.shape != (int(length),):
            protein_id = str(row.get("protein_id", "<unknown>"))
            raise ValueError(f"ESM2 sharded mask shape mismatch for {protein_id}: {mask.shape}")
        return {"node": node, "mask": mask}

    def _open_node(self, path: Path) -> np.memmap:
        key = str(path)
        if key not in self._node_cache:
            assert_offline_path_allowed(path)
            self._node_cache[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._node_cache[key]

    def _open_mask(self, path: Path) -> np.memmap:
        key = str(path)
        if key not in self._mask_cache:
            assert_offline_path_allowed(path)
            self._mask_cache[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._mask_cache[key]


@dataclass
class NpzMirrorStore:
    """Mmap mirror for arrays originally packed inside `.npz` shard files."""

    manifest: str | Path
    _paths: dict[tuple[str, str], str] = field(default_factory=dict, init=False)
    _cache: dict[str, np.memmap] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        manifest_path = Path(self.manifest)
        assert_offline_path_allowed(manifest_path)
        frame = pd.read_parquet(manifest_path)
        for _, row in frame.iterrows():
            array_name = str(row["array_name"])
            mirror_path = str(row["mirror_path"])
            for col in ("source_path", "source_abs_path"):
                value = str(row.get(col, ""))
                if value and value.lower() != "nan":
                    self._paths[(value, array_name)] = mirror_path

    def array(self, source_path: Any, array_name: str, *, dataset_root: Path | None = None) -> np.memmap | None:
        candidates = [str(source_path)]
        path = Path(str(source_path))
        if dataset_root is not None and not path.is_absolute():
            candidates.append(str(dataset_root / path))
        for candidate in candidates:
            mirror_path = self._paths.get((candidate, str(array_name)))
            if mirror_path is not None:
                return self._open(mirror_path)
        return None

    def _open(self, path_value: str) -> np.memmap:
        if path_value not in self._cache:
            path = Path(path_value)
            assert_offline_path_allowed(path)
            self._cache[path_value] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._cache[path_value]
