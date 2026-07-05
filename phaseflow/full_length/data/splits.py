from __future__ import annotations

from pathlib import Path


def resolve_split_ids(data_config: dict, split: str) -> list[str]:
    direct_key = f"{split}_ids"
    file_key = f"{split}_ids_file"
    if direct_key in data_config and data_config[direct_key]:
        return [str(protein_id) for protein_id in data_config[direct_key]]
    if file_key in data_config and data_config[file_key]:
        path = Path(data_config[file_key])
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    manifest = data_config.get("manifest")
    if manifest:
        import pandas as pd

        frame = pd.read_csv(manifest)
        if "split" not in frame.columns:
            raise ValueError(f"Manifest {manifest} has no split column; set {direct_key} or {file_key}")
        aliases = {split}
        if split == "valid":
            aliases.add("val")
        rows = frame.loc[frame["split"].astype(str).isin(aliases)]
        return [str(value) for value in rows["protein_id"].tolist()]
    raise ValueError(f"Could not resolve IDs for split '{split}'; set {direct_key}, {file_key}, or data.manifest")
