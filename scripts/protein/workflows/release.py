"""Public contracts and source-first input builders for protein reproduction."""



# Source: protocol.py

"""Validation for fixed public training protocols."""


from dataclasses import dataclass
from typing import Iterable

import pandas as pd


IDENTITY_COLUMNS = ("protein_id", "sequence_sha256", "dataset_index")
PLAN_COLUMNS = (
    "epoch",
    "global_step",
    "local_rank",
    "local_slot",
    "plan_dataset_index",
    *IDENTITY_COLUMNS,
)


@dataclass(frozen=True)
class TrainingProtocolReport:
    records: int
    epochs: tuple[int, ...]


def validate_llps_training_protocol(
    training_units: pd.DataFrame,
    epoch_plans: Iterable[pd.DataFrame],
    *,
    world_size: int,
    batch_size: int,
) -> TrainingProtocolReport:
    """Require each fixed-plan row to resolve exactly to a public training row."""
    if world_size <= 0 or batch_size <= 0:
        raise ValueError("world_size and batch_size must be positive")
    _require_columns(training_units, IDENTITY_COLUMNS, "training_units")
    if training_units["dataset_index"].duplicated().any():
        raise ValueError("training_units.dataset_index must be unique")

    plans = [frame.reset_index(drop=True) for frame in epoch_plans]
    if not plans:
        raise ValueError("at least one epoch plan is required")
    plan = pd.concat(plans, ignore_index=True)
    _require_columns(plan, PLAN_COLUMNS, "training protocol")

    if (plan["plan_dataset_index"] < 0).any() or (plan["plan_dataset_index"] >= len(training_units)).any():
        raise ValueError("training protocol contains an out-of-range plan_dataset_index")
    positional = training_units.iloc[plan["plan_dataset_index"].to_numpy()].reset_index(drop=True)
    indexed = training_units.set_index("dataset_index").reindex(plan["dataset_index"].to_numpy()).reset_index()
    for source in (positional, indexed):
        for column in ("protein_id", "sequence_sha256"):
            if not source[column].astype(str).equals(plan[column].astype(str)):
                raise ValueError("training protocol identity does not match public training_units")

    expected_rows = int(world_size) * int(batch_size)
    grouped = plan.groupby(["epoch", "global_step"], sort=False)
    for (epoch, step), frame in grouped:
        if len(frame) != expected_rows:
            raise ValueError(f"training protocol epoch={epoch}, step={step} has {len(frame)} rows; expected {expected_rows}")
        if set(frame["local_rank"].astype(int)) != set(range(world_size)):
            raise ValueError(f"training protocol epoch={epoch}, step={step} has incomplete ranks")
        counts = frame.groupby("local_rank")["local_slot"].apply(lambda values: set(values.astype(int)))
        if any(slots != set(range(batch_size)) for slots in counts):
            raise ValueError(f"training protocol epoch={epoch}, step={step} has incomplete local slots")

    return TrainingProtocolReport(
        records=int(len(plan)),
        epochs=tuple(sorted(int(value) for value in plan["epoch"].unique())),
    )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")



# Source: release_paths.py

"""Portable paths for a PhaseFlow protein release run."""


from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleasePaths:
    """Resolve all release paths from explicit user-owned roots."""

    data_root: Path
    work_root: Path
    output_root: Path

    @classmethod
    def from_roots(
        cls,
        *,
        data_root: Path,
        work_root: Path,
        output_root: Path,
    ) -> "ReleasePaths":
        return cls(
            data_root=Path(data_root),
            work_root=Path(work_root),
            output_root=Path(output_root),
        )

    @property
    def llps_raw_root(self) -> Path:
        return self.data_root / "PhaseFlow-LLPS"

    @property
    def dpr_raw_root(self) -> Path:
        return self.data_root / "PhaseFlow-DPR"

    @property
    def llps_cache_root(self) -> Path:
        return self.work_root / "llps"

    @property
    def dpr_cache_root(self) -> Path:
        return self.work_root / "dpr"

    @property
    def dpr_packed_root(self) -> Path:
        return self.dpr_cache_root / "packed"

    @property
    def run_root(self) -> Path:
        return self.output_root



# Source: rebuild.py

"""Public, inspectable plan for rebuilding protein training inputs."""


from dataclasses import dataclass
from pathlib import Path



STAGES = (
    "validate",
    "features",
    "llps-inputs",
    "llps",
    "llps-hidden",
    "dpr-inputs",
    "dpr",
    "refinement",
    "evaluate",
)


@dataclass(frozen=True)
class RebuildPlan:
    """Describe, but do not execute, the public protein workflow."""

    paths: ReleasePaths

    @classmethod
    def from_roots(cls, data_root: Path, work_root: Path, output_root: Path) -> "RebuildPlan":
        return cls(ReleasePaths.from_roots(
            data_root=data_root,
            work_root=work_root,
            output_root=output_root,
        ))

    def stage_names(self) -> tuple[str, ...]:
        return STAGES

    def as_dict(self) -> dict[str, object]:
        return {
            "format": "phaseflow_protein_rebuild_plan_v1",
            "paths": {
                "data_root": str(self.paths.data_root),
                "work_root": str(self.paths.work_root),
                "output_root": str(self.paths.output_root),
                "llps_raw_root": str(self.paths.llps_raw_root),
                "dpr_raw_root": str(self.paths.dpr_raw_root),
                "llps_cache_root": str(self.paths.llps_cache_root),
                "dpr_cache_root": str(self.paths.dpr_cache_root),
            },
            "outputs": {
                "dpr_packed_sidecar": str(self.paths.dpr_packed_root),
            },
            "contracts": {
                "dpr_packed_hidden_key": "phaseflow_llps_hidden",
            },
            "stages": list(self.stage_names()),
        }



# Source: historical_ppmc.py

"""Freeze and validate the historical PPMC feature/label contract."""


import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


def _normalise_sequence(value: object) -> str:
    return str(value).strip().upper()


def _read_h5_record(path: Path, expected_protein_id: str) -> tuple[str, int]:
    if not path.is_file():
        raise ValueError(f"missing feature file for {expected_protein_id}: {path}")
    with h5py.File(path, "r") as handle:
        stored_id = str(handle.attrs.get("protein_id", ""))
        sequence = _normalise_sequence(handle.attrs.get("sequence", ""))
        stored_length = int(handle.attrs.get("length", len(sequence)))
    if stored_id != expected_protein_id:
        raise ValueError(
            f"protein_id mismatch for {expected_protein_id}: H5 stores {stored_id!r} in {path}"
        )
    if not sequence:
        raise ValueError(f"empty H5 sequence for {expected_protein_id}: {path}")
    if stored_length != len(sequence):
        raise ValueError(
            f"length mismatch inside H5 for {expected_protein_id}: attr={stored_length} sequence={len(sequence)}"
        )
    return sequence, stored_length


def _read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"protein_id": str})
    required = {"protein_id", "sequence", "length"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")
    frame = frame.copy()
    frame["protein_id"] = frame["protein_id"].astype(str)
    if frame["protein_id"].duplicated().any():
        examples = frame.loc[frame["protein_id"].duplicated(), "protein_id"].head(10).tolist()
        raise ValueError(f"manifest has duplicated protein_id values: {examples}")
    return frame


def materialize_h5_locked_manifest(
    source_manifest: Path | str,
    feature_dir: Path | str,
    output_manifest: Path | str,
) -> dict[str, Any]:
    """Write a manifest whose sequence fields match the frozen H5 payloads exactly."""

    source_manifest = Path(source_manifest)
    feature_dir = Path(feature_dir)
    output_manifest = Path(output_manifest)
    frame = _read_manifest(source_manifest)
    source_sequence_mismatches = 0
    sequences: list[str] = []
    lengths: list[int] = []
    hashes: list[str] = []
    for row in frame.itertuples(index=False):
        protein_id = str(row.protein_id)
        sequence, length = _read_h5_record(feature_dir / f"{protein_id}.h5", protein_id)
        if _normalise_sequence(row.sequence) != sequence:
            source_sequence_mismatches += 1
        sequences.append(sequence)
        lengths.append(length)
        hashes.append(hashlib.sha256(sequence.encode("ascii")).hexdigest())
    frame["sequence"] = sequences
    frame["length"] = lengths
    frame["sequence_sha256"] = hashes
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_manifest, index=False)
    return {
        "records": int(len(frame)),
        "source_sequence_mismatches": source_sequence_mismatches,
        "manifest": str(output_manifest),
        "feature_dir": str(feature_dir),
    }


def validate_h5_locked_manifest(manifest: Path | str, feature_dir: Path | str) -> dict[str, Any]:
    """Require every manifest record to agree with its frozen H5 feature record."""

    manifest = Path(manifest)
    feature_dir = Path(feature_dir)
    frame = _read_manifest(manifest)
    mismatches: list[str] = []
    for row in frame.itertuples(index=False):
        protein_id = str(row.protein_id)
        sequence, length = _read_h5_record(feature_dir / f"{protein_id}.h5", protein_id)
        manifest_sequence = _normalise_sequence(row.sequence)
        if manifest_sequence != sequence:
            mismatches.append(f"sequence mismatch for {protein_id}")
            continue
        if int(row.length) != length:
            mismatches.append(f"length mismatch for {protein_id}")
            continue
        if hasattr(row, "sequence_sha256"):
            expected_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            if str(row.sequence_sha256) != expected_hash:
                mismatches.append(f"sequence_sha256 mismatch for {protein_id}")
    if mismatches:
        raise ValueError("; ".join(mismatches[:10]))
    return {"records": int(len(frame)), "manifest": str(manifest), "feature_dir": str(feature_dir)}


def score_llps_panel(
    predictions: pd.DataFrame,
    panel_membership: pd.DataFrame,
    *,
    panel_id: str,
    score_column: str = "llps_score",
) -> dict[str, float | int | str]:
    """Compute the historical PPMC headline metrics for one explicitly named panel."""

    required_predictions = {"protein_id", score_column}
    required_panels = {"panel_id", "protein_id", "llps_label"}
    missing_predictions = sorted(required_predictions - set(predictions.columns))
    missing_panels = sorted(required_panels - set(panel_membership.columns))
    if missing_predictions:
        raise ValueError(f"predictions are missing required columns: {missing_predictions}")
    if missing_panels:
        raise ValueError(f"panel membership is missing required columns: {missing_panels}")
    prediction_frame = predictions[["protein_id", score_column]].copy()
    prediction_frame["protein_id"] = prediction_frame["protein_id"].astype(str)
    if prediction_frame["protein_id"].duplicated().any():
        raise ValueError("predictions have duplicated protein_id values")
    panel = panel_membership.loc[panel_membership["panel_id"].astype(str).eq(panel_id)].copy()
    if panel.empty:
        raise ValueError(f"unknown panel_id: {panel_id}")
    panel["protein_id"] = panel["protein_id"].astype(str)
    if panel["protein_id"].duplicated().any():
        raise ValueError(f"panel {panel_id} has duplicated protein_id values")
    merged = panel.merge(prediction_frame, on="protein_id", how="left", validate="one_to_one")
    labels = pd.to_numeric(merged["llps_label"], errors="raise")
    scores = pd.to_numeric(merged[score_column], errors="coerce")
    if scores.isna().any():
        missing_ids = merged.loc[scores.isna(), "protein_id"].head(10).tolist()
        raise ValueError(f"missing predictions for panel {panel_id}: {missing_ids}")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"panel {panel_id} has non-binary LLPS labels")
    labels_np = labels.astype(int).to_numpy()
    scores_np = scores.to_numpy(dtype=float)
    if len(np.unique(labels_np)) != 2:
        raise ValueError(f"panel {panel_id} must contain both LLPS classes")
    if len(np.unique(scores_np)) < 2:
        raise ValueError(f"panel {panel_id} has constant scores")
    predicted = (scores_np >= 0.5).astype(int)
    return {
        "panel_id": panel_id,
        "score_column": score_column,
        "n": int(len(merged)),
        "positive_n": int(labels_np.sum()),
        "negative_n": int(len(labels_np) - labels_np.sum()),
        "auroc": float(roc_auc_score(labels_np, scores_np)),
        "auprc": float(average_precision_score(labels_np, scores_np)),
        "mcc_at_0.5": float(matthews_corrcoef(labels_np, predicted)),
        "f1_at_0.5": float(f1_score(labels_np, predicted, zero_division=0)),
    }



# Source: llps_inputs.py

"""Compile locally regenerated feature caches into LLPS training inputs."""


import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from phaseflow.protein.contracts import FeatureCacheReader
from phaseflow.protein.features import compute_biophys_node
from phaseflow.protein.tokenizer import ProteinTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def compile_llps_inputs(
    *,
    release_root: str | Path,
    feature_root: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    """Create the offline dataset consumed by the fixed public LLPS plan.

    ``feature_root`` must contain one locally generated ``<protein_id>.h5``
    feature cache per record in the release table. The compiler keeps the
    public training-unit order and copies the released rank-local plans
    unchanged after identity validation.
    """

    release_root = Path(release_root).resolve()
    feature_root = Path(feature_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"LLPS output root must be empty: {output_root}")

    proteins = _read_table(release_root, "proteins")
    units = _read_table(release_root, "training_units")
    merged = _merge_release_tables(proteins, units)
    plan_paths, plans = _read_plans(release_root)
    _validate_plans(units, plans)

    processed = output_root / "processed"
    paths = {
        "tables": processed / "tables",
        "configs": processed / "configs",
        "esm2": processed / "features" / "esm2",
        "biophys": processed / "features" / "biophys",
        "protenix": processed / "features" / "protenix",
        "starling": processed / "features" / "starling",
        "graphs": processed / "graphs" / "merged_sparse",
        "plans": output_root / "training" / "plan",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        sample_rows.append(_compile_record(row, feature_root=feature_root, paths=paths))

    sample_index = paths["tables"] / "training_sample_index.parquet"
    pd.DataFrame(sample_rows).to_parquet(sample_index, index=False)
    for plan_path in plan_paths:
        shutil.copy2(plan_path, paths["plans"] / plan_path.name)

    contract = {
        "format": "phaseflow_protein_offline_input_v1",
        "dataset_root": str(processed),
        "sample_index": str(sample_index),
        "graph_contract": {"source": "merged_sparse", "edge_attr_dim": 32, "multigraph": True},
    }
    (paths["configs"] / "offline_input_contract.yaml").write_text(
        yaml.safe_dump(contract, sort_keys=False), encoding="utf-8"
    )
    _write_training_config(output_root=output_root, processed=processed, plan_dir=paths["plans"], sample_index=sample_index)
    return {"records": len(sample_rows), "epochs": sorted(_plan_epochs(plans)), "plans": len(plan_paths)}


def _read_table(release_root: Path, name: str) -> pd.DataFrame:
    path = release_root / "data" / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing public LLPS table: {path}")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"public LLPS table has no records: {path}")
    return frame


def _merge_release_tables(proteins: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    required_proteins = {"protein_id", "sequence", "sequence_sha256", "sequence_length"}
    required_units = {"protein_id", "sequence_sha256", "dataset_index"}
    if missing := sorted(required_proteins - set(proteins.columns)):
        raise ValueError(f"proteins.parquet is missing required columns: {missing}")
    if missing := sorted(required_units - set(units.columns)):
        raise ValueError(f"training_units.parquet is missing required columns: {missing}")
    if proteins["protein_id"].astype(str).duplicated().any() or units["dataset_index"].duplicated().any():
        raise ValueError("public LLPS tables contain duplicate protein_id or dataset_index values")

    merged = units.merge(
        proteins[["protein_id", "sequence", "sequence_sha256", "sequence_length"]],
        on=["protein_id", "sequence_sha256"],
        how="left",
        validate="one_to_one",
    )
    if merged["sequence"].isna().any():
        raise ValueError("training_units.parquet contains protein/sequence pairs absent from proteins.parquet")
    merged = merged.sort_values("dataset_index").reset_index(drop=True)
    expected = np.arange(len(merged), dtype=np.int64)
    if not np.array_equal(merged["dataset_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("training_units dataset_index must be a contiguous zero-based public row order")
    return merged


def _read_plans(release_root: Path) -> tuple[list[Path], list[pd.DataFrame]]:
    paths = sorted((release_root / "data" / "training_plan").glob("batch_plan_epoch_*.parquet"))
    if not paths:
        raise FileNotFoundError("public LLPS release has no fixed batch_plan_epoch_*.parquet files")
    return paths, [pd.read_parquet(path) for path in paths]


def _validate_plans(units: pd.DataFrame, plans: list[pd.DataFrame]) -> None:
    merged = pd.concat(plans, ignore_index=True)
    world_size = int(merged["local_rank"].astype(int).max()) + 1
    batch_size = int(merged["local_slot"].astype(int).max()) + 1
    validate_llps_training_protocol(units, plans, world_size=world_size, batch_size=batch_size)


def _compile_record(row: Any, *, feature_root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    protein_id = str(row.protein_id)
    sequence = str(row.sequence)
    expected_hash = str(row.sequence_sha256)
    if hashlib.sha256(sequence.encode("ascii")).hexdigest() != expected_hash:
        raise ValueError(f"public sequence hash mismatch for {protein_id}")
    cache_path = feature_root / f"{protein_id}.h5"
    if not cache_path.is_file():
        raise FileNotFoundError(f"missing local feature cache for {protein_id}: {cache_path}")
    record = FeatureCacheReader.read_h5(cache_path)
    if record.protein_id != protein_id or record.sequence != sequence:
        raise ValueError(f"local feature cache identity mismatch for {protein_id}")

    length = len(sequence)
    esm2_path = paths["esm2"] / f"{protein_id}.npz"
    np.savez_compressed(
        esm2_path,
        esm2_node=np.asarray(record.plm, dtype=np.float32),
        esm2_available_mask=_available_mask(record.modality_mask, column=0, length=length),
    )
    biophys_path = paths["biophys"] / f"{protein_id}.npz"
    biophys_node, _ = compute_biophys_node(sequence)
    np.savez_compressed(biophys_path, biophys_node=biophys_node)

    graph_path = paths["graphs"] / f"{protein_id}.npz"
    edge_attr = _fit_edge_width(record.edge_attr, width=32)
    np.savez_compressed(
        graph_path,
        edge_index=np.stack((record.edge_src, record.edge_dst), axis=0),
        edge_type=np.asarray(record.edge_type, dtype=np.int64),
        edge_scalar_attr=edge_attr,
    )

    protenix_present = _is_present(record.modality_mask, column=3)
    starling_present = _is_present(record.modality_mask, column=4)
    protenix_path = ""
    starling_paths = "[]"
    if protenix_present:
        path = paths["protenix"] / f"{protein_id}.npz"
        np.savez_compressed(path, protenix_node_embed=np.asarray(record.protenix_embed, dtype=np.float32))
        protenix_path = _relative_to_processed(path, paths["tables"].parent)
    if starling_present:
        path = paths["starling"] / f"{protein_id}.npz"
        np.savez_compressed(path, starling_node_embed=np.asarray(record.starling_embed, dtype=np.float32))
        starling_paths = f'["{_relative_to_processed(path, paths["tables"].parent)}"]'

    processed = paths["tables"].parent
    return {
        **row._asdict(),
        "sample_id": protein_id,
        "seq_len": length,
        "esm2_path": _relative_to_processed(esm2_path, processed),
        "biophys_shard": _relative_to_processed(biophys_path, processed),
        "biophys_offset": 0,
        "has_protenix": protenix_present,
        "protenix_node_shard": protenix_path,
        "protenix_node_offset": 0,
        "has_starling": starling_present,
        "starling_segment_ids": "[0]" if starling_present else "[]",
        "starling_start_0based": "[0]" if starling_present else "[]",
        "starling_end_exclusive_0based": f"[{length}]" if starling_present else "[]",
        "starling_node_shards": starling_paths,
        "starling_node_offsets": "[0]" if starling_present else "[]",
        "starling_segment_lengths": f"[{length}]" if starling_present else "[]",
        "merged_graph_shard": _relative_to_processed(graph_path, processed),
        "merged_graph_offset": 0,
        "graph_num_nodes": length,
        "graph_num_edges": int(len(record.edge_src)),
        "modality_mask_esm2": _is_present(record.modality_mask, column=0),
        "modality_mask_biophys": True,
        "modality_mask_protenix": protenix_present,
        "modality_mask_starling": starling_present,
        "reliability_esm2": _reliability(record.reliability, column=0),
        "reliability_biophys": 1.0,
        "reliability_protenix": _reliability(record.reliability, column=3),
        "reliability_starling": _reliability(record.reliability, column=4),
    }


def _available_mask(mask: np.ndarray, *, column: int, length: int) -> np.ndarray:
    if mask.ndim != 2 or mask.shape[0] != length or mask.shape[1] <= column:
        return np.ones(length, dtype=np.float32)
    return (np.asarray(mask[:, column], dtype=np.float32) < 0.5).astype(np.float32)


def _is_present(mask: np.ndarray, *, column: int) -> bool:
    return bool(np.any(_available_mask(mask, column=column, length=int(mask.shape[0])) > 0.0))


def _reliability(values: np.ndarray, *, column: int) -> float:
    if values.ndim != 2 or values.shape[1] <= column:
        return 0.0
    return float(np.nan_to_num(values[:, column], nan=0.0).mean())


def _fit_edge_width(values: np.ndarray, *, width: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    result = np.zeros((source.shape[0], width), dtype=np.float32)
    result[:, : min(width, source.shape[1])] = source[:, :width]
    return result


def _relative_to_processed(path: Path, processed: Path) -> str:
    return str(path.relative_to(processed))


def _plan_epochs(plans: list[pd.DataFrame]) -> set[int]:
    return {int(frame["epoch"].iloc[0]) for frame in plans}


def _write_training_config(*, output_root: Path, processed: Path, plan_dir: Path, sample_index: Path) -> None:
    template_path = REPOSITORY_ROOT / "configs" / "protein" / "llps.yaml"
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"LLPS configuration template is not a mapping: {template_path}")
    contract_path = processed / "configs" / "offline_input_contract.yaml"
    for section_name in ("data", "dataset"):
        section = config.setdefault(section_name, {})
        section["dataset_root"] = str(processed)
        section["sample_index"] = str(sample_index)
        section["input_contract"] = str(contract_path)
    config["dataset"].update(
        {
            "plan_dir": str(plan_dir),
            "esm2_store_metadata": None,
            "npz_mirror_manifest": None,
        }
    )
    config["output_dir"] = str(output_root / "runs" / "llps")
    (output_root / "llps.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")



# Source: packed.py

"""Write a portable DPR packed sidecar from validated PhaseFlow inputs."""


import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from phaseflow.protein.data import PhaseFlowCollator
from phaseflow.protein.data import PhaseFlowDataset
from phaseflow.protein.data import PHASEFLOW_LLPS_HIDDEN_KEY, RUNTIME_ARRAYS
from phaseflow.protein.contracts import IGNORE_INDEX
from phaseflow.protein.features import compute_biophys_node


_FLOAT16_ARRAYS = {
    "plm",
    "biophys",
    "modality_mask",
    "reliability",
    "edge_attr",
    PHASEFLOW_LLPS_HIDDEN_KEY,
}
_INT16_ARRAYS = {"aa_ids", "neighbor_edge_type"}
_INT64_ARRAYS = {"neighbors"}
_BOOL_ARRAYS = {"neighbor_mask"}
_RUNTIME_TRAILING_SHAPES = {
    "plm": (1280,),
    "biophys": (112,),
    "aa_ids": (),
    "modality_mask": (5,),
    "reliability": (5,),
    "neighbors": (96,),
    "edge_attr": (96, 32),
    "neighbor_mask": (96,),
    "neighbor_edge_type": (96,),
    PHASEFLOW_LLPS_HIDDEN_KEY: (256,),
    "residue_target": (),
    "residue_mask": (),
    "residue_weight": (),
    "core_target": (),
    "core_mask": (),
    "start_target": (),
    "end_target": (),
    "boundary_weight": (),
    "safe_background_mask": (),
    "ignore_mask": (),
}


@dataclass(frozen=True)
class PackedSidecarReport:
    """Summary of a validated PhaseFlow DPR sidecar."""

    output_root: Path
    hidden_key: str
    records: int
    residues: int
    manifest: pd.DataFrame


def build_packed_sidecar(
    *,
    records: Iterable[Mapping[str, object]],
    output_root: str | Path,
) -> PackedSidecarReport:
    """Pack residue-aligned DPR arrays with a PhaseFlow-native LLPS hidden state.

    Each record must contain ``protein_id``, ``sequence``, and an ``arrays``
    mapping for every runtime array.  The destination must be empty so a
    partial invocation cannot silently mix cache generations.
    """

    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Packed sidecar output must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    normalized = [_normalize_record(record) for record in records]
    if not normalized:
        raise ValueError("Packed sidecar requires at least one record")
    _validate_consistent_trailing_shapes(normalized)

    shard_path = output_root / "shard_00000"
    shard_path.mkdir()
    total_residues = sum(record["length"] for record in normalized)
    arrays = _allocate_arrays(shard_path, total_residues, normalized[0]["arrays"])
    manifest_rows: list[dict[str, object]] = []
    offset = 0
    for record in normalized:
        length = int(record["length"])
        end = offset + length
        for name in RUNTIME_ARRAYS:
            arrays[name][offset:end] = record["arrays"][name]
        manifest_rows.append(
            {
                "protein_id": record["protein_id"],
                "sequence": record["sequence"],
                "sequence_sha256": record["sequence_sha256"],
                "length": length,
                "shard_id": 0,
                "residue_offset": offset,
                "sidecar_path": shard_path.name,
                "hidden_key": PHASEFLOW_LLPS_HIDDEN_KEY,
                "hidden_dtype": str(record["arrays"][PHASEFLOW_LLPS_HIDDEN_KEY].dtype),
                "hidden_shape": json.dumps(list(record["arrays"][PHASEFLOW_LLPS_HIDDEN_KEY].shape)),
                "array_sha256": _record_arrays_sha256(record["arrays"]),
            }
        )
        offset = end
    _flush(arrays)

    file_hashes = {name: _sha256_file(shard_path / f"{name}.npy") for name in RUNTIME_ARRAYS}
    metadata = {
        "format": "phaseflow_dpr_packed_sidecar_v1",
        "hidden_key": PHASEFLOW_LLPS_HIDDEN_KEY,
        "records": len(normalized),
        "residues": total_residues,
        "arrays": list(RUNTIME_ARRAYS),
        "files": file_hashes,
    }
    (shard_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = pd.DataFrame(manifest_rows)
    shards = pd.DataFrame(
        [
            {
                "shard_id": 0,
                "path": shard_path.name,
                "num_proteins": len(normalized),
                "total_residues": total_residues,
                "sha256": _sha256_file(shard_path / "metadata.json"),
            }
        ]
    )
    _write_table(manifest, output_root / "manifest")
    _write_table(shards, output_root / "shards")
    (output_root / "sidecar_manifest.json").write_text(
        json.dumps(
            {
                "format": "phaseflow_dpr_packed_sidecar_v1",
                "hidden_key": PHASEFLOW_LLPS_HIDDEN_KEY,
                "records": len(normalized),
                "residues": total_residues,
                "manifest_sha256": _sha256_file(_table_path(output_root / "manifest")),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return PackedSidecarReport(
        output_root=output_root,
        hidden_key=PHASEFLOW_LLPS_HIDDEN_KEY,
        records=len(normalized),
        residues=total_residues,
        manifest=manifest,
    )


def build_packed_sidecar_from_feature_cache(
    *,
    feature_dir: str | Path,
    output_root: str | Path,
    llps_hidden_provider: Any,
    protein_ids: list[str] | None = None,
) -> PackedSidecarReport:
    """Create DPR runtime inputs from PhaseFlow feature caches and frozen LLPS states.

    ``llps_hidden_provider`` receives a one-protein ``PhaseFlowCollator`` batch
    and must return a ``[1, L, 256]`` tensor generated by the selected frozen
    LLPS checkpoint.  Keeping checkpoint loading outside this function makes
    the input boundary directly testable and prevents hidden model defaults.
    """

    dataset = PhaseFlowDataset(feature_dir=feature_dir, protein_ids=protein_ids, read_raw_edges=False)
    collator = PhaseFlowCollator(max_neighbors=96, edge_attr_dim=32, require_precomputed_graph=True)
    records: list[dict[str, object]] = []
    for index in range(len(dataset)):
        batch = collator([dataset[index]])
        length = int(batch["lengths"][0])
        sequence = str(batch["sequences"][0])
        biophys, _ = compute_biophys_node(sequence)
        _validate_feature_cache_biophys(batch, biophys, length, sequence)
        hidden = llps_hidden_provider(batch)
        if not torch.is_tensor(hidden) or hidden.shape != (1, length, 256):
            shape = tuple(hidden.shape) if torch.is_tensor(hidden) else type(hidden).__name__
            raise ValueError(f"LLPS hidden provider must return [1, {length}, 256], got {shape} for {sequence}")
        labels = _runtime_labels(batch, length)
        records.append(
            {
                "protein_id": str(batch["protein_ids"][0]),
                "sequence": sequence,
                "arrays": {
                    "plm": _batch_array(batch, "plm", length),
                    "biophys": biophys,
                    "aa_ids": _aa_ids(sequence),
                    "modality_mask": _batch_array(batch, "modality_mask", length),
                    "reliability": _batch_array(batch, "reliability", length),
                    "neighbors": _batch_array(batch, "neighbors", length),
                    "edge_attr": _batch_array(batch, "edge_attr", length),
                    "neighbor_mask": _batch_array(batch, "neighbor_mask", length),
                    "neighbor_edge_type": _batch_array(batch, "neighbor_edge_type", length),
                    PHASEFLOW_LLPS_HIDDEN_KEY: hidden[0].detach().cpu().numpy(),
                    **labels,
                },
            }
        )
    return build_packed_sidecar(records=records, output_root=output_root)


def make_llps_hidden_provider(*, checkpoint: str | Path, device: str | torch.device) -> Any:
    """Load a frozen LLPS checkpoint and expose its DPR residue representation."""

    from phaseflow.protein.model import load_phaseflow_llps_checkpoint

    resolved_device = torch.device(device)
    model, _ = load_phaseflow_llps_checkpoint(checkpoint, device=resolved_device)

    def provider(batch: Mapping[str, Any]) -> torch.Tensor:
        model_batch = {
            name: value.to(resolved_device) if torch.is_tensor(value) else value
            for name, value in batch.items()
        }
        with torch.inference_mode():
            outputs = model(model_batch)
        return extract_llps_hidden(outputs, seq_mask=model_batch["seq_mask"]).cpu()

    return provider


def extract_llps_hidden(outputs: Mapping[str, Any], *, seq_mask: torch.Tensor) -> torch.Tensor:
    """Use the published mean of LLPS and DPR-aligned frozen residue taps."""

    names = ("llps_residue_repr", "dpr_residue_repr")
    layers = [outputs[name].detach().float() for name in names if torch.is_tensor(outputs.get(name))]
    if not layers:
        raise KeyError(f"LLPS output does not contain any of {names}")
    hidden = torch.stack(layers, dim=0).mean(dim=0)
    if hidden.ndim != 3 or hidden.shape[-1] != 256:
        raise ValueError(f"LLPS hidden state must have shape [B, L, 256], got {tuple(hidden.shape)}")
    return hidden.masked_fill(~seq_mask.bool().unsqueeze(-1), 0.0)


def validate_packed_sidecar(output_root: str | Path) -> dict[str, int]:
    """Verify the immutable identity and file-integrity contract of a sidecar."""

    output_root = Path(output_root)
    sidecar_manifest_path = output_root / "sidecar_manifest.json"
    if not sidecar_manifest_path.exists():
        raise ValueError(f"Packed sidecar manifest is missing: {sidecar_manifest_path}")
    summary = json.loads(sidecar_manifest_path.read_text(encoding="utf-8"))
    if summary.get("hidden_key") != PHASEFLOW_LLPS_HIDDEN_KEY:
        raise ValueError("Packed sidecar must declare phaseflow_llps_hidden")
    manifest = _read_packed_table(output_root / "manifest")
    shards = _read_packed_table(output_root / "shards")
    required = {
        "protein_id",
        "sequence",
        "sequence_sha256",
        "length",
        "shard_id",
        "residue_offset",
        "hidden_key",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Packed sidecar manifest is missing columns: {missing}")
    if manifest.duplicated(["protein_id", "sequence_sha256"]).any():
        raise ValueError("Packed sidecar manifest has duplicate protein identities")
    calculated = manifest["sequence"].astype(str).map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    if not calculated.eq(manifest["sequence_sha256"].astype(str)).all():
        raise ValueError("Packed sidecar sequence_sha256 mismatch")
    if not manifest["length"].astype(int).eq(manifest["sequence"].astype(str).str.len()).all():
        raise ValueError("Packed sidecar sequence length mismatch")
    if not manifest["hidden_key"].astype(str).eq(PHASEFLOW_LLPS_HIDDEN_KEY).all():
        raise ValueError("Packed sidecar contains a non-PhaseFlow hidden key")
    manifest_path = _table_path(output_root / "manifest")
    if summary.get("manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError("Packed sidecar manifest hash mismatch")
    for shard in shards.itertuples(index=False):
        shard_path = output_root / str(shard.path)
        metadata_path = shard_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Packed sidecar shard metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("hidden_key") != PHASEFLOW_LLPS_HIDDEN_KEY:
            raise ValueError(f"Packed sidecar shard has a non-PhaseFlow hidden key: {shard_path}")
        for name in RUNTIME_ARRAYS:
            path = shard_path / f"{name}.npy"
            if not path.exists():
                raise ValueError(f"Packed sidecar array is missing: {path}")
            actual = _sha256_file(path)
            if metadata.get("files", {}).get(name) != actual:
                raise ValueError(f"Packed sidecar array hash mismatch: {path}")
    return {"records": int(len(manifest)), "residues": int(manifest["length"].astype(int).sum())}


def _validate_feature_cache_biophys(batch: Mapping[str, Any], biophys: np.ndarray, length: int, sequence: str) -> None:
    physchem = _batch_array(batch, "physchem", length)
    disorder = _batch_array(batch, "disorder", length)
    if physchem.shape != (length, 90) or disorder.shape != (length, 6):
        raise ValueError(
            f"Feature cache must contain physchem [L,90] and disorder [L,6] for DPR, "
            f"got {physchem.shape} and {disorder.shape} for {sequence}"
        )
    if not np.allclose(biophys[:, :90], physchem) or not np.allclose(biophys[:, 90:96], disorder):
        raise ValueError(f"Feature cache biophys values do not match public deterministic reconstruction for {sequence}")


def _runtime_labels(batch: Mapping[str, Any], length: int) -> dict[str, np.ndarray]:
    raw = _batch_array(batch, "y_dpr", length).astype(np.int64, copy=False)
    keys = _batch_array(batch, "y_key", length).astype(np.int64, copy=False)
    weights = _batch_array(batch, "y_weight", length).astype(np.float32, copy=False)
    residue_mask = raw != IGNORE_INDEX
    core_mask = keys != IGNORE_INDEX
    target = (raw > 0).astype(np.float32) * residue_mask.astype(np.float32)
    core_target = (keys > 0).astype(np.float32) * core_mask.astype(np.float32)
    starts = target.copy()
    starts[1:] *= 1.0 - target[:-1]
    ends = target.copy()
    ends[:-1] *= 1.0 - target[1:]
    return {
        "residue_target": target,
        "residue_mask": residue_mask.astype(np.float32),
        "residue_weight": weights * residue_mask.astype(np.float32),
        "core_target": core_target,
        "core_mask": core_mask.astype(np.float32),
        "start_target": starts,
        "end_target": ends,
        "boundary_weight": residue_mask.astype(np.float32),
        "safe_background_mask": (residue_mask & ~target.astype(bool)).astype(np.float32),
        "ignore_mask": (~residue_mask).astype(np.float32),
    }


def _batch_array(batch: Mapping[str, Any], name: str, length: int) -> np.ndarray:
    value = batch[name]
    if not torch.is_tensor(value):
        raise ValueError(f"Feature batch value {name!r} must be a tensor")
    return value[0, :length].detach().cpu().numpy()


def _aa_ids(sequence: str) -> np.ndarray:
    return ProteinTokenizer().encode(sequence)


def _normalize_record(record: Mapping[str, object]) -> dict[str, Any]:
    try:
        protein_id = str(record["protein_id"])
        sequence = str(record["sequence"])
        raw_arrays = record["arrays"]
    except KeyError as exc:
        raise ValueError(f"Packed record is missing {exc.args[0]!r}") from exc
    if not protein_id or not sequence:
        raise ValueError("Packed records require non-empty protein_id and sequence")
    if not isinstance(raw_arrays, Mapping):
        raise ValueError(f"Packed record arrays must be a mapping for {protein_id}")
    extra = sorted(set(raw_arrays) - set(RUNTIME_ARRAYS))
    missing = sorted(set(RUNTIME_ARRAYS) - set(raw_arrays))
    if missing or extra:
        raise ValueError(f"Packed arrays for {protein_id} differ from runtime contract: missing={missing}, extra={extra}")
    length = len(sequence)
    arrays = {name: _coerce_array(name, raw_arrays[name], length, protein_id) for name in RUNTIME_ARRAYS}
    sequence_sha256 = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    supplied_hash = record.get("sequence_sha256")
    if supplied_hash is not None and str(supplied_hash) != sequence_sha256:
        raise ValueError(f"sequence_sha256 mismatch for {protein_id}")
    return {
        "protein_id": protein_id,
        "sequence": sequence,
        "sequence_sha256": sequence_sha256,
        "length": length,
        "arrays": arrays,
    }


def _coerce_array(name: str, value: object, length: int, protein_id: str) -> np.ndarray:
    if name in _FLOAT16_ARRAYS:
        dtype = np.float16
    elif name in _INT16_ARRAYS:
        dtype = np.int16
    elif name in _INT64_ARRAYS:
        dtype = np.int64
    elif name in _BOOL_ARRAYS:
        dtype = np.bool_
    else:
        dtype = np.float32
    array = np.asarray(value, dtype=dtype)
    if array.ndim < 1 or array.shape[0] != length:
        raise ValueError(f"{name} must have leading length {length} for {protein_id}, got {array.shape}")
    expected = _RUNTIME_TRAILING_SHAPES[name]
    if array.shape[1:] != expected:
        raise ValueError(
            f"{name} must have shape ({length}, {', '.join(str(value) for value in expected)}) "
            f"for DPR, got {array.shape} for {protein_id}"
        )
    return array


def _validate_consistent_trailing_shapes(records: list[dict[str, Any]]) -> None:
    reference = records[0]["arrays"]
    for record in records[1:]:
        for name in RUNTIME_ARRAYS:
            expected = reference[name].shape[1:]
            observed = record["arrays"][name].shape[1:]
            if observed != expected:
                raise ValueError(f"{name} trailing shape mismatch for {record['protein_id']}: {observed} != {expected}")


def _allocate_arrays(path: Path, residues: int, reference: Mapping[str, np.ndarray]) -> dict[str, np.memmap]:
    return {
        name: np.lib.format.open_memmap(
            path / f"{name}.npy",
            mode="w+",
            dtype=reference[name].dtype,
            shape=(residues, *reference[name].shape[1:]),
        )
        for name in RUNTIME_ARRAYS
    }


def _flush(arrays: Mapping[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def _record_arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in RUNTIME_ARRAYS:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_table(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(stem.with_suffix(".parquet"), index=False)
    except (ImportError, ModuleNotFoundError):
        pass


def _table_path(stem: Path) -> Path:
    parquet = stem.with_suffix(".parquet")
    return parquet if parquet.exists() else stem.with_suffix(".csv")


def _read_packed_table(stem: Path) -> pd.DataFrame:
    parquet = stem.with_suffix(".parquet")
    if parquet.exists():
        return pd.read_parquet(parquet)
    csv = stem.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise ValueError(f"Packed sidecar table is missing: {stem}")



# Source: phasepro.py

"""Identity validation for frozen official PhasePro evaluation inputs."""


from typing import Any

import pandas as pd

from scripts.protein.workflows.training import normalize_checkpoint_namespace


def validate_release_sidecar_pairs(
    release: pd.DataFrame,
    sidecar: pd.DataFrame,
    *,
    expected_count: int = 121,
) -> dict[str, Any]:
    release_required = {"protein_id", "sequence_sha256", "sequence_length"}
    sidecar_required = {"protein_id", "sequence_sha256", "length"}
    if missing := sorted(release_required - set(release.columns)):
        raise ValueError(f"release table missing columns: {missing}")
    if missing := sorted(sidecar_required - set(sidecar.columns)):
        raise ValueError(f"sidecar table missing columns: {missing}")
    left = release[["protein_id", "sequence_sha256", "sequence_length"]].copy()
    right = sidecar[["protein_id", "sequence_sha256", "length"]].copy()
    left["protein_id"] = left["protein_id"].astype(str)
    right["protein_id"] = right["protein_id"].astype(str)
    merged = left.merge(right, on="protein_id", how="outer", suffixes=("_release", "_sidecar"), indicator=True)
    bad = merged.loc[
        merged["_merge"].ne("both")
        | merged["sequence_sha256_release"].ne(merged["sequence_sha256_sidecar"])
        | merged["sequence_length"].ne(merged["length"])
    ]
    if len(left) != expected_count or len(right) != expected_count or len(merged) != expected_count or not bad.empty:
        raise ValueError(
            f"PhasePro release/sidecar identity mismatch: release={len(left)} sidecar={len(right)} "
            f"merged={len(merged)} mismatches={len(bad)}"
        )
    return {"release_pairs": int(len(left)), "sidecar_pairs": int(len(right)), "matched_pairs": int(len(merged))}
"""Validate a protein release package and write its training manifest."""


import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

# Support direct invocation from a source checkout without requiring installation.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.protein.workflows.release import validate_llps_training_protocol


REQUIRED_COLUMNS = {
    "llps": {
        "proteins": {"protein_id", "sequence_sha256", "sequence", "sequence_length"},
        "training_units": {"protein_id", "sequence_sha256", "llps_label", "sample_weight", "dataset_index"},
    },
    "dpr": {
        "proteins": {"protein_id", "sequence_sha256", "sequence", "sequence_length"},
        "training_units": {"protein_id", "sequence_sha256", "training_stage"},
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a protein dataset package and write a training manifest.")
    parser.add_argument("--task", choices=tuple(REQUIRED_COLUMNS), required=True)
    parser.add_argument("--package-root", type=Path, required=True, help="Directory containing data/ and metadata/.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON manifest.")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_required_table(package_root: Path, name: str, columns: set[str]) -> tuple[Path, pd.DataFrame]:
    path = package_root / "data" / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"required dataset table is missing: {path}")
    frame = pd.read_parquet(path)
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no records")
    return path, frame


def build_manifest(task: str, package_root: Path) -> dict[str, object]:
    tables: dict[str, pd.DataFrame] = {}
    table_paths: dict[str, Path] = {}
    for name, columns in REQUIRED_COLUMNS[task].items():
        table_paths[name], tables[name] = read_required_table(package_root, name, columns)

    proteins = tables["proteins"]
    units = tables["training_units"]
    protein_ids = set(proteins["protein_id"].astype(str))
    unknown_ids = sorted(set(units["protein_id"].astype(str)).difference(protein_ids))
    if unknown_ids:
        raise ValueError(f"training units reference unknown proteins: {unknown_ids[:10]}")
    if proteins["protein_id"].astype(str).duplicated().any():
        raise ValueError("proteins.parquet contains duplicate protein_id values")

    manifest: dict[str, object] = {
        "format": "phaseflow_protein_dataset_manifest_v1",
        "task": task,
        "package_root": str(package_root),
        "tables": {
            name: {"path": str(path), "rows": int(len(tables[name])), "sha256": sha256_file(path)}
            for name, path in table_paths.items()
        },
        "proteins": int(len(proteins)),
        "training_units": int(len(units)),
    }
    if task == "llps":
        manifest["training_protocol"] = _validate_llps_training_protocol(package_root, units)
    return manifest


def _validate_llps_training_protocol(package_root: Path, training_units: pd.DataFrame) -> dict[str, object]:
    plan_dir = package_root / "data" / "training_plan"
    paths = sorted(plan_dir.glob("batch_plan_epoch_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"LLPS training protocol is missing: {plan_dir}")
    plans = [pd.read_parquet(path) for path in paths]
    merged = pd.concat(plans, ignore_index=True)
    world_size = int(merged["local_rank"].astype(int).max()) + 1
    batch_size = int(merged["local_slot"].astype(int).max()) + 1
    report = validate_llps_training_protocol(
        training_units,
        plans,
        world_size=world_size,
        batch_size=batch_size,
    )
    return {"records": report.records, "epochs": list(report.epochs)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args.task, args.package_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
