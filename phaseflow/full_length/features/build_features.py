from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phaseflow.full_length.data.feature_cache import FeatureCacheWriter
from phaseflow.full_length.data.schemas import FeatureCacheRecord, IGNORE_INDEX
from phaseflow.full_length.features.disorder import compute_disorder_features
from phaseflow.full_length.features.edge_builder import build_edges
from phaseflow.full_length.features.graph_cache import edge_list_to_precomputed_graph
from phaseflow.full_length.features.physchem import compute_physchem_features
from phaseflow.full_length.features.plm_embedder import ESM2Config, ESM2Embedder, clean_protein_sequence, simple_plm_embedding
from phaseflow.full_length.features.starling_runner import (
    STARLING_EMBED_DIM,
    load_starling_distance_contacts,
    load_starling_embedding,
    zero_starling_embedding,
)

DEFAULT_PROTENIX_EMBEDDING_DIM = 512
DEFAULT_PROTENIX_S_DIM = 384
DEFAULT_PROTENIX_Z_DIM = 128
DEFAULT_GRAPH_EDGE_DIM = 13
DEFAULT_STARLING_DISTANCE_TOPK = 48


def build_feature_cache(
    fasta: str | Path,
    out_dir: str | Path,
    protein_labels: str | Path | None = None,
    regions: str | Path | None = None,
    mil_bags: str | Path | None = None,
    candidate_priors: str | Path | None = None,
    teacher_scores: str | Path | None = None,
    mode: str = "simple",
    esm2_dir: str | Path | None = None,
    esm2_config: ESM2Config | None = None,
    structure_dir: str | Path | None = None,
    protenix_embedding_dir: str | Path | None = None,
    protenix_embedding_dim: int = DEFAULT_PROTENIX_EMBEDDING_DIM,
    af3_dir: str | Path | None = None,
    starling_dir: str | Path | None = None,
    starling_embedding_dir: str | Path | None = None,
    starling_distance_dir: str | Path | None = None,
    local_window: int = 16,
    graph_max_neighbors: int | None = 96,
    graph_edge_dim: int = DEFAULT_GRAPH_EDGE_DIM,
    starling_distance_topk: int = DEFAULT_STARLING_DISTANCE_TOPK,
    require_structure: bool = False,
    require_starling: bool = False,
    overwrite: bool = True,
) -> list[Path]:
    out_dir = Path(out_dir)
    records = [(protein_id, clean_protein_sequence(sequence)) for protein_id, sequence in read_fasta(fasta)]
    label_frame = _read_labels(protein_labels)
    region_map = _read_regions(regions)
    mil_bag_map = _read_mil_bags(mil_bags)
    candidate_prior_map = _read_candidate_priors(candidate_priors)
    teacher_score_map = _read_teacher_scores(teacher_scores)
    embedder = ESM2Embedder(esm2_config) if mode == "esm2" and esm2_dir is None else None
    written: list[Path] = []
    for protein_id, sequence in records:
        out_path = out_dir / f"{protein_id}.h5"
        if out_path.exists() and not overwrite:
            written.append(out_path)
            continue
        plm, plm_missing, plm_reliability = _plm_features(
            protein_id=protein_id,
            sequence=sequence,
            mode=mode,
            esm2_dir=esm2_dir,
            embedder=embedder,
        )
        physchem, _ = compute_physchem_features(sequence)
        disorder, _, disorder_missing, disorder_reliability = compute_disorder_features(sequence, mode="simple")
        protenix_embed, protenix_missing, protenix_reliability, structure_metadata = _protenix_embedding_features(
            protein_id,
            sequence,
            protenix_embedding_dir=protenix_embedding_dir,
            protenix_embedding_dim=protenix_embedding_dim,
        )
        starling_embed, starling_missing, starling_reliability, starling_metadata = _starling_embedding_features(
            protein_id,
            sequence,
            starling_embedding_dir or starling_dir,
            require_starling=require_starling,
        )
        star_contacts, distance_metadata = _starling_distance_contacts(
            protein_id,
            sequence,
            starling_distance_dir=starling_distance_dir,
            contact_topk=starling_distance_topk,
        )
        structure_metadata.update(starling_metadata)
        structure_metadata.update(distance_metadata)
        modality_mask = np.stack(
            [
                plm_missing,
                np.zeros(len(sequence), dtype=np.float32),
                disorder_missing,
                protenix_missing,
                starling_missing,
            ],
            axis=1,
        )
        reliability = np.stack(
            [
                plm_reliability,
                np.ones(len(sequence), dtype=np.float32),
                disorder_reliability,
                protenix_reliability,
                starling_reliability,
            ],
            axis=1,
        )
        edges = build_edges(
            len(sequence),
            local_window=local_window,
            af_contacts=None,
            star_contacts=star_contacts,
            physchem=physchem,
            segment_ids=disorder[:, 3],
            edge_dim=graph_edge_dim,
            star_topk=starling_distance_topk,
        )
        graph = _precompute_graph(edges, len(sequence), graph_max_neighbors, edge_dim=graph_edge_dim)
        label_row = _label_row_for(label_frame, protein_id)
        llps_label = _label_for(label_frame, protein_id)
        sample_weight = _sample_weight_for(label_frame, protein_id, llps_label)
        dpr, key, weight, sample_regions, soft = _labels_from_regions(
            protein_id,
            len(sequence),
            llps_label,
            region_map,
            candidate_prior_map=candidate_prior_map,
            teacher_score_map=teacher_score_map,
        )
        bag = _mil_bag_for(mil_bag_map, label_row, protein_id, llps_label)
        cache_record = FeatureCacheRecord(
            protein_id=protein_id,
            sequence=sequence,
            plm=plm,
            physchem=physchem,
            disorder=disorder,
            protenix_embed=protenix_embed,
            starling_embed=starling_embed,
            modality_mask=modality_mask,
            reliability=reliability,
            edge_src=edges.edge_src,
            edge_dst=edges.edge_dst,
            edge_type=edges.edge_type,
            edge_attr=edges.edge_attr,
            graph_neighbors=graph.neighbors if graph is not None else None,
            graph_edge_attr=graph.edge_attr if graph is not None else None,
            graph_neighbor_mask=graph.neighbor_mask if graph is not None else None,
            y_llps=float(llps_label),
            sample_weight=sample_weight,
            y_dpr=dpr,
            y_key=key,
            y_weight=weight,
            teacher_llps=_float_or_nan(label_row.get("teacher_consensus_score", label_row.get("teacher_weighted", np.nan))),
            teacher_llps_weight=_teacher_weight_from_row(label_row),
            self_llps=_float_or_nan(label_row.get("self_training_score", label_row.get("self_llps", np.nan))),
            self_llps_weight=_float_or_zero(label_row.get("self_training_weight", label_row.get("self_llps_weight", 0.0))),
            region_bag_label=bag["region_bag_label"],
            region_bag_weight=bag["region_bag_weight"],
            region_bag_type=str(bag["region_bag_type"]),
            negative_regularization_weight=_negative_regularization_weight(label_row),
            teacher_dpr=soft["teacher_dpr"],
            teacher_dpr_weight=soft["teacher_dpr_weight"],
            self_dpr=soft["self_dpr"],
            self_dpr_weight=soft["self_dpr_weight"],
            candidate_prior=soft["candidate_prior"],
            candidate_prior_weight=soft["candidate_prior_weight"],
            label_quality=str(label_row.get("label_quality", "")),
            negative_type=str(label_row.get("negative_type", "")),
            source=str(label_row.get("source", "")),
            regions=sample_regions,
            structure_metadata=structure_metadata,
        )
        FeatureCacheWriter.write_h5(out_path, cache_record)
        written.append(out_path)
    return written


def build_feature_cache_from_manifest(
    manifest: str | Path,
    out_dir: str | Path,
    regions: str | Path | None = None,
    mil_bags: str | Path | None = None,
    candidate_priors: str | Path | None = None,
    teacher_scores: str | Path | None = None,
    mode: str = "simple",
    esm2_dir: str | Path | None = None,
    esm2_config: ESM2Config | None = None,
    structure_dir: str | Path | None = None,
    protenix_embedding_dir: str | Path | None = None,
    protenix_embedding_dim: int = DEFAULT_PROTENIX_EMBEDDING_DIM,
    af3_dir: str | Path | None = None,
    starling_dir: str | Path | None = None,
    starling_embedding_dir: str | Path | None = None,
    starling_distance_dir: str | Path | None = None,
    local_window: int = 16,
    graph_max_neighbors: int | None = 96,
    graph_edge_dim: int = DEFAULT_GRAPH_EDGE_DIM,
    starling_distance_topk: int = DEFAULT_STARLING_DISTANCE_TOPK,
    require_structure: bool = False,
    require_starling: bool = False,
    overwrite: bool = True,
) -> list[Path]:
    frame = pd.read_csv(manifest)
    required = {"protein_id", "sequence"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    out_dir = Path(out_dir)
    records = [
        (str(row["protein_id"]), clean_protein_sequence(str(row["sequence"])))
        for _, row in frame.iterrows()
    ]
    label_frame = _labels_from_manifest_frame(frame)
    region_map = _read_regions(regions)
    mil_bag_map = _read_mil_bags(mil_bags)
    candidate_prior_map = _read_candidate_priors(candidate_priors)
    teacher_score_map = _read_teacher_scores(teacher_scores)
    embedder = ESM2Embedder(esm2_config) if mode == "esm2" and esm2_dir is None else None
    written: list[Path] = []
    for protein_id, sequence in records:
        out_path = out_dir / f"{protein_id}.h5"
        if out_path.exists() and not overwrite:
            written.append(out_path)
            continue
        plm, plm_missing, plm_reliability = _plm_features(protein_id, sequence, mode, esm2_dir, embedder)
        physchem, _ = compute_physchem_features(sequence)
        disorder, _, disorder_missing, disorder_reliability = compute_disorder_features(sequence, mode="simple")
        protenix_embed, protenix_missing, protenix_reliability, structure_metadata = _protenix_embedding_features(
            protein_id,
            sequence,
            protenix_embedding_dir=protenix_embedding_dir,
            protenix_embedding_dim=protenix_embedding_dim,
        )
        starling_embed, starling_missing, starling_reliability, starling_metadata = _starling_embedding_features(
            protein_id,
            sequence,
            starling_embedding_dir or starling_dir,
            require_starling=require_starling,
        )
        star_contacts, distance_metadata = _starling_distance_contacts(
            protein_id,
            sequence,
            starling_distance_dir=starling_distance_dir,
            contact_topk=starling_distance_topk,
        )
        structure_metadata.update(starling_metadata)
        structure_metadata.update(distance_metadata)
        modality_mask = np.stack(
            [
                plm_missing,
                np.zeros(len(sequence), dtype=np.float32),
                disorder_missing,
                protenix_missing,
                starling_missing,
            ],
            axis=1,
        )
        reliability = np.stack(
            [
                plm_reliability,
                np.ones(len(sequence), dtype=np.float32),
                disorder_reliability,
                protenix_reliability,
                starling_reliability,
            ],
            axis=1,
        )
        edges = build_edges(
            len(sequence),
            local_window=local_window,
            af_contacts=None,
            star_contacts=star_contacts,
            physchem=physchem,
            segment_ids=disorder[:, 3],
            edge_dim=graph_edge_dim,
            star_topk=starling_distance_topk,
        )
        graph = _precompute_graph(edges, len(sequence), graph_max_neighbors, edge_dim=graph_edge_dim)
        label_row = _label_row_for(label_frame, protein_id)
        llps_label = _label_for(label_frame, protein_id)
        sample_weight = _sample_weight_for(label_frame, protein_id, llps_label)
        dpr, key, weight, sample_regions, soft = _labels_from_regions(
            protein_id,
            len(sequence),
            llps_label,
            region_map,
            candidate_prior_map=candidate_prior_map,
            teacher_score_map=teacher_score_map,
        )
        bag = _mil_bag_for(mil_bag_map, label_row, protein_id, llps_label)
        FeatureCacheWriter.write_h5(
            out_path,
            FeatureCacheRecord(
                protein_id=protein_id,
                sequence=sequence,
                plm=plm,
                physchem=physchem,
                disorder=disorder,
                protenix_embed=protenix_embed,
                starling_embed=starling_embed,
                modality_mask=modality_mask,
                reliability=reliability,
                edge_src=edges.edge_src,
                edge_dst=edges.edge_dst,
                edge_type=edges.edge_type,
                edge_attr=edges.edge_attr,
                graph_neighbors=graph.neighbors if graph is not None else None,
                graph_edge_attr=graph.edge_attr if graph is not None else None,
                graph_neighbor_mask=graph.neighbor_mask if graph is not None else None,
                y_llps=float(llps_label),
                sample_weight=sample_weight,
                y_dpr=dpr,
                y_key=key,
                y_weight=weight,
                teacher_llps=_float_or_nan(label_row.get("teacher_consensus_score", label_row.get("teacher_weighted", np.nan))),
                teacher_llps_weight=_teacher_weight_from_row(label_row),
                self_llps=_float_or_nan(label_row.get("self_training_score", label_row.get("self_llps", np.nan))),
                self_llps_weight=_float_or_zero(label_row.get("self_training_weight", label_row.get("self_llps_weight", 0.0))),
                region_bag_label=bag["region_bag_label"],
                region_bag_weight=bag["region_bag_weight"],
                region_bag_type=str(bag["region_bag_type"]),
                negative_regularization_weight=_negative_regularization_weight(label_row),
                teacher_dpr=soft["teacher_dpr"],
                teacher_dpr_weight=soft["teacher_dpr_weight"],
                self_dpr=soft["self_dpr"],
                self_dpr_weight=soft["self_dpr_weight"],
                candidate_prior=soft["candidate_prior"],
                candidate_prior_weight=soft["candidate_prior_weight"],
                label_quality=str(label_row.get("label_quality", "")),
                negative_type=str(label_row.get("negative_type", "")),
                source=str(label_row.get("source", "")),
                regions=sample_regions,
                structure_metadata=structure_metadata,
            ),
        )
        written.append(out_path)
    return written


def _precompute_graph(edges, length: int, graph_max_neighbors: int | None, edge_dim: int):
    if graph_max_neighbors is None or int(graph_max_neighbors) <= 0:
        return None
    return edge_list_to_precomputed_graph(
        length=length,
        edge_src=edges.edge_src,
        edge_dst=edges.edge_dst,
        edge_type=edges.edge_type,
        edge_attr=edges.edge_attr,
        max_neighbors=int(graph_max_neighbors),
        edge_dim=edge_dim,
    )


def _plm_features(
    protein_id: str,
    sequence: str,
    mode: str,
    esm2_dir: str | Path | None,
    embedder: ESM2Embedder | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "simple":
        plm = simple_plm_embedding(sequence, dim=32)
    elif mode == "esm2":
        if esm2_dir is not None:
            plm = _read_esm2_npz(esm2_dir, protein_id, sequence)
        elif embedder is not None:
            plm = embedder.embed(sequence)
        else:
            raise ValueError("mode='esm2' requires either esm2_dir or esm2_config")
    else:
        raise ValueError(f"Unsupported feature mode: {mode}")
    missing = np.zeros(len(sequence), dtype=np.float32)
    reliability = np.ones(len(sequence), dtype=np.float32)
    return plm.astype(np.float32, copy=False), missing, reliability


def _read_esm2_npz(esm2_dir: str | Path, protein_id: str, sequence: str) -> np.ndarray:
    path = Path(esm2_dir) / f"{protein_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing ESM-2 embedding file: {path}")
    with np.load(path, allow_pickle=False) as data:
        embedding = np.asarray(data["embedding_last_hidden_state"], dtype=np.float32)
        cached_sequence = str(data["sequence"].item()) if "sequence" in data else sequence
    if clean_protein_sequence(cached_sequence) != sequence:
        raise ValueError(f"Sequence mismatch for {protein_id}: ESM-2 npz does not match cache sequence")
    if embedding.ndim != 2 or embedding.shape[0] != len(sequence):
        raise ValueError(f"ESM-2 embedding for {protein_id} must have shape [L, D], got {embedding.shape}")
    return embedding


def _protenix_embedding_features(
    protein_id: str,
    sequence: str,
    protenix_embedding_dir: str | Path | None,
    protenix_embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    length = len(sequence)
    if protenix_embedding_dir is None:
        s_dim, z_dim = _default_protenix_split_dims(protenix_embedding_dim)
        return (
            np.zeros((length, s_dim + z_dim), dtype=np.float32),
            np.ones(length, dtype=np.float32),
            np.zeros(length, dtype=np.float32),
            {
                "protenix_embedding_success": "0",
                "protenix_embedding_path": "",
                "protenix_embedding_dim": str(int(s_dim + z_dim)),
            },
        )
    path = Path(protenix_embedding_dir) / f"{protein_id}.npz"
    if not path.exists():
        s_dim, z_dim = _default_protenix_split_dims(protenix_embedding_dim)
        return np.zeros((length, s_dim + z_dim), dtype=np.float32), np.ones(length, dtype=np.float32), np.zeros(length, dtype=np.float32), {
            "protenix_embedding_success": "0",
            "protenix_embedding_path": str(path),
            "protenix_embedding_dim": str(int(s_dim + z_dim)),
        }
    with np.load(path, allow_pickle=False) as data:
        if "s" not in data or "z" not in data:
            raise ValueError(f"Protenix embedding file {path} must contain s and z arrays")
        s = np.asarray(data["s"], dtype=np.float32)
        z = np.asarray(data["z"], dtype=np.float32)
        if s.ndim != 2 or z.ndim != 2 or s.shape[0] != length or z.shape[0] != length:
            raise ValueError(
                f"Protenix embedding for {protein_id} must have s/z shapes [L, D], "
                f"got s={s.shape}, z={z.shape}, L={length}"
            )
        if "single_mask" in data:
            single_mask = np.asarray(data["single_mask"], dtype=np.float32)
            if single_mask.shape != (length,):
                raise ValueError(
                    f"Protenix embedding single_mask for {protein_id} must have shape [{length}], got {single_mask.shape}"
                )
            available = np.clip(single_mask, 0.0, 1.0).astype(np.float32)
        else:
            available = np.ones(length, dtype=np.float32)
    missing = (available <= 0.0).astype(np.float32)
    reliability = available.astype(np.float32, copy=False)
    embedding = np.concatenate([s, z], axis=1).astype(np.float32, copy=False)
    return embedding, missing, reliability, {
        "protenix_embedding_success": "1",
        "protenix_embedding_path": str(path),
        "protenix_embedding_dim": str(embedding.shape[1]),
    }


def _default_protenix_split_dims(protenix_embedding_dim: int) -> tuple[int, int]:
    if int(protenix_embedding_dim) == DEFAULT_PROTENIX_EMBEDDING_DIM:
        return DEFAULT_PROTENIX_S_DIM, DEFAULT_PROTENIX_Z_DIM
    s_dim = int(round(float(protenix_embedding_dim) * 0.75))
    z_dim = int(protenix_embedding_dim) - s_dim
    return max(s_dim, 1), max(z_dim, 1)


def _starling_embedding_features(
    protein_id: str,
    sequence: str,
    starling_embedding_dir: str | Path | None,
    require_starling: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if starling_embedding_dir is None:
        if require_starling:
            raise ValueError("require_starling=True needs --starling-embedding-dir")
        embedding, missing, reliability = zero_starling_embedding(len(sequence), dim=STARLING_EMBED_DIM)
        return embedding, missing, reliability, {"starling_embedding_success": "0", "starling_embedding_path": ""}
    path = Path(starling_embedding_dir) / f"{protein_id}.npz"
    if not path.exists():
        if require_starling:
            raise FileNotFoundError(f"Missing required STARLING embedding file: {path}")
        embedding, missing, reliability = zero_starling_embedding(len(sequence), dim=STARLING_EMBED_DIM)
        return embedding, missing, reliability, {"starling_embedding_success": "0", "starling_embedding_path": str(path)}
    return load_starling_embedding(path, sequence)


def _starling_distance_contacts(
    protein_id: str,
    sequence: str,
    *,
    starling_distance_dir: str | Path | None,
    contact_topk: int,
) -> tuple[np.ndarray | None, dict[str, object]]:
    if starling_distance_dir is None:
        return None, {"starling_distance_success": "0", "starling_distance_path": ""}
    path = Path(starling_distance_dir) / f"{protein_id}.h5"
    if not path.exists():
        return None, {"starling_distance_success": "0", "starling_distance_path": str(path)}
    return load_starling_distance_contacts(path, sequence, contact_topk=contact_topk)


def _labels_from_manifest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"protein_id", "llps_label"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=["protein_id", "llps_label"])
    columns = ["protein_id", "llps_label"]
    for name in (
        "sample_weight",
        "label_confidence",
        "confidence",
        "negative_type",
        "role_label",
        "source",
        "label_quality",
        "evidence_level",
        "teacher_consensus_score",
        "teacher_weighted",
        "teacher_confidence",
        "teacher_agreement",
        "self_training_score",
        "self_training_weight",
        "self_llps",
        "self_llps_weight",
    ):
        if name in frame.columns and name not in columns:
            columns.append(name)
    return frame[columns].copy()


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    protein_id: str | None = None
    chunks: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if protein_id is not None:
                records.append((protein_id, "".join(chunks).upper()))
            protein_id = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if protein_id is not None:
        records.append((protein_id, "".join(chunks).upper()))
    return records


def _read_labels(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["protein_id", "llps_label"])
    path = Path(path)
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _read_regions(path: str | Path | None) -> dict[str, list[dict[str, object]]]:
    if path is None:
        return {}
    path = Path(path)
    region_map: dict[str, list[dict[str, object]]] = {}
    if path.suffix.lower() in {".csv", ".tsv", ".tab"}:
        frame = pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".tab"} else ",")
        if not frame.empty:
            for protein_id, group in frame.groupby("protein_id"):
                regions: list[dict[str, object]] = []
                for _, row in group.iterrows():
                    start_1 = int(row["start"])
                    end_1 = int(row["end"])
                    regions.append(
                        {
                            "protein_id": str(protein_id),
                            "start": max(0, start_1 - 1),
                            "end": max(0, end_1 - 1),
                            "type": str(row.get("region_type") or row.get("type") or "DPR_candidate"),
                            "region_type": str(row.get("region_type") or row.get("type") or "DPR_candidate"),
                            "region_label": row.get("region_label", "unknown"),
                            "confidence": float(row.get("confidence", 1.0)),
                            "soft_label": _float_or_nan(row.get("soft_label", row.get("score", np.nan))),
                            "soft_weight": _float_or_zero(row.get("soft_weight", row.get("sample_weight", row.get("confidence", 0.0)))),
                            "evidence_level": str(row.get("evidence_level") or "candidate"),
                            "source": str(row.get("source") or ""),
                            "assay": str(row.get("assay") or ""),
                            "notes": str(row.get("notes") or ""),
                        }
                    )
                region_map[str(protein_id)] = regions
        return region_map
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        protein_id = str(row["protein_id"])
        if "regions" in row:
            region_map[protein_id] = list(row.get("regions", []))
            continue
        dpr_spans = row.get("dpr_spans", [])
        regions: list[dict[str, object]] = []
        for span in dpr_spans:
            if isinstance(span, dict):
                start = int(span.get("start", 0))
                end = int(span.get("end", start))
                label_tier = str(span.get("label_tier", row.get("label_tier", "gold")))
                source = str(span.get("source", row.get("source", "")))
                confidence = float(span.get("confidence", row.get("sample_weight", 1.0)))
                sample_weight = float(span.get("sample_weight", row.get("sample_weight", confidence)))
            else:
                start, end = int(span[0]), int(span[1])
                label_tier = str(row.get("label_tier", "gold"))
                source = str(row.get("source", ""))
                confidence = float(row.get("sample_weight", 1.0))
                sample_weight = confidence
            regions.append(
                {
                    "protein_id": protein_id,
                    "start": start,
                    "end": end,
                    "type": "DPR_gold" if label_tier == "gold" else "DPR_curated",
                    "region_type": "DPR_gold" if label_tier == "gold" else "DPR_curated",
                    "region_label": "positive",
                    "confidence": confidence,
                    "soft_weight": sample_weight,
                    "evidence_level": label_tier,
                    "source": source,
                }
            )
        if bool(row.get("outside_is_negative", False)):
            for span in row.get("negative_spans", []):
                if isinstance(span, dict):
                    start = int(span.get("start", 0))
                    end = int(span.get("end", start))
                    confidence = float(span.get("sample_weight", row.get("outside_negative_weight", 0.1)))
                else:
                    start, end = int(span[0]), int(span[1])
                    confidence = float(row.get("outside_negative_weight", 0.1))
                regions.append(
                    {
                        "protein_id": protein_id,
                        "start": start,
                        "end": end,
                        "type": "non_DPR_control",
                        "region_type": "non_DPR_control",
                        "region_label": "negative",
                        "confidence": confidence,
                        "soft_weight": confidence,
                        "evidence_level": "negative_control",
                        "source": str(row.get("source", "")),
                    }
                )
        region_map[protein_id] = regions
    return region_map


def _read_mil_bags(path: str | Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    bags: dict[str, dict[str, object]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        bags[str(row["protein_id"])] = row
    return bags


def _read_candidate_priors(path: str | Path | None) -> dict[str, list[dict[str, object]]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    priors: dict[str, list[dict[str, object]]] = {}
    if path.suffix.lower() in {".jsonl", ".json"}:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            protein_id = str(row["protein_id"])
            priors[protein_id] = list(row.get("candidate_spans", row.get("regions", [])))
        return priors
    import h5py

    with h5py.File(path, "r") as handle:
        for protein_id in handle:
            group = handle[protein_id]
            spans = np.asarray(group.get("spans", np.zeros((0, 2))), dtype=np.int64)
            scores = np.asarray(group.get("scores", np.ones((len(spans),))), dtype=np.float32)
            types = group.attrs.get("types_json", "[]")
            if isinstance(types, bytes):
                types = types.decode("utf-8")
            type_values = json.loads(str(types))
            rows = []
            for index, span in enumerate(spans):
                rows.append(
                    {
                        "start": int(span[0]),
                        "end": int(span[1]),
                        "score": float(scores[index]) if index < len(scores) else 1.0,
                        "type": str(type_values[index]) if index < len(type_values) else "candidate_prior",
                    }
                )
            priors[str(protein_id)] = rows
    return priors


def _read_teacher_scores(path: str | Path | None) -> dict[str, dict[str, np.ndarray]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    import h5py

    scores: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as handle:
        for protein_id in handle:
            group = handle[protein_id]
            if "teacher_consensus" not in group:
                continue
            record = {"teacher_consensus": np.asarray(group["teacher_consensus"], dtype=np.float32)}
            if "teacher_uncertainty" in group:
                record["teacher_uncertainty"] = np.asarray(group["teacher_uncertainty"], dtype=np.float32)
            if "teacher_confidence" in group:
                record["teacher_confidence"] = np.asarray(group["teacher_confidence"], dtype=np.float32)
            scores[str(protein_id)] = record
    return scores


def _label_row_for(frame: pd.DataFrame, protein_id: str) -> dict[str, object]:
    if frame.empty:
        return {}
    rows = frame.loc[frame["protein_id"].astype(str) == str(protein_id)]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _label_for(frame: pd.DataFrame, protein_id: str) -> int:
    if frame.empty:
        return IGNORE_INDEX
    rows = frame.loc[frame["protein_id"].astype(str) == str(protein_id)]
    if rows.empty:
        return IGNORE_INDEX
    return int(rows.iloc[0]["llps_label"])


def _sample_weight_for(frame: pd.DataFrame, protein_id: str, llps_label: int) -> float:
    if frame.empty:
        return 1.0
    rows = frame.loc[frame["protein_id"].astype(str) == str(protein_id)]
    if rows.empty:
        return 1.0
    row = rows.iloc[0]
    if llps_label == IGNORE_INDEX:
        return 0.0
    if "sample_weight" in rows.columns and pd.notna(row.get("sample_weight")):
        return float(row["sample_weight"])
    if "label_confidence" in rows.columns and pd.notna(row.get("label_confidence")):
        return float(row["label_confidence"])
    if "confidence" in rows.columns and pd.notna(row.get("confidence")):
        return float(row["confidence"])
    return 1.0


def _labels_from_regions(
    protein_id: str,
    length: int,
    llps_label: int,
    region_map: dict[str, list[dict[str, object]]],
    candidate_prior_map: dict[str, list[dict[str, object]]] | None = None,
    teacher_score_map: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]], dict[str, np.ndarray]]:
    dpr = np.full(length, IGNORE_INDEX, dtype=np.int64)
    key = np.full(length, IGNORE_INDEX, dtype=np.int64)
    weight = np.zeros(length, dtype=np.float32)
    sample_regions = region_map.get(protein_id, [])
    teacher_dpr = np.full(length, np.nan, dtype=np.float32)
    teacher_dpr_weight = np.zeros(length, dtype=np.float32)
    self_dpr = np.full(length, np.nan, dtype=np.float32)
    self_dpr_weight = np.zeros(length, dtype=np.float32)
    candidate_prior = np.zeros(length, dtype=np.float32)
    candidate_prior_weight = np.zeros(length, dtype=np.float32)
    for region in sample_regions:
        start = max(0, int(region["start"]))
        end = min(length - 1, int(region["end"]))
        confidence = float(region.get("confidence", 1.0))
        label_kind = _region_label_kind(region)
        if label_kind == "positive":
            dpr[start : end + 1] = 1
            weight[start : end + 1] = confidence
        elif label_kind == "negative":
            dpr[start : end + 1] = 0
            weight[start : end + 1] = confidence
        elif label_kind == "key":
            key[start : end + 1] = 1
            weight[start : end + 1] = confidence
        soft_label = _region_soft_label(region, label_kind)
        soft_weight = _region_soft_weight(region, confidence, label_kind)
        if soft_label == soft_label and soft_weight > 0.0:
            source = str(region.get("source") or region.get("evidence_level") or "").lower()
            region_type = str(region.get("region_type") or region.get("type") or "").lower()
            if "self" in source or "self" in region_type:
                _write_soft_region(self_dpr, self_dpr_weight, start, end, soft_label, soft_weight)
            elif label_kind == "soft" or "pseudo" in source or "teacher" in source or "pseudo" in region_type:
                _write_soft_region(teacher_dpr, teacher_dpr_weight, start, end, soft_label, soft_weight)
    if candidate_prior_map:
        for prior in candidate_prior_map.get(protein_id, []):
            start = max(0, int(prior.get("start", 0)))
            end = min(length - 1, int(prior.get("end", start)))
            score = _float_or_nan(prior.get("score", prior.get("confidence", 1.0)))
            if score != score:
                score = 1.0
            prior_weight = _float_or_nan(prior.get("weight", prior.get("sample_weight", 0.2)))
            if prior_weight != prior_weight:
                prior_weight = 0.2
            _write_prior_region(candidate_prior, candidate_prior_weight, start, end, score, prior_weight)
    if teacher_score_map and protein_id in teacher_score_map:
        teacher = teacher_score_map[protein_id]
        consensus = _fit_vector_length(teacher["teacher_consensus"], length, fill=np.nan)
        if "teacher_confidence" in teacher:
            confidence = _fit_vector_length(teacher["teacher_confidence"], length, fill=0.0)
        else:
            uncertainty = _fit_vector_length(teacher.get("teacher_uncertainty", np.ones(length, dtype=np.float32)), length, fill=1.0)
            confidence = np.clip(1.0 - uncertainty, 0.0, 1.0)
        valid = np.isfinite(consensus) & (confidence > 0)
        teacher_dpr[valid] = np.clip(consensus[valid], 0.0, 1.0)
        teacher_dpr_weight[valid] = np.maximum(teacher_dpr_weight[valid], confidence[valid])
    soft = {
        "teacher_dpr": teacher_dpr,
        "teacher_dpr_weight": teacher_dpr_weight,
        "self_dpr": self_dpr,
        "self_dpr_weight": self_dpr_weight,
        "candidate_prior": candidate_prior,
        "candidate_prior_weight": candidate_prior_weight,
    }
    return dpr, key, weight, sample_regions, soft


def _region_label_kind(region: dict[str, object]) -> str:
    label = region.get("region_label")
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized in {"1", "positive", "gold", "curated"}:
            return "positive"
        if normalized in {"candidate", "prior"}:
            return "ignore"
        if normalized in {"0", "negative", "control"}:
            return "negative"
        if normalized in {"key", "key_region"}:
            return "key"
        if normalized in {"unknown", "ignore", ""}:
            return "ignore"
    elif isinstance(label, (int, float)):
        if int(label) == 1:
            return "positive"
        if int(label) == 0:
            return "negative"
    region_type = str(region.get("region_type") or region.get("type") or "").strip()
    if region_type in {"DPR_gold", "DPR_curated", "DPR_silver", "DPR_pseudo"}:
        return "positive"
    if region_type in {"DPR_candidate"}:
        return "ignore"
    if region_type in {"non_DPR_control"}:
        return "negative"
    if region_type in {"key_region"}:
        return "key"
    if region_type in {"DPR_soft", "DPR_teacher", "DPR_self_training"}:
        return "soft"
    return "ignore"


def _region_soft_label(region: dict[str, object], label_kind: str) -> float:
    for name in ("soft_label", "score", "mean_residue_score", "confidence"):
        value = _float_or_nan(region.get(name, np.nan))
        if value == value:
            return float(np.clip(value, 0.0, 1.0))
    if label_kind == "positive":
        return 1.0
    if label_kind == "negative":
        return 0.0
    return float("nan")


def _region_soft_weight(region: dict[str, object], confidence: float, label_kind: str) -> float:
    for name in ("soft_weight", "sample_weight", "weight"):
        value = _float_or_nan(region.get(name, np.nan))
        if value == value:
            return float(np.clip(value, 0.0, 1.0))
    if label_kind in {"positive", "negative", "soft"}:
        return float(np.clip(confidence, 0.0, 1.0))
    return 0.0


def _write_soft_region(target: np.ndarray, weight: np.ndarray, start: int, end: int, value: float, new_weight: float) -> None:
    old_weight = weight[start : end + 1]
    old_value = np.nan_to_num(target[start : end + 1], nan=0.0)
    denom = old_weight + new_weight
    merged = np.where(denom > 0.0, (old_value * old_weight + value * new_weight) / denom, value)
    target[start : end + 1] = merged.astype(np.float32)
    weight[start : end + 1] = np.maximum(old_weight, new_weight)


def _write_prior_region(target: np.ndarray, weight: np.ndarray, start: int, end: int, value: float, new_weight: float) -> None:
    target[start : end + 1] = np.maximum(target[start : end + 1], float(np.clip(value, 0.0, 1.0)))
    weight[start : end + 1] = np.maximum(weight[start : end + 1], float(np.clip(new_weight, 0.0, 1.0)))


def _fit_vector_length(values: np.ndarray, length: int, fill: float) -> np.ndarray:
    out = np.full(length, fill, dtype=np.float32)
    n = min(length, int(values.shape[0]))
    if n:
        out[:n] = values[:n].astype(np.float32, copy=False)
    return out


def _mil_bag_for(
    mil_bag_map: dict[str, dict[str, object]],
    label_row: dict[str, object],
    protein_id: str,
    llps_label: int,
) -> dict[str, object]:
    explicit = mil_bag_map.get(protein_id)
    if explicit is not None:
        label = explicit.get("bag_label", explicit.get("region_bag_label", IGNORE_INDEX))
        return {
            "region_bag_label": float(label if label is not None else IGNORE_INDEX),
            "region_bag_weight": float(explicit.get("bag_weight", explicit.get("region_bag_weight", 0.0))),
            "region_bag_type": str(explicit.get("bag_type", explicit.get("region_bag_type", "mask"))),
        }
    role = str(label_row.get("role_label", label_row.get("role_type", ""))).lower()
    tier = str(label_row.get("label_quality", label_row.get("label_tier", label_row.get("evidence_level", "")))).lower()
    negative_type = str(label_row.get("negative_type", "")).lower()
    sample_weight = _float_or_zero(label_row.get("sample_weight", label_row.get("label_confidence", 0.0)))
    if llps_label == 1 and any(token in role for token in ("driver", "scaffold", "self")):
        return {
            "region_bag_label": 1.0,
            "region_bag_weight": sample_weight if sample_weight > 0 else 0.75,
            "region_bag_type": "protein_positive_driver",
        }
    if llps_label == 0 and ("negative" in tier or "negative" in role or "negative" in negative_type):
        bag_type = "negative_disordered" if "disordered" in negative_type or "disordered" in role else "negative_structured"
        return {
            "region_bag_label": 0.0,
            "region_bag_weight": sample_weight if sample_weight > 0 else 0.75,
            "region_bag_type": bag_type,
        }
    return {"region_bag_label": float(IGNORE_INDEX), "region_bag_weight": 0.0, "region_bag_type": "mask"}


def _negative_regularization_weight(label_row: dict[str, object]) -> float:
    negative_type = str(label_row.get("negative_type", "")).lower()
    role = str(label_row.get("role_label", label_row.get("role_type", ""))).lower()
    llps = _float_or_nan(label_row.get("llps_label", np.nan))
    if llps != 0.0:
        return 0.0
    if "disordered" in negative_type or "disordered" in role:
        return 0.4
    if "structured" in negative_type or "structured" in role or "negative" in negative_type:
        return 0.2
    return 0.0


def _teacher_weight_from_row(row: dict[str, object]) -> float:
    explicit = _float_or_nan(row.get("teacher_confidence", row.get("teacher_llps_weight", np.nan)))
    if explicit == explicit:
        return float(np.clip(explicit, 0.0, 1.0))
    score = _float_or_nan(row.get("teacher_consensus_score", row.get("teacher_weighted", np.nan)))
    if score == score:
        agreement = _float_or_nan(row.get("teacher_agreement", np.nan))
        if agreement == agreement:
            return float(np.clip(agreement, 0.0, 1.0))
        return 0.5
    return 0.0


def _float_or_nan(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _float_or_zero(value: object) -> float:
    parsed = _float_or_nan(value)
    if parsed != parsed:
        return 0.0
    return float(np.clip(parsed, 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fasta")
    source.add_argument("--manifest")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--protein-labels")
    parser.add_argument("--regions")
    parser.add_argument("--mil-bags")
    parser.add_argument("--candidate-priors")
    parser.add_argument("--teacher-scores")
    parser.add_argument("--mode", choices=["simple", "esm2"], default="simple")
    parser.add_argument("--esm2-dir")
    parser.add_argument("--esm2-model-name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--esm2-model-dir")
    parser.add_argument("--esm2-device", default="auto")
    parser.add_argument("--esm2-dtype", default="float32")
    parser.add_argument("--esm2-storage-dtype", default="float32")
    parser.add_argument("--esm2-local-files-only", action="store_true")
    parser.add_argument("--esm2-chunk-size", type=int)
    parser.add_argument("--esm2-overlap", type=int, default=128)
    parser.add_argument("--structure-dir")
    parser.add_argument("--protenix-embedding-dir")
    parser.add_argument("--protenix-embedding-dim", type=int, default=DEFAULT_PROTENIX_EMBEDDING_DIM)
    parser.add_argument("--af3-dir")
    parser.add_argument("--starling-dir", help="Deprecated alias for --starling-embedding-dir.")
    parser.add_argument("--starling-embedding-dir")
    parser.add_argument("--starling-distance-dir")
    parser.add_argument("--local-window", type=int, default=16)
    parser.add_argument("--graph-max-neighbors", type=int, default=96)
    parser.add_argument("--graph-edge-dim", type=int, default=DEFAULT_GRAPH_EDGE_DIM)
    parser.add_argument("--starling-distance-topk", type=int, default=DEFAULT_STARLING_DISTANCE_TOPK)
    parser.add_argument("--require-structure", action="store_true")
    parser.add_argument("--require-starling", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()
    esm2_config = ESM2Config(
        model_name=args.esm2_model_name,
        model_dir=args.esm2_model_dir,
        device=args.esm2_device,
        dtype=args.esm2_dtype,
        storage_dtype=args.esm2_storage_dtype,
        local_files_only=args.esm2_local_files_only,
        chunk_size=args.esm2_chunk_size,
        overlap=args.esm2_overlap,
    )
    if args.manifest:
        paths = build_feature_cache_from_manifest(
            manifest=args.manifest,
            out_dir=args.out_dir,
            regions=args.regions,
            mil_bags=args.mil_bags,
            candidate_priors=args.candidate_priors,
            teacher_scores=args.teacher_scores,
            mode=args.mode,
            esm2_dir=args.esm2_dir,
            esm2_config=esm2_config,
            structure_dir=args.structure_dir,
            protenix_embedding_dir=args.protenix_embedding_dir,
            protenix_embedding_dim=args.protenix_embedding_dim,
            af3_dir=args.af3_dir,
            starling_dir=args.starling_dir,
            starling_embedding_dir=args.starling_embedding_dir,
            starling_distance_dir=args.starling_distance_dir,
            local_window=args.local_window,
            graph_max_neighbors=args.graph_max_neighbors,
            graph_edge_dim=args.graph_edge_dim,
            starling_distance_topk=args.starling_distance_topk,
            require_structure=args.require_structure,
            require_starling=args.require_starling,
            overwrite=not args.no_overwrite,
        )
    else:
        paths = build_feature_cache(
            fasta=args.fasta,
            out_dir=args.out_dir,
            protein_labels=args.protein_labels,
            regions=args.regions,
            mil_bags=args.mil_bags,
            candidate_priors=args.candidate_priors,
            teacher_scores=args.teacher_scores,
            mode=args.mode,
            esm2_dir=args.esm2_dir,
            esm2_config=esm2_config,
            structure_dir=args.structure_dir,
            protenix_embedding_dir=args.protenix_embedding_dir,
            protenix_embedding_dim=args.protenix_embedding_dim,
            af3_dir=args.af3_dir,
            starling_dir=args.starling_dir,
            starling_embedding_dir=args.starling_embedding_dir,
            starling_distance_dir=args.starling_distance_dir,
            local_window=args.local_window,
            graph_max_neighbors=args.graph_max_neighbors,
            graph_edge_dim=args.graph_edge_dim,
            starling_distance_topk=args.starling_distance_topk,
            require_structure=args.require_structure,
            require_starling=args.require_starling,
            overwrite=not args.no_overwrite,
        )
    print(f"Wrote {len(paths)} feature caches to {args.out_dir}")


if __name__ == "__main__":
    main()
