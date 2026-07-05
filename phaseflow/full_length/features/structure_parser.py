from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


STRUCTURE_PROVIDER = "protenix"
DEFAULT_STRUCTURE_DIM = 12


@dataclass(slots=True)
class ParsedStructureFeatures:
    node: np.ndarray
    missing_mask: np.ndarray
    reliability: np.ndarray
    contacts: np.ndarray | None
    metadata: dict[str, Any]


def zero_structure_features(length: int, dim: int = DEFAULT_STRUCTURE_DIM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node = np.zeros((length, dim), dtype=np.float32)
    missing = np.ones(length, dtype=np.float32)
    reliability = np.zeros(length, dtype=np.float32)
    return node, missing, reliability


def load_structure_features(
    path: str | Path,
    sequence: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Load provider-agnostic structure features from PhaseFlow intermediate npz.

    The npz is intentionally provider-neutral. Current Protenix parser writes
    `node`, `reliability`, `missing_mask`, optional `contacts`, and metadata
    fields such as `structure_provider`, `raw_cif_path`, and
    `raw_confidence_json_path`.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        cached_sequence = str(data["sequence"].item()) if "sequence" in data else sequence
        if cached_sequence and cached_sequence != sequence:
            raise ValueError(f"Structure sequence mismatch for {path}: cache sequence differs from feature cache sequence")
        node = _optional_array(data, "node", "node_features", "structure_node")
        if node is None:
            raise ValueError(f"Structure feature file {path} has no node features")
        if node.ndim != 2 or node.shape[0] != len(sequence):
            raise ValueError(f"Structure node features in {path} must have shape [L, D], got {node.shape}")
        missing = (
            np.asarray(data["missing_mask"], dtype=np.float32)
            if "missing_mask" in data
            else np.zeros(len(sequence), dtype=np.float32)
        )
        reliability = (
            np.asarray(data["reliability"], dtype=np.float32)
            if "reliability" in data
            else np.clip(node[:, 0], 0.0, 1.0).astype(np.float32)
        )
        contacts = np.asarray(data["contacts"], dtype=np.float32) if "contacts" in data else None
        metadata = _metadata_from_npz(data)
    return node.astype(np.float32, copy=False), missing, reliability, contacts, metadata


def parse_protenix_outputs(
    records: list[tuple[str, str]],
    protenix_output: str | Path,
    out_dir: str | Path,
    contact_topk: int = 32,
    contact_cutoff: float = 8.0,
) -> list[Path]:
    protenix_output = Path(protenix_output)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for protein_id, sequence in records:
        parsed = parse_single_protenix_output(
            protein_id=protein_id,
            sequence=sequence,
            protenix_output=protenix_output,
            contact_topk=contact_topk,
            contact_cutoff=contact_cutoff,
        )
        if parsed is None:
            continue
        payload: dict[str, Any] = {
            "protein_id": np.asarray(protein_id),
            "sequence": np.asarray(sequence),
            "node": parsed.node.astype(np.float32, copy=False),
            "missing_mask": parsed.missing_mask.astype(np.float32, copy=False),
            "reliability": parsed.reliability.astype(np.float32, copy=False),
        }
        if parsed.contacts is not None:
            payload["contacts"] = parsed.contacts.astype(np.float32, copy=False)
        for key, value in parsed.metadata.items():
            if value is not None:
                payload[key] = np.asarray(str(value))
        path = out_dir / f"{protein_id}.npz"
        np.savez_compressed(path, **payload)
        written.append(path)
    return written


def parse_single_protenix_output(
    protein_id: str,
    sequence: str,
    protenix_output: str | Path,
    contact_topk: int = 32,
    contact_cutoff: float = 8.0,
) -> ParsedStructureFeatures | None:
    prediction = _find_best_prediction(Path(protenix_output), protein_id)
    if prediction is None:
        return None
    summary = _read_json(prediction.summary_path) if prediction.summary_path else {}
    coords, plddt, valid = _extract_ca_coordinates(prediction.cif_path, len(sequence))
    if coords is None:
        return None
    global_plddt = _scalar(summary, "plddt")
    if global_plddt is not None and (not np.isfinite(plddt).any() or float(np.nanmax(plddt)) <= 0.0):
        plddt = np.full(len(sequence), _normalize_score(global_plddt), dtype=np.float32)
    else:
        plddt = _normalize_vector(plddt, default=global_plddt)

    contacts = _contacts_from_coordinates(coords, plddt, valid, contact_topk, contact_cutoff)
    contact_degree = _contact_degree(contacts, len(sequence), max(contact_topk, 1))
    mean_contact_conf = _mean_contact_confidence(contacts, len(sequence))
    surface_exposure = np.clip(1.0 - contact_degree, 0.0, 1.0).astype(np.float32)
    helix_hint, strand_hint = _secondary_structure_hints(coords, valid)
    surface_hydrophobicity, surface_charge = _surface_sequence_features(sequence, surface_exposure)
    gpde = _normalize_gpde(_scalar(summary, "gpde"))
    ptm = _normalize_probability(_scalar(summary, "ptm"))
    has_clash = float(bool(_scalar(summary, "has_clash") or 0.0))

    node = np.stack(
        [
            plddt,
            valid.astype(np.float32),
            contact_degree,
            surface_exposure,
            mean_contact_conf,
            np.full(len(sequence), gpde, dtype=np.float32),
            np.full(len(sequence), ptm, dtype=np.float32),
            np.full(len(sequence), has_clash, dtype=np.float32),
            helix_hint,
            strand_hint,
            surface_hydrophobicity,
            surface_charge,
        ],
        axis=1,
    ).astype(np.float32)
    reliability = np.clip(plddt, 0.0, 1.0).astype(np.float32)
    reliability *= valid.astype(np.float32)
    if has_clash:
        reliability *= 0.7
    missing = (~valid).astype(np.float32)
    metadata = {
        "structure_provider": STRUCTURE_PROVIDER,
        "provider_version": _provider_version(),
        "model_name": _metadata_scalar(summary, "model_name"),
        "seed": _seed_from_path(prediction.cif_path),
        "raw_cif_path": str(prediction.cif_path),
        "raw_confidence_json_path": str(prediction.summary_path) if prediction.summary_path else "",
        "raw_full_data_json_path": str(prediction.full_data_path) if prediction.full_data_path else "",
        "mean_confidence": f"{float(np.nanmean(plddt)):.6g}" if len(plddt) else "",
        "gpde": f"{gpde:.6g}",
        "ptm": f"{ptm:.6g}",
        "ranking_score": _metadata_scalar(summary, "ranking_score"),
        "has_clash": str(int(has_clash)),
        "num_recycles": _metadata_scalar(summary, "num_recycles"),
        "use_msa": _metadata_scalar(summary, "use_msa"),
        "use_template": _metadata_scalar(summary, "use_template"),
        "structure_node_dim": str(DEFAULT_STRUCTURE_DIM),
    }
    return ParsedStructureFeatures(node=node, missing_mask=missing, reliability=reliability, contacts=contacts, metadata=metadata)


@dataclass(slots=True)
class _PredictionPaths:
    cif_path: Path
    summary_path: Path | None
    full_data_path: Path | None
    ranking_score: float


def _find_best_prediction(root: Path, protein_id: str) -> _PredictionPaths | None:
    summaries = _summary_candidates(root / protein_id, protein_id)
    if not summaries:
        summaries = _summary_candidates(root, protein_id)
    candidates: list[_PredictionPaths] = []
    for summary_path in summaries:
        summary = _read_json(summary_path)
        rank = _sample_rank(summary_path)
        cif_path = summary_path.with_name(f"{protein_id}_sample_{rank}.cif")
        if not cif_path.exists():
            cif_matches = sorted(summary_path.parent.glob(f"*sample_{rank}.cif"))
            cif_path = cif_matches[0] if cif_matches else cif_path
        if not cif_path.exists():
            continue
        full_data_path = summary_path.with_name(f"{protein_id}_full_data_sample_{rank}.json")
        if not full_data_path.exists():
            full_matches = sorted(summary_path.parent.glob(f"*full_data_sample_{rank}.json"))
            full_data_path = full_matches[0] if full_matches else None
        candidates.append(
            _PredictionPaths(
                cif_path=cif_path,
                summary_path=summary_path,
                full_data_path=full_data_path,
                ranking_score=float(_scalar(summary, "ranking_score") or -math.inf),
            )
        )
    if candidates:
        candidates.sort(key=lambda item: item.ranking_score, reverse=True)
        return candidates[0]

    cifs = _cif_candidates(root / protein_id, protein_id)
    if not cifs:
        cifs = _cif_candidates(root, protein_id)
    if not cifs:
        return None
    return _PredictionPaths(cif_path=cifs[0], summary_path=None, full_data_path=None, ranking_score=-math.inf)


def _summary_candidates(root: Path, protein_id: str) -> list[Path]:
    if not root.exists():
        return []
    summaries = sorted(root.glob(f"**/{protein_id}_summary_confidence_sample_*.json"))
    if summaries:
        return summaries
    prefix = protein_id.lower() + "_"
    return [
        path
        for path in sorted(root.glob("**/*summary_confidence_sample_*.json"))
        if path.name.lower().startswith(prefix)
    ]


def _cif_candidates(root: Path, protein_id: str) -> list[Path]:
    if not root.exists():
        return []
    prefix = protein_id.lower() + "_"
    return [
        path
        for path in sorted(root.glob("**/*_sample_*.cif"))
        if path.name.lower().startswith(prefix)
    ]


def _extract_ca_coordinates(cif_path: Path, length: int) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    try:
        data = MMCIF2Dict(str(cif_path))
    except Exception:
        return None, np.zeros(length, dtype=np.float32), np.zeros(length, dtype=bool)
    atom_ids = _as_list(data.get("_atom_site.label_atom_id") or data.get("_atom_site.auth_atom_id"))
    xs = _as_list(data.get("_atom_site.Cartn_x"))
    ys = _as_list(data.get("_atom_site.Cartn_y"))
    zs = _as_list(data.get("_atom_site.Cartn_z"))
    b_factors = _as_list(data.get("_atom_site.B_iso_or_equiv"))
    seq_ids = _as_list(data.get("_atom_site.label_seq_id") or data.get("_atom_site.auth_seq_id"))
    chains = _as_list(data.get("_atom_site.label_asym_id") or data.get("_atom_site.auth_asym_id"))
    models = _as_list(data.get("_atom_site.pdbx_PDB_model_num"))
    if not atom_ids or not xs or not seq_ids:
        return None, np.zeros(length, dtype=np.float32), np.zeros(length, dtype=bool)

    ca_rows: list[tuple[str, int, int]] = []
    for index, atom_id in enumerate(atom_ids):
        if str(atom_id).strip().upper() != "CA":
            continue
        if models and str(models[index]).strip() not in {"1", ".", "?"}:
            continue
        seq_id = _int_or_none(seq_ids[index])
        if seq_id is None:
            continue
        chain = str(chains[index]).strip() if chains else "A"
        ca_rows.append((chain, seq_id, index))
    if not ca_rows:
        return None, np.zeros(length, dtype=np.float32), np.zeros(length, dtype=bool)
    chain_counts: dict[str, int] = {}
    for chain, _, _ in ca_rows:
        chain_counts[chain] = chain_counts.get(chain, 0) + 1
    main_chain = max(chain_counts, key=chain_counts.get)

    coords = np.full((length, 3), np.nan, dtype=np.float32)
    plddt = np.zeros(length, dtype=np.float32)
    valid = np.zeros(length, dtype=bool)
    for chain, seq_id, index in ca_rows:
        if chain != main_chain:
            continue
        pos = seq_id - 1
        if pos < 0 or pos >= length:
            continue
        xyz = [_float_or_nan(xs[index]), _float_or_nan(ys[index]), _float_or_nan(zs[index])]
        if not all(np.isfinite(xyz)):
            continue
        coords[pos] = np.asarray(xyz, dtype=np.float32)
        if b_factors:
            plddt[pos] = _float_or_nan(b_factors[index])
        valid[pos] = True
    return coords, plddt, valid


def _contacts_from_coordinates(
    coords: np.ndarray,
    reliability: np.ndarray,
    valid: np.ndarray,
    topk: int,
    cutoff: float,
) -> np.ndarray:
    rows: list[tuple[int, int, float, float]] = []
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) < 2:
        return np.zeros((0, 4), dtype=np.float32)
    for src in valid_idx:
        delta = coords[valid_idx] - coords[src]
        dist = np.linalg.norm(delta, axis=1)
        order = np.argsort(dist)
        added = 0
        for order_index in order:
            dst = int(valid_idx[order_index])
            if dst == int(src):
                continue
            distance = float(dist[order_index])
            if not np.isfinite(distance) or distance > cutoff:
                continue
            conf = math.sqrt(float(reliability[src]) * float(reliability[dst]))
            conf *= max(0.0, 1.0 - distance / max(cutoff, 1e-6))
            rows.append((int(src), dst, float(conf), distance))
            added += 1
            if added >= topk:
                break
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


def _contact_degree(contacts: np.ndarray, length: int, topk: int) -> np.ndarray:
    degree = np.zeros(length, dtype=np.float32)
    if contacts.size:
        for src in contacts[:, 0].astype(np.int64):
            if 0 <= src < length:
                degree[src] += 1.0
    return np.clip(degree / max(topk, 1), 0.0, 1.0).astype(np.float32)


def _mean_contact_confidence(contacts: np.ndarray, length: int) -> np.ndarray:
    total = np.zeros(length, dtype=np.float32)
    count = np.zeros(length, dtype=np.float32)
    if contacts.size:
        for src, _, confidence, *_ in contacts.tolist():
            src_i = int(src)
            if 0 <= src_i < length:
                total[src_i] += float(confidence)
                count[src_i] += 1.0
    return np.divide(total, np.maximum(count, 1.0), out=np.zeros_like(total), where=count > 0)


def _secondary_structure_hints(coords: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate helix/strand-like local geometry from CA distances.

    This is intentionally lightweight and provider-neutral; it gives PhaseFlow a
    stable structural context feature without depending on DSSP or Protenix
    internals.
    """
    length = len(valid)
    helix = np.zeros(length, dtype=np.float32)
    strand = np.zeros(length, dtype=np.float32)
    for index in range(length):
        if not valid[index]:
            continue
        if index + 4 < length and valid[index + 4]:
            distance = float(np.linalg.norm(coords[index + 4] - coords[index]))
            helix[index : index + 5] = np.maximum(helix[index : index + 5], _triangular_score(distance, center=6.0, width=2.0))
        if index + 2 < length and valid[index + 2]:
            distance = float(np.linalg.norm(coords[index + 2] - coords[index]))
            strand[index : index + 3] = np.maximum(strand[index : index + 3], _triangular_score(distance, center=6.8, width=2.5))
    return np.clip(helix, 0.0, 1.0), np.clip(strand, 0.0, 1.0)


def _triangular_score(value: float, center: float, width: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(1.0 - abs(value - center) / max(width, 1.0e-6), 0.0, 1.0))


def _surface_sequence_features(sequence: str, exposure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hydrophobic = set("AILMFWV")
    positive = set("KRH")
    negative = set("DE")
    surface_hydrophobicity = np.zeros(len(sequence), dtype=np.float32)
    surface_charge = np.zeros(len(sequence), dtype=np.float32)
    for index, aa in enumerate(sequence.upper()):
        surface_hydrophobicity[index] = exposure[index] * float(aa in hydrophobic)
        signed_charge = float(aa in positive) - float(aa in negative)
        surface_charge[index] = exposure[index] * abs(signed_charge)
    return surface_hydrophobicity, surface_charge


def _metadata_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in (
        "structure_provider",
        "provider_version",
        "model_name",
        "seed",
        "raw_cif_path",
        "raw_confidence_json_path",
        "raw_full_data_json_path",
        "mean_confidence",
        "gpde",
        "ptm",
        "ranking_score",
        "has_clash",
        "num_recycles",
        "use_msa",
        "use_template",
    ):
        if key in data:
            metadata[key] = str(data[key].item())
    if "structure_provider" not in metadata:
        metadata["structure_provider"] = STRUCTURE_PROVIDER
    metadata.setdefault("structure_success", "1")
    return metadata


def _optional_array(data: np.lib.npyio.NpzFile, *names: str) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name], dtype=np.float32)
    return None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _sample_rank(path: Path) -> str:
    match = re.search(r"sample_(\d+)", path.name)
    return match.group(1) if match else "0"


def _seed_from_path(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("seed_"):
            return parent.name.removeprefix("seed_")
    return ""


def _scalar(data: Any, key: str) -> float | None:
    if isinstance(data, dict):
        if key in data:
            return _first_number(data[key])
        for value in data.values():
            found = _scalar(value, key)
            if found is not None:
                return found
    return None


def _metadata_scalar(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None:
        found = _scalar(data, key)
        return "" if found is None else f"{found:.6g}"
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return str(value)


def _first_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_number(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_number(item)
            if found is not None:
                return found
    return None


def _normalize_vector(value: np.ndarray, default: float | None = None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    out = value.copy()
    if np.isfinite(out).any() and float(np.nanmax(out)) > 1.5:
        out = out / 100.0
    if default is not None:
        fill = _normalize_score(default)
        out[~np.isfinite(out)] = fill
        out[out <= 0.0] = fill
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _normalize_score(value: float) -> float:
    value = float(value)
    if value > 1.5:
        value = value / 100.0
    return float(np.clip(value, 0.0, 1.0))


def _normalize_probability(value: float | None) -> float:
    if value is None:
        return 0.0
    return _normalize_score(value)


def _normalize_gpde(value: float | None) -> float:
    if value is None:
        return 0.0
    # gpde is an error-like distance score; lower is better. Map to confidence.
    return float(np.clip(1.0 / (1.0 + max(float(value), 0.0)), 0.0, 1.0))


def _provider_version() -> str:
    try:
        from protenix.version import __version__  # type: ignore

        return str(__version__)
    except Exception:
        return ""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _int_or_none(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if text in {"", ".", "?"}:
            return None
        return int(float(text))
    except ValueError:
        return None


def _float_or_nan(value: Any) -> float:
    try:
        text = str(value).strip()
        if text in {"", ".", "?"}:
            return float("nan")
        return float(text)
    except ValueError:
        return float("nan")
