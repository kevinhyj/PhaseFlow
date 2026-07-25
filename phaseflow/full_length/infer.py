from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

from phaseflow.full_length.data.collator import PhaseFlowCollator
from phaseflow.full_length.data.dataset import PhaseFlowDataset
from phaseflow.full_length.models.phaseflow import PhaseFlowModel
from phaseflow.full_length.phaseflow_fusion import (
    DEFAULT_PHASEFLOW_CHECKPOINT,
    DEFAULT_PHASEFLOW_PYTHON,
    DEFAULT_PHASEFLOW_ROOT,
    PhaseFlowFusionConfig,
    PhaseFlowWindowScorer,
    fuse_window_phaseflow_with_full_length,
    load_phaseflow_profile_jsonl,
    parse_window_sizes,
    run_phaseflow_profile_subprocess,
)
from phaseflow.full_length.postprocess import combine_regions, decoder_regions, scores_to_regions
from phaseflow.full_length.utils import load_yaml, move_batch_to_device, resolve_device


@torch.no_grad()
def run_inference(
    checkpoint_path: str | Path,
    feature_dir: str | Path,
    out: str | Path,
    protein_ids: list[str] | None = None,
    phaseflow_config: PhaseFlowFusionConfig | None = None,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = resolve_device(str(config.get("device", "auto")))
    model = PhaseFlowModel(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    if protein_ids is None:
        protein_ids = sorted(path.stem for path in Path(feature_dir).glob("*.h5"))
    out_path = Path(out)
    phaseflow_profiles = None
    phaseflow_scorer = None
    if phaseflow_config is not None:
        if phaseflow_config.profile_jsonl is not None:
            phaseflow_profiles = load_phaseflow_profile_jsonl(Path(phaseflow_config.profile_jsonl))
        elif phaseflow_config.phaseflow_python is not None:
            profile_out = Path(phaseflow_config.profile_out or _phaseflow_profile_out_path(out_path))
            records = _read_sequences_from_feature_cache(feature_dir, protein_ids)
            phaseflow_profiles = run_phaseflow_profile_subprocess(
                records=records,
                config=phaseflow_config,
                out_path=profile_out,
            )
        else:
            try:
                phaseflow_scorer = PhaseFlowWindowScorer(phaseflow_config)
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "PhaseFlow direct import failed. Run with --phaseflow-python "
                    f"{DEFAULT_PHASEFLOW_PYTHON} or provide --phaseflow-profile-jsonl."
                ) from exc
    loader = DataLoader(
        PhaseFlowDataset(feature_dir, protein_ids),
        batch_size=int(config.get("training", {}).get("batch_size", 2)),
        shuffle=False,
        collate_fn=PhaseFlowCollator(
            max_neighbors=int(config.get("training", {}).get("max_neighbors", 96)),
            require_precomputed_graph=bool(config.get("training", {}).get("require_precomputed_graph", False)),
        ),
    )
    post_config = config.get("postprocess", {})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            llps_output = outputs.get("llps_logits", outputs["llps_logits"])
            llps = torch.sigmoid(llps_output).detach().cpu().numpy()
            dpr = torch.sigmoid(outputs["dpr_logits"]).detach().cpu().numpy()
            key = torch.sigmoid(outputs["key_logits"]).detach().cpu().numpy()
            modality_weights = outputs["modality_weights"].detach().cpu().numpy()
            region_logits = outputs["region_logits"].detach().cpu().numpy()
            region_start = outputs["region_start"].detach().cpu().numpy()
            region_end = outputs["region_end"].detach().cpu().numpy()
            lengths = batch["lengths"].detach().cpu().numpy()
            for index, protein_id in enumerate(batch["protein_ids"]):
                length = int(lengths[index])
                phaseflow_llps = float(llps[index])
                output_llps = phaseflow_llps
                phaseflow_dpr_scores = dpr[index, :length]
                dpr_scores = phaseflow_dpr_scores
                phaseflow_fusion = None
                if phaseflow_config is not None:
                    if phaseflow_profiles is not None:
                        if str(protein_id) not in phaseflow_profiles:
                            raise RuntimeError(f"Missing PhaseFlow profile for {protein_id}")
                        phaseflow_scores, used_windows = phaseflow_profiles[str(protein_id)]
                    elif phaseflow_scorer is not None:
                        phaseflow_scores, used_windows = phaseflow_scorer.score_sequence(batch["sequences"][index])
                    else:
                        raise RuntimeError("PhaseFlow fusion is enabled but no scorer or profile lookup is available.")
                    phaseflow_fusion = fuse_window_phaseflow_with_full_length(
                        full_length_dpr=phaseflow_dpr_scores,
                        full_length_llps_probability=phaseflow_llps,
                        window_scores=phaseflow_scores,
                        config=phaseflow_config,
                        window_sizes=used_windows,
                    )
                    dpr_scores = phaseflow_fusion.dpr_scores
                    output_llps = phaseflow_fusion.llps_probability
                post = scores_to_regions(
                    dpr_scores,
                    threshold=float(post_config.get("threshold", 0.5)),
                    smooth_window=int(post_config.get("smooth_window", 5)),
                    merge_gap=int(post_config.get("merge_gap", 5)),
                    min_region_len=int(post_config.get("min_region_len", 6)),
                )
                if phaseflow_fusion is not None:
                    for region in post:
                        region["source"] = "phaseflow_fused_postprocess"
                dec = decoder_regions(
                    region_logits[index],
                    region_start[index],
                    region_end[index],
                    length,
                    score_threshold=float(post_config.get("decoder_score_threshold", 0.5)),
                )
                regions = combine_regions(dec, post)
                evidence = _evidence_for_sample(
                    modality_weights[index, :length],
                    batch["modality_mask"][index, :length].detach().cpu().numpy(),
                    batch["structure_metadata"][index],
                )
                if phaseflow_fusion is not None:
                    evidence["phaseflow_fusion"] = {
                        "enabled": True,
                        "method": phaseflow_config.dpr_fusion_mode,
                        "dpr_blend_alpha": float(phaseflow_config.dpr_blend_alpha),
                        "window_sizes": list(phaseflow_fusion.window_sizes),
                        "phaseflow_llps_proxy": float(phaseflow_fusion.phaseflow_llps_proxy),
                        "changed_fraction": float(phaseflow_fusion.changed_fraction),
                        "lifted_residues": int(phaseflow_fusion.lifted_residues),
                        "suppressed_residues": int(phaseflow_fusion.suppressed_residues),
                    }
                residue_scores = {
                    "DPR": [float(value) for value in dpr_scores],
                    "key_residue": [float(value) for value in key[index, :length]],
                }
                if phaseflow_fusion is not None:
                    residue_scores["phaseflow_DPR"] = [float(value) for value in phaseflow_dpr_scores]
                    residue_scores["phaseflow_DPR"] = [float(value) for value in phaseflow_fusion.phaseflow_scores]
                    residue_scores["phaseflow_rank"] = [float(value) for value in phaseflow_fusion.phaseflow_rank]
                row = {
                    "protein_id": protein_id,
                    "length": length,
                    "LLPS_probability": float(output_llps),
                    "protein_llps_score": float(output_llps),
                    "phaseflow_LLPS_probability": float(phaseflow_llps),
                    "coordinate_system": "1-based inclusive",
                    "residue_scores": residue_scores,
                    "dpr_regions": _public_regions(regions),
                    "DPR_regions": _public_regions(regions),
                    "evidence": evidence,
                }
                if phaseflow_fusion is not None:
                    row["phaseflow_LLPS_proxy"] = float(phaseflow_fusion.phaseflow_llps_proxy)
                    row["fusion_method"] = f"phaseflow_phaseflow_{phaseflow_config.dpr_fusion_mode}"
                handle.write(json.dumps(row) + "\n")


def _public_regions(regions: list[dict[str, float]]) -> list[dict[str, float]]:
    converted: list[dict[str, float]] = []
    for region in regions:
        row = dict(region)
        row["start"] = int(row["start"]) + 1
        row["end"] = int(row["end"]) + 1
        converted.append(row)
    return converted


def _evidence_for_sample(modality_weights, modality_mask, structure_metadata: dict) -> dict[str, object]:
    names = ["plm", "physchem", "disorder", "protenix_embed", "starling_embed"]
    available = 1.0 - modality_mask.astype(float)
    weighted = modality_weights * available
    means = weighted.mean(axis=0) if weighted.size else [0.0] * len(names)
    order = sorted(range(len(names)), key=lambda index: float(means[index]), reverse=True)
    important = [names[index] for index in order if float(means[index]) > 0.05][:3]
    structure_provider = str(structure_metadata.get("structure_provider", "none"))
    return {
        "important_modalities": important,
        "modality_weights": {names[index]: float(means[index]) for index in range(len(names))},
        "structure_provider": structure_provider,
        "structure_success": str(structure_metadata.get("structure_success", "")),
        "structure_model": str(structure_metadata.get("model_name", "")),
    }


def _phaseflow_profile_out_path(out_path: Path) -> Path:
    if out_path.suffix:
        return out_path.with_suffix(".phaseflow_profiles.jsonl")
    return out_path.parent / f"{out_path.name}.phaseflow_profiles.jsonl"


def _read_sequences_from_feature_cache(feature_dir: str | Path, protein_ids: list[str]) -> dict[str, str]:
    directory = Path(feature_dir)
    records: dict[str, str] = {}
    for protein_id in protein_ids:
        path = directory / f"{protein_id}.h5"
        if not path.exists():
            raise FileNotFoundError(f"Missing feature cache for {protein_id}: {path}")
        with h5py.File(path, "r") as handle:
            value = handle.attrs.get("sequence", "")
            if isinstance(value, bytes):
                sequence = value.decode("utf-8")
            else:
                sequence = str(value)
        records[str(protein_id)] = sequence
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--protein-ids", nargs="*")
    parser.add_argument("--config", help="Accepted for CLI symmetry; checkpoint config is authoritative.")
    parser.add_argument("--phaseflow-fusion", action="store_true", help="Fuse PhaseFlow inference with PhaseFlow short-window evidence.")
    parser.add_argument("--phaseflow-root", type=Path, default=DEFAULT_PHASEFLOW_ROOT)
    parser.add_argument("--phaseflow-checkpoint", type=Path, default=DEFAULT_PHASEFLOW_CHECKPOINT)
    parser.add_argument(
        "--phaseflow-python",
        type=Path,
        default=DEFAULT_PHASEFLOW_PYTHON if DEFAULT_PHASEFLOW_PYTHON.exists() else None,
        help="Optional PhaseFlow environment Python. When set, profiles are generated in a subprocess.",
    )
    parser.add_argument("--phaseflow-profile-jsonl", type=Path, help="Reuse precomputed PhaseFlow profile JSONL.")
    parser.add_argument("--phaseflow-profile-out", type=Path, help="Where subprocess-generated PhaseFlow profiles should be written.")
    parser.add_argument("--phaseflow-device", default="auto")
    parser.add_argument("--phaseflow-batch-size", type=int, default=512)
    parser.add_argument("--phaseflow-window-sizes", default="10,20")
    parser.add_argument("--phaseflow-dpr-mode", default="rank_blend", choices=["rank_blend", "gated_lift"])
    parser.add_argument("--phaseflow-dpr-blend-alpha", type=float, default=0.15)
    parser.add_argument("--phaseflow-phaseflow-low", type=float, default=0.60)
    parser.add_argument("--phaseflow-phaseflow-high", type=float, default=0.68)
    parser.add_argument("--phaseflow-rank-gate", type=float, default=0.70)
    parser.add_argument("--phaseflow-lift", type=float, default=0.70)
    parser.add_argument("--phaseflow-lift-span", type=float, default=0.05)
    parser.add_argument("--phaseflow-llps-gate", type=float, default=0.45)
    parser.add_argument("--phaseflow-llps-max-phaseflow", type=float, default=1.00)
    parser.add_argument("--phaseflow-llps-boost-scale", type=float, default=0.50)
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError("Inference requires a trained checkpoint; no fallback prediction is provided.")
    if args.config:
        load_yaml(args.config)
    phaseflow_config = None
    if args.phaseflow_fusion:
        phaseflow_config = PhaseFlowFusionConfig(
            phaseflow_root=Path(args.phaseflow_root),
            checkpoint=Path(args.phaseflow_checkpoint),
            phaseflow_python=Path(args.phaseflow_python) if args.phaseflow_python else None,
            profile_jsonl=Path(args.phaseflow_profile_jsonl) if args.phaseflow_profile_jsonl else None,
            profile_out=Path(args.phaseflow_profile_out) if args.phaseflow_profile_out else None,
            device=str(args.phaseflow_device),
            batch_size=int(args.phaseflow_batch_size),
            window_sizes=parse_window_sizes(args.phaseflow_window_sizes),
            dpr_fusion_mode=str(args.phaseflow_dpr_mode),
            dpr_blend_alpha=float(args.phaseflow_dpr_blend_alpha),
            phaseflow_low=float(args.phaseflow_phaseflow_low),
            phaseflow_high=float(args.phaseflow_phaseflow_high),
            phaseflow_rank_gate=float(args.phaseflow_rank_gate),
            lift=float(args.phaseflow_lift),
            lift_span=float(args.phaseflow_lift_span),
            llps_gate=float(args.phaseflow_llps_gate),
            llps_max_phaseflow=float(args.phaseflow_llps_max_phaseflow),
            llps_boost_scale=float(args.phaseflow_llps_boost_scale),
        )
    run_inference(args.checkpoint, args.feature_dir, args.out, args.protein_ids, phaseflow_config)
    print(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
