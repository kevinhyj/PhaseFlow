from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from phaseflow.full_length.data.collator import PhaseFlowCollator
from phaseflow.full_length.data.config import resolve_feature_dirs, resolve_phase_targets, validate_forbidden_data_paths
from phaseflow.full_length.data.dataset import PhaseFlowDataset
from phaseflow.full_length.data.splits import resolve_split_ids
from phaseflow.full_length.metrics.key_metrics import key_topk_metrics
from phaseflow.full_length.metrics.protein_metrics import binary_classification_metrics
from phaseflow.full_length.metrics.region_metrics import boundary_f1, region_metrics
from phaseflow.full_length.metrics.residue_metrics import residue_binary_metrics
from phaseflow.full_length.models.phaseflow import PhaseFlowModel
from phaseflow.full_length.postprocess import combine_regions, decoder_regions, scores_to_regions
from phaseflow.full_length.utils import dumps_json, load_yaml, move_batch_to_device, resolve_device, write_json


@torch.no_grad()
def evaluate_model(
    model: PhaseFlowModel,
    loader: DataLoader,
    device: torch.device,
    postprocess_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    postprocess_config = postprocess_config or {}
    llps_labels: list[float] = []
    llps_scores: list[float] = []
    dpr_labels: list[np.ndarray] = []
    dpr_scores: list[np.ndarray] = []
    key_labels: list[np.ndarray] = []
    key_scores: list[np.ndarray] = []
    pred_regions: list[list[dict[str, float]]] = []
    true_regions: list[list[dict[str, object]]] = []
    negative_types: list[str] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(batch)
        llps_labels.extend(batch["y_llps"].detach().cpu().numpy().tolist())
        llps_output = outputs.get("llps_logits", outputs["llps_logits"])
        llps_scores.extend(torch.sigmoid(llps_output).detach().cpu().numpy().tolist())
        negative_types.extend(batch.get("negative_type", [""] * len(batch["protein_ids"])))
        lengths = batch["lengths"].detach().cpu().numpy()
        if "dpr_logits" in outputs and "key_logits" in outputs:
            dpr_prob = torch.sigmoid(outputs["dpr_logits"]).detach().cpu().numpy()
            key_prob = torch.sigmoid(outputs["key_logits"]).detach().cpu().numpy()
            dpr_label = batch["y_dpr"].detach().cpu().numpy()
            key_label = batch["y_key"].detach().cpu().numpy()
            region_logits = outputs.get("region_logits")
            region_start = outputs.get("region_start")
            region_end = outputs.get("region_end")
            if region_logits is not None:
                region_logits = region_logits.detach().cpu().numpy()
                region_start = region_start.detach().cpu().numpy()
                region_end = region_end.detach().cpu().numpy()
            for index, length in enumerate(lengths):
                dpr_labels.append(dpr_label[index, :length])
                dpr_scores.append(dpr_prob[index, :length])
                key_labels.append(key_label[index, :length])
                key_scores.append(key_prob[index, :length])
                post = scores_to_regions(
                    dpr_prob[index, :length],
                    threshold=float(postprocess_config.get("threshold", 0.5)),
                    smooth_window=int(postprocess_config.get("smooth_window", 5)),
                    merge_gap=int(postprocess_config.get("merge_gap", 5)),
                    min_region_len=int(postprocess_config.get("min_region_len", 6)),
                )
                if region_logits is not None and bool(postprocess_config.get("use_decoder_regions", False)):
                    dec = decoder_regions(
                        region_logits[index],
                        region_start[index],
                        region_end[index],
                        int(length),
                        score_threshold=float(postprocess_config.get("decoder_score_threshold", 0.5)),
                    )
                    pred_regions.append(combine_regions(dec, post))
                else:
                    pred_regions.append(post)
            true_regions.extend(batch["regions"])

    metrics = {}
    metrics.update(binary_classification_metrics(np.asarray(llps_labels), np.asarray(llps_scores)))
    if dpr_labels and dpr_scores:
        metrics.update(residue_binary_metrics(np.concatenate(dpr_labels), np.concatenate(dpr_scores)))
        metrics.update(key_topk_metrics(key_labels, key_scores, k=10))
        metrics.update(region_metrics(pred_regions, true_regions, iou_threshold=0.3))
        metrics.update(region_metrics(pred_regions, true_regions, iou_threshold=0.5))
        metrics.update(boundary_f1(pred_regions, true_regions))
    metrics.update(_hard_negative_fpr(np.asarray(llps_labels), np.asarray(llps_scores), negative_types))
    return metrics


def _hard_negative_fpr(labels: np.ndarray, scores: np.ndarray, negative_types: list[str], threshold: float = 0.5) -> dict[str, float]:
    result: dict[str, float] = {}
    normalized = [str(value).lower() for value in negative_types]
    groups = {
        "NP": ("structured", "np"),
        "ND": ("disordered", "nd"),
        "LCR": ("lcr",),
        "long_IDR": ("long_idr", "long-idr"),
    }
    preds = scores >= threshold
    labels = np.asarray(labels)
    for name, tokens in groups.items():
        mask = np.asarray([any(token in value for token in tokens) for value in normalized]) & (labels == 0)
        result[f"FPR_on_{name}"] = float(np.mean(preds[mask])) if np.any(mask) else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--out")
    args = parser.parse_args()
    config = load_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PhaseFlowModel(checkpoint.get("config", config))
    model.load_state_dict(checkpoint["model"])
    device = resolve_device(str(config.get("device", "auto")))
    model.to(device)
    ids = resolve_split_ids(config["data"], args.split)
    feature_dirs = resolve_feature_dirs(config["data"])
    validate_forbidden_data_paths(config["data"], feature_dirs)
    loader = DataLoader(
        PhaseFlowDataset(
            feature_dirs,
            ids,
            phase_targets=resolve_phase_targets(config["data"]),
            region_targets=config["data"].get("region_targets"),
        ),
        batch_size=int(config["training"].get("batch_size", 2)),
        collate_fn=PhaseFlowCollator(
            max_neighbors=int(config["training"].get("max_neighbors", 96)),
            require_precomputed_graph=bool(config["training"].get("require_precomputed_graph", False)),
        ),
    )
    metrics = evaluate_model(model, loader, device, config.get("postprocess", {}))
    if args.out:
        write_json(args.out, metrics)
    print(dumps_json(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
