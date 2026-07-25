from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from phaseflow.full_length.data.collator import PhaseFlowCollator
from phaseflow.full_length.data.dataset import PhaseFlowDataset
from phaseflow.full_length.data.splits import resolve_split_ids
from phaseflow.full_length.metrics.protein_metrics import binary_classification_metrics
from phaseflow.full_length.metrics.region_metrics import region_metrics
from phaseflow.full_length.metrics.residue_metrics import residue_binary_metrics
from phaseflow.full_length.models.phaseflow import PhaseFlowModel
from phaseflow.full_length.postprocess import combine_regions, decoder_regions, scores_to_regions
from phaseflow.full_length.utils import dumps_json, load_yaml, move_batch_to_device, resolve_device, write_json


@torch.no_grad()
def run_acceptance(
    checkpoint_path: str | Path,
    config_path: str | Path,
    split: str,
    out_dir: str | Path,
    batch_size: int | None = None,
    ablations: list[str] | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    out_dir = Path(out_dir)
    device = resolve_device(str(config.get("device", "auto")))
    ids = resolve_split_ids(config["data"], split)
    loader = DataLoader(
        PhaseFlowDataset(config["data"]["feature_dir"], ids),
        batch_size=int(batch_size or config["training"].get("batch_size", 2)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=PhaseFlowCollator(max_neighbors=int(config["training"].get("max_neighbors", 96))),
    )
    manifest = pd.read_csv(config["data"]["manifest"]).set_index("protein_id", drop=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    full_predictions = _predict(
        checkpoint=checkpoint,
        fallback_config=config,
        loader=loader,
        device=device,
        postprocess_config=config.get("postprocess", {}),
        ablation_name=None,
    )
    full_metrics = _metrics_from_predictions(full_predictions)
    hard_negative = _hard_negative_metrics(full_predictions, manifest)
    baselines = _baseline_metrics(full_predictions)

    ablation_metrics: dict[str, Any] = {}
    for ablation_name in ablations or []:
        predictions = _predict(
            checkpoint=checkpoint,
            fallback_config=config,
            loader=loader,
            device=device,
            postprocess_config=config.get("postprocess", {}),
            ablation_name=ablation_name,
        )
        ablation_metrics[ablation_name] = _metrics_from_predictions(predictions)

    feature_status = _feature_status(config["data"]["feature_dir"], ids)
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "split": split,
        "sample_count": len(ids),
        "feature_status": feature_status,
        "metrics": full_metrics,
        "hard_negative": hard_negative,
        "baselines": baselines,
        "inference_ablations": ablation_metrics,
        "acceptance": _acceptance_status(full_metrics, hard_negative, baselines, feature_status),
    }
    write_json(out_dir / f"{split}_acceptance.json", result)
    _write_markdown(out_dir / f"{split}_acceptance.md", result)
    return result


def _load_model(
    checkpoint: dict[str, Any],
    fallback_config: dict[str, Any],
    device: torch.device,
    ablation_name: str | None,
) -> torch.nn.Module:
    model_config = copy.deepcopy(checkpoint.get("config", fallback_config))
    if ablation_name:
        model_config.setdefault("model", {}).setdefault("ablation", {})["name"] = ablation_name
    model = PhaseFlowModel(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def _predict(
    checkpoint: dict[str, Any],
    fallback_config: dict[str, Any],
    loader: DataLoader,
    device: torch.device,
    postprocess_config: dict[str, Any],
    ablation_name: str | None,
) -> dict[str, Any]:
    model = _load_model(checkpoint, fallback_config, device, ablation_name)
    records: list[dict[str, Any]] = []
    dpr_labels: list[np.ndarray] = []
    dpr_scores: list[np.ndarray] = []
    pred_regions: list[list[dict[str, float]]] = []
    post_regions: list[list[dict[str, float]]] = []
    decoder_only_regions: list[list[dict[str, float]]] = []
    disorder_regions: list[list[dict[str, float]]] = []
    true_regions: list[list[dict[str, object]]] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        llps_output = outputs.get("llps_logits", outputs["llps_logits"])
        llps_scores = torch.sigmoid(llps_output).detach().cpu().numpy()
        dpr_prob = torch.sigmoid(outputs["dpr_logits"]).detach().cpu().numpy()
        dpr_label = batch["y_dpr"].detach().cpu().numpy()
        region_logits = outputs["region_logits"].detach().cpu().numpy()
        region_start = outputs["region_start"].detach().cpu().numpy()
        region_end = outputs["region_end"].detach().cpu().numpy()
        lengths = batch["lengths"].detach().cpu().numpy()
        labels = batch["y_llps"].detach().cpu().numpy()
        disorder = batch["disorder"].detach().cpu().numpy()

        for index, length_raw in enumerate(lengths):
            length = int(length_raw)
            scores = dpr_prob[index, :length]
            p_disorder = disorder[index, :length, 0]
            p_lcr = disorder[index, :length, 1] if disorder.shape[-1] > 1 else np.zeros(length, dtype=np.float32)
            p_prld = disorder[index, :length, 2] if disorder.shape[-1] > 2 else np.zeros(length, dtype=np.float32)
            post = scores_to_regions(
                scores,
                threshold=float(postprocess_config.get("threshold", 0.5)),
                smooth_window=int(postprocess_config.get("smooth_window", 5)),
                merge_gap=int(postprocess_config.get("merge_gap", 5)),
                min_region_len=int(postprocess_config.get("min_region_len", 6)),
            )
            dec = decoder_regions(
                region_logits[index],
                region_start[index],
                region_end[index],
                length,
                score_threshold=float(postprocess_config.get("decoder_score_threshold", 0.5)),
            )
            disorder_post = scores_to_regions(
                p_disorder,
                threshold=0.5,
                smooth_window=int(postprocess_config.get("smooth_window", 5)),
                merge_gap=int(postprocess_config.get("merge_gap", 5)),
                min_region_len=int(postprocess_config.get("min_region_len", 6)),
            )
            pred_regions.append(combine_regions(dec, post))
            post_regions.append(post)
            decoder_only_regions.append(dec)
            disorder_regions.append(disorder_post)
            true_regions.append(batch["regions"][index])
            dpr_labels.append(dpr_label[index, :length])
            dpr_scores.append(scores)
            records.append(
                {
                    "protein_id": str(batch["protein_ids"][index]),
                    "label": float(labels[index]),
                    "score": float(llps_scores[index]),
                    "length": length,
                    "mean_disorder": float(np.mean(p_disorder)) if length else 0.0,
                    "idr_fraction": float(np.mean(p_disorder >= 0.5)) if length else 0.0,
                    "mean_lcr": float(np.mean(p_lcr)) if length else 0.0,
                    "max_lcr": float(np.max(p_lcr)) if length else 0.0,
                    "mean_prld": float(np.mean(p_prld)) if length else 0.0,
                    "disorder_score": float(np.mean(p_disorder)) if length else 0.0,
                    "lcr_score": float(np.mean(p_lcr)) if length else 0.0,
                    "prld_score": float(np.mean(p_prld)) if length else 0.0,
                }
            )

    return {
        "records": records,
        "dpr_labels": dpr_labels,
        "dpr_scores": dpr_scores,
        "pred_regions": pred_regions,
        "post_regions": post_regions,
        "decoder_regions": decoder_only_regions,
        "disorder_regions": disorder_regions,
        "true_regions": true_regions,
    }


def _metrics_from_predictions(predictions: dict[str, Any]) -> dict[str, float]:
    records = predictions["records"]
    labels = np.asarray([record["label"] for record in records], dtype=np.float32)
    scores = np.asarray([record["score"] for record in records], dtype=np.float32)
    metrics = binary_classification_metrics(labels, scores)
    metrics.update(residue_binary_metrics(np.concatenate(predictions["dpr_labels"]), np.concatenate(predictions["dpr_scores"])))
    metrics.update(region_metrics(predictions["pred_regions"], predictions["true_regions"], iou_threshold=0.3))
    metrics.update(region_metrics(predictions["pred_regions"], predictions["true_regions"], iou_threshold=0.5))
    post_03 = region_metrics(predictions["post_regions"], predictions["true_regions"], iou_threshold=0.3)
    post_05 = region_metrics(predictions["post_regions"], predictions["true_regions"], iou_threshold=0.5)
    decoder_03 = region_metrics(predictions["decoder_regions"], predictions["true_regions"], iou_threshold=0.3)
    decoder_05 = region_metrics(predictions["decoder_regions"], predictions["true_regions"], iou_threshold=0.5)
    metrics.update({f"threshold_only_{key}": value for key, value in {**post_03, **post_05}.items()})
    metrics.update({f"decoder_only_{key}": value for key, value in {**decoder_03, **decoder_05}.items()})
    return metrics


def _baseline_metrics(predictions: dict[str, Any]) -> dict[str, Any]:
    records = predictions["records"]
    labels = np.asarray([record["label"] for record in records], dtype=np.float32)
    baselines: dict[str, Any] = {}
    for name in ("disorder_score", "lcr_score", "prld_score"):
        scores = np.asarray([record[name] for record in records], dtype=np.float32)
        baselines[name] = binary_classification_metrics(labels, scores)
    disorder_03 = region_metrics(predictions["disorder_regions"], predictions["true_regions"], iou_threshold=0.3)
    disorder_05 = region_metrics(predictions["disorder_regions"], predictions["true_regions"], iou_threshold=0.5)
    baselines["disorder_region_threshold"] = {**disorder_03, **disorder_05}
    return baselines


def _hard_negative_metrics(predictions: dict[str, Any], manifest: pd.DataFrame) -> dict[str, Any]:
    records = []
    for record in predictions["records"]:
        protein_id = record["protein_id"]
        manifest_row = manifest.loc[protein_id] if protein_id in manifest.index else {}
        enriched = dict(record)
        enriched["negative_type"] = str(_manifest_value(manifest_row, "negative_type", "unknown")).strip().lower()
        enriched["role_label"] = str(_manifest_value(manifest_row, "role_label", "unknown")).strip().lower()
        records.append(enriched)

    groups: dict[str, list[dict[str, Any]]] = {
        "all_negatives": [record for record in records if int(record["label"]) == 0],
        "NP_structured": [
            record
            for record in records
            if int(record["label"]) == 0 and record["negative_type"] in {"structured", "np_structured"}
        ],
        "ND_disordered": [
            record
            for record in records
            if int(record["label"]) == 0 and record["negative_type"] in {"disordered", "nd_disordered"}
        ],
        "LCR_enriched_heuristic": [
            record
            for record in records
            if int(record["label"]) == 0 and (record["mean_lcr"] >= 0.20 or record["max_lcr"] >= 0.45)
        ],
        "long_IDR_heuristic": [
            record
            for record in records
            if int(record["label"]) == 0 and record["length"] >= 200 and record["idr_fraction"] >= 0.30
        ],
    }
    return {name: _negative_group_metrics(group) for name, group in groups.items()}


def _negative_group_metrics(records: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, float]:
    if not records:
        return {"n": 0, "fpr": float("nan"), "mean_score": float("nan"), "p90_score": float("nan")}
    scores = np.asarray([record["score"] for record in records], dtype=np.float32)
    return {
        "n": len(records),
        "fpr": float(np.mean(scores >= threshold)),
        "mean_score": float(np.mean(scores)),
        "p90_score": float(np.quantile(scores, 0.90)),
    }


def _feature_status(feature_dir: str | Path, ids: list[str]) -> dict[str, Any]:
    feature_dir = Path(feature_dir)
    missing = [protein_id for protein_id in ids if not (feature_dir / f"{protein_id}.h5").exists()]
    return {"feature_dir": str(feature_dir), "missing_count": len(missing), "missing_preview": missing[:10]}


def _acceptance_status(
    metrics: dict[str, float],
    hard_negative: dict[str, Any],
    baselines: dict[str, Any],
    feature_status: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    if feature_status["missing_count"]:
        issues.append(f"missing feature files: {feature_status['missing_count']}")
    if metrics.get("auc", 0.0) < 0.85:
        issues.append("LLPS AUROC below 0.85")
    if metrics.get("prauc", 0.0) < 0.90:
        issues.append("LLPS AUPRC below 0.90")
    if hard_negative.get("ND_disordered", {}).get("n", 0) and hard_negative["ND_disordered"]["fpr"] > 0.40:
        issues.append("ND_disordered FPR above 0.40")
    disorder_auc = baselines.get("disorder_score", {}).get("auc")
    if disorder_auc == disorder_auc and metrics.get("auc", 0.0) - disorder_auc < 0.03:
        issues.append("model AUROC is less than 0.03 above disorder baseline")
    model_iou = metrics.get("region_iou@0.5_recall")
    threshold_iou = metrics.get("threshold_only_region_iou@0.5_recall")
    if model_iou == model_iou and threshold_iou == threshold_iou and model_iou < threshold_iou:
        issues.append("combined region IoU@0.5 recall below threshold-only residue baseline")
    return {"status": "pass" if not issues else "conditional", "issues": issues}


def _manifest_value(row: Any, name: str, default: Any) -> Any:
    if isinstance(row, pd.Series) and name in row:
        value = row[name]
        if pd.isna(value):
            return default
        return value
    return default


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    hard_negative = result["hard_negative"]
    baselines = result["baselines"]
    lines = [
        f"# PhaseFlow Acceptance Report: {result['split']}",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- samples: {result['sample_count']}",
        f"- feature missing: {result['feature_status']['missing_count']}",
        f"- acceptance status: **{result['acceptance']['status']}**",
        "",
        "## Main Metrics",
        "",
        "| AUROC | AUPRC | F1 | MCC | FPR | Residue AUC | Residue AUPRC | IoU@0.3 recall | IoU@0.5 recall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {_fmt(metrics.get('auc'))} | {_fmt(metrics.get('prauc'))} | {_fmt(metrics.get('f1'))} | "
            f"{_fmt(metrics.get('mcc'))} | {_fmt(metrics.get('fpr'))} | {_fmt(metrics.get('residue_auc'))} | "
            f"{_fmt(metrics.get('residue_prauc'))} | {_fmt(metrics.get('region_iou@0.3_recall'))} | "
            f"{_fmt(metrics.get('region_iou@0.5_recall'))} |"
        ),
        "",
        "## Hard Negative Stress",
        "",
        "| Group | N | FPR | Mean score | P90 score |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in hard_negative.items():
        lines.append(
            f"| {name} | {values.get('n', 0)} | {_fmt(values.get('fpr'))} | "
            f"{_fmt(values.get('mean_score'))} | {_fmt(values.get('p90_score'))} |"
        )
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| Baseline | AUROC | AUPRC | FPR | Region IoU@0.5 recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, values in baselines.items():
        lines.append(
            f"| {name} | {_fmt(values.get('auc'))} | {_fmt(values.get('prauc'))} | "
            f"{_fmt(values.get('fpr'))} | {_fmt(values.get('region_iou@0.5_recall'))} |"
        )
    if result["acceptance"]["issues"]:
        lines.extend(["", "## Acceptance Issues", ""])
        lines.extend(f"- {issue}" for issue in result["acceptance"]["issues"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:
        return "NA"
    return f"{numeric:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--ablations",
        default="no_starling,no_disorder,no_af",
        help="Comma-separated inference-time modality ablations.",
    )
    args = parser.parse_args()
    ablations = [item.strip() for item in args.ablations.split(",") if item.strip()]
    result = run_acceptance(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        ablations=ablations,
    )
    print(dumps_json(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
