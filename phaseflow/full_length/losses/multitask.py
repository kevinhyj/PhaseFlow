from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from phaseflow.full_length.losses.consistency_loss import smoothness_loss
from phaseflow.full_length.losses.dice import dice_loss_with_logits
from phaseflow.full_length.losses.focal import focal_loss_with_logits
from phaseflow.full_length.losses.final_region import (
    boundary_transition_loss,
    residue_contrastive_margin_loss,
    weighted_soft_bce_logits,
)
from phaseflow.full_length.losses.masked_bce import masked_bce_with_logits, protein_bce_class_normalized_stats, protein_bce_with_logits
from phaseflow.full_length.losses.phase_aux import phase_diagram_loss
from phaseflow.full_length.losses.region_loss import region_coverage_loss, region_query_loss
from phaseflow.full_length.losses.teacher import (
    calibration_loss_with_logits,
    negative_region_regularization,
    nnpu_loss_with_logits,
    region_bag_mil_loss,
    soft_bce_with_logits,
    teacher_distillation_mse,
    weak_region_mil_loss,
)


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    protein_logit_key = str(weights.get("protein_logit_key", "loss_llps_logits"))
    protein_logits = outputs.get(protein_logit_key)
    if protein_logits is None:
        protein_logits = outputs.get("loss_llps_logits", outputs["llps_logits"])
    protein_logits = protein_logits.float()
    if bool(weights.get("protein_bce_class_normalized", False)):
        llps, llps_stats = protein_bce_class_normalized_stats(
            protein_logits,
            batch["y_llps"],
            batch.get("sample_weight"),
            alpha_pos=float(weights.get("protein_bce_alpha_pos", 0.5)),
            alpha_neg=float(weights.get("protein_bce_alpha_neg", 0.5)),
        )
    else:
        llps = protein_bce_with_logits(protein_logits, batch["y_llps"], batch.get("sample_weight"))
        zero = llps.detach() * 0.0
        llps_stats = {
            "protein_loss_pos": zero,
            "protein_loss_neg": zero,
            "protein_loss_pos_count": zero,
            "protein_loss_neg_count": zero,
            "protein_loss_missing_class": zero,
        }
    weighted_focal_bce = (
        weighted_focal_bce_with_logits(
            protein_logits,
            batch,
            gamma=float(weights.get("focal_gamma", 1.5)),
            positive_weight=float(weights.get("positive_weight", 1.0)),
            negative_weight=float(weights.get("negative_weight", 1.0)),
            client_weight=float(weights.get("client_weight", 1.0)),
            nd_weight=float(weights.get("nd_weight", 1.0)),
        )
        if float(weights.get("weighted_focal_bce", 0.0)) > 0.0
        else protein_logits.sum() * 0.0
    )
    teacher_llps = soft_bce_with_logits(protein_logits, batch["teacher_llps"], batch["teacher_llps_weight"])
    self_llps = soft_bce_with_logits(protein_logits, batch["self_llps"], batch["self_llps_weight"])
    nnpu = nnpu_loss_with_logits(
        protein_logits,
        batch["y_llps"],
        batch.get("sample_weight"),
        positive_prior=float(weights.get("positive_prior", 0.1)),
    )
    calibration = calibration_loss_with_logits(protein_logits, batch["y_llps"])
    zero_dpr_loss = protein_logits.sum() * 0.0
    dpr_logits = outputs.get("dpr_logits")
    if torch.is_tensor(dpr_logits):
        zero_dpr_loss = zero_dpr_loss + dpr_logits.sum() * 0.0
    region_gold_weight = float(weights.get("region_gold", weights.get("dpr", 0.0)))
    region_gold = (
        0.5 * masked_bce_with_logits(outputs["dpr_logits"], batch["y_dpr"], batch["seq_mask"], batch["y_weight"])
        + 0.3 * dice_loss_with_logits(outputs["dpr_logits"], batch["y_dpr"], batch["seq_mask"])
        + 0.2 * focal_loss_with_logits(outputs["dpr_logits"], batch["y_dpr"], batch["seq_mask"], batch["y_weight"])
        if region_gold_weight > 0.0
        else zero_dpr_loss
    )
    region_mil_weight = float(weights.get("region_mil", weights.get("region_MIL", 0.0)))
    region_mil = (
        region_bag_mil_loss(
            outputs["region_global_logits"],
            batch["region_bag_label"],
            batch["region_bag_weight"],
        )
        if region_mil_weight > 0.0
        else zero_dpr_loss
    )
    teacher_dpr_bce = (
        weak_region_mil_loss(outputs["dpr_logits"], batch["teacher_dpr"], batch["teacher_dpr_weight"], batch["seq_mask"])
        if float(weights.get("teacher_dpr", 0.0)) > 0.0
        else zero_dpr_loss
    )
    teacher_distill = (
        teacher_distillation_mse(
            outputs["dpr_logits"],
            batch["teacher_dpr"],
            batch["teacher_dpr_weight"],
            batch["seq_mask"],
        )
        if float(weights.get("teacher_distill", 0.0)) > 0.0
        else zero_dpr_loss
    )
    self_dpr = (
        weak_region_mil_loss(outputs["dpr_logits"], batch["self_dpr"], batch["self_dpr_weight"], batch["seq_mask"])
        if float(weights.get("self_dpr", 0.0)) > 0.0
        else zero_dpr_loss
    )
    region = (
        region_query_loss(outputs["region_logits"], outputs["region_start"], outputs["region_end"], batch["regions"], batch["lengths"])
        if float(weights.get("region", 0.0)) > 0.0
        else zero_dpr_loss
    )
    coverage = (
        region_coverage_loss(outputs["region_logits"], outputs["region_start"], outputs["region_end"], batch["regions"], batch["lengths"])
        if float(weights.get("coverage", 0.0)) > 0.0
        else zero_dpr_loss
    )
    key = (
        masked_bce_with_logits(outputs["key_logits"], batch["y_key"], batch["seq_mask"], batch["y_weight"])
        if float(weights.get("key", 0.0)) > 0.0
        else zero_dpr_loss
    )
    smooth_weight = float(weights.get("smoothness", 0.05))
    smooth = smoothness_loss(outputs["dpr_logits"], batch["seq_mask"]) if smooth_weight > 0.0 else zero_dpr_loss
    negative_regularization_weight = float(weights.get("negative_regularization", 0.0))
    negative_regularization = (
        negative_region_regularization(
            outputs["dpr_logits"],
            batch["seq_mask"],
            batch["negative_regularization_weight"],
        )
        if negative_regularization_weight > 0.0
        else zero_dpr_loss
    )
    phase_aux_weight = float(weights.get("phase_aux", weights.get("phase", 0.0)))
    phase_aux = (
        phase_diagram_loss(
            outputs,
            batch,
            pssi_min=float(weights.get("phase_pssi_min", -2.17)),
            pssi_max=float(weights.get("phase_pssi_max", 1.64)),
            mean_weight=float(weights.get("phase_mean_weight", 0.25)),
        )
        if phase_aux_weight > 0.0
        else zero_dpr_loss
    )
    final_region_teacher = (
        weighted_soft_bce_logits(
            outputs["dpr_logits"],
            batch["region_teacher_target"],
            batch["region_teacher_weight"],
            batch["seq_mask"],
        )
        if float(weights.get("final_region_teacher", 0.0)) > 0.0
        else zero_dpr_loss
    )
    final_key_teacher = (
        weighted_soft_bce_logits(outputs["key_logits"], batch["region_key_target"], batch["region_key_weight"], batch["seq_mask"])
        if float(weights.get("final_key_teacher", 0.0)) > 0.0
        else zero_dpr_loss
    )
    final_boundary = (
        boundary_transition_loss(
            outputs["dpr_logits"],
            batch["region_boundary_target"],
            batch["region_boundary_weight"],
            batch["seq_mask"],
        )
        if float(weights.get("final_boundary", 0.0)) > 0.0
        else zero_dpr_loss
    )
    final_contrastive = (
        residue_contrastive_margin_loss(
            outputs["dpr_logits"],
            batch["region_contrast_target"],
            batch["region_contrast_weight"],
            batch["seq_mask"],
            margin=float(weights.get("final_region_contrastive_margin", 0.35)),
        )
        if float(weights.get("final_contrastive", 0.0)) > 0.0
        else zero_dpr_loss
    )
    ranking_weight = float(weights.get("ranking_loss_weight", 0.0))
    top_negative_ranking = (
        top_negative_ranking_loss(
            protein_logits,
            batch,
            margin=float(weights.get("ranking_loss_margin", 0.7)),
            topk_negatives=int(weights.get("ranking_loss_topk_negatives", 4)),
            positive_pool_names=weights.get("ranking_positive_pool_names"),
            negative_pool_names=weights.get("ranking_negative_pool_names"),
        )
        if bool(weights.get("ranking_loss_enabled", False)) and ranking_weight > 0.0
        else protein_logits.sum() * 0.0
    )
    hard_negative_focal_weight = float(weights.get("hard_negative_focal_weight", 0.0))
    hard_negative_focal = (
        hard_negative_focal_loss(
            protein_logits,
            batch,
            gamma=float(weights.get("hard_negative_focal_gamma", 2.0)),
            pool_names=weights.get("hard_negative_focal_pool_names"),
        )
        if hard_negative_focal_weight > 0.0
        else protein_logits.sum() * 0.0
    )
    pairwise_rank_weight = float(weights.get("pairwise_rank_loss_weight", weights.get("rank_loss_weight", 0.0)))
    pairwise_rank = (
        pairwise_logistic_ranking_loss(
            protein_logits,
            batch,
            topk_negatives=int(weights.get("rank_loss_topk_negatives", 16)),
            positive_pool_names=weights.get("ranking_positive_pool_names"),
            negative_pool_names=weights.get("ranking_negative_pool_names"),
            client_nd_weight=float(weights.get("client_nd_rank_weight", 1.5)),
            nd_pair_weight=float(weights.get("nd_rank_weight", 1.5)),
        )
        if pairwise_rank_weight > 0.0
        else protein_logits.sum() * 0.0
    )
    driver_aux_weight = float(weights.get("driver_head", weights.get("driver_loss_weight", 0.0)))
    client_aux_weight = float(weights.get("client_head", weights.get("client_loss_weight", 0.0)))
    negtype_aux_weight = float(weights.get("negtype_head", weights.get("negtype_loss_weight", 0.0)))
    driver_aux = (
        role_bce_aux_loss(outputs.get("driver_logits"), batch, role="driver")
        if driver_aux_weight > 0.0
        else protein_logits.sum() * 0.0
    )
    client_aux = (
        role_bce_aux_loss(outputs.get("client_logits"), batch, role="client")
        if client_aux_weight > 0.0
        else protein_logits.sum() * 0.0
    )
    negtype_aux = (
        negtype_ce_aux_loss(outputs.get("negtype_logits"), batch)
        if negtype_aux_weight > 0.0
        else protein_logits.sum() * 0.0
    )
    total = (
        float(weights.get("llps", 1.0)) * llps
        + float(weights.get("weighted_focal_bce", 0.0)) * weighted_focal_bce
        + float(weights.get("teacher_llps", 0.0)) * teacher_llps
        + float(weights.get("self_llps", 0.0)) * self_llps
        + float(weights.get("nnpu", 0.0)) * nnpu
        + float(weights.get("calibration", 0.0)) * calibration
        + region_gold_weight * region_gold
        + region_mil_weight * region_mil
        + float(weights.get("teacher_dpr", 0.0)) * teacher_dpr_bce
        + float(weights.get("teacher_distill", 0.0)) * teacher_distill
        + float(weights.get("self_dpr", 0.0)) * self_dpr
        + float(weights.get("region", 0.0)) * region
        + float(weights.get("coverage", 0.0)) * coverage
        + float(weights.get("key", 0.0)) * key
        + smooth_weight * smooth
        + negative_regularization_weight * negative_regularization
        + phase_aux_weight * phase_aux
        + float(weights.get("final_region_teacher", 0.0)) * final_region_teacher
        + float(weights.get("final_key_teacher", 0.0)) * final_key_teacher
        + float(weights.get("final_boundary", 0.0)) * final_boundary
        + float(weights.get("final_contrastive", 0.0)) * final_contrastive
        + ranking_weight * top_negative_ranking
        + hard_negative_focal_weight * hard_negative_focal
        + pairwise_rank_weight * pairwise_rank
        + driver_aux_weight * driver_aux
        + client_aux_weight * client_aux
        + negtype_aux_weight * negtype_aux
    )
    values = {
        "loss": float(total.detach().cpu()),
        "llps": float(llps.detach().cpu()),
        "protein_loss_pos": float(llps_stats["protein_loss_pos"].detach().cpu()),
        "protein_loss_neg": float(llps_stats["protein_loss_neg"].detach().cpu()),
        "protein_loss_pos_count": float(llps_stats["protein_loss_pos_count"].detach().cpu()),
        "protein_loss_neg_count": float(llps_stats["protein_loss_neg_count"].detach().cpu()),
        "protein_loss_missing_class": float(llps_stats["protein_loss_missing_class"].detach().cpu()),
        "teacher_llps": float(teacher_llps.detach().cpu()),
        "self_llps": float(self_llps.detach().cpu()),
        "nnpu": float(nnpu.detach().cpu()),
        "calibration": float(calibration.detach().cpu()),
        "dpr": float(region_gold.detach().cpu()),
        "region_gold": float(region_gold.detach().cpu()),
        "region_mil": float(region_mil.detach().cpu()),
        "teacher_dpr": float(teacher_dpr_bce.detach().cpu()),
        "teacher_distill": float(teacher_distill.detach().cpu()),
        "self_dpr": float(self_dpr.detach().cpu()),
        "region": float(region.detach().cpu()),
        "coverage": float(coverage.detach().cpu()),
        "key": float(key.detach().cpu()),
        "smoothness": float(smooth.detach().cpu()),
        "negative_regularization": float(negative_regularization.detach().cpu()),
        "phase_aux": float(phase_aux.detach().cpu()),
        "final_region_teacher": float(final_region_teacher.detach().cpu()),
        "final_key_teacher": float(final_key_teacher.detach().cpu()),
        "final_boundary": float(final_boundary.detach().cpu()),
        "final_contrastive": float(final_contrastive.detach().cpu()),
        "top_negative_ranking": float(top_negative_ranking.detach().cpu()),
        "hard_negative_focal": float(hard_negative_focal.detach().cpu()),
        "weighted_focal_bce": float(weighted_focal_bce.detach().cpu()),
        "pairwise_rank": float(pairwise_rank.detach().cpu()),
        "driver_aux": float(driver_aux.detach().cpu()),
        "client_aux": float(client_aux.detach().cpu()),
        "negtype_aux": float(negtype_aux.detach().cpu()),
    }
    return total, values


def weighted_focal_bce_with_logits(
    protein_logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    gamma: float,
    positive_weight: float,
    negative_weight: float,
    client_weight: float,
    nd_weight: float,
) -> torch.Tensor:
    logits = protein_logits.float().reshape(-1)
    labels = batch["y_llps"].to(device=logits.device, dtype=torch.float32).reshape(-1)
    valid = labels.ge(0.0) & labels.le(1.0)
    if not torch.any(valid):
        return logits.sum() * 0.0
    logits = logits[valid]
    labels = labels[valid]
    weights = batch.get("sample_weight")
    if torch.is_tensor(weights):
        sample_weight = weights.to(device=logits.device, dtype=torch.float32).reshape(-1)[valid]
    else:
        sample_weight = torch.ones_like(labels)
    texts = _row_texts(batch)
    text_valid = [texts[index] for index, keep in enumerate(valid.detach().cpu().tolist()) if keep]
    role_weight = torch.ones_like(labels)
    role_weight = torch.where(labels.eq(1.0), role_weight * float(positive_weight), role_weight * float(negative_weight))
    client_mask = torch.tensor(["client" in text or "member" in text for text in text_valid], device=logits.device)
    nd_mask = torch.tensor(["disordered" in text or "nd" in text or "n_disordered" in text for text in text_valid], device=logits.device)
    role_weight = torch.where(client_mask & labels.eq(1.0), role_weight * float(client_weight), role_weight)
    role_weight = torch.where(nd_mask & labels.eq(0.0), role_weight * float(nd_weight), role_weight)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    pt = torch.exp(-bce).clamp(min=1.0e-6, max=1.0)
    loss = ((1.0 - pt).pow(float(gamma)) * bce) * sample_weight * role_weight
    denom = (sample_weight * role_weight).sum().clamp(min=1.0)
    return loss.sum() / denom


def pairwise_logistic_ranking_loss(
    protein_logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    topk_negatives: int,
    positive_pool_names: Any = None,
    negative_pool_names: Any = None,
    client_nd_weight: float = 1.5,
    nd_pair_weight: float = 1.5,
) -> torch.Tensor:
    scores = protein_logits.float().reshape(-1)
    labels = batch["y_llps"].to(device=scores.device, dtype=torch.float32).reshape(-1)
    pools = [str(item) for item in batch.get("plan_pool_name", [])]
    if len(pools) != scores.numel():
        pools = ["" for _ in range(int(scores.numel()))]
    positive_pools = _pool_name_set(positive_pool_names, set())
    negative_pools = _pool_name_set(negative_pool_names, set())
    pos_mask = labels.eq(1.0)
    if positive_pools:
        pos_mask &= torch.tensor([pool in positive_pools for pool in pools], device=scores.device, dtype=torch.bool)
    neg_mask = labels.eq(0.0)
    if negative_pools:
        neg_mask &= torch.tensor([pool in negative_pools for pool in pools], device=scores.device, dtype=torch.bool)
    positives = scores[pos_mask]
    negatives_all = scores[neg_mask]
    if positives.numel() == 0 or negatives_all.numel() == 0:
        return scores.sum() * 0.0
    k = max(1, min(int(topk_negatives), int(negatives_all.numel())))
    neg_values, neg_indices_local = torch.topk(negatives_all, k=k, largest=True)
    pos_texts = [text for text, keep in zip(_row_texts(batch), pos_mask.detach().cpu().tolist(), strict=False) if keep]
    neg_texts_all = [text for text, keep in zip(_row_texts(batch), neg_mask.detach().cpu().tolist(), strict=False) if keep]
    neg_texts = [neg_texts_all[int(index)] for index in neg_indices_local.detach().cpu().tolist()]
    pair_loss = F.softplus(-(positives.unsqueeze(1) - neg_values.unsqueeze(0)))
    pair_weight = torch.ones_like(pair_loss)
    pos_client = torch.tensor(
        ["client" in text or "member" in text for text in pos_texts],
        device=scores.device,
        dtype=torch.bool,
    ).unsqueeze(1)
    neg_nd = torch.tensor(
        ["disordered" in text or "nd" in text or "n_disordered" in text for text in neg_texts],
        device=scores.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    pair_weight = torch.where(neg_nd, pair_weight * float(nd_pair_weight), pair_weight)
    pair_weight = torch.where(pos_client & neg_nd, pair_weight * float(client_nd_weight), pair_weight)
    return (pair_loss * pair_weight).sum() / pair_weight.sum().clamp(min=1.0)


def role_bce_aux_loss(logits: torch.Tensor | None, batch: dict[str, Any], *, role: str) -> torch.Tensor:
    if logits is None:
        fallback = batch["y_llps"].float().sum() * 0.0
        return fallback
    out = logits.float().reshape(-1)
    labels = batch["y_llps"].to(device=out.device, dtype=torch.float32).reshape(-1)
    positive = labels.eq(1.0)
    if not torch.any(positive):
        return out.sum() * 0.0
    texts = _row_texts(batch)
    if role == "driver":
        targets = ["driver" in text or "scaffold" in text or "p_gold" in text for text in texts]
    elif role == "client":
        targets = ["client" in text or "member" in text for text in texts]
    else:
        raise ValueError(f"Unsupported role aux target: {role}")
    target = torch.tensor(targets, device=out.device, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(out[positive], target[positive])


def negtype_ce_aux_loss(logits: torch.Tensor | None, batch: dict[str, Any]) -> torch.Tensor:
    if logits is None:
        return batch["y_llps"].float().sum() * 0.0
    out = logits.float()
    labels = batch["y_llps"].to(device=out.device, dtype=torch.float32).reshape(-1)
    texts = _row_texts(batch)
    structured = ["structured" in text or "np" in text or "n_structured" in text for text in texts]
    disordered = ["disordered" in text or "nd" in text or "n_disordered" in text for text in texts]
    mask = labels.eq(0.0) & torch.tensor(
        [s or d for s, d in zip(structured, disordered, strict=False)],
        device=out.device,
        dtype=torch.bool,
    )
    if not torch.any(mask):
        return out.sum() * 0.0
    target = torch.tensor([1 if d else 0 for d in disordered], device=out.device, dtype=torch.long)
    return F.cross_entropy(out[mask], target[mask])


def _row_texts(batch: dict[str, Any]) -> list[str]:
    fields = [
        batch.get("plan_pool_name", []),
        batch.get("plan_tier", []),
        batch.get("plan_negative_type", []),
        batch.get("negative_type", []),
        batch.get("label_quality", []),
        batch.get("llps_role", []),
        batch.get("source", []),
    ]
    n = 0
    for field in fields:
        try:
            n = max(n, len(field))
        except TypeError:
            continue
    out: list[str] = []
    for index in range(n):
        parts = []
        for field in fields:
            try:
                value = field[index]
            except Exception:
                value = ""
            parts.append(str(value).lower())
        out.append(" ".join(parts))
    return out


def top_negative_ranking_loss(
    protein_logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    margin: float = 0.7,
    topk_negatives: int = 4,
    positive_pool_names: Any = None,
    negative_pool_names: Any = None,
) -> torch.Tensor:
    scores = torch.sigmoid(protein_logits.float()).reshape(-1)
    labels = batch["y_llps"].to(device=scores.device, dtype=torch.float32).reshape(-1)
    pools = [str(item) for item in batch.get("plan_pool_name", [])]
    if len(pools) != scores.numel():
        pools = ["" for _ in range(int(scores.numel()))]

    strong_positive_pools = _pool_name_set(
        positive_pool_names,
        {"P_gold", "P_curated", "P_pseudo_high", "P_mixed_curated_high"},
    )
    negative_pools = _pool_name_set(negative_pool_names, set())
    positive_mask = torch.tensor(
        [pool in strong_positive_pools for pool in pools],
        device=scores.device,
        dtype=torch.bool,
    ) & labels.eq(1.0)
    negative_mask = torch.tensor(
        [(pool in negative_pools) if negative_pools else pool.startswith("N_") for pool in pools],
        device=scores.device,
        dtype=torch.bool,
    ) & labels.eq(0.0)

    positives = scores[positive_mask]
    negatives = scores[negative_mask]
    if positives.numel() == 0 or negatives.numel() == 0:
        return scores.sum() * 0.0
    k = max(1, min(int(topk_negatives), int(negatives.numel())))
    hard_negatives = torch.topk(negatives, k=k, largest=True).values
    pairwise = float(margin) - positives.unsqueeze(1) + hard_negatives.unsqueeze(0)
    return F.relu(pairwise).mean()


def hard_negative_focal_loss(
    protein_logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    gamma: float = 2.0,
    pool_names: Any = None,
) -> torch.Tensor:
    logits = protein_logits.float().reshape(-1)
    labels = batch["y_llps"].to(device=logits.device, dtype=torch.float32).reshape(-1)
    pools = [str(item) for item in batch.get("plan_pool_name", [])]
    if len(pools) != logits.numel():
        pools = ["" for _ in range(int(logits.numel()))]
    hard_pools = _pool_name_set(pool_names, {"N_hard"})
    hard_mask = torch.tensor(
        [pool in hard_pools for pool in pools],
        device=logits.device,
        dtype=torch.bool,
    ) & labels.eq(0.0)
    if not torch.any(hard_mask):
        return logits.sum() * 0.0
    selected = logits[hard_mask]
    target = torch.zeros_like(selected)
    bce = F.binary_cross_entropy_with_logits(selected, target, reduction="none")
    prob = torch.sigmoid(selected)
    return (prob.pow(float(gamma)) * bce).mean()


def _pool_name_set(value: Any, default: set[str]) -> set[str]:
    if value is None:
        return set(default)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set(default)
        return {item.strip() for item in text.split(",") if item.strip()}
    try:
        items = {str(item).strip() for item in value if str(item).strip()}
    except TypeError:
        return set(default)
    return items or set(default)
