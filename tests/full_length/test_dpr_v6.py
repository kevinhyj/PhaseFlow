from __future__ import annotations

from collections import Counter

import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from phaseflow.full_length.data.dpr_v6 import build_fixed_schedule, validate_schedule
from phaseflow.full_length.models.dpr_v6 import (
    DPRV6Head,
    DPRV6LossConfig,
    bag_from_profiles,
    dpr_v6_loss,
    masked_avg_pool1d_same,
)
from phaseflow.full_length.training.dpr_v6 import reduce_rows
from scripts.full_length.evaluation.analyze_dpr_v6_threshold_curves import safe_spearman


def test_masked_avg_pool_edges_short_and_padding() -> None:
    x = torch.arange(1, 6, dtype=torch.float32).view(1, 5, 1)
    mask = torch.tensor([[True, True, True, False, False]])
    out = masked_avg_pool1d_same(x, mask, kernel_size=5).squeeze(0).squeeze(-1)
    expected = torch.tensor([
        (1 + 2 + 3) / 3,
        (1 + 2 + 3) / 3,
        (1 + 2 + 3) / 3,
        0.0,
        0.0,
    ])
    assert torch.allclose(out, expected)


def test_bag_hard_is_mean_of_three_scale_maxima() -> None:
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    p33 = torch.tensor([[0.1, 0.7, 0.2, 0.0], [0.3, 0.4, 0.0, 0.0]])
    p129 = torch.tensor([[0.5, 0.2, 0.4, 0.0], [0.6, 0.2, 0.0, 0.0]])
    p257 = torch.tensor([[0.9, 0.1, 0.2, 0.0], [0.1, 0.8, 0.0, 0.0]])
    bag = bag_from_profiles([p33, p129, p257], mask)
    expected = torch.stack([torch.tensor([0.7, 0.4]), torch.tensor([0.5, 0.6]), torch.tensor([0.9, 0.8])]).mean(dim=0)
    assert torch.allclose(bag["bag_hard"], expected)


def test_tiny_head_no_bypass_and_gradients() -> None:
    torch.manual_seed(7)
    h = torch.randn(2, 8, 6, requires_grad=True)
    mask = torch.ones(2, 8, dtype=torch.bool)
    head = DPRV6Head(6, head_type="tiny")
    out = head(h, mask)
    recomputed = (out["p33"].max(dim=1).values + out["p129"].max(dim=1).values + out["p257"].max(dim=1).values) / 3.0
    assert torch.allclose(out["bag_hard"], recomputed)
    loss = out["bag_hard"].sum()
    loss.backward()
    assert h.grad is not None
    assert float(h.grad.abs().sum()) > 0.0


def test_constant_profile_loss_is_not_near_zero_and_mflat_active() -> None:
    mask = torch.ones(5, 10, dtype=torch.bool)
    z = torch.zeros(5, 10, requires_grad=True)
    p = torch.sigmoid(z)
    bag = bag_from_profiles([p, p, p], mask)
    out = {
        "z33": z,
        "p33": p,
        "bag_hard": bag["bag_hard"],
        "bag_topk": bag["bag_topk"],
        "seq_mask": mask,
    }
    batch = {
        "v3_tiers": ["S", "W", "M", "ND", "NP"],
        "seq_mask": mask,
        "residue_target": F.pad(torch.ones(5, 2), (0, 8)),
    }
    loss, parts = dpr_v6_loss(out, batch, cfg=DPRV6LossConfig(objective="mflat"))
    assert float(parts["L_bag_hard"]) > 0.65
    assert float(parts["L_M_peak"]) > 0.10
    assert float(loss) > 0.70
    loss.backward()
    assert z.grad is not None
    assert float(z.grad.abs().sum()) > 0.0


def test_strong_supervision_uses_safe_background_not_full_sequence_zero() -> None:
    z = torch.zeros(1, 80, requires_grad=True)
    p = torch.sigmoid(z)
    mask = torch.ones(1, 80, dtype=torch.bool)
    bag = bag_from_profiles([p, p, p], mask)
    target = torch.zeros(1, 80)
    target[:, 30:35] = 1.0
    out = {"z33": z, "p33": p, "bag_hard": bag["bag_hard"], "bag_topk": bag["bag_topk"], "seq_mask": mask}
    batch = {"v3_tiers": ["S"], "seq_mask": mask, "residue_target": target}
    loss, parts = dpr_v6_loss(out, batch, cfg=DPRV6LossConfig(objective="strong", bag=0.0, s_dice=0.0, s_rank=0.0))
    assert int(parts["active_S_bce"]) == 1
    loss.backward()
    grad = z.grad.detach().squeeze(0)
    ambiguous = torch.zeros(80, dtype=torch.bool)
    ambiguous[13:52] = True
    ambiguous[30:35] = False
    assert float(grad[ambiguous].abs().max()) == 0.0


def test_v6_schedule_two_step_composition_and_rotation() -> None:
    rows = []
    counts = {"S": 3, "W": 3, "M": 12, "ND": 4, "NP": 12}
    for tier, n in counts.items():
        for i in range(n):
            rows.append({"protein_id": f"{tier}_{i}", "sequence_sha256": f"hash_{tier}_{i}", "length": 100 + i, "v3_tier": tier})
    schedule = build_fixed_schedule(pd.DataFrame(rows), updates=100, seed=20260616)
    audit = validate_schedule(schedule, updates=100)
    assert audit["violation_count"] == 0
    even = Counter(schedule.loc[schedule["update"].eq(2), "v3_tier"])
    odd = Counter(schedule.loc[schedule["update"].eq(1), "v3_tier"])
    assert even == Counter({"S": 1, "W": 1, "M": 2, "ND": 1, "NP": 3})
    assert odd == Counter({"M": 4, "ND": 1, "NP": 3})
    assert all(len(set(schedule.loc[schedule["rank"].eq(rank), "v3_tier"])) > 1 for rank in range(8))


def test_reduce_rows_reports_tier_exposure() -> None:
    rows = [
        {"update": 1, "rank": 0, "tier": "S", "loss": 1.0},
        {"update": 1, "rank": 1, "tier": "NP", "loss": 3.0},
    ]
    reduced = reduce_rows(rows)
    assert reduced["tier_exposure"] == {"NP": 1, "S": 1}
    assert reduced["loss"] == 2.0


def test_fast_spearman_matches_scipy_with_ties() -> None:
    y = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.float32).numpy()
    score = torch.tensor([0.2, 0.9, 0.9, 0.1, 0.5, 0.5, 0.0, 0.7], dtype=torch.float32).numpy()
    expected = float(spearmanr(score, y).statistic)
    actual = safe_spearman(y, score)
    assert abs(actual - expected) < 1.0e-12
