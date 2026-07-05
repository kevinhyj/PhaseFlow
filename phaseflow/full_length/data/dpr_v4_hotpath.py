from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from phaseflow.full_length.data.dpr_v3_hotpath import DPRV3NoStarlingSidecar, dpr_v3_collate, read_table


STAGE0B_RATIOS: dict[str, dict[str, int]] = {
    "A": {"S": 50, "W": 15, "ND": 20, "NP": 15, "M": 0},
    "B": {"M": 50, "S": 20, "W": 10, "ND": 10, "NP": 10},
    "C": {"M": 30, "S": 20, "W": 10, "ND": 25, "NP": 15},
}

STAGE0A_RATIOS: dict[str, int] = {"S": 60, "ND": 20, "NP": 20}
M_CONF_RATIOS: dict[str, int] = {"gold": 30, "pseudo_positive_high": 50, "pseudo_positive_weak_preserved": 20}


class DPRV4Stage0Dataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        sidecar: DPRV3NoStarlingSidecar,
        tier_manifest: pd.DataFrame,
        mode: str,
        rank: int,
        world_size: int,
        updates: int,
        seed: int,
        batch_slots: int,
        tiny_json: Path | None = None,
    ) -> None:
        self.sidecar = sidecar
        self.mode = str(mode)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.updates = int(updates)
        self.seed = int(seed)
        self.batch_slots = max(1, int(batch_slots))
        self.tier_manifest = tier_manifest.reset_index(drop=True).copy()
        if self.mode == "stage0a":
            if tiny_json is None:
                raise ValueError("tiny_json is required for stage0a")
            self.tier_manifest = self._filter_tiny(Path(tiny_json))
        self.by_tier: dict[str, pd.DataFrame] = {}
        for tier in ("S", "W", "M", "ND", "NP"):
            sub = self.tier_manifest.loc[self.tier_manifest["v3_tier"].astype(str).eq(tier)].copy()
            if self.mode == "stage0a" and tier in {"W", "M"}:
                continue
            if sub.empty and not (self.mode == "stage0a" and tier in {"W", "M"}):
                raise RuntimeError(f"DPR v4 tier {tier} is empty for {self.mode}")
            self.by_tier[tier] = _stable_shuffle(sub, self.seed + self.rank + _tier_offset(tier)).reset_index(drop=True)
        self.m_by_conf: dict[str, pd.DataFrame] = {}
        if "M" in self.by_tier and not self.by_tier["M"].empty:
            for conf in M_CONF_RATIOS:
                sub = self.by_tier["M"].loc[self.by_tier["M"]["mil_confidence"].astype(str).eq(conf)].copy()
                if sub.empty:
                    raise RuntimeError(f"DPR v4 MIL confidence group is empty: {conf}")
                self.m_by_conf[conf] = _stable_shuffle(sub, self.seed + self.rank + _tier_offset(conf)).reset_index(drop=True)
        self.plan = self._build_plan()

    def __len__(self) -> int:
        return self.updates

    def __getitem__(self, index: int) -> dict[str, Any]:
        items = self.plan[int(index)]
        samples = []
        for item in items:
            samples.append(self.sidecar.sample_from_tier_row(item["row"], view=str(item["view"])))
        batch = dpr_v3_collate(samples)
        batch["stage0_mode"] = self.mode
        batch["stage"] = items[0]["stage"]
        batch["update"] = int(index) + 1
        batch["rank"] = self.rank
        return batch

    def _filter_tiny(self, path: Path) -> pd.DataFrame:
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = {(str(item["protein_id"]), str(item["sequence_sha256"])) for item in payload["proteins"]}
        mask = [
            (str(row.protein_id), str(row.sequence_sha256)) in keys
            for row in self.tier_manifest.itertuples(index=False)
        ]
        out = self.tier_manifest.loc[mask].reset_index(drop=True).copy()
        if len(out) != len(keys):
            raise RuntimeError(f"Stage0A tiny mismatch: json={len(keys)} manifest={len(out)}")
        return out

    def _build_plan(self) -> list[list[dict[str, Any]]]:
        counters: Counter[str] = Counter()
        m_counters: Counter[str] = Counter()
        plan: list[list[dict[str, Any]]] = []
        for update in range(1, self.updates + 1):
            stage = stage_for_update(update, mode=self.mode)
            rows = []
            for slot in range(self.batch_slots):
                global_slot = ((update - 1) * self.world_size + self.rank) * self.batch_slots + slot
                tier = _cycle_choice(ratios_for(stage, self.mode), global_slot + self.seed)
                if tier == "M":
                    conf = _cycle_choice(M_CONF_RATIOS, counters["M"] + self.seed)
                    frame = self.m_by_conf[conf]
                    idx = (m_counters[conf] * self.world_size + self.rank) % len(frame)
                    row = frame.iloc[int(idx)]
                    m_counters[conf] += 1
                else:
                    frame = self.by_tier[tier]
                    idx = (counters[tier] * self.world_size + self.rank) % len(frame)
                    row = frame.iloc[int(idx)]
                counters[tier] += 1
                view = "base_protenix"
                if self.mode == "stage0b" and update > 1500 and ((global_slot + self.seed) % 10 == 0):
                    view = "base_only"
                rows.append({"stage": stage, "tier": tier, "row": row, "view": view})
            plan.append(rows)
        return plan


def load_tier_manifest(path: str | Path) -> pd.DataFrame:
    return read_table(Path(path))


def stage_for_update(update: int, *, mode: str) -> str:
    if str(mode) == "stage0a":
        return "tiny"
    if int(update) <= 600:
        return "A"
    if int(update) <= 2500:
        return "B"
    return "C"


def ratios_for(stage: str, mode: str) -> dict[str, int]:
    if str(mode) == "stage0a":
        return STAGE0A_RATIOS
    return STAGE0B_RATIOS[str(stage)]


def _cycle_choice(weights: dict[str, int], index: int) -> str:
    total = int(sum(weights.values()))
    if total <= 0:
        raise ValueError("empty cycle weights")
    pos = int(index) % total
    cursor = 0
    for key, weight in weights.items():
        cursor += int(weight)
        if pos < cursor:
            return key
    return next(reversed(weights))


def _stable_shuffle(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rng = np.random.default_rng(int(seed) % (2**32 - 1))
    order = np.arange(len(frame))
    rng.shuffle(order)
    return frame.iloc[order].reset_index(drop=True)


def _tier_offset(name: str) -> int:
    return sum(ord(ch) for ch in str(name)) * 997
