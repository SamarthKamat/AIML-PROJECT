"""Network exposure model: expected contribution margin at risk if a supplier is disrupted.

For supplier i and Tier-1 material m delivered to the manufacturer:

    reach_i(m) = sum over paths i -> ... -> j -> M of  prod_edges w_e
    where w_e = dependency share of the edge divided by the buyer's number of
    distinct input materials (bottleneck assumption: a buyer's output falls in
    proportion to the lost share of any one input).

    Exposure_i = sum_m reach_i(m) * CM_day(m) * E[ S * max(0, D - c_m - b*(tier_i - 1)) ]

    CM_day(m): daily contribution margin of all products whose BOM contains m
    c_m      : manufacturer days of cover for m (median of inventory records)
    b        : intermediate-tier input buffer (assumption: 10 days per tier)
    S, D     : severity and duration of qualifying disruptions (empirical, training period)

All monetary values are SIMULATED INR on synthetic data, not company losses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.common.config import SYN, TRAIN_END

TIER_BUFFER_DAYS = 10.0


class ExposureModel:
    def __init__(self, syn_dir=SYN, graph: str = "observed", extra_edges: pd.DataFrame | None = None,
                 history_end: str = TRAIN_END):
        self.sup = pd.read_csv(syn_dir / "suppliers.csv").set_index("supplier_id")
        rel = pd.read_csv(syn_dir / "supplier_relationships.csv")
        if graph == "observed":
            rel = rel[rel.visibility == "observed"]
        if extra_edges is not None and len(extra_edges):
            rel = pd.concat([rel, extra_edges], ignore_index=True)
        self.rel = rel
        mats = pd.read_csv(syn_dir / "materials.csv").set_index("material_id")
        self.mats = mats
        prod = pd.read_csv(syn_dir / "products.csv").set_index("product_id")
        bom = pd.read_csv(syn_dir / "bill_of_materials.csv")
        production = pd.read_csv(syn_dir / "production.csv")
        weekly = production.groupby("product_id").planned_qty.mean()
        cm_day = (weekly * prod.price_inr * prod.contribution_margin_pct / 7.0)
        self.cm_day = bom.assign(cm=bom.product_id.map(cm_day)).groupby("material_id").cm.sum()
        inv = pd.read_csv(syn_dir / "inventory.csv", parse_dates=["week_start"])
        inv = inv[inv.week_start < pd.Timestamp(history_end)]
        self.cover = inv.groupby("material_id").days_of_cover.median()
        ev = pd.read_csv(syn_dir / "disruptions.csv", parse_dates=["start_date", "reported_date"])
        ev = ev[(ev.is_qualifying == 1) & (ev.reported_date < pd.Timestamp(history_end))]
        self._sev = ev.capacity_loss_pct.values
        self._dur = ev.duration_days.values
        self.n_inputs = {s: max(1, len(str(mats.loc[m, "input_materials"]).split(";")))
                         if isinstance(mats.loc[m, "input_materials"], str) else 1
                         for s, m in self.sup.primary_material_id.items()}
        self._reach = self._compute_reach()

    def expected_uncovered(self, cover_days: float, sev_scale: float = 1.0, dur_scale: float = 1.0) -> float:
        return float(np.mean(np.clip(self._sev * sev_scale, 0, 1) * np.maximum(0, self._dur * dur_scale - cover_days)))

    def _compute_reach(self):
        """reach[i][m]: fraction of manufacturer's supply of Tier-1 material m that depends on supplier i."""
        reach = {}
        into_m = self.rel[self.rel.buyer_id == "C001"]
        for _, e in into_m.iterrows():
            reach.setdefault(e.supplier_id, {})
            reach[e.supplier_id][e.material_id] = reach[e.supplier_id].get(e.material_id, 0) + e.dependency_pct
        deeper = self.rel[self.rel.buyer_id != "C001"]
        buyer_tier = deeper.buyer_id.map(self.sup.tier)
        for tier in (1, 2):  # edges into Tier-1 buyers give Tier-2 reach, then Tier-2 buyers give Tier-3 reach
            for _, e in deeper[buyer_tier == tier].iterrows():
                if e.buyer_id not in reach:
                    continue
                w = e.dependency_pct / self.n_inputs.get(e.buyer_id, 1)
                d = reach.setdefault(e.supplier_id, {})
                for m, r in reach[e.buyer_id].items():
                    d[m] = min(1.0, d.get(m, 0) + w * r)
        return reach

    def reach(self, supplier_id: str) -> dict:
        return self._reach.get(supplier_id, {})

    def exposure(self, supplier_id: str, sev_scale=1.0, dur_scale=1.0) -> dict:
        tier = int(self.sup.loc[supplier_id, "tier"])
        total = 0.0
        detail = {}
        for m, r in self.reach(supplier_id).items():
            c = float(self.cover.get(m, 21.0)) + TIER_BUFFER_DAYS * (tier - 1)
            v = r * float(self.cm_day.get(m, 0.0)) * self.expected_uncovered(c, sev_scale, dur_scale)
            detail[m] = dict(reach=round(r, 4), cover_days=round(c, 1), cm_day_inr=round(float(self.cm_day.get(m, 0)), 0),
                             expected_loss_inr=round(v, 0))
            total += v
        return dict(supplier_id=supplier_id, tier=tier, exposure_inr=round(total, 0), materials=detail)

    def all_exposures(self) -> pd.Series:
        return pd.Series({s: self.exposure(s)["exposure_inr"] for s in self.sup.index}, name="exposure_inr")
