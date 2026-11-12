"""Cascading-disruption simulator (daily material-flow model over the TRUE synthetic network).

State per supplier s and day t: availability a_s(t) in [0, 1] (share of nominal output).
For every (buyer b, input material i) slot:
    arrival_bi(t)   = sum_u share_ub * a_u(t - L_ub) * (1 + surge * 1[inv_bi < B])
    available_bi(t) = inv_bi(t) + arrival_bi(t)                      (units: days of nominal need)
    a_b(t)          = min( 1 - loss_b(t),  min_i min(1, available_bi(t)) )      (bottleneck rule)
    inv_bi(t+1)     = clip(available_bi(t) - a_b(t), 0, B)
Manufacturer, Tier-1 material m (units):
    arrival_m(t) = R_m * sum_s share_sm * a_s(t - L_s) * (1 + surge*1[I_m<I0_m]) + recovery_m(t)
    fill_m(t)    = min(1, (I_m + arrival_m) / need_m(t));  frac_p(t) = min_{m in BOM(p)} fill_m(t)
    unmet_p(t)   = D_p(t) * (1 - frac_p(t))
Monte Carlo draws (shared across policies = common random numbers): disruption duration
multiplier, daily demand noise, alternate-supplier lead-time multipliers and yields, and
independent failures of alternate suppliers sampled from calibrated 30-day risk.
All outputs are SIMULATED values on SYNTHETIC data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.common.config import SYN

MANU = "C001"
SLOT_BUFFER_DAYS = 10.0
SURGE = 0.15
STORAGE_CAP_DAYS = 60.0  # manufacturer storage capacity in days of requirement


@dataclass
class Disruption:
    supplier_ids: list
    capacity_loss: float          # 0.25 / 0.5 / 1.0
    start_day: int = 0
    duration_days: int = 30
    label: str = ""


@dataclass
class Draws:
    dur_mult: float
    demand_noise: np.ndarray       # (H, n_products)
    lead_mult: np.ndarray          # (n_suppliers,)
    yield_: np.ndarray             # (n_suppliers,)
    alt_fail_start: np.ndarray     # (n_suppliers,) day of independent failure or -1
    alt_fail_loss: np.ndarray


@dataclass
class Network:
    sup: pd.DataFrame
    rel: pd.DataFrame
    mats: pd.DataFrame
    products: pd.DataFrame
    bom: pd.DataFrame
    R: pd.Series                   # daily nominal need per tier-1 material (units)
    I0: pd.Series                  # initial manufacturer inventory (units)
    D: pd.Series                   # daily demand per product (units)
    cm_unit: pd.Series             # contribution margin per product unit
    price_unit: pd.Series
    spare_daily: pd.Series         # tier-1 spare capacity per day (units)
    extra: dict = field(default_factory=dict)


def load_network(syn_dir=SYN, start="2025-07-01", graph="full") -> Network:
    sup = pd.read_csv(syn_dir / "suppliers.csv").set_index("supplier_id")
    rel = pd.read_csv(syn_dir / "supplier_relationships.csv")
    if graph == "observed":
        rel = rel[rel.visibility == "observed"]
    mats = pd.read_csv(syn_dir / "materials.csv").set_index("material_id")
    products = pd.read_csv(syn_dir / "products.csv").set_index("product_id")
    bom = pd.read_csv(syn_dir / "bill_of_materials.csv")
    prod = pd.read_csv(syn_dir / "production.csv", parse_dates=["week_start"])
    inv = pd.read_csv(syn_dir / "inventory.csv", parse_dates=["week_start"])
    st = pd.Timestamp(start)
    win = prod[(prod.week_start >= st) & (prod.week_start < st + pd.Timedelta(days=91))]
    D = win.groupby("product_id").planned_qty.mean() / 7.0
    R = bom.assign(q=bom.product_id.map(D) * bom.qty_per_unit).groupby("material_id").q.sum()
    snap = inv[inv.week_start == inv.week_start[inv.week_start <= st].max()]
    I0 = snap.groupby("material_id").opening_qty.sum().reindex(R.index).fillna(0)
    t1 = sup[sup.tier == 1]
    spare = t1.capacity_units_per_month * (1 - t1.capacity_utilisation) / 30.0
    return Network(sup=sup, rel=rel, mats=mats, products=products, bom=bom, R=R, I0=I0, D=D,
                   cm_unit=products.price_inr * products.contribution_margin_pct, price_unit=products.price_inr,
                   spare_daily=spare)


class CascadeSimulator:
    def __init__(self, net: Network, horizon=120):
        self.net = net
        self.H = horizon
        sup = net.sup
        self.ids = list(sup.index)
        self.si = {s: i for i, s in enumerate(self.ids)}
        n = len(self.ids)
        self.n = n
        rel = net.rel
        inner = rel[rel.buyer_id != MANU]
        slots = inner.groupby(["buyer_id", "material_id"]).size().index.tolist()
        self.slot_of = {k: j for j, k in enumerate(slots)}
        self.slot_buyer = np.array([self.si[b] for b, _ in slots], dtype=int)
        self.e_src = np.array([self.si[s] for s in inner.supplier_id], dtype=int)
        self.e_slot = np.array([self.slot_of[(b, m)] for b, m in zip(inner.buyer_id, inner.material_id)], dtype=int)
        tot = np.bincount(self.e_slot, weights=inner.dependency_pct.values, minlength=len(slots))
        self.e_share = inner.dependency_pct.values / tot[self.e_slot]       # normalise shares within slot
        self.e_lag = np.clip(inner.lead_time_days.values, 1, 45).astype(int)
        top = rel[rel.buyer_id == MANU]
        self.mlist = list(net.R.index)
        self.mi = {m: i for i, m in enumerate(self.mlist)}
        self.t_src = np.array([self.si[s] for s in top.supplier_id], dtype=int)
        self.t_mat = np.array([self.mi[m] for m in top.material_id], dtype=int)
        tt = np.bincount(self.t_mat, weights=top.dependency_pct.values, minlength=len(self.mlist))
        self.t_share = top.dependency_pct.values / tt[self.t_mat]
        self.t_lag = np.clip(top.lead_time_days.values, 1, 45).astype(int)
        self.R = net.R.values
        self.I0 = net.I0.values
        self.plist = list(net.D.index)
        self.D = net.D.values
        B = np.zeros((len(self.plist), len(self.mlist)))
        bom = net.bom[net.bom.product_id.isin(self.plist)]
        for p, m, q in zip(bom.product_id, bom.material_id, bom.qty_per_unit):
            B[self.plist.index(p), self.mi[m]] = q
        self.BOM = B
        self.uses = B > 0
        self.plant = net.products.loc[self.plist, "plant_id"].values
        self.price_m = net.mats.loc[self.mlist, "unit_price_inr"].values

    # ------------------------------------------------------------------
    def make_draws(self, rng: np.random.Generator, alt_risk: pd.Series | None = None) -> Draws:
        H, n = self.H, self.n
        fail_start = -np.ones(n, dtype=int)
        fail_loss = np.zeros(n)
        if alt_risk is not None:
            p30 = alt_risk.reindex(self.ids).fillna(0).values
            p_h = 1 - (1 - p30) ** (H / 30.0)
            hit = rng.random(n) < p_h
            fail_start[hit] = rng.integers(0, H, hit.sum())
            fail_loss[hit] = np.clip(rng.beta(2.0, 2.2, hit.sum()), 0.05, 1)
        else:
            rng.random(n)
            rng.integers(0, H, 0)
        rel = self.net.sup.reliability_score.reindex(self.ids).values
        return Draws(dur_mult=float(rng.lognormal(0, 0.35)),
                     demand_noise=rng.normal(1, 0.05, (H, len(self.plist))).clip(0.8, 1.2),
                     lead_mult=rng.uniform(0.8, 1.4, n),
                     yield_=np.clip(rng.beta(rel * 40, (1 - rel) * 40 + 1e-3), 0.5, 1.0),
                     alt_fail_start=fail_start, alt_fail_loss=fail_loss)

    # ------------------------------------------------------------------
    def run(self, disruptions: list[Disruption], policy=None, draws: Draws | None = None, record=False):
        """policy: object with .on_day(t, sim_state) -> list of recovery orders, or None (no intervention)."""
        H, n = self.H, self.n
        loss = np.zeros((H, n))
        actual_dur = {}
        for d in disruptions:
            dur = int(round(d.duration_days * (draws.dur_mult if draws else 1.0)))
            dur = max(1, dur)
            for s in d.supplier_ids:
                i = self.si[s]
                loss[d.start_day:min(H, d.start_day + dur), i] = np.maximum(loss[d.start_day:min(H, d.start_day + dur), i], d.capacity_loss)
                actual_dur[s] = dur
        # independent failures of alternates only reduce RECOVERY deliveries (not the base network)
        alt_loss = np.zeros((H, n))
        if draws is not None:
            for i in np.where(draws.alt_fail_start >= 0)[0]:
                st = draws.alt_fail_start[i]
                alt_loss[st:min(H, st + 10), i] = draws.alt_fail_loss[i]
        A = np.ones((H, n))
        inv = np.full(len(self.slot_buyer), SLOT_BUFFER_DAYS)
        I = self.I0.copy()
        dem_noise = draws.demand_noise if draws else np.ones((H, len(self.plist)))
        recovery_arrivals = np.zeros((H, len(self.mlist)))
        pending = []
        out = dict(fill=np.ones(H), unmet=np.zeros((H, len(self.plist))), mat_short=np.zeros((H, len(self.mlist))),
                   arrival_ratio=np.ones((H, len(self.mlist))), I=np.zeros((H, len(self.mlist))))
        for t in range(H):
            lagged = np.where(t - self.e_lag >= 0, A[np.maximum(t - self.e_lag, 0), self.e_src], 1.0)
            low = inv < SLOT_BUFFER_DAYS
            surge = 1 + SURGE * (low[self.e_slot] & (lagged >= 0.999))
            arr = np.bincount(self.e_slot, weights=self.e_share * lagged * surge, minlength=len(inv))
            avail = inv + arr
            ratio = np.minimum(1.0, avail)
            a = 1.0 - loss[t]
            np.minimum.at(a, self.slot_buyer, ratio)
            A[t] = a
            inv = np.clip(avail - a[self.slot_buyer], 0, SLOT_BUFFER_DAYS)
            # manufacturer
            lag_t1 = np.where(t - self.t_lag >= 0, A[np.maximum(t - self.t_lag, 0), self.t_src], 1.0)
            reg_ratio = np.bincount(self.t_mat, weights=self.t_share * lag_t1, minlength=len(self.mlist))
            low_m = I < self.I0
            reg_ratio_s = reg_ratio * (1 + SURGE * (low_m & (reg_ratio >= 0.999)))
            if policy is not None:
                orders = policy.on_day(t, dict(I=I.copy(), R=self.R, reg_ratio=reg_ratio, A=A, sim=self))
                for o in orders:
                    pending.append(self._new_order(o, t, draws))
            # pending recovery orders deliver up to spare capacity x the alternate's availability today
            for o in pending:
                if o["remaining"] <= 1e-9 or t < o["first_delivery_day"]:
                    continue
                q = min(o["remaining"], o["rate_cap"] * A[t, o["i"]] * (1 - alt_loss[t, o["i"]]))
                recovery_arrivals[t, o["m"]] += q * o["yield"] * o.get("compat", 1.0)
                o["remaining"] -= q
                o["delivered_qty"] += q
            arrival = self.R * reg_ratio_s + recovery_arrivals[t]
            need = (self.D * dem_noise[t]) @ self.BOM
            avail_m = I + arrival
            fill = np.where(need > 0, np.minimum(1.0, avail_m / np.maximum(need, 1e-9)), 1.0)
            frac = np.where(self.uses, fill[None, :], 1.0).min(axis=1)
            demand = self.D * dem_noise[t]
            cons = (demand * frac) @ self.BOM
            I = np.minimum(np.maximum(avail_m - cons, 0), self.R * STORAGE_CAP_DAYS)  # warehouse capacity
            out["fill"][t] = (demand * frac).sum() / demand.sum()
            out["unmet"][t] = demand * (1 - frac)
            out["mat_short"][t] = np.maximum(0, need - avail_m)
            out["arrival_ratio"][t] = np.where(self.R > 0, (self.R * reg_ratio + recovery_arrivals[t]) / self.R, 1)
            out["I"][t] = I
        out["A"] = A
        for o in pending:
            o["procurement_cost"] = o["delivered_qty"] * o["unit_price"]
            o["transport_cost"] = o["delivered_qty"] * o["unit_price"] * o["freight_rate"]
        out["recovery_log"] = pending
        out["actual_duration"] = actual_dur
        out["loss"] = loss
        return out

    def _new_order(self, o, t, draws):
        """o = dict(supplier_id, material_id (delivered as), qty, lead_days, unit_price, freight_rate, compat, fixed_cost)."""
        i = self.si[o["supplier_id"]]
        lead = int(round(o["lead_days"] * (draws.lead_mult[i] if draws else 1.0)))
        return dict(o, i=i, m=self.mi[o["material_id"]], day_ordered=t, first_delivery_day=t + max(1, lead),
                    rate_cap=float(self.net.spare_daily.get(o["supplier_id"], 0.0)) * o.get("capacity_share", 1.0),
                    **{"yield": float(draws.yield_[i]) if draws else 1.0},
                    remaining=float(o["qty"]), delivered_qty=0.0)
