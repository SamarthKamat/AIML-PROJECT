"""Alternative-supplier recovery: candidate generation, rule-based allocation and MILP optimisation.

MILP (solved with PuLP / CBC):
  decision  x[j,k] >= 0  quantity ordered from candidate j with transport mode k in {std, exp}
            z[j] in {0,1} onboarding of a substitute / non-approved candidate
            s[m] >= 0    expected unmet material requirement of material m
  minimise  sum_jk (price_j + freight_jk) x[j,k] + h * price_j * x[j,k] * W/2 + sum_j F z[j] + sum_m pi_m s[m]
  s.t.      sum_{j in J_m,k} rho_j (1 - p_j) kappa_j avail_j x[j,k] + s[m] >= deficit_m      (expected-yield demand cover)
            sum_k x[j,k] <= spare_j * max(0, W - L_jk)                            (capacity before deadline)
            x[j,k] <= spare_j * max(0, W - L_jk) * 1[W > L_jk]                    (delivery deadline)
            sum_k x[j,k] <= a_max * deficit_m   (if |J_m| > 1)                    (allocation-share limit)
            x[j,k] = 0 if p_j > p_max                                            (risk limit)
            sum_k x[j,k] <= M z[j]  for candidates requiring onboarding
  pi_m = contribution margin lost per unit of material shortage (+ shortage penalty), derived from BOM.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

ASSUMPTIONS_PATH = Path(__file__).resolve().parents[2] / "experiments" / "scenario_assumptions.json"

DEFAULT_ASSUMPTIONS = {
    "currency": "INR",
    "status": "ILLUSTRATIVE ASSUMPTIONS for a synthetic demonstration - not verified company or market figures",
    "base_material_prices": "materials.csv unit_price_inr (illustrative)",
    "alt_price_premium_existing_supplier": 0.06,
    "alt_price_premium_substitute_supplier": 0.10,
    "onboarding_fixed_cost_inr": 150000,
    "substitute_qualification_days": 7,
    "freight_rate_standard_road": 0.025,
    "freight_rate_standard_sea": 0.04,
    "expedite_freight_multiplier": 3.0,
    "expedite_lead_time_factor": 0.5,
    "holding_cost_annual_rate": 0.24,
    "shortage_penalty_pct_of_price": 0.15,
    "production_interruption_cost_per_plant_day_inr": 75000,
    "production_interruption_fill_threshold": 0.90,
    "recovery_management_cost_per_day_inr": 25000,
    "risk_limit_calibrated_p30": 0.35,
    "max_allocation_share": 0.60,
    "detection_day": 2,
    "nominal_disruption_duration_days": 60,
    "horizon_days": 180,
    "reactive_trigger_cover_days": 10,
    "safety_cover_days": 7,
    "replications": 500,
    "base_seed": 20250701,
    "scenario_start_date": "2025-07-01",
}


def load_assumptions():
    if ASSUMPTIONS_PATH.exists():
        a = json.loads(ASSUMPTIONS_PATH.read_text())
        return {**DEFAULT_ASSUMPTIONS, **a}
    return dict(DEFAULT_ASSUMPTIONS)


def material_penalty(net, A) -> pd.Series:
    """pi_m: INR lost per unit of material m short = max over products using m of (CM + penalty) per unit / BOM qty."""
    bom = net.bom[net.bom.product_id.isin(net.D.index)].copy()
    per_unit = (net.cm_unit + A["shortage_penalty_pct_of_price"] * net.price_unit)
    bom["pen"] = bom.product_id.map(per_unit) / bom.qty_per_unit
    return bom.groupby("material_id").pen.median()


def build_candidates(net, material: str, exclude: set, risk: pd.Series, A, subs: pd.DataFrame | None = None) -> list[dict]:
    sup = net.sup
    rows = []
    price_m = float(net.mats.loc[material, "unit_price_inr"])
    pools = [(material, 1.0, False)]
    if subs is not None:
        for _, r in subs[subs.material_id == material].iterrows():
            pools.append((r.substitute_material_id, float(r.compatibility_score), True))
    for mat, kappa, onboard in pools:
        for sid, s in sup[(sup.tier == 1) & (sup.primary_material_id == mat)].iterrows():
            if sid in exclude:
                continue
            spare = float(net.spare_daily.get(sid, 0.0))
            if spare <= 0:
                continue
            sea = s.import_lead_extra_days > 0
            lead_std = int(10 + s.import_lead_extra_days + (A["substitute_qualification_days"] if onboard else 0))
            fr = A["freight_rate_standard_sea"] if sea else A["freight_rate_standard_road"]
            prem = A["alt_price_premium_substitute_supplier"] if onboard else A["alt_price_premium_existing_supplier"]
            base_price = float(net.mats.loc[mat, "unit_price_inr"])
            rows.append(dict(supplier_id=sid, supplier_name=s.supplier_name, for_material=material, supplies_material=mat,
                             compat=kappa, onboarding=onboard, spare_daily=spare, reliability=float(s.reliability_score),
                             risk_p30=float(risk.get(sid, 0.0)), region=s.region,
                             unit_price=base_price * (1 + prem), base_price=price_m,
                             modes={"standard": dict(lead=lead_std, freight=fr),
                                    "expedited": dict(lead=max(2, int(np.ceil(lead_std * A["expedite_lead_time_factor"]))),
                                                      freight=fr * A["expedite_freight_multiplier"])}))
    return rows


def rule_based_allocation(deficits: dict, cand_by_mat: dict, window: int) -> list[dict]:
    """Procurement rule of thumb: same material only, approved suppliers, most reliable first,
    standard transport, fill up to spare capacity; no risk screen, no substitutes."""
    orders = []
    for m, need in deficits.items():
        remaining = need
        for c in sorted([c for c in cand_by_mat.get(m, []) if not c["onboarding"]], key=lambda c: -c["reliability"]):
            if remaining <= 0:
                break
            md = c["modes"]["standard"]
            cap = c["spare_daily"] * max(0, window - md["lead"])
            q = min(remaining / max(c["reliability"], 1e-6), cap)
            if q <= 0:
                continue
            orders.append(dict(supplier_id=c["supplier_id"], material_id=m, qty=q, lead_days=md["lead"], mode="standard",
                               unit_price=c["unit_price"], freight_rate=md["freight"], compat=c["compat"],
                               base_price=c["base_price"], fixed_cost=0.0))
            remaining -= q * c["reliability"]
    return orders


def optimise_recovery(deficits: dict, cand_by_mat: dict, window: int, penalty: pd.Series, A, time_limit=20):
    prob = pulp.LpProblem("recovery", pulp.LpMinimize)
    h_day = A["holding_cost_annual_rate"] / 365.0
    x, z, s = {}, {}, {}
    obj = []
    for m, need in deficits.items():
        if need <= 0:
            continue
        s[m] = pulp.LpVariable(f"s_{m}", lowBound=0)
        obj.append(float(penalty.get(m, 0.0)) * s[m])
        cover = []
        cands = cand_by_mat.get(m, [])
        for c in cands:
            j = c["supplier_id"]
            if c["risk_p30"] > A["risk_limit_calibrated_p30"]:
                continue
            for k, md in c["modes"].items():
                cap = c["spare_daily"] * max(0, window - md["lead"])
                if cap <= 0:
                    continue
                v = pulp.LpVariable(f"x_{m}_{j}_{k}", lowBound=0, upBound=cap)
                x[(m, j, k)] = (v, c, md)
                unit = c["unit_price"] * (1 + md["freight"]) + h_day * c["unit_price"] * window / 2
                obj.append(unit * v)
                cover.append(c["reliability"] * (1 - c["risk_p30"]) * c["compat"] * c.get("avail", 1.0) * v)
            vs = [x[(m, j, k)][0] for k in c["modes"] if (m, j, k) in x]
            if vs:
                capmax = c["spare_daily"] * window
                prob += pulp.lpSum(vs) <= capmax, f"cap_{m}_{j}"
                if len(cands) > 1:
                    prob += pulp.lpSum(vs) <= A["max_allocation_share"] * need / max(c["reliability"], 0.5), f"share_{m}_{j}"
                if c["onboarding"]:
                    z[(m, j)] = pulp.LpVariable(f"z_{m}_{j}", cat="Binary")
                    obj.append(A["onboarding_fixed_cost_inr"] * z[(m, j)])
                    prob += pulp.lpSum(vs) <= capmax * z[(m, j)], f"onb_{m}_{j}"
        prob += pulp.lpSum(cover) + s[m] >= need, f"demand_{m}"
    if not obj:
        return [], {"status": "no_deficit", "objective": 0.0}
    prob += pulp.lpSum(obj)
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    status = pulp.LpStatus[prob.status]
    orders = []
    for (m, j, k), (v, c, md) in x.items():
        q = v.value() or 0.0
        if q > 1e-6:
            orders.append(dict(supplier_id=j, material_id=m, qty=q, lead_days=md["lead"], mode=k, unit_price=c["unit_price"],
                               freight_rate=md["freight"], compat=c["compat"], base_price=c["base_price"],
                               fixed_cost=A["onboarding_fixed_cost_inr"] if c["onboarding"] else 0.0,
                               supplies_material=c["supplies_material"]))
    info = {"status": status, "objective": float(pulp.value(prob.objective) or 0.0),
            "expected_shortage": {m: float(v.value() or 0.0) for m, v in s.items()},
            "n_variables": len(prob.variables()), "n_constraints": len(prob.constraints)}
    return orders, info
