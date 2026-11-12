"""Generate the SYNTHETIC chocolate & malt-beverage supply-chain ecosystem.

Every table produced here is fictional demonstration data. The generative process
is documented in documentation/research/DATA_PROVENANCE.md and summarised below.

1.  Suppliers are created per material and tier (Tier-1 -> manufacturer,
    Tier-2 -> Tier-1, Tier-3 -> Tier-2) in regions plausible for that material.
2.  Ground-truth relationships: a buyer that needs input material m picks k
    sources among suppliers of m with probability proportional to
    size^alpha * exp(-distance/lambda) (preferential attachment + proximity).
    Dependency shares are Dirichlet distributed.  Only part of the deeper network
    is *observed* by the manufacturer (Tier-2: 45 %, Tier-3: 15 %); the rest is
    labelled visibility='hidden' and is used ONLY as ground truth.
3.  Disruptions follow a daily hazard that depends on latent financial risk,
    capacity utilisation, regional geo-risk, monthly seasonality (monsoon,
    cyclone, West-African lean season), an unobserved frailty term and
    CONTAGION from currently disrupted upstream suppliers over the TRUE network.
4.  Purchase orders and shipments are generated for every Tier-1 -> manufacturer
    edge from the production plan; delays depend on reliability, seasonality and
    the supplier's disruption state at ship time.
5.  Inventory and production are produced by a weekly material-flow recursion
    so that shortages are consistent with receipts and bills of material.

Run:  python -m ml.data_generation.generate_synthetic
"""
from __future__ import annotations

import json
import math
from datetime import timedelta

import numpy as np
import pandas as pd

from ml.common.config import SEED, SIM_END, SIM_START, SYN, QUAL_MIN_DURATION, QUAL_MIN_SEVERITY
from ml.data_generation.catalog import (CYCLONE, MATERIALS, MONSOON, NAME_PREFIX, NAME_SUFFIX, PRODUCT_FAMILIES,
                                        REGIONS, SITES, TIER1_COUNTS, TIER2_COUNTS, TIER3_COUNTS, WEST_AFRICA)

rng = np.random.default_rng(SEED)
MANUFACTURER = "C001"
DATES = pd.date_range(SIM_START, SIM_END, freq="D")
N_DAYS = len(DATES)
OBS_P = {1: 1.0, 2: 0.45, 3: 0.15}  # probability that an edge into a buyer of tier (k-1) is visible


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def seasonal_index(region: str, month_idx: int) -> float:
    country, _, _, _, mons, cyc, _ = REGIONS[region]
    if country in ("Cote d'Ivoire", "Ghana"):
        return 0.8 * WEST_AFRICA[month_idx] + 0.2 * mons * MONSOON[month_idx]
    return mons * MONSOON[month_idx] + cyc * CYCLONE[month_idx]


# ---------------------------------------------------------------------------
# 1. Materials, products, sites, company
# ---------------------------------------------------------------------------
def build_materials():
    rows = []
    for mid, (name, cat, lvl, unit, price, crit, inputs, regions) in MATERIALS.items():
        rows.append(dict(material_id=mid, material_name=name, category=cat, bom_level=lvl, unit=unit,
                         unit_price_inr=price, criticality=crit, input_materials=";".join(inputs),
                         typical_regions=";".join(regions), data_label="synthetic"))
    return pd.DataFrame(rows)


def build_products():
    rows, boms = [], []
    sizes = [0.5, 0.75, 1.0, 1.0, 1.25, 1.5, 2.0, 1.0, 0.8, 1.2]
    variants = ["Classic", "Lite", "Family", "Festive", "Value", "Premium", "Mega", "Kids", "Mini", "Gold"]
    pid = 1
    for fam_i, (fam, bom, price, margin) in enumerate(PRODUCT_FAMILIES):
        for v in range(10):
            p = f"P{pid:03d}"
            scale = sizes[v]
            plant = "W01" if (fam_i + v) % 3 != 0 else "W02"
            base_weekly = float(np.round(rng.lognormal(mean=math.log(9000 / max(price, 20) * 30), sigma=0.5)))
            seasonal = "festive_peak" if "chocolate" in fam.lower() or "spread" in fam.lower() else "winter_peak"
            rows.append(dict(product_id=p, product_name=f"{fam} - {variants[v]}", family=fam, pack_scale=scale,
                             price_inr=round(price * scale * rng.uniform(0.95, 1.05), 2),
                             contribution_margin_pct=round(margin + rng.normal(0, 0.015), 3),
                             plant_id=plant, base_weekly_demand_units=base_weekly, seasonality=seasonal,
                             data_label="synthetic"))
            for m, q in bom.items():
                if q <= 0:
                    continue
                per = q * (scale if MATERIALS[m][3] == "kg" else 1.0)
                boms.append(dict(product_id=p, material_id=m, qty_per_unit=round(per, 6)))
            pid += 1
    return pd.DataFrame(rows), pd.DataFrame(boms)


def weekly_demand_factor(seasonality: str, month_idx: int) -> float:
    if seasonality == "festive_peak":  # Diwali / Christmas peak, summer dip for chocolate
        return [1.0, 1.05, 0.95, 0.85, 0.75, 0.75, 0.85, 0.95, 1.05, 1.30, 1.35, 1.20][month_idx]
    return [1.15, 1.10, 1.00, 0.90, 0.85, 0.90, 1.00, 1.00, 1.00, 1.05, 1.10, 1.15][month_idx]


# ---------------------------------------------------------------------------
# 2. Suppliers
# ---------------------------------------------------------------------------
def build_suppliers(material_monthly_need):
    rows = []
    sid = 1
    used_names = set()
    for tier, counts in ((1, TIER1_COUNTS), (2, TIER2_COUNTS), (3, TIER3_COUNTS)):
        for mid, n in counts.items():
            name_m, cat, *_rest = MATERIALS[mid]
            regions = MATERIALS[mid][7]
            w = np.array([1.0 / (k + 1) ** 0.5 for k in range(len(regions))])
            for _ in range(n):
                region = regions[rng.choice(len(regions), p=w / w.sum())]
                country, lat, lon, geo, *_ = REGIONS[region]
                for _try in range(50):
                    nm = f"{rng.choice(NAME_PREFIX)} {NAME_SUFFIX[cat]}"
                    if nm in used_names:
                        nm = f"{nm} {rng.integers(2, 99)}"
                    if nm not in used_names:
                        break
                used_names.add(nm)
                fin_true = float(rng.beta(2, 5))
                util = float(np.clip(rng.beta(6, 3), 0.45, 0.97))
                reliability = float(np.clip(0.985 - 0.25 * rng.beta(2, 6) - 0.08 * fin_true, 0.70, 0.995))
                size = float(rng.lognormal(0, 0.6))
                rows.append(dict(
                    supplier_id=f"S{sid:04d}", supplier_name=nm + " (Synthetic)", tier=tier, primary_material_id=mid,
                    category=cat, region=region, country=country,
                    latitude=round(lat + rng.normal(0, 0.8), 4), longitude=round(lon + rng.normal(0, 0.8), 4),
                    size_index=round(size, 4), capacity_utilisation=round(util, 4), reliability_score=round(reliability, 4),
                    financial_risk_score=round(float(np.clip(fin_true + rng.normal(0, 0.08), 0, 1)), 4),
                    geo_risk_index=geo, certified_fssc22000=bool(rng.random() < (0.85 if tier == 1 else 0.5)),
                    onboarding_year=int(rng.integers(2008, 2023)), import_lead_extra_days=REGIONS[region][6],
                    # latent variables (ground truth only; never used as model features)
                    _fin_true=fin_true, _frailty=float(rng.normal(0, 0.45)),
                    data_label="synthetic"))
                sid += 1
    sup = pd.DataFrame(rows)
    return sup


# ---------------------------------------------------------------------------
# 3. Ground-truth relationships
# ---------------------------------------------------------------------------
def build_relationships(sup: pd.DataFrame):
    edges = []
    by_mat = {m: g for m, g in sup.groupby("primary_material_id")}
    # Tier-1 -> manufacturer
    for mid, g in sup[sup.tier == 1].groupby("primary_material_id"):
        shares = rng.dirichlet(np.ones(len(g)) * 1.2 + g.size_index.values)
        for (_, s), sh in zip(g.iterrows(), shares):
            edges.append(dict(supplier_id=s.supplier_id, buyer_id=MANUFACTURER, material_id=mid, dependency_pct=sh,
                              buyer_tier=0))
    # deeper tiers
    for tier in (1, 2):
        for _, buyer in sup[sup.tier == tier].iterrows():
            inputs = MATERIALS[buyer.primary_material_id][6]
            for im in inputs:
                cands = by_mat.get(im)
                if cands is None or len(cands) == 0:
                    continue
                ks = np.array([2, 3, 4, 5, 6])
                pk = np.array([0.20, 0.30, 0.25, 0.15, 0.10])
                ok = ks <= len(cands)
                k = int(rng.choice(ks[ok], p=pk[ok] / pk[ok].sum()))
                d = np.array([haversine(buyer.latitude, buyer.longitude, c.latitude, c.longitude) for _, c in cands.iterrows()])
                score = cands.size_index.values ** 1.6 * np.exp(-d / 2500.0) + 1e-6
                pick = rng.choice(len(cands), size=k, replace=False, p=score / score.sum())
                shares = rng.dirichlet(np.ones(k) * 1.5)
                for j, sh in zip(pick, shares):
                    edges.append(dict(supplier_id=cands.iloc[j].supplier_id, buyer_id=buyer.supplier_id,
                                      material_id=im, dependency_pct=sh, buyer_tier=tier))
    e = pd.DataFrame(edges)
    e["relationship_id"] = [f"R{i + 1:05d}" for i in range(len(e))]
    tiers = sup.set_index("supplier_id").tier
    lead = []
    vis = []
    for _, r in e.iterrows():
        s_t = tiers[r.supplier_id]
        extra = sup.loc[sup.supplier_id == r.supplier_id, "import_lead_extra_days"].iloc[0]
        lead.append(int(rng.integers(5, 15) + extra))
        vis.append("observed" if rng.random() < OBS_P[s_t] else "hidden")
    e["lead_time_days"] = lead
    e["visibility"] = vis
    e["relationship_status"] = "active"
    e["dependency_pct"] = e.dependency_pct.round(4)
    e["data_label"] = "synthetic"
    return e[["relationship_id", "supplier_id", "buyer_id", "material_id", "dependency_pct", "lead_time_days",
              "relationship_status", "visibility", "buyer_tier", "data_label"]]


# ---------------------------------------------------------------------------
# 4. Disruption process (daily, contagion over the TRUE network)
# ---------------------------------------------------------------------------
EVENT_TYPES = ["weather", "labour_strike", "quality_failure", "equipment_failure", "financial_distress",
               "logistics_port", "regulatory"]
DUR_MU = {"weather": 12, "labour_strike": 8, "quality_failure": 6, "equipment_failure": 10, "financial_distress": 28,
          "logistics_port": 7, "regulatory": 15, "upstream_material_shortage": 10}


def simulate_disruptions(sup, rel, base_rate):
    ids = sup.supplier_id.tolist()
    idx = {s: i for i, s in enumerate(ids)}
    n = len(ids)
    W = np.zeros((n, n))
    sup_rel = rel[rel.buyer_id != MANUFACTURER]
    for (buyer, mat), g in sup_rel.groupby(["buyer_id", "material_id"]):
        n_inputs = len(MATERIALS[sup.loc[idx[buyer], "primary_material_id"]][6])
        for _, r in g.iterrows():
            W[idx[buyer], idx[r.supplier_id]] += r.dependency_pct / n_inputs
    fin = sup._fin_true.values
    util = sup.capacity_utilisation.values
    geo = sup.geo_risk_index.values
    frail = sup._frailty.values
    static = np.exp(1.8 * fin + 3.0 * (util - 0.75) + 1.5 * geo + frail)
    season = np.array([[seasonal_index(r, m) for m in range(12)] for r in sup.region])
    sev = np.zeros(n)
    remaining = np.zeros(n, dtype=int)
    events = []
    kc = 0.035
    for t, day in enumerate(DATES):
        m = day.month - 1
        contagion = W @ sev
        h_own = base_rate * static * np.exp(1.4 * season[:, m])
        h = h_own + kc * contagion
        u = rng.random(n)
        new = (remaining <= 0) & (u < h)
        # recovery
        remaining -= 1
        ended = remaining <= 0
        sev[ended] = 0.0
        for i in np.where(new)[0]:
            upstream = rng.random() < (kc * contagion[i]) / h[i]
            if upstream:
                etype = "upstream_material_shortage"
                severity = float(np.clip(contagion[i] * rng.uniform(0.6, 1.0) + rng.normal(0, 0.05), 0.1, 1.0))
                origin = ids[int(np.argmax(W[i] * sev))]
            else:
                p = np.array([0.6 + 3.0 * season[i, m], 0.5, 0.5, 0.6, 0.25 + 2.0 * fin[i], 0.45, 0.25])
                etype = EVENT_TYPES[rng.choice(len(p), p=p / p.sum())]
                severity = float(np.clip(rng.beta(2.0, 2.2), 0.05, 1.0))
                origin = ""
            dur = int(max(1, round(rng.lognormal(math.log(DUR_MU[etype]), 0.55))))
            sev[i] = severity
            remaining[i] = dur
            start = day
            events.append(dict(supplier_id=ids[i], disruption_type=etype, start_date=start.date(),
                               end_date=(start + timedelta(days=dur - 1)).date(), duration_days=dur,
                               capacity_loss_pct=round(severity, 4),
                               reported_date=(start + timedelta(days=int(rng.poisson(1.5)))).date(),
                               origin_supplier_id=origin))
    ev = pd.DataFrame(events)
    return ev, W


# ---------------------------------------------------------------------------
# 5. External risks (synthetic seasonal forecasts)
# ---------------------------------------------------------------------------
def build_external_risks():
    rows = []
    for region, (country, *_r) in REGIONS.items():
        for mstart in pd.date_range(SIM_START, SIM_END, freq="MS"):
            s = seasonal_index(region, mstart.month - 1)
            rows.append(dict(region=region, country=country, month=mstart.date(),
                             weather_risk_forecast=round(float(np.clip(s + rng.normal(0, 0.1), 0, 1.5)), 3),
                             port_congestion_index=round(float(np.clip((0.3 if REGIONS[region][6] > 0 else 0.1) +
                                                                       rng.normal(0, 0.08), 0, 1)), 3),
                             geopolitical_index=round(float(np.clip(REGIONS[region][3] + rng.normal(0, 0.05), 0, 1)), 3),
                             data_label="synthetic"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Procurement, shipments, inventory, production
# ---------------------------------------------------------------------------
def build_operations(sup, rel, ev, products, bom):
    weeks = pd.date_range(SIM_START, SIM_END, freq="W-MON")
    if weeks[0] > pd.Timestamp(SIM_START):
        weeks = weeks.insert(0, pd.Timestamp(SIM_START))
    weeks = weeks[weeks <= pd.Timestamp(SIM_END) - pd.Timedelta(days=6)]
    # planned production per product-week
    plan = []
    for _, p in products.iterrows():
        for w in weeks:
            f = weekly_demand_factor(p.seasonality, w.month - 1)
            plan.append(dict(product_id=p.product_id, week_start=w, plant_id=p.plant_id,
                             planned_qty=float(round(p.base_weekly_demand_units * f * rng.lognormal(0, 0.08)))))
    plan = pd.DataFrame(plan)
    req = plan.merge(bom, on="product_id")
    req["req"] = req.planned_qty * req.qty_per_unit
    plant_mat_week = req.groupby(["plant_id", "material_id", "week_start"]).req.sum()
    monthly_need = req.assign(month=req.week_start.dt.to_period("M")).groupby(["material_id", "month"]).req.sum()

    # supplier disruption state by day
    sev_day = {}
    for _, e in ev.iterrows():
        for d in pd.date_range(e.start_date, e.end_date):
            k = (e.supplier_id, d)
            sev_day[k] = max(sev_day.get(k, 0.0), e.capacity_loss_pct)
    supi = sup.set_index("supplier_id")
    t1 = rel[rel.buyer_id == MANUFACTURER]
    pos, ships = [], []
    po_id = 1
    sh_id = 1
    cum_assigned = {}
    plant_share = req.groupby(["material_id", "plant_id"]).req.sum().unstack(fill_value=0)
    plant_share = plant_share.div(plant_share.sum(axis=1), axis=0)
    for _, e in t1.iterrows():
        s = supi.loc[e.supplier_id]
        price0 = MATERIALS[e.material_id][4]
        price = price0 * rng.uniform(0.92, 1.10)
        mode = "sea" if s.import_lead_extra_days > 0 else "road"
        for month, need in monthly_need.loc[e.material_id].items():
            qty_month = need * e.dependency_pct * 1.05  # 5% replenishment buffer (documented ordering rule)
            n_po = int(rng.choice([1, 2, 3], p=[0.45, 0.38, 0.17]))
            for j in range(n_po):
                order_date = month.start_time + pd.Timedelta(days=int(rng.integers(0, 8)) + j * 9) - pd.Timedelta(days=int(e.lead_time_days))
                qty = qty_month / n_po
                promised = order_date + pd.Timedelta(days=int(e.lead_time_days))
                pos.append(dict(po_id=f"PO{po_id:05d}", supplier_id=e.supplier_id, material_id=e.material_id,
                                relationship_id=e.relationship_id, order_date=order_date.date(),
                                promised_date=promised.date(), ordered_qty=round(qty, 2),
                                unit_price_inr=round(price, 2), po_value_inr=round(qty * price, 2),
                                currency="INR", data_label="synthetic"))
                k = int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25]))
                plants = plant_share.loc[e.material_id]
                for part in range(k):
                    planned = promised + pd.Timedelta(days=part * 3)
                    sv = sev_day.get((e.supplier_id, planned.normalize()), 0.0)
                    seas = seasonal_index(s.region, planned.month - 1)
                    mu = -0.6 + 6.0 * (1 - s.reliability_score) + 2.2 * seas + (4 + 10 * sv if sv > 0 else 0) + \
                        (2.0 if mode == "sea" else 0.0) * rng.random()
                    delay = int(max(0, round(rng.normal(mu, 1.3))))
                    q = qty / k
                    delivered = q * (1 - sv * rng.uniform(0.3, 1.0)) if sv > 0 else q * rng.uniform(0.97, 1.0)
                    # greedy allocation: send to the plant furthest below its target share
                    ca = cum_assigned.setdefault(e.material_id, {pl: 0.0 for pl in plants.index})
                    tot = sum(ca.values()) + q
                    plant = max(plants.index, key=lambda pl: plants[pl] * tot - ca[pl])
                    ca[plant] += q
                    ships.append(dict(shipment_id=f"SH{sh_id:05d}", po_id=f"PO{po_id:05d}", supplier_id=e.supplier_id,
                                      material_id=e.material_id, destination_site_id=plant, transport_mode=mode,
                                      ship_date=(planned - pd.Timedelta(days=int(e.lead_time_days))).date(),
                                      promised_date=planned.date(),
                                      actual_delivery_date=(planned + pd.Timedelta(days=delay)).date(),
                                      delay_days=delay, is_delayed=int(delay >= 2), shipped_qty=round(q, 2),
                                      delivered_qty=round(delivered, 2),
                                      freight_cost_inr=round(q * price * (0.04 if mode == "sea" else 0.025), 2),
                                      data_label="synthetic"))
                    sh_id += 1
                po_id += 1
    pos = pd.DataFrame(pos)
    ships = pd.DataFrame(ships)

    # weekly inventory recursion per plant-material
    ships["arr_week"] = pd.to_datetime(ships.actual_delivery_date).dt.to_period("W-SUN").dt.start_time
    receipts = ships.groupby(["destination_site_id", "material_id", "arr_week"]).delivered_qty.sum()
    inv_rows, prod_rows = [], []
    on_hand = {}
    for (pl, m), g in plant_mat_week.groupby(level=[0, 1]):
        on_hand[(pl, m)] = float(g.iloc[:4].mean() * 3.0)  # 3 weeks of cover at start
    bom_by_p = bom.groupby("product_id")
    plan_by_week = plan.groupby("week_start")
    for w in weeks:
        pw = plan_by_week.get_group(w)
        fill = {}
        for (pl, m) in on_hand:
            need = plant_mat_week.get((pl, m, w), 0.0)
            rec = receipts.get((pl, m, w), 0.0)
            avail = on_hand[(pl, m)] + rec
            fill[(pl, m)] = 1.0 if need <= 0 else min(1.0, avail / need)
        cons = {k: 0.0 for k in on_hand}
        for _, pr in pw.iterrows():
            b = bom_by_p.get_group(pr.product_id)
            frac = min(fill[(pr.plant_id, mm)] for mm in b.material_id)
            actual = pr.planned_qty * frac
            prod_rows.append(dict(product_id=pr.product_id, plant_id=pr.plant_id, week_start=w.date(),
                                  planned_qty=pr.planned_qty, actual_qty=round(actual, 1),
                                  shortfall_qty=round(pr.planned_qty - actual, 1), data_label="synthetic"))
            for _, br in b.iterrows():
                cons[(pr.plant_id, br.material_id)] += actual * br.qty_per_unit
        for (pl, m) in on_hand:
            need = plant_mat_week.get((pl, m, w), 0.0)
            rec = receipts.get((pl, m, w), 0.0)
            start = on_hand[(pl, m)]
            end = max(0.0, start + rec - cons[(pl, m)])
            # finite storage: excess receipts beyond 6 weeks cover are not booked (returned / deferred)
            end = min(end, max(need, 1e-9) * 6)
            inv_rows.append(dict(site_id=pl, material_id=m, week_start=w.date(), opening_qty=round(start, 2),
                                 receipts_qty=round(rec, 2), consumption_qty=round(cons[(pl, m)], 2),
                                 required_qty=round(need, 2), closing_qty=round(end, 2),
                                 material_shortage_qty=round(max(0.0, need - (start + rec)), 2),
                                 days_of_cover=round(7 * end / need, 2) if need > 0 else None,
                                 data_label="synthetic"))
            on_hand[(pl, m)] = end
    inv = pd.DataFrame(inv_rows)
    prod = pd.DataFrame(prod_rows)
    inv.insert(0, "inventory_id", [f"I{i + 1:05d}" for i in range(len(inv))])
    prod.insert(0, "production_id", [f"PR{i + 1:05d}" for i in range(len(prod))])
    ships = ships.drop(columns=["arr_week"])
    return pos, ships, inv, prod


def main():
    materials = build_materials()
    products, bom = build_products()
    # rough monthly need per material for capacity sizing (from base demand)
    sup = build_suppliers(None)
    rel = build_relationships(sup)
    # calibrate base hazard to ~1,000 disruption events over two years
    base = 0.0011
    for _ in range(4):
        state = rng.bit_generator.state
        ev, W = simulate_disruptions(sup, rel, base)
        if 930 <= len(ev) <= 1070:
            break
        base *= 1000 / max(len(ev), 1)
        rng.bit_generator.state = state
    ev = ev.sort_values(["start_date", "supplier_id"]).reset_index(drop=True)
    ev.insert(0, "disruption_id", [f"D{i + 1:05d}" for i in range(len(ev))])
    ev["is_qualifying"] = ((ev.capacity_loss_pct >= QUAL_MIN_SEVERITY) & (ev.duration_days >= QUAL_MIN_DURATION)).astype(int)
    ev["data_label"] = "synthetic"
    ext = build_external_risks()
    pos, ships, inv, prod = build_operations(sup, rel, ev, products, bom)

    # absolute capacities for Tier-1 suppliers (units/month) from manufacturer demand
    t1 = rel[rel.buyer_id == MANUFACTURER]
    mneed = (prod.merge(bom, on="product_id").assign(r=lambda d: d.planned_qty * d.qty_per_unit)
             .groupby("material_id").r.sum() / 24.0)
    cap = []
    for _, s in sup.iterrows():
        if s.tier == 1:
            share = t1.loc[t1.supplier_id == s.supplier_id, "dependency_pct"].sum()
            f_us = rng.uniform(0.25, 0.6)
            cap.append(round(mneed.get(s.primary_material_id, 0) * share / f_us / s.capacity_utilisation, 1))
        else:
            cap.append(round(float(rng.lognormal(math.log(50000), 0.7) * s.size_index), 1))
    sup["capacity_units_per_month"] = cap
    sup["capacity_unit"] = [MATERIALS[m][3] for m in sup.primary_material_id]

    company = pd.DataFrame([dict(company_id=MANUFACTURER, company_name="Cocoa & Malt Foods Ltd (fictional)",
                                 industry="FMCG - chocolate and malt-based beverages", country="India",
                                 headquarters="Pune, Maharashtra", data_label="synthetic")])
    sites = pd.DataFrame(SITES, columns=["site_id", "site_name", "site_type", "region", "city", "latitude", "longitude"])
    sites["company_id"] = MANUFACTURER
    sites["storage_capacity_tonnes"] = [8000, 6000, 12000, 9000]
    sites["data_label"] = "synthetic"
    subs = pd.DataFrame([
        ("M01", "M02", 0.9, "Spec approval needed (alkalisation changes colour)"),
        ("M02", "M01", 0.9, "Spec approval needed"),
        ("M05", "M06", 0.7, "Fat re-balancing with vegetable fat"),
        ("M06", "M05", 0.8, "Reformulation for fat content"),
        ("M11", "M12", 0.8, "Viscosity adjustment"),
        ("M12", "M11", 0.8, "Moisture adjustment"),
        ("M22", "M23", 0.6, "Lower barrier; shelf life reduced"),
        ("M24", "M25", 0.7, "Heavier pack; line changeover"),
        ("M25", "M24", 0.9, "Line changeover"),
        ("M19", "M20", 0.5, "Different SKU taste profile"),
    ], columns=["material_id", "substitute_material_id", "compatibility_score", "note"])
    subs["data_label"] = "synthetic"

    out = {
        "companies": company, "suppliers": sup.drop(columns=["_fin_true", "_frailty"]),
        "supplier_relationships": rel, "materials": materials, "products": products, "bill_of_materials": bom,
        "inventory": inv, "shipments": ships, "disruptions": ev, "procurement": pos,
        "external_risks": ext, "warehouses": sites, "production": prod, "material_substitutes": subs,
    }
    SYN.mkdir(parents=True, exist_ok=True)
    for k, df in out.items():
        df.to_csv(SYN / f"{k}.csv", index=False)
    # latent ground truth stored separately and NEVER used as features
    gt = SYN / "ground_truth"
    gt.mkdir(exist_ok=True)
    sup[["supplier_id", "_fin_true", "_frailty"]].rename(columns=lambda c: c.strip("_")).to_csv(gt / "supplier_latent.csv", index=False)
    summary = {k: int(len(v)) for k, v in out.items()}
    summary["hazard_base_rate"] = base
    summary["observed_edges"] = int((rel.visibility == "observed").sum())
    summary["hidden_edges"] = int((rel.visibility == "hidden").sum())
    summary["qualifying_disruptions"] = int(ev.is_qualifying.sum())
    summary["seed"] = SEED
    (SYN / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
