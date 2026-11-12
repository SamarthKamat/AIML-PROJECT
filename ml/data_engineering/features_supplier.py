"""Point-in-time (leakage-safe) supplier feature construction for Experiment 1.

Prediction timestamp t = 00:00 on the first day of each month.
Rules enforced here (and unit-tested in tests/test_leakage.py):
  * disruption records are used only if reported_date < t
  * shipment outcomes are used only if actual_delivery_date < t
  * external risk uses the monthly *forecast* published for month(t)
  * network features use ONLY visibility == 'observed' relationships
  * the label is a qualifying disruption whose start_date lies in (t, t + 30 days]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.common.config import (HORIZON_DAYS, QUAL_MIN_DURATION, QUAL_MIN_SEVERITY, SNAPSHOT_FIRST, SNAPSHOT_LAST, SYN)

FEATURES = [
    "tier", "reliability_score", "financial_risk_score", "capacity_utilisation", "geo_risk_index",
    "weather_risk_forecast", "port_congestion_index", "material_criticality", "max_dependency_pct",
    "n_buyers_observed", "n_sources_observed", "single_sourced_input", "alt_supplier_count",
    "disruptions_90d", "disruptions_365d", "days_since_last_disruption", "max_severity_365d",
    "currently_disrupted_reported", "upstream_disruptions_90d", "upstream_disrupted_now_share",
    "has_shipment_history", "delay_rate_90d", "mean_delay_days_90d", "fill_rate_90d", "import_supplier",
]
CRIT_MAP = {"A": 3, "B": 2, "C": 1}


def load_tables(syn_dir=SYN):
    t = {}
    for name in ["suppliers", "supplier_relationships", "disruptions", "shipments", "external_risks", "materials"]:
        t[name] = pd.read_csv(syn_dir / f"{name}.csv")
    d = t["disruptions"]
    for c in ["start_date", "end_date", "reported_date"]:
        d[c] = pd.to_datetime(d[c])
    s = t["shipments"]
    for c in ["ship_date", "promised_date", "actual_delivery_date"]:
        s[c] = pd.to_datetime(s[c])
    t["external_risks"]["month"] = pd.to_datetime(t["external_risks"]["month"])
    return t


def observed_edges(rel: pd.DataFrame) -> pd.DataFrame:
    """The ONLY relationship view allowed for feature construction."""
    return rel[rel.visibility == "observed"].copy()


def static_network_features(t) -> pd.DataFrame:
    sup = t["suppliers"].set_index("supplier_id")
    rel = observed_edges(t["supplier_relationships"])
    mats = t["materials"].set_index("material_id")
    f = pd.DataFrame(index=sup.index)
    f["tier"] = sup.tier
    for c in ["reliability_score", "financial_risk_score", "capacity_utilisation", "geo_risk_index"]:
        f[c] = sup[c]
    f["import_supplier"] = (sup.import_lead_extra_days > 0).astype(int)
    f["material_criticality"] = sup.primary_material_id.map(mats.criticality).map(CRIT_MAP)
    f["max_dependency_pct"] = rel.groupby("supplier_id").dependency_pct.max().reindex(f.index).fillna(0)
    f["n_buyers_observed"] = rel.groupby("supplier_id").buyer_id.nunique().reindex(f.index).fillna(0)
    f["n_sources_observed"] = rel.groupby("buyer_id").supplier_id.nunique().reindex(f.index).fillna(0)
    src_per_mat = rel.groupby(["buyer_id", "material_id"]).supplier_id.nunique()
    single = (src_per_mat == 1).groupby(level=0).any()
    f["single_sourced_input"] = single.reindex(f.index).fillna(False).astype(int)
    f["alt_supplier_count"] = sup.groupby("primary_material_id").primary_material_id.transform("size") - 1
    return f


def build_snapshot_features(t, snapshots=None) -> pd.DataFrame:
    if snapshots is None:
        snapshots = pd.date_range(SNAPSHOT_FIRST, SNAPSHOT_LAST, freq="MS")
    sup = t["suppliers"].set_index("supplier_id")
    stat = static_network_features(t)
    rel = observed_edges(t["supplier_relationships"])
    up = rel[rel.buyer_id != "C001"][["supplier_id", "buyer_id"]]
    d = t["disruptions"]
    s = t["shipments"]
    ext = t["external_risks"].set_index(["region", "month"])
    qual = d[(d.capacity_loss_pct >= QUAL_MIN_SEVERITY) & (d.duration_days >= QUAL_MIN_DURATION)]
    rows = []
    for ts in snapshots:
        known = d[d.reported_date < ts]
        k90 = known[known.start_date >= ts - pd.Timedelta(days=90)]
        k365 = known[known.start_date >= ts - pd.Timedelta(days=365)]
        ongoing = known[known.end_date >= ts]
        f = stat.copy()
        f["disruptions_90d"] = k90.groupby("supplier_id").size().reindex(f.index).fillna(0)
        f["disruptions_365d"] = k365.groupby("supplier_id").size().reindex(f.index).fillna(0)
        last = known.groupby("supplier_id").start_date.max().reindex(f.index)
        f["days_since_last_disruption"] = ((ts - last).dt.days).fillna(730).clip(upper=730)
        f["max_severity_365d"] = k365.groupby("supplier_id").capacity_loss_pct.max().reindex(f.index).fillna(0)
        disrupted_now = set(ongoing.supplier_id)
        f["currently_disrupted_reported"] = f.index.isin(disrupted_now).astype(int)
        # observed-upstream disruption exposure
        u90 = up.merge(k90.groupby("supplier_id").size().rename("n").reset_index(), on="supplier_id", how="left").fillna({"n": 0})
        f["upstream_disruptions_90d"] = u90.groupby("buyer_id").n.sum().reindex(f.index).fillna(0)
        un = up.assign(dis=up.supplier_id.isin(disrupted_now).astype(float))
        f["upstream_disrupted_now_share"] = un.groupby("buyer_id").dis.mean().reindex(f.index).fillna(0)
        # shipment history (only deliveries completed before ts)
        sh = s[(s.actual_delivery_date < ts) & (s.actual_delivery_date >= ts - pd.Timedelta(days=90))]
        g = sh.groupby("supplier_id")
        f["has_shipment_history"] = f.index.isin(sh.supplier_id.unique()).astype(int)
        f["delay_rate_90d"] = g.is_delayed.mean().reindex(f.index)
        f["mean_delay_days_90d"] = g.delay_days.mean().reindex(f.index)
        f["fill_rate_90d"] = (g.delivered_qty.sum() / g.shipped_qty.sum()).reindex(f.index)
        # external risk forecast for month(ts)
        f["weather_risk_forecast"] = [ext.loc[(sup.region[i], ts), "weather_risk_forecast"] for i in f.index]
        f["port_congestion_index"] = [ext.loc[(sup.region[i], ts), "port_congestion_index"] for i in f.index]
        # label
        win = qual[(qual.start_date > ts) & (qual.start_date <= ts + pd.Timedelta(days=HORIZON_DAYS))]
        f["disruption_next_30_days"] = f.index.isin(win.supplier_id.unique()).astype(int)
        f["snapshot_ts"] = ts
        rows.append(f.reset_index())
    out = pd.concat(rows, ignore_index=True)
    return out[["supplier_id", "snapshot_ts"] + FEATURES + ["disruption_next_30_days"]]


if __name__ == "__main__":
    from ml.common.config import EXPDATA
    tables = load_tables()
    df = build_snapshot_features(tables)
    df.to_csv(EXPDATA / "exp1_supplier_snapshots.csv", index=False)
    print(df.shape, df.disruption_next_30_days.mean())
    print(df.groupby("snapshot_ts").disruption_next_30_days.mean().round(3).to_string())
