"""Service layer shared by REST routes and the AI assistant tools.

Every number returned here is computed from database records or model/simulation outputs.
"""
from __future__ import annotations

import json
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd
from flask import current_app
from sqlalchemy import delete, insert, select, text

from backend.app import db
from ml.common.config import EXPERIMENTS

MANU = "C001"


def q(sql: str, **params) -> pd.DataFrame:
    with db.get_engine().connect() as c:
        return pd.read_sql(text(sql), c, params=params)


def as_of() -> pd.Timestamp:
    return pd.Timestamp(current_app.config["AS_OF"])


def suppliers_df() -> pd.DataFrame:
    return q("SELECT * FROM suppliers")


def relationships_df(include_hidden=False) -> pd.DataFrame:
    df = q("SELECT * FROM supplier_relationships")
    return df if include_hidden else df[df.visibility == "observed"]


# ------------------------------------------------------------------ predictions
def ensure_predictions(ts: pd.Timestamp | None = None) -> pd.DataFrame:
    """Score all suppliers at ts (1st of month) with the calibrated model and persist; idempotent."""
    from ml.decision.engine import MODEL_VERSION, _bundle, _exposure, predict_risk
    from ml.decision.policy import choose_action
    ts = (ts or as_of()).normalize().replace(day=1)
    existing = _preds_at(ts)
    if len(existing) >= 500:
        return existing
    f = predict_risk(ts)
    em, em_pred = _exposure()
    rows = []
    for _, r in f.iterrows():
        e = em_pred.exposure(r.supplier_id)["exposure_inr"]
        a = choose_action(float(r.p_cal), e)
        rows.append(dict(supplier_id=r.supplier_id, prediction_timestamp=ts.to_pydatetime(), model_version=MODEL_VERSION,
                         calibration_method=_bundle()["calibration_method"], probability_raw=float(r.p_raw),
                         probability_calibrated=float(r.p_cal), risk_category=a["risk_category"], recommended_action=a["action"],
                         exposure_inr=float(e)))
    with db.get_engine().begin() as c:
        c.execute(delete(db.predictions).where(db.predictions.c.prediction_timestamp == ts.to_pydatetime()))
        c.execute(insert(db.predictions), rows)
    return _preds_at(ts)


def _preds_at(ts: pd.Timestamp) -> pd.DataFrame:
    stmt = select(db.predictions).where(db.predictions.c.prediction_timestamp == ts.to_pydatetime())
    with db.get_engine().connect() as c:
        return pd.DataFrame(c.execute(stmt).mappings().all(), columns=[col.name for col in db.predictions.columns])


def risk_band(p: float) -> str:
    """Display band on calibrated probability (relative to ~6.5 % base rate); actions use expected cost."""
    return "Critical" if p >= 0.30 else "High" if p >= 0.15 else "Medium" if p >= 0.07 else "Low"


# ------------------------------------------------------------------ dashboard
def dashboard_summary() -> dict:
    t = as_of()
    sup = suppliers_df()
    rel = relationships_df()
    pred = ensure_predictions(t)
    pred["band"] = pred.probability_calibrated.map(risk_band)
    dis = q("SELECT * FROM disruptions")
    for c in ["start_date", "end_date", "reported_date"]:
        dis[c] = pd.to_datetime(dis[c])
    active = dis[(dis.reported_date < t) & (dis.end_date >= t)]
    inv = q("SELECT * FROM inventory")
    inv["week_start"] = pd.to_datetime(inv.week_start)
    last = inv[inv.week_start == inv.week_start[inv.week_start < t].max()]
    cover = last.groupby("material_id").apply(lambda g: 7 * g.closing_qty.sum() / max(g.required_qty.sum(), 1e-9), include_groups=False)
    mats = q("SELECT material_id, material_name, criticality, unit_price_inr FROM materials").set_index("material_id")
    low_cover = cover[cover < 14].sort_values()
    # critical dependencies: Tier-1 materials with one supplier or HHI > 0.4 on observed edges
    t1 = rel[rel.buyer_id == MANU]
    hhi = t1.groupby("material_id").dependency_pct.apply(lambda s: float(((s / s.sum()) ** 2).sum()))
    crit = hhi[hhi > 0.4].sort_values(ascending=False)
    m = pred.merge(sup[["supplier_id", "region", "tier", "supplier_name"]], on="supplier_id")
    geo = m.groupby("region").agg(mean_risk=("probability_calibrated", "mean"), suppliers=("supplier_id", "size"),
                                  exposure=("exposure_inr", "sum")).reset_index().sort_values("mean_risk", ascending=False)
    lat = sup.groupby("region")[["latitude", "longitude"]].mean().reset_index()
    geo = geo.merge(lat, on="region")
    trend = dis[dis.start_date < t].assign(month=lambda d: d.start_date.dt.to_period("M").astype(str)).groupby(["month", "disruption_type"]).size().unstack(fill_value=0)
    exp_loss = float((m.probability_calibrated * m.exposure_inr).sum())
    top = m.sort_values("probability_calibrated", ascending=False).head(10)
    return dict(
        as_of=str(t.date()), data_label="SYNTHETIC demonstration data",
        totals=dict(suppliers=int(len(sup)), tier1=int((sup.tier == 1).sum()), tier2=int((sup.tier == 2).sum()), tier3=int((sup.tier == 3).sum()),
                    observed_relationships=int(len(rel)), materials=int(len(mats)),
                    high_risk_suppliers=int(pred.band.isin(["High", "Critical"]).sum()),
                    critical_dependencies=int(len(crit)), active_disruptions=int(len(active)),
                    low_cover_materials=int(len(low_cover)), expected_30d_loss_inr=round(exp_loss, 0),
                    total_exposure_inr=round(float(m.exposure_inr.sum()), 0)),
        risk_distribution=pred.band.value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0).astype(int).to_dict(),
        action_distribution=pred.recommended_action.value_counts().to_dict(),
        risk_histogram=np.histogram(pred.probability_calibrated, bins=np.linspace(0, max(0.5, pred.probability_calibrated.max()), 11))[0].tolist(),
        top_risk_suppliers=top[["supplier_id", "supplier_name", "tier", "region", "probability_calibrated", "exposure_inr", "risk_category", "recommended_action"]].round(4).to_dict(orient="records"),
        critical_dependencies=[dict(material_id=k, material_name=mats.loc[k, "material_name"], hhi=round(v, 3),
                                    suppliers=int((t1.material_id == k).sum())) for k, v in crit.items()],
        inventory_exposure=[dict(material_id=k, material_name=mats.loc[k, "material_name"], days_of_cover=round(float(v), 1),
                                 criticality=mats.loc[k, "criticality"]) for k, v in cover.sort_values().head(12).items()],
        geographic_risk=geo.round(4).to_dict(orient="records"),
        active_disruptions=active.merge(sup[["supplier_id", "supplier_name", "tier"]], on="supplier_id")[
            ["disruption_id", "supplier_id", "supplier_name", "tier", "disruption_type", "start_date", "end_date", "capacity_loss_pct"]]
        .assign(start_date=lambda d: d.start_date.dt.date.astype(str), end_date=lambda d: d.end_date.dt.date.astype(str)).to_dict(orient="records"),
        disruption_trend=[dict(month=i, **{k: int(v) for k, v in r.items()}) for i, r in trend.iterrows()],
    )


# ------------------------------------------------------------------ graph
def build_graph(include_predicted=False, include_hidden=False) -> nx.DiGraph:
    sup = suppliers_df().set_index("supplier_id")
    rel = relationships_df(include_hidden)
    G = nx.DiGraph()
    G.add_node(MANU, tier=0, label="Cocoa & Malt Foods (fictional)")
    for s, r in sup.iterrows():
        G.add_node(s, tier=int(r.tier), label=r.supplier_name, material=r.primary_material_id, region=r.region)
    for _, e in rel.iterrows():
        G.add_edge(e.supplier_id, e.buyer_id, material=e.material_id, dependency=float(e.dependency_pct), status=e.visibility,
                   lead_time=int(e.lead_time_days))
    if include_predicted:
        for _, c in candidate_links(top=300).iterrows():
            a, b = c.supplier_id, c.buyer_id
            if not G.has_edge(a, b):
                G.add_edge(a, b, material=sup.loc[a, "primary_material_id"], dependency=None, status="predicted", score=float(c.score))
    return G


@lru_cache(maxsize=1)
def _candidates_raw():
    p = EXPERIMENTS / "exp2a_link_prediction" / "candidate_links.csv"
    if not p.exists():
        return pd.DataFrame(columns=["node_a", "node_b", "score", "rank"])
    return pd.read_csv(p)


def candidate_links(top=500, supplier_id=None) -> pd.DataFrame:
    c = _candidates_raw().copy()
    if c.empty:
        return pd.DataFrame(columns=["supplier_id", "buyer_id", "score", "rank", "status"])
    sup = suppliers_df().set_index("supplier_id")
    tier = sup.tier.to_dict()
    tier[MANU] = 0
    c["supplier_id"] = [a if tier[a] > tier[b] else b for a, b in zip(c.node_a, c.node_b)]
    c["buyer_id"] = [b if tier[a] > tier[b] else a for a, b in zip(c.node_a, c.node_b)]
    if supplier_id:
        c = c[(c.supplier_id == supplier_id) | (c.buyer_id == supplier_id)]
    c = c.sort_values("rank").head(top)
    c["status"] = "candidate (predicted, unverified)"
    return c[["supplier_id", "buyer_id", "score", "rank", "status"]]


def compute_graph_metrics() -> pd.DataFrame:
    G = build_graph()
    U = G.to_undirected()
    deg = nx.degree_centrality(U)
    btw = nx.betweenness_centrality(U, normalized=True)
    pr = nx.pagerank(G.reverse(), alpha=0.85)  # importance flows upstream from the manufacturer
    reach = {n: len(nx.descendants(G, n)) for n in G.nodes}
    rows = [dict(supplier_id=n, degree_centrality=deg[n], betweenness_centrality=btw[n], pagerank=pr[n], downstream_reach=reach[n],
                 graph_view="observed") for n in G.nodes if n != MANU]
    with db.get_engine().begin() as c:
        c.execute(delete(db.graph_metrics))
        c.execute(insert(db.graph_metrics), rows)
    return pd.DataFrame(rows)


def graph_metrics_df() -> pd.DataFrame:
    df = q("SELECT * FROM graph_metrics")
    return df if len(df) else compute_graph_metrics()


def network_json(include_predicted=False, max_tier=3) -> dict:
    G = build_graph(include_predicted)
    pred = ensure_predictions().set_index("supplier_id")
    gm = graph_metrics_df().set_index("supplier_id")
    nodes = []
    for n, a in G.nodes(data=True):
        if a.get("tier", 0) > max_tier:
            continue
        p = float(pred.probability_calibrated.get(n, 0)) if n != MANU else 0.0
        nodes.append(dict(id=n, label=a.get("label"), tier=a.get("tier", 0), material=a.get("material"), region=a.get("region"),
                          risk=round(p, 4), band=risk_band(p) if n != MANU else "Manufacturer",
                          pagerank=round(float(gm.pagerank.get(n, 0)), 5) if n != MANU else None, degree=G.degree(n)))
    ids = {x["id"] for x in nodes}
    links = [dict(source=u, target=v, material=d.get("material"), dependency=d.get("dependency"), status=d.get("status"),
                  score=d.get("score")) for u, v, d in G.edges(data=True) if u in ids and v in ids]
    return dict(nodes=nodes, links=links, legend=dict(observed="relationship known to the manufacturer",
                                                      predicted="candidate relationship from link prediction (unverified)"))


def supplier_detail(sid: str) -> dict:
    sup = suppliers_df().set_index("supplier_id")
    if sid not in sup.index:
        raise KeyError(f"unknown supplier {sid}")
    s = sup.loc[sid]
    rel = relationships_df()
    mats = q("SELECT material_id, material_name, criticality FROM materials").set_index("material_id")
    up = rel[rel.buyer_id == sid].merge(sup[["supplier_name", "tier"]], left_on="supplier_id", right_index=True)
    down = rel[rel.supplier_id == sid]
    pred = ensure_predictions().set_index("supplier_id")
    gm = graph_metrics_df().set_index("supplier_id")
    alts = sup[(sup.primary_material_id == s.primary_material_id) & (sup.index != sid)].copy()
    alts["risk"] = alts.index.map(pred.probability_calibrated)
    alts = alts.sort_values("risk").head(8)
    hist = q("SELECT disruption_id, disruption_type, start_date, duration_days, capacity_loss_pct FROM disruptions "
             "WHERE supplier_id = :s AND reported_date < :t ORDER BY start_date DESC", s=sid, t=as_of().date())
    ships = q("SELECT COUNT(*) n, AVG(is_delayed) delay_rate, AVG(delay_days) mean_delay FROM shipments WHERE supplier_id = :s "
              "AND actual_delivery_date < :t", s=sid, t=as_of().date()).iloc[0]
    return dict(
        supplier=dict(supplier_id=sid, **{k: (v.item() if hasattr(v, "item") else v) for k, v in s.items()},
                      material_name=mats.loc[s.primary_material_id, "material_name"]),
        risk=dict(calibrated_probability=float(pred.probability_calibrated.get(sid, np.nan)), band=risk_band(float(pred.probability_calibrated.get(sid, 0))),
                  recommended_action=pred.recommended_action.get(sid), exposure_inr=float(pred.exposure_inr.get(sid, 0)),
                  prediction_timestamp=str(pred.prediction_timestamp.get(sid))),
        centrality=dict(degree=float(gm.degree_centrality.get(sid, 0)), betweenness=float(gm.betweenness_centrality.get(sid, 0)),
                        pagerank=float(gm.pagerank.get(sid, 0)), downstream_reach=int(gm.downstream_reach.get(sid, 0))),
        upstream=up[["supplier_id", "supplier_name", "tier", "material_id", "dependency_pct", "lead_time_days"]].to_dict(orient="records"),
        downstream=down[["buyer_id", "material_id", "dependency_pct", "lead_time_days"]].to_dict(orient="records"),
        candidate_links=candidate_links(20, sid).to_dict(orient="records"),
        alternatives=[dict(supplier_id=i, supplier_name=r.supplier_name, region=r.region, reliability=r.reliability_score,
                           capacity_utilisation=r.capacity_utilisation, risk=round(float(r.risk), 4) if pd.notna(r.risk) else None)
                      for i, r in alts.iterrows()],
        disruption_history=hist.assign(start_date=hist.start_date.astype(str)).to_dict(orient="records"),
        shipment_history=dict(n=int(ships.n or 0), delay_rate=None if pd.isna(ships.delay_rate) else float(ships.delay_rate),
                              mean_delay_days=None if pd.isna(ships.mean_delay) else float(ships.mean_delay)),
    )


def experiment_results() -> dict:
    out = {}
    for d in sorted(EXPERIMENTS.glob("*/run.json")):
        out[d.parent.name] = json.loads(d.read_text())
    return out


def nine_scenarios() -> dict:
    p = EXPERIMENTS / "scenario_results.csv"
    if not p.exists():
        return dict(available=False)
    s = pd.read_csv(p)
    c = pd.read_csv(EXPERIMENTS / "scenario_comparison.csv")
    a = json.loads((EXPERIMENTS / "scenario_assumptions.json").read_text())
    out = dict(available=True, results=s.round(4).to_dict(orient="records"), paired=c.round(2).to_dict(orient="records"), assumptions=a)
    st = EXPERIMENTS / "statistics.json"
    if st.exists():
        j = json.loads(st.read_text())
        out["tests"] = j.get("scenarios", [])          # full nine-comparison family (Wilcoxon, Holm)
        out["ablation_tests"] = j.get("ablation", [])  # link-information ablation family
    rob = EXPERIMENTS / "robustness"
    if (rob / "ablation_summary.csv").exists():
        out["ablation"] = pd.read_csv(rob / "ablation_summary.csv").round(2).to_dict(orient="records")
    if (rob / "sensitivity.csv").exists():
        out["sensitivity"] = pd.read_csv(rob / "sensitivity.csv").round(2).to_dict(orient="records")
    return out
