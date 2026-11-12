"""Model inference, simulation, optimisation, decision and reporting endpoints."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from flask import Blueprint, g, jsonify, request
from sqlalchemy import insert

from backend.app import db, services as S
from backend.app.auth import login_required
from backend.app.routes_core import _clean
from ml.common.config import MODELS

bp = Blueprint("ml", __name__, url_prefix="/api")
DISRUPTION_TYPES = ["supplier_failure", "capacity_reduction", "weather", "transportation", "labour_strike", "quality_failure",
                    "financial_distress"]


def _body():
    return request.get_json(silent=True) or {}


# ------------------------------------------------------------------ ML
@bp.post("/ml/supplier-risk")
@login_required
def supplier_risk():
    """Score suppliers at a prediction timestamp: {timestamp?, supplier_ids?, with_explanation?}."""
    from ml.decision.engine import MODEL_VERSION, _bundle, _shap_row, predict_risk
    d = _body()
    ts = pd.Timestamp(d.get("timestamp", S.as_of())).normalize().replace(day=1)
    if ts < pd.Timestamp("2024-04-01") or ts > pd.Timestamp("2025-12-01"):
        raise ValueError("timestamp must be between 2024-04-01 and 2025-12-01 (history needed for point-in-time features)")
    ids = d.get("supplier_ids")
    pred = S.ensure_predictions(ts)
    if ids:
        pred = pred[pred.supplier_id.isin(ids)]
    out = pred.sort_values("probability_calibrated", ascending=False)
    res = dict(prediction_timestamp=str(ts.date()), model_version=MODEL_VERSION, calibration=_bundle()["calibration_method"],
               information_cutoff="records reported/delivered strictly before the timestamp",
               predictions=out.assign(risk_band=out.probability_calibrated.map(S.risk_band)).to_dict(orient="records"))
    if d.get("with_explanation") and ids:
        f = predict_risk(ts, ids)
        res["explanations"] = {sid: _shap_row(f[f.supplier_id == sid]) for sid in ids if (f.supplier_id == sid).any()}
    return jsonify(_clean(res))


@bp.get("/ml/predictions/<sid>/history")
@login_required
def prediction_history(sid):
    """Calibrated risk trajectory of a supplier over all monthly snapshots up to AS_OF."""
    from ml.decision.engine import predict_risk
    rows = []
    for ts in pd.date_range("2025-01-01", S.as_of(), freq="MS"):
        f = predict_risk(ts, [sid])
        if len(f):
            rows.append(dict(timestamp=str(ts.date()), p_cal=float(f.p_cal.iloc[0]), p_raw=float(f.p_raw.iloc[0]),
                             actual_disruption_next_30d=int(f.disruption_next_30_days.iloc[0])))
    return jsonify(rows)


@bp.post("/ml/link-prediction")
@login_required
def link_prediction():
    """Candidate (unverified) supplier relationships from the Exp 2A link predictor: {supplier_id?, top?}."""
    d = _body()
    c = S.candidate_links(int(d.get("top", 50)), d.get("supplier_id"))
    sup = S.suppliers_df().set_index("supplier_id")
    c["supplier_name"] = c.supplier_id.map(sup.supplier_name)
    c["buyer_name"] = c.buyer_id.map(sup.supplier_name).fillna("Cocoa & Malt Foods (manufacturer)")
    import json as _json
    from ml.common.config import EXPERIMENTS
    sel = _json.loads((EXPERIMENTS / "exp2a_link_prediction" / "run.json").read_text()).get("candidate_model", "?")
    return jsonify(_clean(dict(model=f"{sel} pair model selected by validation AP (Exp 2A, seed 0)", disclaimer="Candidate relationships are model "
                               "predictions to be verified with suppliers; they are NOT confirmed relationships.",
                               candidates=c.to_dict(orient="records"))))


@lru_cache(maxsize=1)
def _delay_model():
    import joblib
    return joblib.load(MODELS / "exp3_dataco_xgb.joblib")


@bp.post("/ml/delivery-delay")
@login_required
def delivery_delay():
    """Predict late delivery for an order (DataCo-trained, real data). Body: order-time attributes."""
    m = _delay_model()
    d = _body()
    defaults = {"Type": "DEBIT", "Shipping Mode": "Standard Class", "Market": "Pacific Asia", "Order Region": "South Asia",
                "Customer Segment": "Consumer", "Department Name": "Fan Shop", "Category Name": "Cleats",
                "Days for shipment (scheduled)": 4, "Order Item Quantity": 1, "Order Item Product Price": 59.99,
                "Order Item Discount Rate": 0.1, "Sales": 59.99, "Latitude": 18.5, "Longitude": 73.8,
                "order_month": 7, "order_dow": 2, "order_hour": 12}
    row = {**defaults, **{k: v for k, v in d.items() if k in defaults}}
    X = pd.DataFrame([row])[m["features_cat"] + m["features_num"]]
    p = float(m["classifier"].predict_proba(X)[:, 1][0])
    delay = float(m["regressor"].predict(X)[0])
    return jsonify(dict(late_probability=round(p, 4), expected_delay_days=round(delay, 2), input=row,
                        model="XGBoost trained on DataCo (real, CC BY 4.0) chronological train split",
                        note="Uncalibrated probability; see Exp 3 for test metrics"))


# ------------------------------------------------------------------ simulation
@lru_cache(maxsize=2)
def _sim(start="2025-07-01", horizon=120):
    from ml.simulation.cascade import CascadeSimulator, load_network
    net = load_network(start=start)
    return net, CascadeSimulator(net, horizon=horizon)


def _resolve_targets(d, net):
    sup = net.sup
    ids = d.get("supplier_ids") or ([d["supplier_id"]] if d.get("supplier_id") else [])
    if d.get("region"):
        ids = list(set(ids) | set(sup[sup.region == d["region"]].index))
    if d.get("disruption_type") == "transportation" and not ids:
        ids = list(sup[sup.import_lead_extra_days > 0].index)
    bad = [s for s in ids if s not in sup.index]
    if bad:
        raise ValueError(f"unknown supplier ids: {bad[:5]}")
    if not ids:
        raise ValueError("provide supplier_id, supplier_ids or region")
    return ids


@bp.post("/simulation/run")
@login_required
def simulation_run():
    """Run the cascade simulator. Body: {supplier_id|supplier_ids|region, disruption_type, capacity_reduction (0-1),
    duration_days, policy (A|B|C), replications (1-100)}."""
    from ml.optimisation.recovery import load_assumptions
    from ml.simulation.cascade import Disruption
    from ml.simulation.scenarios import knowledge_graphs, latest_risk, make_policy, metrics
    d = _body()
    net, sim = _sim()
    A = load_assumptions()
    ids = _resolve_targets(d, net)
    loss = float(d.get("capacity_reduction", 1.0))
    if not 0 < loss <= 1:
        raise ValueError("capacity_reduction must be in (0, 1]")
    dur = int(d.get("duration_days", 30))
    if not 1 <= dur <= 110:
        raise ValueError("duration_days must be 1..110")
    reps = int(d.get("replications", 20))
    reps = max(1, min(100, reps))
    pol = {"A": "A: No intervention", "B": "B: Rule-based", "C": "C: Optimisation"}[d.get("policy", "A")]
    risk = latest_risk(net)
    obs, predicted = knowledge_graphs(net)
    subs = S.q("SELECT * FROM material_substitutes")
    rows, first = [], None
    A2 = dict(A, nominal_disruption_duration_days=dur)
    for r in range(reps):
        draws = sim.make_draws(np.random.default_rng(A["base_seed"] + r), alt_risk=risk)
        policy = make_policy(pol, net, A2, obs, predicted, risk, ids, loss, subs)
        o = sim.run([Disruption(ids, loss, 0, dur)], policy, draws)
        rows.append(metrics(o, sim, net, A2, ids, pol))
        if first is None:
            first = o
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number")
    summary = num.mean().round(2).to_dict()
    summary.update({c + "_p97.5": round(float(num[c].quantile(0.975)), 2) for c in ["financial_impact_inr", "unmet_demand_units"]})
    summary["feasibility_full_recovery_share"] = float((df.feasibility == "full recovery").mean())
    mats = net.mats
    series = dict(day=list(range(sim.H)), service_level=np.round(first["fill"], 4).tolist(),
                  materials={m: np.round(first["arrival_ratio"][:, j], 4).tolist() for j, m in enumerate(sim.mlist)
                             if first["arrival_ratio"][:, j].min() < 0.99},
                  cover_days={m: np.round(first["I"][:, j] / max(sim.R[j], 1e-9), 2).tolist() for j, m in enumerate(sim.mlist)
                              if first["arrival_ratio"][:, j].min() < 0.99})
    affected = [dict(supplier_id=sim.ids[i], supplier_name=net.sup.supplier_name.iloc[i], tier=int(net.sup.tier.iloc[i]),
                     min_availability=round(float(first["A"][:, i].min()), 4)) for i in np.where(first["A"].min(0) < 0.99)[0]]
    res = dict(parameters=dict(targets=ids, disruption_type=d.get("disruption_type", "supplier_failure"), capacity_reduction=loss,
                               duration_days=dur, policy=pol, replications=reps),
               summary=summary, series=series, affected_suppliers=sorted(affected, key=lambda x: x["min_availability"])[:60],
               material_names={m: mats.loc[m, "material_name"] for m in series["materials"]},
               recovery_orders=[{k: v for k, v in o.items() if k in ("supplier_id", "material_id", "qty", "mode", "lead_days", "delivered_qty",
                                                                    "unit_price", "day_ordered")} for o in first["recovery_log"]],
               data_label="SIMULATED outcomes on SYNTHETIC data")
    with db.get_engine().begin() as c:
        r = c.execute(insert(db.simulation_runs).values(user_id=g.user.get("user_id"), run_type="custom", parameters=_clean(res["parameters"]),
                                                         results=_clean(summary)))
    res["run_id"] = r.inserted_primary_key[0]
    return jsonify(_clean(res))


@bp.get("/simulation/scenarios")
@login_required
def scenarios_get():
    """Stored nine-scenario evaluation (500 CRN replications), paired comparisons and assumptions."""
    return jsonify(_clean(S.nine_scenarios()))


@bp.get("/simulation/impact-types")
@login_required
def impact_types():
    """Experiment 4: no-intervention consequences per disruption type (stored Monte Carlo means)."""
    from ml.common.config import EXPERIMENTS
    p = EXPERIMENTS / "exp4_disruption_types.csv"
    if not p.exists():
        raise ValueError("run python -m ml.simulation.scenarios first")
    return jsonify(_clean(pd.read_csv(p).to_dict(orient="records")))


@bp.post("/simulation/scenarios")
@login_required
def scenarios_rerun():
    """Reproduce the nine-scenario matrix with fewer replications: {replications (5-100)}."""
    from ml.simulation.scenarios import run_nine, summarise
    reps = max(5, min(100, int(_body().get("replications", 20))))
    res, ctx = run_nine(reps=reps, verbose=False)
    s = summarise(res)
    with db.get_engine().begin() as c:
        c.execute(insert(db.simulation_runs).values(user_id=g.user.get("user_id"), run_type="nine_scenarios", parameters=dict(replications=reps),
                                                     results=_clean(s[["scenario", "financial_impact_inr", "unmet_demand_units"]].to_dict(orient="records"))))
    return jsonify(_clean(dict(replications=reps, disrupted=ctx["disrupted"], results=s.round(4).to_dict(orient="records"))))


# ------------------------------------------------------------------ recovery & decision
@bp.post("/recovery/optimise")
@login_required
def recovery_optimise():
    """MILP recovery plan. Body: {material_id, quantity, deadline_days, exclude_supplier_ids?, compare_rule_based?}."""
    from ml.decision.engine import predict_risk
    from ml.optimisation.recovery import build_candidates, load_assumptions, material_penalty, optimise_recovery, rule_based_allocation
    d = _body()
    net, sim = _sim()
    A = load_assumptions()
    m = d.get("material_id")
    if m not in net.R.index:
        raise ValueError("material_id must be a Tier-1 material delivered to the manufacturer (M01-M28)")
    qty = float(d.get("quantity", 0))
    if qty <= 0:
        raise ValueError("quantity must be > 0")
    window = int(d.get("deadline_days", 30))
    exclude = set(d.get("exclude_supplier_ids", []))
    risk = predict_risk(S.as_of()).set_index("supplier_id").p_cal
    subs = S.q("SELECT * FROM material_substitutes")
    cands = {m: build_candidates(net, m, exclude, risk, A, subs)}
    orders, info = optimise_recovery({m: qty}, cands, window, material_penalty(net, A), A)
    cost = lambda os: sum(o["qty"] * o["unit_price"] * (1 + o["freight_rate"]) + o.get("fixed_cost", 0) for o in os)  # noqa: E731
    exp_cov = lambda os: sum(o["qty"] * next(c for c in cands[m] if c["supplier_id"] == o["supplier_id"])["reliability"] *  # noqa: E731
                             o["compat"] for o in os)
    res = dict(material_id=m, material_name=net.mats.loc[m, "material_name"], quantity=qty, deadline_days=window,
               solver=info, allocation=orders, total_cost_inr=round(cost(orders), 2), expected_delivered=round(exp_cov(orders), 2),
               candidates=[{k: v for k, v in c.items() if k != "modes"} | {"lead_standard": c["modes"]["standard"]["lead"],
                                                                         "lead_expedited": c["modes"]["expedited"]["lead"]} for c in cands[m]])
    if d.get("compare_rule_based", True):
        rb = rule_based_allocation({m: qty}, cands, window)
        res["rule_based"] = dict(allocation=rb, total_cost_inr=round(cost(rb), 2), expected_delivered=round(exp_cov(rb), 2))
    with db.get_engine().begin() as c:
        c.execute(insert(db.recovery_recommendations).values(user_id=g.user.get("user_id"), material_id=m, solver_status=info["status"],
                                                              allocation=_clean(orders), total_cost_inr=res["total_cost_inr"]))
    return jsonify(_clean(res))


@bp.post("/decision/evaluate")
@login_required
def decision_evaluate():
    """Executable decision workflow for one supplier: {supplier_id, timestamp?, use_predicted_links?}."""
    from ml.decision.engine import evaluate
    d = _body()
    if not d.get("supplier_id"):
        raise ValueError("supplier_id is required")
    r = evaluate(d["supplier_id"], d.get("timestamp", str(S.as_of().date())), bool(d.get("use_predicted_links", True)))
    with db.get_engine().begin() as c:
        c.execute(insert(db.decisions).values(user_id=g.user.get("user_id"), supplier_id=r["supplier_id"],
                                              prediction_timestamp=pd.Timestamp(r["prediction_timestamp"]).to_pydatetime(),
                                              calibrated_probability=r["calibrated_probability"], risk_category=r["risk_category"],
                                              recommended_action=r["recommended_action"], rationale=r["decision_rationale"], payload=_clean(r)))
    return jsonify(_clean(r))


# ------------------------------------------------------------------ reports
@bp.get("/reports")
@login_required
def reports():
    """History of decisions, simulations, recovery plans, uploads and chat."""
    lim = int(request.args.get("limit", 50))
    out = {}
    for name, sql in [("decisions", "SELECT decision_id, supplier_id, prediction_timestamp, calibrated_probability, risk_category, recommended_action, rationale, created_at FROM decisions"),
                      ("simulations", "SELECT run_id, run_type, parameters, results, created_at FROM simulation_runs"),
                      ("recovery_plans", "SELECT recommendation_id, material_id, solver_status, total_cost_inr, created_at FROM recovery_recommendations"),
                      ("uploads", "SELECT upload_id, filename, target_table, row_count, status, created_at FROM upload_history"),
                      ("chats", "SELECT chat_id, question, engine, created_at FROM chat_logs")]:
        key = sql.split("SELECT ")[1].split(",")[0]
        out[name] = S.q(f"{sql} ORDER BY {key} DESC LIMIT {lim}").astype(object).to_dict(orient="records")
    return jsonify(_clean(out))


@bp.get("/experiments")
@login_required
def experiments():
    """All logged experiment runs (metrics, splits, configuration) from experiments/*/run.json."""
    return jsonify(_clean(S.experiment_results()))
