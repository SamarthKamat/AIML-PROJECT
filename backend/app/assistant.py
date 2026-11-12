"""AI Assistant: query understanding -> tool selection -> database retrieval / model inference /
simulation / optimisation -> evidence-backed answer.

Engine 1 (if GEMINI_API_KEY is set): Gemini function calling over the tool registry below.  The
LLM may only phrase answers from tool outputs, which are returned to the UI as evidence.
Engine 2 (fallback, always available): deterministic intent router calling the same tools.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import requests
from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy import insert

from backend.app import db, services as S
from backend.app.auth import login_required
from backend.app.routes_core import _clean

bp = Blueprint("chat", __name__, url_prefix="/api/chat")
SID_RE = re.compile(r"\bS\d{4}\b", re.I)
MID_RE = re.compile(r"\bM\d{2}\b", re.I)


# ======================================================================= tools
def _sup():
    return S.suppliers_df().set_index("supplier_id")


def tool_most_critical_suppliers(n: int = 5) -> dict:
    """Rank suppliers by expected 30-day loss = calibrated probability x network exposure (INR)."""
    p = S.ensure_predictions().merge(S.suppliers_df()[["supplier_id", "supplier_name", "tier", "region"]], on="supplier_id")
    gm = S.graph_metrics_df()[["supplier_id", "pagerank", "downstream_reach"]]
    p = p.merge(gm, on="supplier_id", how="left")
    p["expected_loss_inr"] = p.probability_calibrated * p.exposure_inr
    top = p.sort_values("expected_loss_inr", ascending=False).head(int(n))
    return dict(as_of=str(S.as_of().date()), ranking_basis="calibrated p(30d) x network exposure",
                suppliers=top[["supplier_id", "supplier_name", "tier", "region", "probability_calibrated", "exposure_inr",
                               "expected_loss_inr", "pagerank", "recommended_action"]].round(4).to_dict(orient="records"))


def tool_supplier_failure_impact(supplier_id: str, capacity_reduction: float = 1.0, duration_days: int = 30) -> dict:
    """Simulate a disruption of one supplier with no intervention and report cascade consequences."""
    from backend.app.routes_ml import _sim
    from ml.simulation.cascade import Disruption
    from ml.simulation.scenarios import metrics
    from ml.optimisation.recovery import load_assumptions
    net, sim = _sim()
    if supplier_id not in net.sup.index:
        raise KeyError(f"unknown supplier {supplier_id}")
    A = load_assumptions()
    o = sim.run([Disruption([supplier_id], float(capacity_reduction), 0, int(duration_days))])
    m = metrics(o, sim, net, A, [supplier_id], "A: No intervention")
    hit = [sim.mlist[j] for j in range(len(sim.mlist)) if o["arrival_ratio"][:, j].min() < 0.99]
    return dict(supplier_id=supplier_id, supplier_name=net.sup.loc[supplier_id, "supplier_name"], tier=int(net.sup.loc[supplier_id, "tier"]),
                assumption=f"{capacity_reduction:.0%} capacity loss for {duration_days} days, no intervention (deterministic run)",
                affected_suppliers=m["affected_suppliers"], affected_materials=hit,
                material_names={x: net.mats.loc[x, "material_name"] for x in hit}, unmet_demand_units=round(m["unmet_demand_units"], 0),
                lost_contribution_margin_inr=round(m["production_loss_cm_inr"], 0), financial_impact_inr=round(m["financial_impact_inr"], 0),
                min_service_level=round(m["min_service_level"], 4), recovery_time_days=m["recovery_time_days"],
                data_label="simulated on synthetic data")


def tool_materials_at_risk(n: int = 8) -> dict:
    """Tier-1 materials ranked by inventory cover and the calibrated risk of the suppliers that reach them."""
    from ml.decision.engine import _exposure
    d = S.dashboard_summary()
    cover = {x["material_id"]: x["days_of_cover"] for x in d["inventory_exposure"]}
    em = _exposure()[1]
    pred = S.ensure_predictions().set_index("supplier_id").probability_calibrated
    risk_by_m = {}
    for s, p in pred.items():
        for m, r in em.reach(s).items():
            risk_by_m[m] = risk_by_m.get(m, 0) + p * r  # expected share of supply disrupted in 30 days
    mats = S.q("SELECT material_id, material_name, criticality FROM materials").set_index("material_id")
    rows = [dict(material_id=m, material_name=mats.loc[m, "material_name"], criticality=mats.loc[m, "criticality"],
                 expected_supply_share_disrupted_30d=round(v, 4), days_of_cover=cover.get(m)) for m, v in risk_by_m.items()]
    rows.sort(key=lambda r: -r["expected_supply_share_disrupted_30d"])
    return dict(as_of=d["as_of"], materials=rows[: int(n)], low_cover=d["inventory_exposure"][:5])


def tool_inventory_coverage(material_id: str | None = None) -> dict:
    """Days of inventory cover per Tier-1 material at the latest inventory week before AS_OF."""
    inv = S.q("SELECT * FROM inventory")
    inv["week_start"] = pd.to_datetime(inv.week_start)
    wk = inv.week_start[inv.week_start < S.as_of()].max()
    last = inv[inv.week_start == wk]
    mats = S.q("SELECT material_id, material_name FROM materials").set_index("material_id")
    g_ = last.groupby("material_id").agg(closing=("closing_qty", "sum"), weekly_need=("required_qty", "sum"))
    g_["days_of_cover"] = (7 * g_.closing / g_.weekly_need.replace(0, np.nan)).round(1)
    if material_id:
        g_ = g_.loc[[material_id.upper()]]
    g_ = g_.sort_values("days_of_cover")
    return dict(week=str(wk.date()), materials=[dict(material_id=k, material_name=mats.loc[k, "material_name"],
                                                     days_of_cover=float(r.days_of_cover), closing_qty=round(float(r.closing), 1))
                                                for k, r in g_.iterrows()])


def tool_alternative_suppliers(supplier_id: str | None = None, material_id: str | None = None) -> dict:
    """Qualified alternatives (same or substitute material), screened by calibrated risk."""
    from backend.app.routes_ml import _sim
    from ml.optimisation.recovery import build_candidates, load_assumptions
    net, _ = _sim()
    if supplier_id:
        material_id = net.sup.loc[supplier_id, "primary_material_id"]
    if not material_id or material_id not in net.R.index:
        raise ValueError("need a Tier-1 supplier or a Tier-1 material (M01-M28)")
    risk = S.ensure_predictions().set_index("supplier_id").probability_calibrated
    c = build_candidates(net, material_id, {supplier_id} if supplier_id else set(), risk, load_assumptions(), S.q("SELECT * FROM material_substitutes"))
    c = sorted(c, key=lambda x: (x["risk_p30"] > 0.35, -x["reliability"] * (1 - x["risk_p30"]) * x["compat"]))
    return dict(material_id=material_id, material_name=net.mats.loc[material_id, "material_name"],
                alternatives=[dict(supplier_id=x["supplier_id"], supplier_name=x["supplier_name"], supplies=x["supplies_material"],
                                   compatibility=x["compat"], reliability=x["reliability"], calibrated_risk_p30=round(x["risk_p30"], 4),
                                   spare_capacity_per_day=round(x["spare_daily"], 1), unit_price_inr=round(x["unit_price"], 2),
                                   lead_time_days=x["modes"]["standard"]["lead"], onboarding_required=x["onboarding"]) for x in c[:8]])


def tool_compare_scenarios(severity_a: str = "Mild", severity_b: str = "Severe") -> dict:
    """Compare stored nine-scenario results (500 CRN replications) between two severities."""
    s = S.nine_scenarios()
    if not s.get("available"):
        raise ValueError("scenario results not available - run python -m ml.simulation.scenarios")
    df = pd.DataFrame(s["results"])
    keep = ["scenario", "severity", "policy", "financial_impact_inr", "unmet_demand_units", "recovery_time_days",
            "procurement_spend_inr", "min_service_level", "feasibility_full_recovery_share"]
    sel = df[df.severity.isin([severity_a.title(), severity_b.title()])][keep]
    return dict(rows=sel.round(2).to_dict(orient="records"), note="means over 500 replications; simulated INR on synthetic data")


def tool_explain_supplier_risk(supplier_id: str) -> dict:
    """SHAP explanation of the calibrated risk of one supplier at AS_OF."""
    from ml.decision.engine import _shap_row, predict_risk
    f = predict_risk(S.as_of().normalize().replace(day=1), [supplier_id])
    if not len(f):
        raise KeyError(f"unknown supplier {supplier_id}")
    p = S.ensure_predictions().set_index("supplier_id")
    return dict(supplier_id=supplier_id, supplier_name=_sup().loc[supplier_id, "supplier_name"],
                calibrated_probability=round(float(f.p_cal.iloc[0]), 4), raw_probability=round(float(f.p_raw.iloc[0]), 4),
                risk_band=S.risk_band(float(f.p_cal.iloc[0])), recommended_action=p.recommended_action.get(supplier_id),
                top_contributions=_shap_row(f, 6), base_rate_train=0.0653)


def tool_supplier_profile(supplier_id: str) -> dict:
    """Database profile of a supplier: attributes, upstream/downstream links, history."""
    d = S.supplier_detail(supplier_id)
    return dict(supplier=d["supplier"], risk=d["risk"], centrality=d["centrality"], upstream=d["upstream"][:10],
                downstream=d["downstream"][:10], disruption_history=d["disruption_history"][:5], shipment_history=d["shipment_history"])


def tool_decision_recommendation(supplier_id: str) -> dict:
    """Full executable decision workflow (prediction -> exposure -> action -> recovery plan)."""
    from ml.decision.engine import evaluate
    r = evaluate(supplier_id, str(S.as_of().date()))
    return {k: r[k] for k in ["supplier_id", "supplier_name", "calibrated_probability", "risk_category", "recommended_action",
                              "network_exposure_inr", "inventory_coverage_days", "recovery_plan", "decision_rationale"] if k in r}


TOOLS = {
    "most_critical_suppliers": (tool_most_critical_suppliers, {"n": "integer"}),
    "supplier_failure_impact": (tool_supplier_failure_impact, {"supplier_id": "string", "capacity_reduction": "number", "duration_days": "integer"}),
    "materials_at_risk": (tool_materials_at_risk, {"n": "integer"}),
    "inventory_coverage": (tool_inventory_coverage, {"material_id": "string"}),
    "alternative_suppliers": (tool_alternative_suppliers, {"supplier_id": "string", "material_id": "string"}),
    "compare_scenarios": (tool_compare_scenarios, {"severity_a": "string", "severity_b": "string"}),
    "explain_supplier_risk": (tool_explain_supplier_risk, {"supplier_id": "string"}),
    "supplier_profile": (tool_supplier_profile, {"supplier_id": "string"}),
    "decision_recommendation": (tool_decision_recommendation, {"supplier_id": "string"}),
}
REQUIRED = {"supplier_failure_impact": ["supplier_id"], "explain_supplier_risk": ["supplier_id"], "supplier_profile": ["supplier_id"],
            "decision_recommendation": ["supplier_id"]}


def call_tool(name, args):
    fn, schema = TOOLS[name]
    args = {k: v for k, v in (args or {}).items() if k in schema and v not in (None, "")}
    for k in ("supplier_id", "material_id"):
        if k in args:
            args[k] = str(args[k]).upper()
    return _clean(fn(**args))


# ======================================================================= rule-based router
def _find_supplier(text: str):
    m = SID_RE.search(text)
    if m:
        return m.group(0).upper()
    sup = S.suppliers_df()
    low = text.lower()
    for _, r in sup.iterrows():
        base = r.supplier_name.replace(" (Synthetic)", "").lower()
        if base in low:
            return r.supplier_id
    return None


def route(question: str):
    t = question.lower()
    sid = _find_supplier(question)
    mid = MID_RE.search(question)
    mid = mid.group(0).upper() if mid else None
    pct = re.search(r"(\d{1,3})\s*%", t)
    days = re.search(r"(\d{1,3})\s*days?", t)
    if any(w in t for w in ["compare", "versus", " vs ", "scenario"]):
        sev = [s for s in ["mild", "moderate", "severe"] if s in t] or ["mild", "severe"]
        return "compare_scenarios", dict(severity_a=sev[0], severity_b=sev[-1] if len(sev) > 1 else "severe")
    if sid and any(w in t for w in ["why", "explain", "classified", "reason", "driver"]):
        return "explain_supplier_risk", dict(supplier_id=sid)
    if sid and any(w in t for w in ["fail", "happens", "what if", "disrupt", "outage", "goes down", "impact"]):
        return "supplier_failure_impact", dict(supplier_id=sid, capacity_reduction=int(pct.group(1)) / 100 if pct else 1.0,
                                               duration_days=int(days.group(1)) if days else 30)
    if any(w in t for w in ["alternative", "alternate", "backup", "replace", "substitute"]):
        return "alternative_suppliers", dict(supplier_id=sid, material_id=mid)
    if sid and any(w in t for w in ["recommend", "action", "decision", "should we", "what to do"]):
        return "decision_recommendation", dict(supplier_id=sid)
    if any(w in t for w in ["inventory", "coverage", "cover", "stock"]):
        return "inventory_coverage", dict(material_id=mid)
    if any(w in t for w in ["raw material", "materials at risk", "material", "ingredient"]):
        return "materials_at_risk", {}
    if any(w in t for w in ["critical", "riskiest", "most risk", "top risk", "highest risk", "important supplier"]):
        n = re.search(r"top\s*(\d+)", t)
        return "most_critical_suppliers", dict(n=int(n.group(1)) if n else 5)
    if sid:
        return "supplier_profile", dict(supplier_id=sid)
    return None, {}


def inr(x):
    return f"INR {x:,.0f}"


def compose(tool, r) -> str:
    if tool == "most_critical_suppliers":
        lines = [f"{i + 1}. {s['supplier_name']} ({s['supplier_id']}, Tier-{s['tier']}, {s['region']}): p30 = {s['probability_calibrated']:.1%}, "
                 f"exposure {inr(s['exposure_inr'])}, expected loss {inr(s['expected_loss_inr'])} -> {s['recommended_action']}"
                 for i, s in enumerate(r["suppliers"])]
        return f"Most critical suppliers as of {r['as_of']} (ranked by calibrated probability x network exposure):\n" + "\n".join(lines)
    if tool == "supplier_failure_impact":
        mats = ", ".join(f"{k} {v}" for k, v in r["material_names"].items()) or "none"
        return (f"If {r['supplier_name']} ({r['supplier_id']}, Tier-{r['tier']}) suffers {r['assumption']}: {r['affected_suppliers']} suppliers are "
                f"affected, Tier-1 materials hit: {mats}. Unmet demand {r['unmet_demand_units']:,.0f} units, lost contribution margin "
                f"{inr(r['lost_contribution_margin_inr'])}, total simulated financial impact {inr(r['financial_impact_inr'])}, minimum service level "
                f"{r['min_service_level']:.1%}, recovery {r['recovery_time_days']} days. (Simulated on synthetic data.)")
    if tool == "materials_at_risk":
        lines = [f"- {m['material_name']} ({m['material_id']}, class {m['criticality']}): expected {m['expected_supply_share_disrupted_30d']:.1%} of supply "
                 f"disrupted in 30 days; cover {m['days_of_cover'] if m['days_of_cover'] is not None else 'n/a'} days" for m in r["materials"]]
        return f"Raw materials at risk (as of {r['as_of']}):\n" + "\n".join(lines)
    if tool == "inventory_coverage":
        lines = [f"- {m['material_name']} ({m['material_id']}): {m['days_of_cover']} days" for m in r["materials"][:12]]
        return f"Inventory cover at week {r['week']} (lowest first):\n" + "\n".join(lines)
    if tool == "alternative_suppliers":
        lines = [f"- {a['supplier_name']} ({a['supplier_id']}): supplies {a['supplies']} (compatibility {a['compatibility']}), reliability "
                 f"{a['reliability']:.2f}, risk {a['calibrated_risk_p30']:.1%}, spare {a['spare_capacity_per_day']}/day, lead {a['lead_time_days']} d, "
                 f"{inr(a['unit_price_inr'])}/unit{' (onboarding needed)' if a['onboarding_required'] else ''}" for a in r["alternatives"]]
        return f"Alternatives for {r['material_name']} ({r['material_id']}):\n" + "\n".join(lines or ["- none with spare capacity"])
    if tool == "compare_scenarios":
        lines = [f"- {x['scenario']} {x['severity']} / {x['policy']}: impact {inr(x['financial_impact_inr'])}, unmet {x['unmet_demand_units']:,.0f} units, "
                 f"recovery {x['recovery_time_days']:.0f} d, min service {x['min_service_level']:.1%}" for x in r["rows"]]
        return "Nine-scenario results (means of 500 common-random-number replications):\n" + "\n".join(lines)
    if tool == "explain_supplier_risk":
        lines = [f"- {c['feature']} = {c['value']}: {'raises' if c['shap_log_odds'] > 0 else 'lowers'} risk ({c['shap_log_odds']:+.3f} log-odds)"
                 for c in r["top_contributions"]]
        return (f"{r['supplier_name']} ({r['supplier_id']}) has calibrated 30-day disruption probability {r['calibrated_probability']:.1%} "
                f"(band {r['risk_band']}; training base rate {r['base_rate_train']:.1%}); action: {r['recommended_action']}. SHAP drivers:\n" + "\n".join(lines))
    if tool == "supplier_profile":
        s, k = r["supplier"], r["risk"]
        return (f"{s['supplier_name']} ({s['supplier_id']}), Tier-{s['tier']} supplier of {s['material_name']} in {s['region']}. Reliability "
                f"{s['reliability_score']:.2f}, utilisation {s['capacity_utilisation']:.0%}, calibrated risk {k['calibrated_probability']:.1%} "
                f"({k['band']}), exposure {inr(k['exposure_inr'])}. Observed upstream sources: {len(r['upstream'])}, buyers: {len(r['downstream'])}.")
    if tool == "decision_recommendation":
        return r["decision_rationale"]
    return json.dumps(r)[:1500]


HELP = ("I answer questions from the database, the calibrated risk model, the cascade simulator and the recovery optimiser. Try: "
        "'Which supplier is most critical?', 'What happens if S0122 fails for 45 days?', 'Which raw materials are at risk?', "
        "'What is the current inventory coverage?', 'Alternative supplier for S0003', 'Compare mild and severe scenarios', "
        "'Why is S0122 high risk?'")


# ======================================================================= Gemini
GEMINI_BUDGET_S = 30.0
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-2.5-flash-lite"]


def _gemini(question: str):
    key = current_app.config["GEMINI_API_KEY"]
    model = current_app.config["GEMINI_MODEL"]
    decls = [dict(name=n, description=(f.__doc__ or "").strip(),
                  parameters=dict(type="OBJECT", properties={k: {"type": v.upper()} for k, v in sch.items()}, required=REQUIRED.get(n, [])))
             for n, (f, sch) in TOOLS.items()]
    sys_txt = ("You are the Supply Chain Black Box analyst assistant for a fictional chocolate & malt-beverage manufacturer (synthetic data). "
               "Always call a tool to obtain numbers; never invent values. Supplier ids look like S0001, materials M01. Cite the numbers returned by "
               "tools, state that monetary values are simulated, and keep answers concise.")
    contents = [dict(role="user", parts=[dict(text=question)])]
    calls = []
    models = [model] + [m for m in FALLBACK_MODELS if m != model]
    import time as _t
    deadline = _t.time() + GEMINI_BUDGET_S  # total time budget before falling back to the deterministic router

    def post(body):
        """Try each model; on quota (429) / overload (503) rotate models, then wait once for the server-suggested delay."""
        import re as _re
        import time as _time
        last = None
        for attempt in range(2):
            wait = 0.0
            for mdl in list(models):
                left = deadline - _time.time()
                if left < 3:
                    raise TimeoutError("Gemini time budget exhausted")
                r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent",
                                  headers={"x-goog-api-key": key}, timeout=min(20.0, left), json=body)
                if r.ok:
                    models[:] = [mdl] + [m for m in models if m != mdl]  # prefer the model that worked
                    return r.json()
                last = r
                if r.status_code in (429, 500, 503) or (r.status_code in (400, 404) and "model" in r.text.lower()):
                    m = _re.search(r"retry in ([0-9.]+)s", r.text)
                    wait = max(wait, float(m.group(1)) if m else 3.0)
                    continue
                r.raise_for_status()
            if attempt == 0:
                _time.sleep(max(0.0, min(wait, 8.0, deadline - _time.time() - 5)))
        last.raise_for_status()

    for _ in range(5):
        data = post(dict(contents=contents, tools=[dict(function_declarations=decls)],
                         system_instruction=dict(parts=[dict(text=sys_txt)]), generation_config=dict(temperature=0.2)))
        cand = data["candidates"][0]["content"]
        parts = cand.get("parts", [])
        fcs = [p["functionCall"] for p in parts if "functionCall" in p]
        if not fcs:
            return "".join(p.get("text", "") for p in parts), calls
        contents.append(cand)
        resp_parts = []
        for fc in fcs:
            try:
                out = call_tool(fc["name"], fc.get("args", {}))
            except Exception as e:  # noqa: BLE001
                out = {"error": str(e)}
            calls.append(dict(tool=fc["name"], args=fc.get("args", {}), result=out))
            resp_parts.append(dict(functionResponse=dict(name=fc["name"], response=dict(content=out))))
        contents.append(dict(role="user", parts=resp_parts))
    return "I could not complete the request within the tool-call limit.", calls


@bp.post("/query")
@login_required
def query():
    """Ask the assistant: {question}. Returns answer, engine, tool calls and evidence."""
    question = str((request.get_json(silent=True) or {}).get("question", "")).strip()[:1000]
    if not question:
        raise ValueError("question is required")
    engine, calls, answer = "rule-based router", [], None
    if current_app.config["GEMINI_API_KEY"]:
        try:
            answer, calls = _gemini(question)
            engine = "gemini + tools"
        except Exception as e:  # noqa: BLE001
            engine = f"rule-based router (Gemini unavailable: {type(e).__name__})"
    if answer is None:
        tool, args = route(question)
        if tool is None:
            answer = HELP
        else:
            try:
                res = call_tool(tool, args)
                calls = [dict(tool=tool, args=args, result=res)]
                answer = compose(tool, res)
            except (KeyError, ValueError) as e:
                answer = f"I could not answer that: {e}. {HELP}"
    with db.get_engine().begin() as c:
        c.execute(insert(db.chat_logs).values(user_id=g.user.get("user_id"), question=question, engine=engine,
                                               tools_called=_clean([dict(tool=x["tool"], args=x["args"]) for x in calls]), answer=answer))
    return jsonify(_clean(dict(question=question, answer=answer, engine=engine, tool_calls=calls)))


@bp.get("/tools")
@login_required
def tools():
    """List assistant tools."""
    return jsonify([dict(name=n, description=(f.__doc__ or "").strip(), parameters=s) for n, (f, s) in TOOLS.items()])
