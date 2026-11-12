"""Nine-scenario business evaluation (3 severities x 3 policies) + Experiment 4 disruption-type sweep.

Fairness controls
  * identical network, demand plan, initial inventory, economic assumptions for every policy
  * common random numbers: replication r uses seed base_seed + r for ALL nine scenarios, so a
    policy comparison within a severity sees the same duration, demand, lead-time, yield and
    alternate-failure draws
  * 500 replications per scenario; mean, 95 % percentile interval and standard error reported

Run: python -m ml.simulation.scenarios
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ml.common.config import EXPERIMENTS, MODELS, SYN
from ml.optimisation.recovery import DEFAULT_ASSUMPTIONS, load_assumptions
from ml.simulation.cascade import CascadeSimulator, Disruption, load_network
from ml.simulation.policies import NoIntervention, Optimised, RuleBased

EXP_ID = "exp4_5_scenarios"
SEVERITIES = {"Mild": 0.25, "Moderate": 0.50, "Severe": 1.00}
POLICIES = ["A: No intervention", "B: Rule-based", "C: Optimisation"]


def scenario_suppliers(net, k=5):
    """Five largest Tier-2 cocoa-bean (M29) traders - concentrated, partly hidden dependency."""
    c = net.sup[(net.sup.tier == 2) & (net.sup.primary_material_id == "M29")].sort_values("size_index", ascending=False)
    return list(c.index[:k])


def latest_risk(net) -> pd.Series:
    """Calibrated 30-day disruption probability per supplier at the scenario start (Exp 1 primary model)."""
    import joblib
    from ml.data_engineering.features_supplier import build_snapshot_features, load_tables
    A = load_assumptions()
    bundle = joblib.load(MODELS / "exp1_xgboost.joblib")
    f = build_snapshot_features(load_tables(), [pd.Timestamp(A["scenario_start_date"])])
    p = bundle["calibrator"].predict(bundle["model"].predict_proba(f[bundle["features"]])[:, 1])
    return pd.Series(p, index=f.supplier_id.values)


def knowledge_graphs(net):
    rel = pd.read_csv(SYN / "supplier_relationships.csv")
    observed = rel[rel.visibility == "observed"]
    cand = pd.read_csv(EXPERIMENTS / "exp2a_link_prediction" / "candidate_links.csv").sort_values("rank").head(500)
    tier = net.sup.tier.to_dict()
    tier["C001"] = 0
    rows = []
    for a, b in zip(cand.node_a, cand.node_b):
        s, buyer = (a, b) if tier[a] > tier[b] else (b, a)
        mat = net.sup.loc[s, "primary_material_id"]
        rows.append(dict(supplier_id=s, buyer_id=buyer, material_id=mat, dependency_pct=0.5, visibility="predicted"))
    predicted = pd.concat([observed, pd.DataFrame(rows)], ignore_index=True)
    return observed, predicted


def metrics(out, sim, net, A, disrupted, policy_name):
    H = sim.H
    cm = net.cm_unit.reindex(sim.plist).values
    price = net.price_unit.reindex(sim.plist).values
    unmet_units = out["unmet"].sum(0)
    lost_cm = float((unmet_units * cm).sum())
    penalty = float((unmet_units * price).sum() * A["shortage_penalty_pct_of_price"])
    demand_total = float((sim.D.sum()) * H)
    rec = out["recovery_log"]
    proc_spend = sum(o["delivered_qty"] * o["unit_price"] for o in rec)
    premium = sum(o["delivered_qty"] * (o["unit_price"] - o["base_price"]) for o in rec)
    transport = sum(o["delivered_qty"] * o["unit_price"] * o["freight_rate"] for o in rec)
    fixed = sum(o.get("fixed_cost", 0.0) for o in rec if o["delivered_qty"] > 0 or o.get("fixed_cost", 0) > 0)
    excess = np.maximum(0, out["I"] - sim.I0[None, :])
    holding = float((excess * sim.price_m[None, :]).sum() * A["holding_cost_annual_rate"] / 365)
    plants = sorted(set(sim.plant))
    plant_fill = {}
    for pl in plants:
        m = sim.plant == pl
        dem = sim.D[m].sum()
        plant_fill[pl] = 1 - out["unmet"][:, m].sum(1) / dem
    interruption_days = sum(int((f < A["production_interruption_fill_threshold"]).sum()) for f in plant_fill.values())
    interruption = interruption_days * A["production_interruption_cost_per_plant_day_inr"]
    ok = (out["fill"] >= 0.99) & (out["arrival_ratio"].min(1) >= 0.99)
    bad = np.where(~ok)[0]
    recovery_days = int(bad.max() + 1) if len(bad) else 0
    censored = bool(len(bad) and bad.max() == H - 1)
    recovery_mgmt = recovery_days * A["recovery_management_cost_per_day_inr"]
    short_value = float((out["mat_short"] * sim.price_m[None, :]).sum())
    spare_used = [o["delivered_qty"] / max(o["rate_cap"] * max(1, H - o["first_delivery_day"]), 1e-9) for o in rec]
    disrupted_idx = [sim.si[s] for s in disrupted]
    cap_avail = 1 - out["loss"][:, disrupted_idx].mean()
    total = lost_cm + penalty + premium + transport + holding + fixed + interruption + recovery_mgmt
    feas = "full recovery" if unmet_units.sum() <= 1e-6 else ("partial" if unmet_units.sum() / demand_total < 0.02 else "insufficient")
    return dict(
        policy=policy_name,
        disrupted_capacity_available=round(float(cap_avail), 4),
        affected_suppliers=int((out["A"].min(0) < 0.99).sum()),
        affected_materials=int((out["arrival_ratio"].min(0) < 0.99).sum()),
        affected_production_sites=int(sum((1 - f).max() > 1e-6 for f in plant_fill.values())),
        inventory_shortage_value_inr=short_value,
        unmet_demand_units=float(unmet_units.sum()),
        unmet_demand_pct=float(unmet_units.sum() / demand_total * 100),
        production_loss_cm_inr=lost_cm,
        shortage_penalty_inr=penalty,
        procurement_spend_inr=float(proc_spend),
        procurement_premium_inr=float(premium),
        transportation_cost_inr=float(transport),
        holding_cost_inr=holding,
        onboarding_cost_inr=float(fixed),
        interruption_cost_inr=float(interruption),
        recovery_management_cost_inr=float(recovery_mgmt),
        financial_impact_inr=float(total),
        recovery_time_days=recovery_days,
        recovery_censored=censored,
        alt_supplier_utilisation=float(np.mean(spare_used)) if spare_used else 0.0,
        n_recovery_orders=len(rec),
        feasibility=feas,
        min_service_level=float(out["fill"].min()),
    )


def make_policy(name, net, A, observed, predicted, risk, disrupted, sev, subs):
    reported = dict(supplier_ids=disrupted, severity=sev)
    if name.startswith("A"):
        return NoIntervention()
    if name.startswith("B"):
        return RuleBased(net, observed, reported, A, risk)
    return Optimised(net, predicted, reported, A, risk, subs)


def run_nine(reps=None, verbose=True):
    A = load_assumptions()
    reps = reps or A["replications"]
    net = load_network(start=A["scenario_start_date"])
    sim = CascadeSimulator(net, horizon=A["horizon_days"])
    disrupted = scenario_suppliers(net)
    risk = latest_risk(net)
    observed, predicted = knowledge_graphs(net)
    subs = pd.read_csv(SYN / "material_substitutes.csv")
    rows = []
    t0 = time.time()
    sid = 1
    for sev_name, sev in SEVERITIES.items():
        for pol in POLICIES:
            for r in range(reps):
                rng = np.random.default_rng(A["base_seed"] + r)            # common random numbers
                draws = sim.make_draws(rng, alt_risk=risk)
                policy = make_policy(pol, net, A, observed, predicted, risk, disrupted, sev, subs)
                out = sim.run([Disruption(disrupted, sev, 0, A["nominal_disruption_duration_days"])], policy, draws)
                m = metrics(out, sim, net, A, disrupted, pol)
                m.update(scenario=f"S{sid}", severity=sev_name, capacity_loss=sev, replication=r,
                         realised_duration=int(round(A["nominal_disruption_duration_days"] * draws.dur_mult)))
                rows.append(m)
            if verbose:
                print(f"S{sid} {sev_name} {pol} done ({time.time() - t0:.0f}s)", flush=True)
            sid += 1
    res = pd.DataFrame(rows)
    return res, dict(disrupted=disrupted, A=A, net=net, sim=sim, risk=risk, observed=observed, predicted=predicted)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in res.columns if res[c].dtype.kind in "fi" and c not in ("replication", "capacity_loss")]
    g = res.groupby(["scenario", "severity", "policy"], sort=False)
    mean = g[num].mean()
    out = mean.copy()
    for c in ["financial_impact_inr", "unmet_demand_units", "recovery_time_days", "production_loss_cm_inr"]:
        out[c + "_p2.5"] = g[c].quantile(0.025)
        out[c + "_p97.5"] = g[c].quantile(0.975)
        out[c + "_se"] = g[c].std(ddof=1) / np.sqrt(g[c].count())
    out["feasibility_full_recovery_share"] = g.feasibility.apply(lambda s: (s == "full recovery").mean())
    out["recovery_censored_share"] = g.recovery_censored.mean()
    out["replications"] = g.size()
    return out.reset_index()


def paired_comparison(res: pd.DataFrame) -> pd.DataFrame:
    """Paired (CRN) differences of financial impact between policies within each severity."""
    rows = []
    for sev, g in res.groupby("severity", sort=False):
        piv = g.pivot(index="replication", columns="policy", values="financial_impact_inr")
        for a, b in [("A: No intervention", "B: Rule-based"), ("A: No intervention", "C: Optimisation"), ("B: Rule-based", "C: Optimisation")]:
            d = piv[a] - piv[b]
            rows.append(dict(severity=sev, comparison=f"{a} minus {b}", mean_saving_inr=d.mean(),
                             ci95_low=d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d)),
                             ci95_high=d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)),
                             share_reps_b_better=float((d > 0).mean())))
    return pd.DataFrame(rows)


def exp4_types(ctx, reps=200):
    """Experiment 4: no-intervention cascade analysis for seven disruption types."""
    net, sim, A, risk = ctx["net"], ctx["sim"], ctx["A"], ctx["risk"]
    sup = net.sup
    t1 = sup[sup.tier == 1]
    from ml.decision.exposure import ExposureModel
    E = ExposureModel(graph="full").all_exposures()
    top_t1 = E[sup.tier == 1].idxmax()
    top_t2 = E[sup.tier == 2].idxmax()
    top_t3 = E[sup.tier == 3].idxmax()
    imports_ci = list(sup[sup.country.isin(["Cote d'Ivoire", "Ghana", "Indonesia", "Malaysia"]) & (sup.import_lead_extra_days > 0)].index)
    kerala = list(sup[sup.region == "Kerala"].index)
    top_risk = list(risk.sort_values(ascending=False).index[:3])
    cases = {
        "Direct Tier-1 failure (highest exposure, 100%, 45d)": [Disruption([top_t1], 1.0, 0, 45)],
        "Tier-2 failure (highest exposure, 100%, 45d)": [Disruption([top_t2], 1.0, 0, 45)],
        "Tier-3 failure (highest exposure, 100%, 45d)": [Disruption([top_t3], 1.0, 0, 45)],
        "Transportation / port disruption (imports via sea, 60%, 30d)": [Disruption(imports_ci, 0.6, 0, 30)],
        "Weather: Kerala monsoon flood (all suppliers in region, 70%, 21d)": [Disruption(kerala, 0.7, 0, 21)],
        "Capacity reduction (top-exposure Tier-1, 50%, 60d)": [Disruption([top_t1], 0.5, 0, 60)],
        "Multiple simultaneous failures (3 highest-risk suppliers, 100%, 30d)": [Disruption(top_risk, 1.0, 0, 30)],
    }
    rows = []
    for name, dis in cases.items():
        for r in range(reps):
            draws = sim.make_draws(np.random.default_rng(A["base_seed"] + r), alt_risk=None)
            out = sim.run(dis, None, draws)
            m = metrics(out, sim, net, A, dis[0].supplier_ids, "A: No intervention")
            m.update(case=name, replication=r, n_disrupted=sum(len(d.supplier_ids) for d in dis))
            rows.append(m)
    df = pd.DataFrame(rows)
    cols = ["n_disrupted", "affected_suppliers", "affected_materials", "affected_production_sites", "inventory_shortage_value_inr",
            "unmet_demand_units", "production_loss_cm_inr", "financial_impact_inr", "recovery_time_days", "min_service_level"]
    s = df.groupby("case", sort=False)[cols].mean()
    s["financial_impact_p97.5"] = df.groupby("case", sort=False).financial_impact_inr.quantile(0.975)
    return s.reset_index(), dict(top_t1=top_t1, top_t2=top_t2, top_t3=top_t3, imports=len(imports_ci), kerala=len(kerala), top_risk=top_risk)


def convergence(res):
    g = res[(res.severity == "Severe") & (res.policy == "A: No intervention")].sort_values("replication").financial_impact_inr
    run_mean = g.expanding().mean()
    se = g.expanding().std() / np.sqrt(np.arange(1, len(g) + 1))
    return pd.DataFrame(dict(n=np.arange(1, len(g) + 1), running_mean=run_mean.values, se=se.values,
                             rel_se=(se / run_mean).values))


def main():
    from ml.common.tracking import log_run
    d = EXPERIMENTS / EXP_ID
    d.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS / "scenario_assumptions.json").write_text(json.dumps(DEFAULT_ASSUMPTIONS, indent=2))
    res, ctx = run_nine()
    res.to_csv(d / "scenario_replications.csv", index=False)
    summ = summarise(res)
    summ.to_csv(EXPERIMENTS / "scenario_results.csv", index=False)
    comp = paired_comparison(res)
    comp.to_csv(EXPERIMENTS / "scenario_comparison.csv", index=False)
    conv = convergence(res)
    conv.to_csv(d / "mc_convergence.csv", index=False)
    e4, e4info = exp4_types(ctx)
    e4.to_csv(EXPERIMENTS / "exp4_disruption_types.csv", index=False)
    net = ctx["net"]
    disrupted = ctx["disrupted"]
    rel = pd.read_csv(SYN / "supplier_relationships.csv")
    hub_edges = rel[rel.supplier_id.isin(disrupted)]
    log_run(EXP_ID, dict(
        data_label="synthetic + simulated outcomes", disrupted_suppliers=disrupted,
        disrupted_names=net.sup.loc[disrupted, "supplier_name"].tolist(),
        hub_edges=dict(observed=int((hub_edges.visibility == "observed").sum()), hidden=int((hub_edges.visibility == "hidden").sum())),
        assumptions=ctx["A"], replications=int(res.replication.nunique()),
        crn="replication r uses np.random.default_rng(base_seed + r) for all scenarios",
        mc_convergence_final_relative_se=float(conv.rel_se.iloc[-1]),
        exp4_cases=e4info, scenario_summary=summ[["scenario", "severity", "policy", "financial_impact_inr", "unmet_demand_units",
                                                   "recovery_time_days"]].to_dict(orient="records"),
        paired=comp.to_dict(orient="records"),
        summary={"severe_A_impact": float(summ.loc[summ.scenario == "S7", "financial_impact_inr"].iloc[0]),
                 "severe_C_impact": float(summ.loc[summ.scenario == "S9", "financial_impact_inr"].iloc[0])}))
    pd.set_option("display.width", 250)
    print(summ[["scenario", "severity", "policy", "financial_impact_inr", "unmet_demand_units", "recovery_time_days",
                "procurement_spend_inr", "affected_suppliers", "feasibility_full_recovery_share"]].round(0).to_string())
    print(comp.round(0).to_string())
    print(e4.round(0).to_string())


if __name__ == "__main__":
    main()
