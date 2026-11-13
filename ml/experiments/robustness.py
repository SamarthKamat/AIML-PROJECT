"""Robustness analyses requested in review.

1. LINK-INFORMATION ABLATION (identifies the value of predicted links separately from the optimiser).
   2 x 2 factorial  {rule-based, MILP} x {observed graph, observed + predicted graph}  plus an ORACLE arm
   (MILP with the true graph, an upper bound).  Everything else - costs, timing, triggers, candidate
   generation, risk inputs and common random numbers - is identical.  500 replications per cell.
2. ONE-AT-A-TIME PARAMETER SENSITIVITY of the policy ranking (A none, B rule-based, C optimisation):
   recovery-management overhead, shortage penalty, interruption cost, alternate spare capacity,
   alternate failure risk, disruption duration and network visibility; Mild and Severe shocks,
   200 CRN replications per cell.

Writes experiments/robustness/{ablation_replications.csv, ablation_summary.csv, sensitivity.csv}.
Run: python -m ml.experiments.robustness [ablation|sensitivity|all]
"""
from __future__ import annotations

import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ml.common.config import EXPERIMENTS, SYN
from ml.optimisation.recovery import load_assumptions
from ml.simulation import scenarios as S
from ml.simulation.cascade import CascadeSimulator, Disruption, load_network
from ml.simulation.policies import NoIntervention, Optimised, RuleBased

OUT = EXPERIMENTS / "robustness"
OUT.mkdir(exist_ok=True)


def _graphs(net):
    observed, predicted = S.knowledge_graphs(net)
    true = pd.read_csv(SYN / "supplier_relationships.csv")
    return dict(observed=observed, predicted=predicted, oracle=true)


def _run_cells(cells, sev, reps, A=None, spare_mult=1.0, risk_mult=1.0, visibility=None):
    """cells: list of (label, kind in {A, B, C}, graph key). Returns per-replication rows."""
    A = A or load_assumptions()
    net = load_network(start=A["scenario_start_date"])
    if spare_mult != 1.0:
        net.spare_daily = net.spare_daily * spare_mult
    sim = CascadeSimulator(net, horizon=A["horizon_days"])
    disrupted = S.scenario_suppliers(net)
    risk = S.latest_risk(net)
    true_risk = (risk * risk_mult).clip(0, 1)
    g = _graphs(net)
    if visibility is not None:   # change how much of the true network the buyer knows (B and C alike)
        rel = g["oracle"]
        hidden = rel[rel.visibility != "observed"]
        obs = rel[rel.visibility == "observed"]
        if visibility > 0:
            known = pd.concat([obs, hidden.sample(frac=visibility, random_state=1)])
        else:
            deep = obs[obs.buyer_id != "C001"]
            known = pd.concat([obs[obs.buyer_id == "C001"], deep.sample(frac=1 + visibility, random_state=1)])
        extra = g["predicted"][g["predicted"].visibility == "predicted"]
        g["observed"], g["predicted"] = known, pd.concat([known, extra], ignore_index=True)
    subs = pd.read_csv(SYN / "material_substitutes.csv")
    rows = []
    for label, kind, gk in cells:
        for r in range(reps):
            rng = np.random.default_rng(A["base_seed"] + r)
            draws = sim.make_draws(rng, alt_risk=true_risk)
            rep = dict(supplier_ids=disrupted, severity=sev)
            pol = NoIntervention() if kind == "A" else (RuleBased(net, g[gk], rep, A, risk) if kind == "B"
                                                       else Optimised(net, g[gk], rep, A, risk, subs))
            out = sim.run([Disruption(disrupted, sev, 0, A["nominal_disruption_duration_days"])], pol, draws)
            m = S.metrics(out, sim, net, A, disrupted, label)
            m.update(replication=r, capacity_loss=sev)
            rows.append(m)
    return rows


ARMS = [("A: No intervention", "A", "observed"),
        ("B-obs: rule, observed graph", "B", "observed"),
        ("B-pred: rule, obs.+predicted", "B", "predicted"),
        ("C-obs: MILP, observed graph", "C", "observed"),
        ("C-pred: MILP, obs.+predicted", "C", "predicted"),
        ("C-oracle: MILP, true graph", "C", "oracle")]


def _abl_job(args):
    sev_name, sev, reps = args
    rows = _run_cells(ARMS, sev, reps)
    for r in rows:
        r["severity"] = sev_name
    return rows


def ablation(reps=500):
    jobs = [(k, v, reps) for k, v in S.SEVERITIES.items()]
    with ProcessPoolExecutor(3) as ex:
        rows = [r for part in ex.map(_abl_job, jobs) for r in part]
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "ablation_replications.csv", index=False)
    summ = res.groupby(["severity", "policy"], sort=False).agg(
        impact_inr=("financial_impact_inr", "mean"), impact_se=("financial_impact_inr", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        unmet_units=("unmet_demand_units", "mean"), min_service=("min_service_level", "mean")).reset_index()
    summ.to_csv(OUT / "ablation_summary.csv", index=False)
    print(summ.round(0).to_string())


SETTINGS = [("base", {}, {}),
            ("overhead 0", {"recovery_management_cost_per_day_inr": 0}, {}),
            ("overhead 50k", {"recovery_management_cost_per_day_inr": 50000}, {}),
            ("penalty 5%", {"shortage_penalty_pct_of_price": 0.05}, {}),
            ("penalty 30%", {"shortage_penalty_pct_of_price": 0.30}, {}),
            ("interruption 25k", {"production_interruption_cost_per_plant_day_inr": 25000}, {}),
            ("interruption 150k", {"production_interruption_cost_per_plant_day_inr": 150000}, {}),
            ("duration 30 d", {"nominal_disruption_duration_days": 30}, {}),
            ("duration 90 d", {"nominal_disruption_duration_days": 90}, {}),
            ("spare capacity x0.5", {}, {"spare_mult": 0.5}),
            ("spare capacity x1.5", {}, {"spare_mult": 1.5}),
            ("alternate risk x0.5", {}, {"risk_mult": 0.5}),
            ("alternate risk x2", {}, {"risk_mult": 2.0}),
            ("visibility -50% deep links", {}, {"visibility": -0.5}),
            ("visibility +50% hidden links", {}, {"visibility": 0.5})]
POL3 = [("A", "A", "observed"), ("B", "B", "observed"), ("C", "C", "predicted")]


def _sens_job(args):
    name, a_over, kw, reps = args
    A = copy.deepcopy(load_assumptions())
    A.update(a_over)
    out = []
    for sev_name in ("Mild", "Severe"):
        rows = pd.DataFrame(_run_cells(POL3, S.SEVERITIES[sev_name], reps, A=A, **kw))
        piv = rows.pivot(index="replication", columns="policy", values="financial_impact_inr")
        mean = piv.mean()
        d = piv["A"] - piv["C"]
        out.append(dict(setting=name, severity=sev_name, A=mean["A"], B=mean["B"], C=mean["C"], best=mean.idxmin(),
                        save_C_over_A=d.mean(), save_lo=d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d)),
                        save_hi=d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)), reps=reps))
    return out


def sensitivity(reps=200):
    jobs = [(n, a, k, reps) for n, a, k in SETTINGS]
    with ProcessPoolExecutor(4) as ex:
        rows = [r for part in ex.map(_sens_job, jobs) for r in part]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sensitivity.csv", index=False)
    print(df.assign(A=df.A / 1e5, B=df.B / 1e5, C=df.C / 1e5, save=df.save_C_over_A / 1e5)[
        ["setting", "severity", "A", "B", "C", "best", "save"]].round(2).to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("ablation", "all"):
        ablation()
    if what in ("sensitivity", "all"):
        sensitivity()
    (OUT / "README.json").write_text(json.dumps({"ablation_reps": 500, "sensitivity_reps": 200,
                                                 "note": "simulated INR on synthetic data; assumption-conditional"}, indent=1))
