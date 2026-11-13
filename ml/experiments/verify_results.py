"""Independent verification of reported results (run after all experiments).

Checks
  * dataset counts reconcile (raw -> processed -> splits)
  * confusion matrices sum to test size; precision/recall/specificity/F1 recomputed from TP/FP/TN/FN
  * reported prevalence equals test prevalence
  * calibrators were fitted on the calibration split (snapshot ranges disjoint from test)
  * metrics recomputed from saved prediction files match the logged metrics
  * nine scenarios: 9 scenarios x N replications, identical CRN draws per replication, policies share baseline
Writes experiments/VERIFICATION_REPORT.md and exits non-zero on any failure.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.common.config import EXPERIMENTS, PROC, SYN
from ml.common.tracking import binary_metrics

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


def main():
    # ---------------- Experiment 1
    r1 = json.loads((EXPERIMENTS / "exp1_supplier_disruption" / "run.json").read_text())
    pred = pd.read_csv(EXPERIMENTS / "exp1_supplier_disruption" / "test_predictions.csv", parse_dates=["snapshot_ts"])
    sp = r1["splits"]
    check("Exp1 split rows reconcile with snapshot file (incl. embargo month)",
          sp["train"]["rows"] + sp["calibration"]["rows"] + sp["test"]["rows"] + 500 == 21 * 500,
          f"{sp['train']['rows']}+{sp['calibration']['rows']}+{sp['test']['rows']}+500 embargo = 10,500")
    check("Exp1 test predictions file has test size", len(pred) == sp["test"]["rows"])
    check("Exp1 test prevalence matches", abs(pred.disruption_next_30_days.mean() - sp["test"]["prevalence"]) < 1e-4)
    check("Exp1 calibration window precedes test", pd.Timestamp(sp["calibration"]["snapshots"][1]) < pd.Timestamp(sp["test"]["snapshots"][0]))
    best = r1["results"][r1["primary_model"]][r1["primary_calibration"]]
    m = binary_metrics(pred.disruption_next_30_days, pred.p_cal, 0.5)
    for k in ["tp", "fp", "tn", "fn", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier"]:
        check(f"Exp1 {r1['primary_model']}+{r1['primary_calibration']} {k} recomputed from predictions", abs(m[k] - best[k]) < 1e-5, f"{m[k]} vs {best[k]}")
    for model, d in r1["results"].items():
        for cal, v in d.items():
            n = v["tp"] + v["fp"] + v["tn"] + v["fn"]
            prec = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else 0.0
            rec = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else 0.0
            spec = v["tn"] / (v["tn"] + v["fp"])
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            check(f"Exp1 {model}/{cal} confusion matrix sums to test size", n == sp["test"]["rows"])
            check(f"Exp1 {model}/{cal} P/R/Spec/F1 consistent", max(abs(prec - v["precision"]), abs(rec - v["recall"]),
                                                                   abs(spec - v["specificity"]), abs(f1 - v["f1"])) < 1e-5)
    pc = r1["decision_policy"]["policy_confusion"]
    check("Exp1 decision-policy confusion matrix sums to test size", pc["tp"] + pc["fp"] + pc["tn"] + pc["fn"] == sp["test"]["rows"])
    check("Exp1 alerts = TP + FP", pc["alerts"] == pc["tp"] + pc["fp"])

    # ---------------- Experiment 3
    r3 = json.loads((EXPERIMENTS / "exp3_delivery_delay_dataco" / "run.json").read_text())
    p3 = pd.read_csv(EXPERIMENTS / "exp3_delivery_delay_dataco" / "test_predictions.csv")
    summ = json.loads((PROC / "processing_summary.json").read_text())["dataco"]
    s3 = r3["splits"]
    check("Exp3 raw rows - cancelled = processed", summ["raw_rows"] - summ["cancelled_removed"] == summ["processed_rows"])
    check("Exp3 splits sum to processed rows", s3["train"]["rows"] + s3["validation"]["rows"] + s3["test"]["rows"] == summ["processed_rows"])
    check("Exp3 splits chronological", s3["train"]["end"] < s3["validation"]["start"] and s3["validation"]["end"] < s3["test"]["start"])
    auc = roc_auc_score(p3.is_late, p3.p_xgboost)
    check("Exp3 XGBoost ROC-AUC recomputed", abs(auc - r3["classification"]["XGBoost"]["none"]["roc_auc"]) < 1e-5, f"{auc:.6f}")
    ap = average_precision_score(p3.is_late, p3.p_xgboost)
    check("Exp3 XGBoost PR-AUC recomputed", abs(ap - r3["classification"]["XGBoost"]["none"]["pr_auc"]) < 1e-5, f"{ap:.6f}")
    v = r3["classification"]["XGBoost"]["none"]
    check("Exp3 XGBoost confusion matrix sums to test size", v["tp"] + v["fp"] + v["tn"] + v["fn"] == s3["test"]["rows"])

    # ---------------- Experiment 2A
    r2 = json.loads((EXPERIMENTS / "exp2a_link_prediction" / "run.json").read_text())
    rel = pd.read_csv(SYN / "supplier_relationships.csv")
    check("Exp2A observed/hidden counts match data", r2["observed_edges"] == int((rel.visibility == "observed").sum()) and
          r2["hidden_edges"] == int((rel.visibility == "hidden").sum()))
    sp2 = r2["A1_split"]
    check("Exp2A edge split sums to observed edges", sp2["train"] + sp2["val"] + sp2["test"] == r2["observed_edges"])
    cand = pd.read_csv(EXPERIMENTS / "exp2a_link_prediction" / "candidate_links.csv")
    check("Exp2A candidate table size", len(cand) == r2["A2_candidates"]["n"])
    check("Exp2A candidates exclude observed edges",
          not set(map(frozenset, zip(cand.node_a, cand.node_b))) & set(map(frozenset, zip(rel[rel.visibility == "observed"].supplier_id,
                                                                                         rel[rel.visibility == "observed"].buyer_id))))
    check("Exp2A candidate model chosen by validation AP", r2["candidate_model"] == max(
        {k: v["val_ap"] for k, v in r2["A2_results"].items() if "val_ap" in v}.items(), key=lambda kv: kv[1])[0])

    # ---------------- Scenarios
    reps = pd.read_csv(EXPERIMENTS / "exp4_5_scenarios" / "scenario_replications.csv")
    summ9 = pd.read_csv(EXPERIMENTS / "scenario_results.csv")
    check("Nine scenarios present", sorted(summ9.scenario) == sorted(f"S{i}" for i in range(1, 10)))
    n_rep = reps.groupby("scenario").size()
    check("Equal replications per scenario", n_rep.nunique() == 1, f"{n_rep.iloc[0]} each")
    same = reps.groupby(["severity", "replication"]).realised_duration.nunique().max() == 1
    check("Common random numbers: identical realised duration across policies per replication", same)
    for sev, g in summ9.groupby("severity"):
        check(f"{sev}: disrupted capacity identical across policies", g.disrupted_capacity_available.nunique() == 1)
    a = summ9[summ9.policy.str.startswith("A")]
    check("Policy A places no recovery orders", (a.n_recovery_orders == 0).all() and (a.procurement_spend_inr == 0).all())
    mean_check = reps.groupby("scenario").financial_impact_inr.mean().reindex(summ9.scenario).values
    check("Scenario means recomputed from replications", np.allclose(mean_check, summ9.financial_impact_inr.values))
    comps = ["lost", "pen"]
    parts = ["production_loss_cm_inr", "shortage_penalty_inr", "procurement_premium_inr", "transportation_cost_inr", "holding_cost_inr",
             "onboarding_cost_inr", "interruption_cost_inr", "recovery_management_cost_inr"]
    check("Financial impact = sum of documented components", np.allclose(reps[parts].sum(1), reps.financial_impact_inr))

    # ---- robustness: ablation arms that coincide with main policies must reproduce them exactly (CRN integrity)
    abf = EXPERIMENTS / "robustness" / "ablation_replications.csv"
    if abf.exists():
        ab = pd.read_csv(abf)
        same = {"A: No intervention": "A: No intervention", "B-obs: rule, observed graph": "B: Rule-based", "C-pred: MILP, obs.+predicted": "C: Optimisation"}
        for arm, pol in same.items():
            for sev in ["Mild", "Moderate", "Severe"]:
                x = ab[(ab.policy == arm) & (ab.severity == sev)].sort_values("replication").financial_impact_inr.values
                y = reps[(reps.policy == pol) & (reps.severity == sev)].sort_values("replication").financial_impact_inr.values
                check(f"Ablation arm '{arm[:6]}' reproduces main policy ({sev})", len(x) == len(y) and np.allclose(x, y), f"n={len(x)}")
        check("Ablation impact = sum of documented components", np.allclose(ab[parts].sum(1), ab.financial_impact_inr))
    st = EXPERIMENTS / "statistics.json"
    if st.exists():
        sj = json.loads(st.read_text())
        check("Statistics: nine policy comparisons in Holm family", len(sj["scenarios"]) == 9)
        check("Statistics: Holm p >= raw p", all(r["p_holm"] >= r["p_raw"] - 1e-12 for r in sj["scenarios"] + sj.get("ablation", [])))

    ok = all(c[1] for c in checks)
    lines = ["# Results Verification Report", "", f"Generated by `python -m ml.experiments.verify_results`. **{sum(c[1] for c in checks)} / {len(checks)} checks passed.**", "",
             "| # | Check | Result | Detail |", "|---:|---|---|---|"]
    for i, (n, o, d) in enumerate(checks, 1):
        lines.append(f"| {i} | {n} | {'PASS' if o else '**FAIL**'} | {d} |")
    (EXPERIMENTS / "VERIFICATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"{sum(c[1] for c in checks)}/{len(checks)} checks passed")
    for n, o, d in checks:
        if not o:
            print("FAIL:", n, d)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
