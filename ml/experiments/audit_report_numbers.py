"""Independent audit of every headline number quoted in the IEEE report.

Each check recomputes the value from raw data, saved models or per-replication result files (not from the
paper's generated macros) and compares it with the value printed in the report. A second block checks that the
macros in documentation/final_report/tables/numbers.tex agree with the recomputed values.
Writes experiments/REPORT_NUMBER_AUDIT.md.   Run: python -m ml.experiments.audit_report_numbers
"""
from __future__ import annotations

import json
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.common.config import EXPDATA, EXPERIMENTS, MODELS, ROOT, SYN, TEST_START
from ml.common.tracking import ece

rows = []


def chk(claim, reported, computed, tol=0.0, fmt="{}"):
    ok = abs(float(reported) - float(computed)) <= tol + 1e-12
    rows.append((claim, fmt.format(reported), fmt.format(computed) if not isinstance(computed, str) else computed, ok))


def main():
    sup = pd.read_csv(SYN / "suppliers.csv")
    rel = pd.read_csv(SYN / "supplier_relationships.csv")
    dis = pd.read_csv(SYN / "disruptions.csv")
    chk("Suppliers", 500, len(sup))
    for t, n in [(1, 120), (2, 200), (3, 180)]:
        chk(f"Tier-{t} suppliers", n, int((sup.tier == t).sum()))
    chk("Supplier links (total)", 1500, len(rel))
    chk("Observed links", 461, int((rel.visibility == "observed").sum()))
    chk("Hidden links", 1039, int((rel.visibility != "observed").sum()))
    chk("Hidden-link share (%)", 69, 100 * (rel.visibility != "observed").mean(), tol=0.5, fmt="{:.1f}")
    chk("Graph nodes = suppliers + manufacturer", 501, len(set(rel.supplier_id) | set(rel.buyer_id) | set(sup.supplier_id)))
    chk("Disruptions 2024-2025", 1048, len(dis))
    chk("Qualifying disruptions", 782, int(dis.is_qualifying.sum()))

    snap = pd.read_csv(EXPDATA / "exp1_supplier_snapshots.csv", parse_dates=["snapshot_ts"])
    te = snap[snap.snapshot_ts >= TEST_START]
    y = te.disruption_next_30_days.values
    chk("E1 test supplier-months", 3000, len(te))
    chk("E1 test positives", 223, int(y.sum()))
    chk("E1 test prevalence (%)", 7.4, 100 * y.mean(), tol=0.05, fmt="{:.2f}")
    p = {}
    for name, f in [("LR", "exp1_logistic_regression"), ("XGB", "exp1_xgboost"), ("DT", "exp1_decision_tree")]:
        b = joblib.load(MODELS / f"{f}.joblib")
        raw = b["model"].predict_proba(te[b["features"]])[:, 1]
        p[name] = (raw, b["calibrator"].predict(raw))
    chk("LR ROC-AUC", 0.666, roc_auc_score(y, p["LR"][1]), tol=0.0005, fmt="{:.3f}")
    chk("LR PR-AUC", 0.150, average_precision_score(y, p["LR"][1]), tol=0.0005, fmt="{:.3f}")
    chk("XGBoost ROC-AUC", 0.626, roc_auc_score(y, p["XGB"][1]), tol=0.0005, fmt="{:.3f}")
    chk("XGBoost PR-AUC", 0.122, average_precision_score(y, p["XGB"][1]), tol=0.0005, fmt="{:.3f}")
    chk("XGBoost ECE raw", 0.020, ece(y, p["XGB"][0]), tol=0.0005, fmt="{:.4f}")
    chk("XGBoost ECE Platt", 0.010, ece(y, p["XGB"][1]), tol=0.0005, fmt="{:.4f}")
    chk("Decision tree ECE raw", 0.037, ece(y, p["DT"][0]), tol=0.0005, fmt="{:.4f}")
    chk("Decision tree ECE Platt", 0.011, ece(y, p["DT"][1]), tol=0.0005, fmt="{:.4f}")

    cand = pd.read_csv(EXPERIMENTS / "exp2a_link_prediction" / "candidate_links.csv")
    chk("A2 candidate pairs", 8863, len(cand))
    chk("A2 hidden links among candidates", 1039, int(cand.ground_truth_hidden_edge.sum()))
    chk("A2 base rate", 0.117, cand.ground_truth_hidden_edge.mean(), tol=0.0005, fmt="{:.3f}")
    a2 = json.loads((EXPERIMENTS / "exp2a_link_prediction" / "run.json").read_text())["A2_results"]["MLP"]
    chk("A2 MLP AP, 5-seed mean (run log)", 0.227, a2["ap"], tol=0.0005, fmt="{:.3f}")
    chk("A2 MLP P@100, 5-seed mean (run log)", 0.32, a2["P@100"], tol=0.005, fmt="{:.2f}")
    chk("A2 seed-0 list used downstream: AP (recomputed)", 0.220, average_precision_score(cand.ground_truth_hidden_edge, cand.score), tol=0.0005, fmt="{:.3f}")
    top = cand.sort_values("score", ascending=False)
    chk("A2 seed-0 list used downstream: P@100 (recomputed)", 0.29, top.ground_truth_hidden_edge.head(100).mean(), tol=0.005, fmt="{:.2f}")

    ex = json.loads((EXPERIMENTS / "exp1_supplier_disruption" / "run.json").read_text())["exposure_summary"]
    for t, v in [("2", 2.1), ("3", 10.8)]:
        chk(f"Tier-{t} exposure understatement (x)", v, ex["true_graph_mean_by_tier"][t] / ex["observed_graph_mean_by_tier"][t], tol=0.05, fmt="{:.2f}")

    reps = pd.read_csv(EXPERIMENTS / "exp4_5_scenarios" / "scenario_replications.csv")
    chk("Replications per scenario", 500, int(reps.groupby(["severity", "policy"]).size().min()))
    sv = reps[reps.severity == "Severe"].pivot(index="replication", columns="policy", values="financial_impact_inr")
    for pol, v in [("A: No intervention", 4.72), ("B: Rule-based", 3.98), ("C: Optimisation", 1.16)]:
        chk(f"Severe impact, {pol[0]} (INR crore)", v, sv[pol].mean() / 1e7, tol=0.005, fmt="{:.3f}")
    d = sv["A: No intervention"] - sv["C: Optimisation"]
    chk("Severe saving C over A (INR crore)", 3.56, d.mean() / 1e7, tol=0.005, fmt="{:.3f}")
    chk("Severe saving C over B (INR crore)", 2.82, (sv["B: Rule-based"] - sv["C: Optimisation"]).mean() / 1e7, tol=0.005, fmt="{:.3f}")
    chk("Severe reduction C vs A (%)", 75, 100 * (1 - sv["C: Optimisation"].mean() / sv["A: No intervention"].mean()), tol=0.5, fmt="{:.1f}")
    md = reps[reps.severity == "Moderate"].pivot(index="replication", columns="policy", values="financial_impact_inr")
    dm = md["A: No intervention"] - md["C: Optimisation"]
    chk("Moderate mean saving C over A (L)", 16.7, dm.mean() / 1e5, tol=0.05, fmt="{:.2f}")
    chk("Moderate median saving C over A (L)", -11.8, dm.median() / 1e5, tol=0.05, fmt="{:.2f}")
    mi = reps[reps.severity == "Mild"].pivot(index="replication", columns="policy", values="financial_impact_inr")
    dmi = mi["A: No intervention"] - mi["C: Optimisation"]
    chk("Mild extra cost of C over A (L)", 0.82, -dmi.mean() / 1e5, tol=0.005, fmt="{:.3f}")
    chk("Mild: C dearer in every replication (share)", 1.0, float((dmi < 0).mean()))

    ab = pd.read_csv(EXPERIMENTS / "robustness" / "ablation_replications.csv")
    s = ab[ab.severity == "Severe"].pivot(index="replication", columns="policy", values="financial_impact_inr")
    chk("Ablation severe C-obs (INR crore)", 3.64, s["C-obs: MILP, observed graph"].mean() / 1e7, tol=0.005, fmt="{:.3f}")
    chk("Ablation severe C-oracle (INR crore)", 1.21, s["C-oracle: MILP, true graph"].mean() / 1e7, tol=0.005, fmt="{:.3f}")
    chk("Ablation severe B-pred (INR crore)", 4.73, s["B-pred: rule, obs.+predicted"].mean() / 1e7, tol=0.005, fmt="{:.3f}")
    lv = (s["C-obs: MILP, observed graph"] - s["C-pred: MILP, obs.+predicted"]).mean() / 1e7
    chk("Ablation: value of predicted links (INR crore)", 2.48, lv, tol=0.005, fmt="{:.3f}")
    gain = (s["B-obs: rule, observed graph"] - s["C-pred: MILP, obs.+predicted"]).mean() / 1e7
    chk("Ablation: share of C-over-B gain from links (%)", 88, 100 * lv / gain, tol=0.5, fmt="{:.1f}")
    chk("Ablation: MILP alone, observed graph (INR crore)", 0.34, (s["B-obs: rule, observed graph"] - s["C-obs: MILP, observed graph"]).mean() / 1e7, tol=0.005, fmt="{:.3f}")

    # statistics.json and sensitivity: every printed CI / p in Tables IV, VII, VIII
    st = json.loads((EXPERIMENTS / "statistics.json").read_text())
    tex = {n: (ROOT / "documentation" / "final_report" / "tables" / f"{n}.tex").read_text() for n in ["stats", "ablation", "sens"]}
    pf = lambda q: "$<$0.001" if q < 0.001 else f"{q:.3f}"  # noqa: E731
    for r in st["scenarios"]:
        tok = f"{pf(r['p_raw'])} & {pf(r['p_holm'])}"
        rows.append((f"Table IV p/p_Holm {r['severity']} {r['comparison']}", tok, "present" if tok in tex["stats"] else "MISSING", tok in tex["stats"]))
    sens = pd.read_csv(EXPERIMENTS / "robustness" / "sensitivity.csv")
    for _, r in sens[sens.severity == "Severe"].iterrows():
        tok = f"{r.save_C_over_A / 1e7:.2f} [{r.save_lo / 1e7:.2f}, {r.save_hi / 1e7:.2f}]"
        rows.append((f"Table VIII severe saving '{r.setting}'", tok, "present" if tok in tex["sens"] else "MISSING", tok in tex["sens"]))
    for r in st["ablation"]:
        if r["first"].startswith("C-pred"):
            continue  # oracle contrast is reported in text (statistics.json), not printed in Table VII
        tok = f"{r['mean_saving'] / 1e5:+.1f}/{r['median'] / 1e5:+.1f}"
        rows.append((f"Table VII contrast {r['severity']} {r['first'][:6]}->{r['second'][:8]}", tok, "present" if tok in tex["ablation"] else "MISSING", tok in tex["ablation"]))

    # macros vs recomputed (guards against stale numbers.tex)
    mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (ROOT / "documentation" / "final_report" / "tables" / "numbers.tex").read_text()))
    for k, v in [("SevA", sv["A: No intervention"].mean() / 1e7), ("SevC", sv["C: Optimisation"].mean() / 1e7), ("SaveAC", d.mean() / 1e7),
                 ("AblLinkVal", lv), ("LrAUC", roc_auc_score(y, p["LR"][1])), ("XgbAUC", roc_auc_score(y, p["XGB"][1]))]:
        chk(f"Macro \\{k} matches recomputation", float(mac[k]), v, tol=0.0051, fmt="{:.3f}")

    ok = all(r[3] for r in rows)
    lines = ["# Report Number Audit", "", f"Generated by `python -m ml.experiments.audit_report_numbers`. **{sum(r[3] for r in rows)} / {len(rows)} checks passed.**",
             "", "Each value is recomputed from raw data, saved models or per-replication results, not from the paper's macros.", "",
             "| # | Claim | In report | Recomputed | Result |", "|---:|---|---|---|---|"]
    for i, (c, a, b, o) in enumerate(rows, 1):
        lines.append(f"| {i} | {c} | {a} | {b} | {'PASS' if o else '**FAIL**'} |")
    (EXPERIMENTS / "REPORT_NUMBER_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(f"{sum(r[3] for r in rows)}/{len(rows)} report-number checks passed")
    for r in rows:
        if not r[3]:
            print("FAIL:", r)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
