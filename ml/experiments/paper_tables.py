"""Write LaTeX tables for the IEEE paper directly from saved experiment outputs.

Run: python -m ml.experiments.paper_tables  -> documentation/final_report/tables/*.tex
"""
from __future__ import annotations

import json

import pandas as pd

from ml.common.config import EXPERIMENTS, ROOT

OUT = ROOT / "documentation" / "final_report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def j(name):
    return json.loads((EXPERIMENTS / name / "run.json").read_text())


def f3(x):
    return f"{x:.3f}"


def exp1():
    r = j("exp1_supplier_disruption")
    rows = []
    for m, d in r["results"].items():
        for cal, lab in [("none", "--"), ("platt", "Platt"), ("isotonic", "Iso.")]:
            v = d[cal]
            bold = m == r["primary_model"] and cal == r["primary_calibration"]
            line = f"{m.replace('Logistic Regression', 'Logistic reg.')} & {lab} & {f3(v['roc_auc'])} & {f3(v['pr_auc'])} & {v['brier']:.4f} & {v['ece_uniform10']:.4f} \\\\"
            rows.append(("\\rowcolor{gray!15}" if bold else "") + line)
    s = r["splits"]
    tex = (r"""\begin{table}[t]\centering\caption{Exp.~1: 30-day supplier disruption prediction on the untouched test split (""" +
           f"{s['test']['rows']:,} supplier-months, prevalence {100 * s['test']['prevalence']:.1f}\\%" +
           r"""). Calibrators fitted on the calibration split only; shaded = deployed.}\label{tab:exp1}
\scriptsize\setlength{\tabcolsep}{3.5pt}\begin{tabular}{llcccc}\toprule
Model & Calib. & ROC-AUC & PR-AUC & Brier & ECE$_{10}$\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}""")
    (OUT / "exp1.tex").write_text(tex)


def graph():
    a = j("exp2a_link_prediction")
    b = j("exp2b_graph_risk_classification")
    names = list(a["A2_results"])
    rows = []
    for n in names:
        h = a["A1_results"][n]["hard_compatible_negatives"]
        d = a["A2_results"][n]
        sd = lambda v, k: f"{v[k]:.3f}" + (f"$\\pm${v[k + '_sd']:.3f}" if k + "_sd" in v else "")  # noqa: E731
        rows.append(f"{n.replace('Preferential Attachment', 'Pref.~attach.').replace('Common Neighbours', 'Common neigh.')} & {sd(h, 'roc_auc')} & {sd(h, 'ap')} & {f3(d['ap'])} & {d['P@100']:.2f} & {d['R@500']:.3f} \\\\")
    tex = (r"""\begin{table}[t]\centering\caption{Exp.~2A link prediction (synthetic, 5 seeds for learned models). A1: held-out observed edges vs.\ material-compatible negatives. A2: ranking all """ +
           f"{a['A2_candidates']['n']:,} compatible unobserved pairs against {a['A2_candidates']['positives']:,} ground-truth hidden edges (base rate {a['A2_candidates']['base_rate']:.3f})" +
           r""".}\label{tab:exp2a}\scriptsize\setlength{\tabcolsep}{2.6pt}\begin{tabular}{lccccc}\toprule
 & \multicolumn{2}{c}{A1 (held-out)} & \multicolumn{3}{c}{A2 (hidden-edge discovery)}\\\cmidrule(lr){2-3}\cmidrule(lr){4-6}
Method & ROC-AUC & AP & AP & P@100 & R@500\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}""")
    (OUT / "exp2a.tex").write_text(tex)
    rows = []
    p1, p2 = b["part1_synthetic"]["results"], b["part2_supplygraph_real"]["results"]
    for k, v in p1.items():
        rows.append(f"Synthetic & {k.replace('observed+predicted(top500)', 'obs.+pred.').replace('true graph (oracle)', 'oracle').replace('(tabular, no graph)', '(no graph)')} & {v['pr_auc']:.3f}$\\pm${v['pr_auc_sd']:.3f} & {f3(v['roc_auc'])} & {v['brier']:.4f}\\\\")
    rows.append("\\midrule")
    for k, v in p2.items():
        rows.append(f"SupplyGraph & {k} & {v['pr_auc']:.3f}$\\pm${v['pr_auc_sd']:.3f} & {f3(v['roc_auc'])} & {v['brier']:.4f}\\\\")
    tex = (r"""\begin{table}[t]\centering\caption{Exp.~2B: graph vs.\ tabular risk classification (test split, Platt-calibrated; mean$\pm$sd over 5 seeds). Synthetic: 30-day disruption (prevalence """ +
           f"{100 * b['part1_synthetic']['info']['test_prevalence']:.1f}\\%); SupplyGraph (real FMCG): 7-day service shortfall (prevalence {100 * b['part2_supplygraph_real']['info']['prevalence']['test']:.1f}\\%)" +
           r""".}\label{tab:exp2b}\scriptsize\setlength{\tabcolsep}{2.4pt}\begin{tabular}{llccc}\toprule
Data & Model [graph] & PR-AUC & ROC-AUC & Brier\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}""")
    (OUT / "exp2b.tex").write_text(tex)


def exp3():
    r = j("exp3_delivery_delay_dataco")
    rows = []
    for k, d in r["classification"].items():
        v = d["none"]
        rows.append(f"{k.replace('Baseline: shipping-mode late rate', 'Shipping-mode rate')} & {f3(v['roc_auc'])} & {f3(v['pr_auc'])} & {f3(v['f1'])} & {v['brier']:.4f}\\\\")
    reg = r["regression"]
    rr = " ; ".join(f"{k.replace('Baseline: ', '')}: {v['mae']:.3f}/{v['r2']:.3f}" for k, v in reg.items())
    tex = (r"""\begin{table}[t]\centering\caption{Exp.~3 (REAL DataCo): late-delivery prediction, chronological test split (""" +
           f"{r['splits']['test']['rows']:,} order lines, late rate {100 * r['splits']['test']['late_rate']:.1f}\\%" +
           r"""), threshold 0.5.}\label{tab:exp3}\scriptsize\setlength{\tabcolsep}{3.5pt}\begin{tabular}{lcccc}\toprule
Model & ROC-AUC & PR-AUC & F1 & Brier\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}""")
    (OUT / "exp3.tex").write_text(tex)
    (OUT / "exp3_reg.txt").write_text(rr)


def scen():
    s = pd.read_csv(EXPERIMENTS / "scenario_results.csv")
    cr = lambda v: f"{v / 1e7:.2f}"  # noqa: E731
    rows = []
    for _, r in s.iterrows():
        rows.append(f"{r.scenario} & {r.severity[:3]}. & {r.policy[0]} & {r.affected_suppliers:.1f} & {r.affected_materials:.1f} & {r.unmet_demand_units / 1e3:.1f} & "
                    f"{r.inventory_shortage_value_inr / 1e5:.1f} & {r.procurement_spend_inr / 1e5:.1f} & {r.transportation_cost_inr / 1e5:.2f} & "
                    f"{cr(r.financial_impact_inr)} [{cr(r['financial_impact_inr_p2.5'])}, {cr(r['financial_impact_inr_p97.5'])}] & {r.recovery_time_days:.0f} & "
                    f"{100 * r.alt_supplier_utilisation:.0f} & {100 * r.feasibility_full_recovery_share:.0f}\\\\" + ("\\midrule" if r.scenario in ("S3", "S6") else ""))
    tex = (r"""\begin{table*}[t]\centering\caption{Nine-scenario evaluation (means of """ + f"{int(s.replications.iloc[0])}" +
           r""" common-random-number replications; simulated INR on synthetic data). Severity = capacity loss of the five largest Tier-2 cocoa-bean traders for a nominal 60 days (log-normal). Policy A none, B rule-based, C optimisation. Disrupted capacity available: 75\,\% / 50\,\% / 0\,\% for all policies.}\label{tab:scen}
\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lllrrrrrrlrrr}\toprule
Scen. & Sev. & Pol. & Aff.\ sup. & Aff.\ mat. & Unmet (k units) & Inv.\ short.\ (L) & Procure.\ (L) & Transport (L) & Impact, Cr [95\% PI] & Recov.\ d & Alt.\ util.\ \% & Full rec.\ \%\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table*}""")
    (OUT / "scenarios.tex").write_text(tex)


if __name__ == "__main__":
    exp1()
    graph()
    exp3()
    scen()
    print("tables written to", OUT)
