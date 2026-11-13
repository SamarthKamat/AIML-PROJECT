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
        n, p_, i_ = d["none"], d["platt"], d["isotonic"]
        eces = {"none": n["ece_uniform10"], "platt": p_["ece_uniform10"], "isotonic": i_["ece_uniform10"]}
        best = min(eces, key=eces.get)
        e = {k: (f"\\textbf{{{v:.4f}}}" if k == best else f"{v:.4f}") for k, v in eces.items()}
        line = (f"{m.replace('Logistic Regression', 'Logistic reg.')} & {f3(p_['roc_auc'])} & {f3(p_['pr_auc'])} & {n['brier']:.4f} & {p_['brier']:.4f} & "
                f"{e['none']} & {e['platt']} & {e['isotonic']}\\\\")
        rows.append(("\\rowcolor{gray!15}" if m == r["primary_model"] else "") + line)
    s = r["splits"]
    tex = (r"""\begin{table}[t]\centering\caption{E1: 30-day supplier disruption prediction on the untouched test split (""" +
           f"{s['test']['rows']:,} supplier-months, prevalence {100 * s['test']['prevalence']:.1f}\\%" +
           r"""). AUCs after Platt scaling; calibrators fitted on the calibration split; bold = lowest ECE; shaded = pre-specified model.}\label{tab:exp1}
\scriptsize\setlength{\tabcolsep}{2.4pt}\begin{tabular}{@{}lcccccccc@{}}\toprule
 & & & \multicolumn{2}{c}{Brier} & \multicolumn{3}{c}{ECE$_{10}$}\\\cmidrule(lr){4-5}\cmidrule(lr){6-8}
Model & ROC-AUC & PR-AUC & raw & Platt & raw & Platt & Iso.\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}""")
    (OUT / "exp1.tex").write_text(tex)


def graph():
    a = j("exp2a_link_prediction")
    b = j("exp2b_graph_risk_classification")
    names = list(a["A2_results"])
    rows = []
    for n in names:
        if n in ("Adamic-Adar", "GraphSAGE+skip"):
            continue
        h = a["A1_results"][n]["hard_compatible_negatives"]
        d = a["A2_results"][n]
        sd = lambda v, k: f"{v[k]:.3f}" + (f"$\\pm${v[k + '_sd']:.3f}" if k + "_sd" in v else "")  # noqa: E731
        rows.append(f"{n.replace('Preferential Attachment', 'Pref.~attach.').replace('Common Neighbours', 'Common neigh.')} & {sd(h, 'roc_auc')} & {sd(h, 'ap')} & {f3(d['ap'])} & {d['P@100']:.2f} & {d['R@500']:.3f} \\\\")
    tex = (r"""\begin{table}[t]\centering\caption{E2A link prediction (synthetic, 5 seeds for learned models). A1: held-out observed edges vs.\ material-compatible negatives. A2: ranking all """ +
           f"{a['A2_candidates']['n']:,} compatible unobserved pairs against {a['A2_candidates']['positives']:,} ground-truth hidden edges (base rate {a['A2_candidates']['base_rate']:.3f})" +
           r"""). Adamic--Adar equals common neighbours. P@100 = precision of top 100; R@500 = recall of top 500.}\label{tab:exp2a}\scriptsize\setlength{\tabcolsep}{2.6pt}\begin{tabular}{lccccc}\toprule
 & \multicolumn{2}{c}{A1 (held-out)} & \multicolumn{3}{c}{A2 (hidden-edge discovery)}\\\cmidrule(lr){2-3}\cmidrule(lr){4-6}
Method & ROC-AUC & AP & AP & P@100 & R@500\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}""")
    (OUT / "exp2a.tex").write_text(tex)
    p1, p2 = b["part1_synthetic"]["results"], b["part2_supplygraph_real"]["results"]
    fmt = lambda v: f"{v['pr_auc']:.3f}" + (f"$\\pm${v['pr_auc_sd']:.3f}" if v.get("pr_auc_sd", 0) > 0 else "")  # noqa: E731
    best = lambda tag: max(((k, v) for k, v in p1.items() if tag in k), key=lambda kv: kv[1]["pr_auc"])  # noqa: E731
    rows = [f"Logistic regression (tabular) & -- & {fmt(p2['Logistic Regression'])}\\\\",
            f"XGBoost (tabular) & {fmt(p1['XGBoost (tabular)'])} & {fmt(p2['XGBoost (tabular)'])}\\\\",
            f"MLP (features, no graph) & {fmt(p1['MLP (tabular, no graph)'])} & {fmt(p2['MLP (no graph)'])}\\\\"]
    for g in ["GCN", "GraphSAGE", "GAT"]:
        rows.append(f"{g} [observed graph] & {fmt(p1[g + ' [observed]'])} & {fmt(p2[g])}\\\\")
    for tag, lab in [("obs", "Best GNN [obs.+predicted links]"), ("oracle", "Best GNN [oracle true graph]")]:
        k, v = best("observed+predicted" if tag == "obs" else "oracle")
        rows.append(f"{lab} ({k.split()[0]}) & {fmt(v)} & --\\\\")
    tex = (r"""\begin{table}[t]\centering\caption{E2B: PR-AUC of graph vs.\ tabular risk models on the test split (Platt-calibrated; mean$\pm$sd over 5 seeds). Synthetic: 30-day disruption, prevalence """ +
           f"{100 * b['part1_synthetic']['info']['test_prevalence']:.1f}\\%; SupplyGraph (real FMCG): 7-day service shortfall, prevalence {100 * b['part2_supplygraph_real']['info']['prevalence']['test']:.1f}\\%" +
           r""". The best GNN variant is reported to favour graph models.}\label{tab:exp2b}\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lcc}\toprule
Model [graph] & Synthetic & SupplyGraph (real)\\\midrule
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
    tex = (r"""\begin{table}[t]\centering\caption{E3 (real DataCo): late-delivery prediction, chronological test split (""" +
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
           r""" common-random-number replications; simulated INR on synthetic data). Severity = capacity loss of the five largest Tier-2 cocoa-bean traders for a nominal 60 days (log-normal). Policies: A none, B rule-based, C optimisation. L = lakh, Cr = crore; PI = 2.5--97.5th percentile over replications; Full rec.\ = replications with zero unmet demand; Alt.\ util.\ = share of alternates' spare capacity used.}\label{tab:scen}
\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lllrrrrrrlrrr}\toprule
Scen. & Sev. & Pol. & Aff.\ sup. & Aff.\ mat. & Unmet (k units) & Inv.\ short.\ (L) & Procure.\ (L) & Transport (L) & Impact, Cr [95\% PI] & Recov.\ d & Alt.\ util.\ \% & Full rec.\ \%\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table*}""")
    (OUT / "scenarios.tex").write_text(tex)


def macros():
    """Numbers quoted in the prose, written as LaTeX macros so text and results cannot diverge."""
    m = {}
    r1 = j("exp1_supplier_disruption")
    x = r1["results"]["XGBoost"]
    m.update(XgbRaw=f"{x['none']['ece_uniform10']:.3f}", XgbCal=f"{x[r1['primary_calibration']]['ece_uniform10']:.3f}",
             XgbAUC=f"{x[r1['primary_calibration']]['roc_auc']:.3f}", XgbPR=f"{x[r1['primary_calibration']]['pr_auc']:.3f}",
             LrAUC=f"{r1['results']['Logistic Regression']['platt']['roc_auc']:.3f}", LrPR=f"{r1['results']['Logistic Regression']['platt']['pr_auc']:.3f}",
             DtRaw=f"{r1['results']['Decision Tree']['none']['ece_uniform10']:.3f}", DtCal=f"{r1['results']['Decision Tree']['platt']['ece_uniform10']:.3f}",
             TestPrev=f"{100 * r1['splits']['test']['prevalence']:.1f}", TestRows=f"{r1['splits']['test']['rows']:,}")
    ex = r1["exposure_summary"]
    m.update(GapTwo=f"{ex['true_graph_mean_by_tier']['2'] / ex['observed_graph_mean_by_tier']['2']:.1f}",
             GapThree=f"{ex['true_graph_mean_by_tier']['3'] / ex['observed_graph_mean_by_tier']['3']:.1f}")
    b = r1["business_outcome_test_inr"]
    pc = r1["decision_policy"]["policy_confusion"]
    m.update(BizNone=f"{b['no_action_monitor_only'] / 1e5:.1f}", BizCal=f"{b['expected_cost_policy (calibrated)'] / 1e5:.1f}",
             BizUncal=f"{b['expected_cost_policy (uncalibrated)'] / 1e5:.1f}", BizOracle=f"{b['oracle (perfect foresight, upper bound)'] / 1e5:.1f}",
             BizAll=f"{b['contingency_review_for_all'] / 1e5:.1f}", PolAlerts=str(pc["alerts"]), PolTP=str(pc["tp"]))
    a = j("exp2a_link_prediction")
    sel = a["candidate_model"]
    m.update(LinkSel=sel, LinkSelAP=f"{a['A2_results'][sel]['ap']:.3f}", LinkBase=f"{a['A2_candidates']['base_rate']:.3f}",
             LinkSAGEAP=f"{a['A2_results']['GraphSAGE']['ap']:.3f}", LinkSelP=f"{a['A2_results'][sel]['P@100']:.2f}",
             LinkCand=f"{a['A2_candidates']['n']:,}", LinkHid=f"{a['A2_candidates']['positives']:,}")
    r3 = j("exp3_delivery_delay_dataco")
    c = r3["classification"]
    m.update(DcXgbAUC=f"{c['XGBoost']['none']['roc_auc']:.3f}", DcXgbPR=f"{c['XGBoost']['none']['pr_auc']:.3f}",
             DcBaseAUC=f"{c['Baseline: shipping-mode late rate']['none']['roc_auc']:.3f}",
             DcRegR=f"{r3['regression']['XGBoost']['r2']:.2f}", DcBaseR=f"{r3['regression']['Baseline: shipping-mode mean']['r2']:.2f}",
             DcTest=f"{r3['splits']['test']['rows']:,}")
    s = pd.read_csv(EXPERIMENTS / "scenario_results.csv").set_index("scenario")
    cp = pd.read_csv(EXPERIMENTS / "scenario_comparison.csv")
    g = lambda sev, comp: cp[(cp.severity == sev) & (cp.comparison == comp)].iloc[0]  # noqa: E731
    ac, bc, ab = (g("Severe", "A: No intervention minus C: Optimisation"), g("Severe", "B: Rule-based minus C: Optimisation"),
                  g("Severe", "A: No intervention minus B: Rule-based"))
    m.update(SevA=f"{s.loc['S7', 'financial_impact_inr'] / 1e7:.2f}", SevB=f"{s.loc['S8', 'financial_impact_inr'] / 1e7:.2f}",
             SevC=f"{s.loc['S9', 'financial_impact_inr'] / 1e7:.2f}", SaveAC=f"{ac.mean_saving_inr / 1e7:.2f}",
             SaveACl=f"{ac.ci95_low / 1e7:.2f}", SaveACh=f"{ac.ci95_high / 1e7:.2f}", SaveBC=f"{bc.mean_saving_inr / 1e7:.2f}",
             SaveAB=f"{ab.mean_saving_inr / 1e7:.2f}", SevUnmetA=f"{s.loc['S7', 'unmet_demand_units'] / 1e3:.0f}",
             SevUnmetB=f"{s.loc['S8', 'unmet_demand_units'] / 1e3:.0f}", SevUnmetC=f"{s.loc['S9', 'unmet_demand_units'] / 1e3:.0f}",
             SevSvcA=f"{100 * s.loc['S7', 'min_service_level']:.0f}", SevSpendC=f"{s.loc['S9', 'procurement_spend_inr'] / 1e7:.2f}",
             ModExtraC=f"{(s.loc['S6', 'financial_impact_inr'] - s.loc['S4', 'financial_impact_inr']) / 1e5:.1f}",
             MildExtraC=f"{(s.loc['S3', 'financial_impact_inr'] - s.loc['S1', 'financial_impact_inr']) / 1e5:.2f}",
             ModSaveAC=f"{g('Moderate', 'A: No intervention minus C: Optimisation').mean_saving_inr / 1e5:.1f}",
             ModA=f"{s.loc['S4', 'financial_impact_inr'] / 1e5:.1f}", ModB=f"{s.loc['S5', 'financial_impact_inr'] / 1e5:.1f}",
             ModC=f"{s.loc['S6', 'financial_impact_inr'] / 1e5:.1f}", SevRed=f"{100 * (1 - s.loc['S9', 'financial_impact_inr'] / s.loc['S7', 'financial_impact_inr']):.0f}",
             FullRecC=f"{100 * s.loc['S9', 'feasibility_full_recovery_share']:.0f}", Reps=str(int(s.replications.iloc[0])),
             Censored=f"{100 * s.recovery_censored_share.max():.0f}")
    reps = pd.read_csv(EXPERIMENTS / "exp4_5_scenarios" / "scenario_replications.csv")
    win = reps.loc[reps.groupby(["severity", "replication"]).financial_impact_inr.idxmin()].groupby("severity").policy.value_counts(normalize=True)
    m.update(ModAWins=f"{100 * win.get(('Moderate', 'A: No intervention'), 0):.0f}", SevCWins=f"{100 * win.get(('Severe', 'C: Optimisation'), 0):.0f}")
    run = j("exp4_5_scenarios")
    m.update(HubObs=str(run["hub_edges"]["observed"]), HubHid=str(run["hub_edges"]["hidden"]), McSE=f"{100 * run['mc_convergence_final_relative_se']:.1f}")
    (OUT / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n")


def stats_table():
    st = json.loads((EXPERIMENTS / "statistics.json").read_text())
    e1, e3 = st["exp1"]["paired"], st["exp3"]["delta_auc_xgb_minus_mode"]
    pf = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"  # noqa: E731
    L = lambda v: f"{v / 1e5:+.2f}" if abs(v) < 1e6 else f"{v / 1e5:+.1f}"  # noqa: E731
    Lc = lambda c: f"[{c[0] / 1e5:.3f}, {c[1] / 1e5:.3f}]" if max(abs(c[0]), abs(c[1])) < 1e5 else f"[{c[0] / 1e5:.2f}, {c[1] / 1e5:.2f}]" if max(abs(c[0]), abs(c[1])) < 1e6 else f"[{c[0] / 1e5:.1f}, {c[1] / 1e5:.1f}]"  # noqa: E731
    rows = [r"\multicolumn{8}{@{}l}{\textit{Models: bootstrap, $B$=1,000, two-sided}}\\",
            f"E1 & LR$-$XGB, ROC-AUC & {e1['ROC-AUC LR - XGB']['mean']:+.3f} & [{e1['ROC-AUC LR - XGB']['ci'][0]:.3f}, {e1['ROC-AUC LR - XGB']['ci'][1]:.3f}] & -- & -- & {pf(e1['ROC-AUC LR - XGB']['p'])} & --\\\\",
            f"E1 & LR$-$XGB, PR-AUC & {e1['PR-AUC LR - XGB']['mean']:+.3f} & [{e1['PR-AUC LR - XGB']['ci'][0]:.3f}, {e1['PR-AUC LR - XGB']['ci'][1]:.3f}] & -- & -- & {pf(e1['PR-AUC LR - XGB']['p'])} & --\\\\",
            f"E1 & ECE, raw$-$Platt & {e1['ECE raw - Platt (XGB)']['mean']:+.4f} & [{e1['ECE raw - Platt (XGB)']['ci'][0]:.3f}, {e1['ECE raw - Platt (XGB)']['ci'][1]:.3f}] & -- & -- & {pf(e1['ECE raw - Platt (XGB)']['p'])} & --\\\\",
            f"E3 & XGB$-$mode, AUC & {e3['mean']:+.3f} & [{e3['ci'][0]:.3f}, {e3['ci'][1]:.3f}] & -- & -- & {pf(e3['p'])} & --\\\\",
            r"\midrule\multicolumn{8}{@{}l}{\textit{Policies: saving in INR lakh; two-sided Wilcoxon}}\\"]
    for r in st["scenarios"]:
        rows.append(f"{r['severity'][:3]}. & {r['comparison'].replace(' vs ', ' vs.\\ ')} & {L(r['mean_saving'])} & {Lc(r['ci'])} & {L(r['median'])} & {100 * r['share_second_cheaper']:.0f} & {pf(r['p_raw'])} & {pf(r['p_holm'])}\\\\")
    tex = (r"""\begin{table}[t]\centering\caption{Statistical tests. E1: bootstrap over 500 suppliers; E3: over 16,840 orders. Policies: saving = impact(first)$-$impact(second) over 500 CRN pairs, Holm over 9; CI of the mean; Med.\ = median; Win = \% of replications where the second policy is cheaper.}\label{tab:stats}
\scriptsize\setlength{\tabcolsep}{1.2pt}\begin{tabular}{@{}llrcrrrr@{}}\toprule
 & Comparison & Mean & 95\% CI & Med. & Win\% & $p$ & $p_{\mathrm{Holm}}$\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}""")
    (OUT / "stats.tex").write_text(tex)


def ablation_table():
    f = EXPERIMENTS / "robustness" / "ablation_summary.csv"
    if not f.exists():
        return
    a = pd.read_csv(f)
    st = {(r["severity"], r["first"], r["second"]): r for r in json.loads((EXPERIMENTS / "statistics.json").read_text())["ablation"]}
    arms = list(dict.fromkeys(a.policy))
    lab = {arms[0]: ("A", "--", "--"), arms[1]: ("B-obs", "rule", "observed"), arms[2]: ("B-pred", "rule", "obs.+pred."),
           arms[3]: ("C-obs", "MILP", "observed"), arms[4]: ("C-pred = C", "MILP", "obs.+pred."), arms[5]: ("C-oracle", "MILP", "true")}
    rows = []
    for arm in arms:
        cells = []
        for sv in ["Mild", "Moderate", "Severe"]:
            r = a[(a.severity == sv) & (a.policy == arm)].iloc[0]
            cells.append(f"{r.impact_inr / 1e5:.1f}$\\pm${1.96 * r.impact_se / 1e5:.1f}")
        n, lg, g = lab[arm]
        rows.append(f"{n} & {lg} & {g} & " + " & ".join(cells) + r"\\")
    pf = lambda p: "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0")  # noqa: E731
    con = []
    for x, y, t in [(arms[3], arms[4], "Links, MILP: C-obs$-$C"), (arms[1], arms[2], "Links, rule: B-obs$-$B-pred"),
                    (arms[1], arms[3], "MILP, observed: B-obs$-$C-obs")]:
        v = [st[(sv, x, y)] for sv in ["Mild", "Moderate", "Severe"]]
        con.append(f"\\multicolumn{{3}}{{@{{}}l}}{{{t}}} & " + " & ".join(f"{q['mean_saving'] / 1e5:+.1f}/{q['median'] / 1e5:+.1f}" for q in v) + r"\\")
        con.append(r"\multicolumn{3}{@{}r}{\textit{$p_{\mathrm{Holm}}$}} & " + " & ".join(pf(q["p_holm"]) for q in v) + r"\\")
    tex = (r"""\begin{table}[t]\centering\caption{Link-information ablation (RQ4b). Mean simulated impact, INR lakh $\pm$ 95\% CI (500 CRN replications); only logic or known graph changes. Lower block: paired saving, mean/median, Holm-adjusted Wilcoxon $p$ over 12 comparisons.}\label{tab:abl}
\scriptsize\setlength{\tabcolsep}{2.2pt}\begin{tabular}{@{}lllrrr@{}}\toprule
Arm & Logic & Graph known & Mild & Moderate & Severe\\\midrule
""" + "\n".join(rows) + r"\midrule" + "\n" + "\n".join(con) + r"""
\bottomrule\end{tabular}\end{table}""")
    (OUT / "ablation.tex").write_text(tex)


def sens_table():
    f = EXPERIMENTS / "robustness" / "sensitivity.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    rows = []
    for name in dict.fromkeys(d.setting):
        sv = d[(d.setting == name) & (d.severity == "Severe")].iloc[0]
        mi = d[(d.setting == name) & (d.severity == "Mild")].iloc[0]
        nm = {"base": "Base (as Table~\\ref{tab:scen})", "overhead 0": "Mgmt.\\ overhead 0/day", "overhead 50k": "Mgmt.\\ overhead 50k/day",
              "penalty 5%": "Shortage penalty 5\\%", "penalty 30%": "Shortage penalty 30\\%", "interruption 25k": "Interruption 25k/plant-day",
              "interruption 150k": "Interruption 150k/plant-day", "duration 30 d": "Nominal duration 30 d", "duration 90 d": "Nominal duration 90 d",
              "spare capacity x0.5": "Alt.\\ spare capacity $\\times$0.5", "spare capacity x1.5": "Alt.\\ spare capacity $\\times$1.5",
              "alternate risk x0.5": "Alt.\\ failure risk $\\times$0.5$^\\dagger$", "alternate risk x2": "Alt.\\ failure risk $\\times$2$^\\dagger$",
              "visibility -50% deep links": "Known deep links $-$50\\%", "visibility +50% hidden links": "Known hidden links $+$50\\%"}.get(name, name)
        rows.append(f"{nm} & {sv.save_C_over_A / 1e7:.2f} [{sv.save_lo / 1e7:.2f}, {sv.save_hi / 1e7:.2f}] & {sv.best} & {mi.save_C_over_A / 1e5:+.2f} & {mi.best}\\\\")
    tex = (r"""\begin{table}[t]\centering\caption{One-at-a-time sensitivity (200 CRN replications per cell). Saving = impact(A)$-$impact(C); ties go to A. Known-link rows change what B and C know, not the true network. $^\dagger$Changes the random stream; compare within row.}\label{tab:sens}
\scriptsize\setlength{\tabcolsep}{2pt}\begin{tabular}{@{}lccrc@{}}\toprule
 & \multicolumn{2}{c}{Severe} & \multicolumn{2}{c}{Mild}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Setting & C$-$A saving, Cr [95\% CI] & Best & Saving, L & Best\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}""")
    (OUT / "sens.tex").write_text(tex)


def stats_macros():
    st = json.loads((EXPERIMENTS / "statistics.json").read_text())
    e1 = st["exp1"]
    m = dict(CiLrRoc=f"{e1['LR']['roc'][0]:.3f}--{e1['LR']['roc'][1]:.3f}", CiXgbRoc=f"{e1['XGB']['roc'][0]:.3f}--{e1['XGB']['roc'][1]:.3f}",
             CiLrPr=f"{e1['LR']['pr'][0]:.3f}--{e1['LR']['pr'][1]:.3f}", CiXgbPr=f"{e1['XGB']['pr'][0]:.3f}--{e1['XGB']['pr'][1]:.3f}",
             DRoc=f"{e1['paired']['ROC-AUC LR - XGB']['mean']:.3f}", DRocP=f"{e1['paired']['ROC-AUC LR - XGB']['p']:.3f}",
             DEce=f"{e1['paired']['ECE raw - Platt (XGB)']['mean']:.4f}", DEceL=f"{e1['paired']['ECE raw - Platt (XGB)']['ci'][0]:.3f}",
             DEceH=f"{e1['paired']['ECE raw - Platt (XGB)']['ci'][1]:.3f}", DEceP=f"{e1['paired']['ECE raw - Platt (XGB)']['p']:.2f}",
             DcCi=f"{st['exp3']['xgb_auc_ci'][0]:.3f}--{st['exp3']['xgb_auc_ci'][1]:.3f}", DcDelta=f"{st['exp3']['delta_auc_xgb_minus_mode']['mean']:.3f}")
    sens = EXPERIMENTS / "sensitivity_lr_risk" / "scenario_results_lr_risk.csv"
    if sens.exists():
        sl = pd.read_csv(sens).set_index("scenario")
        cp = pd.read_csv(EXPERIMENTS / "sensitivity_lr_risk" / "scenario_comparison_lr_risk.csv")
        sv = cp[(cp.severity == "Severe") & (cp.comparison == "A: No intervention minus C: Optimisation")].iloc[0]
        m.update(SensSevC=f"{sl.loc['S9', 'financial_impact_inr'] / 1e7:.2f}", SensSaveAC=f"{sv.mean_saving_inr / 1e7:.2f}",
                 SensModC=f"{sl.loc['S6', 'financial_impact_inr'] / 1e5:.1f}")
    m["DcCi"] = f"{st['exp3']['xgb_auc_ci'][0]:.3f}--{st['exp3']['xgb_auc_ci'][1]:.3f}"
    m["DcOrders"] = f"{st['exp3']['n_orders']:,}"
    sc = {(r["severity"], r["comparison"]): r for r in st["scenarios"]}
    m.update(ModACmed=f"{-sc[('Moderate', 'A vs C')]['median'] / 1e5:.1f}", ModACwin=f"{100 * sc[('Moderate', 'A vs C')]['share_second_cheaper']:.0f}",
             SevACmed=f"{sc[('Severe', 'A vs C')]['median'] / 1e7:.2f}", SevACwin=f"{100 * sc[('Severe', 'A vs C')]['share_second_cheaper']:.0f}",
             MildACsd=f"{1e-3 * 0:.0f}")
    ab = EXPERIMENTS / "robustness" / "ablation_summary.csv"
    if ab.exists():
        a = pd.read_csv(ab).set_index(["severity", "policy"])["impact_inr"]
        g = lambda sv, k: a[[i for i in a.index if i[0] == sv and i[1].startswith(k)][0]]  # noqa: E731
        abl = {(r["severity"], r["first"][:6], r["second"][:6]): r for r in st["ablation"]}
        lv = abl[("Severe", "C-obs:", "C-pred")]
        lm = abl[("Moderate", "C-obs:", "C-pred")]
        mv = abl[("Severe", "B-obs:", "C-obs:")]
        m.update(AblCobs=f"{g('Severe', 'C-obs') / 1e7:.2f}", AblCpred=f"{g('Severe', 'C-pred') / 1e7:.2f}", AblOracle=f"{g('Severe', 'C-oracle') / 1e7:.2f}",
                 AblBpred=f"{g('Severe', 'B-pred') / 1e7:.2f}", AblBobs=f"{g('Severe', 'B-obs') / 1e7:.2f}",
                 AblLinkVal=f"{lv['mean_saving'] / 1e7:.2f}", AblLinkValL=f"{lv['ci'][0] / 1e7:.2f}", AblLinkValH=f"{lv['ci'][1] / 1e7:.2f}",
                 AblLinkShare=f"{100 * lv['mean_saving'] / (g('Severe', 'B-obs') - g('Severe', 'C-pred')):.0f}",
                 AblMilpVal=f"{mv['mean_saving'] / 1e7:.2f}", AblModMean=f"{lm['mean_saving'] / 1e5:.1f}", AblModMed=f"{-lm['median'] / 1e5:.1f}",
                 AblModOracle=f"{g('Moderate', 'C-oracle') / 1e5:.1f}", AblModCpred=f"{g('Moderate', 'C-pred') / 1e5:.1f}", AblModCobs=f"{g('Moderate', 'C-obs') / 1e5:.1f}")
    sp = EXPERIMENTS / "robustness" / "sensitivity.csv"
    if sp.exists():
        d = pd.read_csv(sp)
        sv = d[d.severity == "Severe"]
        nd = sv[~sv.setting.str.startswith("duration")]
        m.update(SensLo=f"{sv.save_C_over_A.min() / 1e7:.2f}", SensHi=f"{sv.save_C_over_A.max() / 1e7:.2f}",
                 SensNdLo=f"{nd.save_C_over_A.min() / 1e7:.2f}", SensNdHi=f"{nd.save_C_over_A.max() / 1e7:.2f}",
                 SensN=str(d.setting.nunique()), SensSevCbest=str(int((sv.best == "C").sum())),
                 SensMildAbest=str(int((d[d.severity == "Mild"].best == "A").sum())))
    with open(OUT / "numbers.tex", "a") as f:
        f.write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))


def decision():
    """Worked decision example from experiments/decision_example.json (output of ml.decision.engine.evaluate)."""
    d = json.loads((EXPERIMENTS / "decision_example.json").read_text())
    names = {"routine_monitoring": "Routine monitoring", "enhanced_monitoring": "Enhanced monitoring",
             "contingency_review": "Contingency review", "escalate_recovery": "Escalate \\& recover"}
    cost = d["expected_cost_by_action_inr"]
    rows = [f"{names[k]} & {v / 1e3:.1f}{' $\\leftarrow$' if k == d['recommended_action'] else ''}\\\\" for k, v in cost.items()]
    be = list(d["break_even_probabilities"].values())
    drivers = ", ".join(f"{c['feature'].replace('_', ' ').replace('mean delay days 90d', 'mean delay (90 d)').replace('max dependency pct', 'dependency share')} ({c['shap_log_odds']:+.2f})" for c in d["top_feature_contributions"][:3])
    rp = d.get("recovery_plan") or {}
    tex = (r"""\begin{table}[t]\centering\caption{Worked decision for supplier """ + d["supplier_id"] + r""" (Tier-""" + str(d["tier"]) +
           r""", synthetic) at $t$=""" + d["prediction_timestamp"][:10] + r""", produced end-to-end by the decision engine.}\label{tab:decision}
\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}p{0.47\columnwidth}p{0.47\columnwidth}@{}}\toprule
\multicolumn{2}{@{}p{0.97\columnwidth}@{}}{Raw / calibrated 30-day risk: """ + f"{100 * d['disruption_probability']:.1f}\\,\\% / \\textbf{{{100 * d['calibrated_probability']:.1f}\\,\\%}}" +
           r""" \quad Exposure $E_i$: INR """ + f"{d['network_exposure_inr'] / 1e5:.2f}" + r""" lakh}\\
\multicolumn{2}{@{}p{0.97\columnwidth}@{}}{Break-even $q^*$: """ + " / ".join(f"{100 * b:.1f}\\,\\%" for b in be) + r""" \quad Cover: """ +
           ", ".join(f"{v:.0f} d" for v in d["inventory_coverage_days"].values()) + r"""}\\
\multicolumn{2}{@{}p{0.97\columnwidth}@{}}{SHAP drivers: """ + drivers + r"""}\\\midrule
Action & Expected cost (INR '000)\\\midrule
""" + "\n".join(rows) + r"""
\midrule\multicolumn{2}{@{}p{0.97\columnwidth}@{}}{Recovery MILP: """ + f"{rp.get('solver', {}).get('status', 'n/a')}, cost INR {rp.get('estimated_recovery_cost_inr', 0) / 1e3:.1f}k" + r"""}\\
\bottomrule\end{tabular}\end{table}""")
    (OUT / "decision.tex").write_text(tex)


if __name__ == "__main__":
    macros()
    stats_macros()
    stats_table()
    ablation_table()
    sens_table()
    decision()
    exp1()
    graph()
    exp3()
    scen()
    print("tables written to", OUT)
