"""Build presentation/Supply_Chain_Black_Box.pptx from saved results, figures and screenshots.

Run: python presentation/build_pptx.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
E = ROOT / "experiments"
FIG = ROOT / "documentation" / "final_report" / "figures"
DIA = ROOT / "documentation" / "diagrams"
SS = ROOT / "documentation" / "testing" / "screenshots"
NAVY, PANEL, BLUE, CYAN, WHITE, MUTED, AMBER, GREEN, RED = (RGBColor(0x05, 0x0D, 0x22), RGBColor(0x0F, 0x1F, 0x45), RGBColor(0x4F, 0x8D, 0xFF),
                                                           RGBColor(0x22, 0xD3, 0xEE), RGBColor(0xE7, 0xEE, 0xFB), RGBColor(0x8E, 0xA3, 0xC7),
                                                           RGBColor(0xFB, 0xBF, 0x24), RGBColor(0x34, 0xD3, 0x99), RGBColor(0xF8, 0x71, 0x71))
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, H = prs.slide_width, prs.slide_height


def j(n):
    return json.loads((E / n / "run.json").read_text())


def bg(s):
    f = s.background.fill
    f.solid()
    f.fore_color.rgb = NAVY


def text(s, x, y, w, h, t, size=18, color=WHITE, bold=False, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    lines = t if isinstance(t, list) else [t]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.alignment = align
        p.space_after = Pt(6)
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
            r.font.name = "Calibri"
    return tb


def panel(s, x, y, w, h):
    sh = s.shapes.add_shape(1, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = PANEL
    sh.line.color.rgb = RGBColor(0x2A, 0x45, 0x80)
    return sh


def slide(title, sub=None, synthetic=True):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg(s)
    bar = s.shapes.add_shape(1, 0, 0, Inches(0.12), H)
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    text(s, Inches(0.5), Inches(0.3), Inches(10.5), Inches(0.8), title, 30, WHITE, True)
    if sub:
        text(s, Inches(0.5), Inches(0.95), Inches(11.5), Inches(0.6), sub, 15, MUTED)
    if synthetic:
        text(s, Inches(10.9), Inches(0.35), Inches(2.3), Inches(0.4), "synthetic + real data, labelled", 11, AMBER, align=PP_ALIGN.RIGHT)
    return s


def bullets(s, items, x=Inches(0.6), y=Inches(1.7), w=Inches(12), size=19):
    return text(s, x, y, w, Inches(5.5), ["•  " + i for i in items], size)


def img(s, p, x, y, w=None, h=None):
    if Path(p).exists():
        return s.shapes.add_picture(str(p), x, y, width=w, height=h)


def table(s, df, x, y, w, size=12, colw=None):
    rows, cols = df.shape[0] + 1, df.shape[1]
    t = s.shapes.add_table(rows, cols, x, y, w, Inches(0.3) * rows).table
    for c, name in enumerate(df.columns):
        t.cell(0, c).text = str(name)
    for r in range(df.shape[0]):
        for c in range(cols):
            t.cell(r + 1, c).text = str(df.iat[r, c])
    for r in range(rows):
        for c in range(cols):
            cell = t.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL if r else RGBColor(0x1A, 0x33, 0x6B)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size)
                    run.font.color.rgb = WHITE
    if colw:
        for i, cw in enumerate(colw):
            t.columns[i].width = cw
    return t


r1, r2a, r2b, r3, rs = j("exp1_supplier_disruption"), j("exp2a_link_prediction"), j("exp2b_graph_risk_classification"), j("exp3_delivery_delay_dataco"), j("exp4_5_scenarios")
scen = pd.read_csv(E / "scenario_results.csv")
comp = pd.read_csv(E / "scenario_comparison.csv")

# 1 title
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s)
img(s, SS / "00_login.png", Inches(6.6), 0, h=H)
text(s, Inches(0.6), Inches(1.0), Inches(6), Inches(2.0), ["Supply Chain", "Black Box"], 44, WHITE, True)
text(s, Inches(0.6), Inches(2.9), Inches(6), Inches(2), "AI-powered hidden supplier dependency discovery, cascading risk prediction and intelligent supply chain recovery", 20, CYAN)
text(s, Inches(0.6), Inches(4.4), Inches(6), Inches(2), ["Advanced Machine Learning for Business Transformation", "PGDM Big Data Analytics · Goa Institute of Management",
                                                          "Shraddha Shete (B2026102) · Shirsodeep Biswas (B2026100) · Jayant Kumawat (B2026078)", "FMCG: chocolate & malt-based beverages"], 15, MUTED)

# 2 problem
s = slide("The business problem", "A manufacturer sees its Tier-1 suppliers — not the Tier-2/3 suppliers they quietly share")
bullets(s, ["A malt-beverage jar needs malt extract, milk powder, cocoa, sugar, vitamin premix and PET jars; each Tier-1 input depends on Tier-2 commodity traders.",
            "One hidden cocoa-bean trader can starve several 'independent' cocoa processors at once — a ripple effect through invisible links.",
            "Disruptions destroy value: announced supply chain disruptions are followed by lower long-run returns and higher equity risk (Hendricks & Singhal, 2005).",
            "Decision supported: each month, which mitigation action per supplier? If a shock hits, which alternates get which quantity, by which mode?"], size=16)
img(s, DIA / "04_supplier_dependency_network.png", Inches(2.4), Inches(4.3), h=Inches(3.1))

# 3 gap & objectives
s = slide("Research gap, objectives and contributions")
bullets(s, ["Gap: discovery (link prediction), risk prediction and recovery optimisation are studied separately; few works report calibrated, leakage-controlled, policy-level business outcomes.",
            "Objectives: discover candidate hidden links · predict calibrated 30-day disruption risk · simulate cascades · quantify inventory/production/₹ exposure · optimise recovery · explain.",
            "Contribution 1: an executable, traceable pipeline from prediction → exposure → cost-optimal action → MILP plan.",
            "Contribution 2: leakage-audited evaluation (prediction timestamp, embargoed chronological splits, calibration-only split, test edges removed).",
            "Contribution 3: quantification of the 'black-box gap' between observed and true network exposure.",
            "Contribution 4: predefined nine-scenario policy evaluation with 500 common-random-number replications and explicit economics.",
            "Not claimed: novelty of GNNs, calibration or MILP individually."], size=18)

# 4 literature
s = slide("Literature review")
lit = pd.DataFrame([["Baryannis et al. 2019", "AI for SCRM review", "few studies close the loop to response"],
                    ["Brintrup et al. 2018", "hidden-link prediction (automotive)", "discovery only"],
                    ["Kosasih & Brintrup 2022", "GNN link prediction", "no risk/recovery integration"],
                    ["Wasi et al. 2024 (SupplyGraph)", "real FMCG GNN benchmark", "product graph, no suppliers"],
                    ["Platt 1999; Zadrozny & Elkan 2002; Niculescu-Mizil & Caruana 2005", "probability calibration", "used for decision thresholds"],
                    ["Kipf 2017; Hamilton 2017; Veličković 2018; Grover 2016", "GCN, GraphSAGE, GAT, node2vec", "comparators"],
                    ["Snyder et al. 2016; Ivanov et al. 2019", "disruption OR models, ripple effect", "recovery & propagation"]],
                   columns=["Work", "Contribution", "Relevance / gap"])
table(s, lit, Inches(0.6), Inches(1.6), Inches(12.1), 14)

# 5 data
s = slide("Data and provenance", "Real and synthetic data are kept separate and labelled everywhere")
d = pd.DataFrame([["DataCo Smart Supply Chain", "REAL (CC BY 4.0, Mendeley DOI; GitHub mirror, SHA-256)", "180,519 order lines → 172,765 after removing cancelled", "Exp 3 shipment delay"],
                  ["SupplyGraph", "REAL FMCG (LGPL-2.1, pinned commit)", "40 products, 684 edges, 221 days (1 duplicate node removed)", "Exp 2B real graph"],
                  ["Cocoa & Malt Foods (fictional)", "SYNTHETIC (seed 42)", "500 suppliers · 1,500 links (461 observed) · 1,048 disruptions · 9,579 shipments", "Exps 1, 2, 4, 5, app"]],
                 columns=["Dataset", "Provenance", "Size", "Use"])
table(s, d, Inches(0.6), Inches(1.7), Inches(12.1), 14)
bullets(s, ["Hidden edges are ground truth for the simulator and evaluation only — never model features.",
            "196 validation checks pass (keys, foreign keys, dates, ranges, logical quantity rules); outliers flagged, not deleted.",
            "AnonNet, SynDelay, UN Comtrade, NOAA investigated: not verifiable / not reachable — not used."], y=Inches(4.3), size=17)

# 6 architecture
s = slide("System architecture")
img(s, DIA / "01_system_architecture.png", Inches(0.8), Inches(1.4), w=Inches(11.7))

# 7 leakage
s = slide("Methodology: prediction timestamp and leakage control")
bullets(s, ["Prediction timestamp t = 00:00 on the 1st of each month; label = qualifying disruption (≥25 % capacity loss, ≥3 days) starting in (t, t+30 d].",
            "Features use disruptions with reported_date < t, deliveries completed before t, the monthly risk forecast, and OBSERVED edges only (25 features).",
            "Chronological split: train 2024-04…2025-02 (5,500) · embargo 2025-03 · calibration 2025-04…06 (1,500) · test 2025-07…12 (3,000).",
            "Calibrators (Platt, isotonic) fitted on the calibration split only; method chosen by 5-fold CV Brier inside that split.",
            "Unit tests tamper with future records and hidden edges and assert features do not change."], size=19)
img(s, DIA / "05_supplier_risk_workflow.png", Inches(0.8), Inches(5.3), w=Inches(11.7))

# 8 exp1
s = slide("Experiment 1 — supplier disruption prediction", f"Test: {r1['splits']['test']['rows']:,} supplier-months, prevalence {100 * r1['splits']['test']['prevalence']:.1f} %")
rows = [[m, c, f"{v['roc_auc']:.3f}", f"{v['pr_auc']:.3f}", f"{v['brier']:.4f}", f"{v['ece_uniform10']:.4f}"] for m, d in r1["results"].items() for c, v in d.items() if c in ("none", "platt")]
table(s, pd.DataFrame(rows, columns=["Model", "Calibration", "ROC-AUC", "PR-AUC", "Brier", "ECE"]), Inches(0.5), Inches(1.6), Inches(6.6), 12)
img(s, FIG / "exp1_reliability.png", Inches(7.3), Inches(1.5), w=Inches(5.7))
text(s, Inches(7.3), Inches(6.0), Inches(5.7), Inches(1.2), "Modest discrimination for rare, contagion-driven events; Platt calibration halves XGBoost's ECE.", 15, CYAN)

# 9 decision rule
s = slide("From probability to action: cost-based decision rule")
bullets(s, ["a* = argmin_a  C_a + q·(1 − ρ_a)·E_i   (Bayes-optimal when q is calibrated)",
            "Actions: routine monitoring (₹0, 0 %) · enhanced monitoring (₹5k, 15 %) · contingency review (₹20k, 45 %) · escalate & recover (₹60k, 70 %)",
            "E_i = network exposure: Σ_m reach × contribution margin × E[severity × uncovered days]",
            f"Test months: {r1['decision_policy']['policy_confusion']['alerts']} contingency alerts; realised cost ₹{r1['business_outcome_test_inr']['expected_cost_policy (calibrated)'] / 1e5:.1f} L vs ₹{r1['business_outcome_test_inr']['no_action_monitor_only'] / 1e5:.1f} L monitor-only, ₹{r1['business_outcome_test_inr']['expected_cost_policy (uncalibrated)'] / 1e5:.1f} L uncalibrated, ₹{r1['business_outcome_test_inr']['contingency_review_for_all'] / 1e5:.1f} L review-all.",
            "Honest finding: 3–4 weeks of inventory absorb most single-supplier events → value concentrates in network-level shocks."], size=19)

# 10 black-box gap & links
ex = r1["exposure_summary"]
s = slide("Experiment 2A — the black-box gap and hidden-link discovery")
text(s, Inches(0.6), Inches(1.5), Inches(5.5), Inches(2.2), [f"Observed graph understates mean exposure of Tier-2 suppliers {ex['true_graph_mean_by_tier']['2'] / ex['observed_graph_mean_by_tier']['2']:.1f}× and Tier-3 suppliers {ex['true_graph_mean_by_tier']['3'] / ex['observed_graph_mean_by_tier']['3']:.1f}×.",
                                                             f"Discovery: {r2a['A2_candidates']['n']:,} compatible unobserved pairs, {r2a['A2_candidates']['positives']:,} hidden edges (base rate {r2a['A2_candidates']['base_rate']:.3f}).",
                                                             f"Validation-selected model: {r2a['candidate_model']} (AP {r2a['A2_results'][r2a['candidate_model']]['ap']:.3f}); GraphSAGE AP {r2a['A2_results']['GraphSAGE']['ap']:.3f}; CN/Adamic-Adar uninformative on layered networks.",
                                                             "Outputs are candidate links for supplier verification — never confirmed relationships."], 17)
img(s, FIG / "exp2a_links.png", Inches(6.2), Inches(1.6), w=Inches(6.9))

# 11 GNN vs tabular
s = slide("Experiment 2B — graph vs tabular risk models", "PR-AUC on the test split, mean ± sd over 5 seeds (Platt-calibrated)")
img(s, FIG / "exp2b_graph_risk.png", Inches(0.6), Inches(1.6), w=Inches(12))
text(s, Inches(0.6), Inches(6.2), Inches(12), Inches(1), "Negative result, reported transparently: GNNs did not beat tabular models on synthetic data (even with the oracle graph) or on real SupplyGraph, where logistic regression is best.", 16, CYAN)

# 12 DataCo
s = slide("Experiment 3 — real shipment delays (DataCo)", f"Chronological test: {r3['splits']['test']['rows']:,} order lines, late rate {100 * r3['splits']['test']['late_rate']:.1f} %", synthetic=False)
rows = [[k, f"{v['none']['roc_auc']:.3f}", f"{v['none']['pr_auc']:.3f}", f"{v['none']['f1']:.3f}", f"{v['none']['brier']:.4f}"] for k, v in r3["classification"].items()]
table(s, pd.DataFrame(rows, columns=["Model", "ROC-AUC", "PR-AUC", "F1", "Brier"]), Inches(0.6), Inches(1.6), Inches(7), 14)
rows = [[k, f"{v['mae']:.3f}", f"{v['rmse']:.3f}", f"{v['r2']:.3f}"] for k, v in r3["regression"].items()]
table(s, pd.DataFrame(rows, columns=["Delay regression", "MAE", "RMSE", "R²"]), Inches(0.6), Inches(4.2), Inches(7), 14)
bullets(s, ["Order-time features only; outcome columns (real shipping days, delivery status) excluded.", "XGBoost beats the shipping-mode baseline on ranking; delay-days regression adds nothing beyond shipping mode."], x=Inches(7.9), y=Inches(1.7), w=Inches(5), size=16)

# 13 cascade
e4 = pd.read_csv(E / "exp4_disruption_types.csv")
s = slide("Experiment 4 — cascading-disruption simulator", "Daily availability over the TRUE network; buffers, lead-time lags and BOM bottlenecks")
t4 = e4[["case", "affected_suppliers", "unmet_demand_units", "financial_impact_inr", "recovery_time_days"]].copy()
t4["affected_suppliers"] = t4.affected_suppliers.round(1)
t4["unmet_demand_units"] = (t4.unmet_demand_units / 1e3).round(1).astype(str) + "k"
t4["financial_impact_inr"] = (t4.financial_impact_inr / 1e5).round(1).astype(str) + " L"
t4["recovery_time_days"] = t4.recovery_time_days.round(0).astype(int)
t4.columns = ["Disruption (no intervention)", "Affected suppliers", "Unmet", "Impact (₹)", "Recovery (d)"]
table(s, t4, Inches(0.6), Inches(1.6), Inches(12.1), 13, [Inches(6.3), Inches(1.5), Inches(1.4), Inches(1.5), Inches(1.4)])
img(s, DIA / "08_cascade_simulation.png", Inches(1.6), Inches(5.6), w=Inches(10))

# 14 nine-scenario design
s = slide("Nine-scenario evaluation design")
img(s, DIA / "09_nine_scenario_architecture.png", Inches(0.6), Inches(1.4), w=Inches(7.2))
bullets(s, ["Shock: five largest Tier-2 cocoa-bean traders (19 observed + 25 hidden outgoing links) lose 25 / 50 / 100 % capacity for a nominal 60 days.",
            "A: no intervention · B: rule-based (observed graph, same material, reliability-ranked, standard freight, reactive) · C: MILP (observed + predicted graph, derated availability, substitutes, expedited option).",
            "500 replications per scenario with common random numbers; 180-day horizon; illustrative INR economics documented."], x=Inches(8), y=Inches(1.5), w=Inches(5), size=16)

# 15 scenario results
s = slide("Nine-scenario results", "Means and 95 % percentile intervals; simulated INR on synthetic data")
img(s, FIG / "scenarios_nine.png", Inches(0.4), Inches(1.6), w=Inches(7.4))
t = scen[["scenario", "severity", "policy", "unmet_demand_units", "financial_impact_inr", "recovery_time_days"]].copy()
t["unmet_demand_units"] = (t.unmet_demand_units / 1e3).round(1)
t["financial_impact_inr"] = (t.financial_impact_inr / 1e7).round(2)
t["recovery_time_days"] = t.recovery_time_days.round(0).astype(int)
t.columns = ["Scen.", "Severity", "Policy", "Unmet (k)", "Impact (₹ Cr)", "Recovery (d)"]
table(s, t, Inches(7.95), Inches(1.6), Inches(5.2), 10, [Inches(0.5), Inches(0.85), Inches(1.65), Inches(0.7), Inches(0.8), Inches(0.7)])

# 16 business impact
s = slide("Business impact")
g = lambda sev, c: comp[(comp.severity == sev) & (comp.comparison == c)].iloc[0]  # noqa: E731
sev_ac, mod_ac, sev_bc = g("Severe", "A: No intervention minus C: Optimisation"), g("Moderate", "A: No intervention minus C: Optimisation"), g("Severe", "B: Rule-based minus C: Optimisation")
bullets(s, [f"Severe shock: optimisation saves ₹{sev_ac.mean_saving_inr / 1e7:.2f} Cr vs no intervention (95 % CI {sev_ac.ci95_low / 1e7:.2f}–{sev_ac.ci95_high / 1e7:.2f}) and ₹{sev_bc.mean_saving_inr / 1e7:.2f} Cr vs rule-based.",
            f"Moderate shock: ₹{mod_ac.mean_saving_inr / 1e5:.1f} L mean saving — tail protection; doing nothing is cheapest in most individual replications.",
            "Mild shock: buffers suffice; optimisation over-hedges (small extra cost) → use a tiered policy.",
            "Every number is reproducible from experiments/scenario_results.csv and checked by the verification script (61/61 checks).",
            "Financial values are simulated on synthetic data with illustrative assumptions — not observed company losses."], w=Inches(7.6), size=18)
img(s, FIG / "scenarios_cost_vs_risk.png", Inches(8.4), Inches(1.8), w=Inches(4.7))

# 17-18 app demo
for title, pics in [("Application demo — control tower", ["01_dashboard.png", "03_network.png"]),
                    ("Application demo — simulation, scenarios, recovery", ["05_simulator.png", "06_scenarios.png"]),
                    ("Application demo — decision engine and AI assistant", ["04b_decision.png", "09_assistant.png"])]:
    s = slide(title)
    for k, pic in enumerate(pics):
        pth = SS / pic
        from PIL import Image
        iw, ih = Image.open(pth).size
        wid = min(Inches(6.2), int(Inches(4.9) * iw / ih))
        img(s, pth, Inches(0.4) + k * Inches(6.35), Inches(1.5), w=wid)
    text(s, Inches(0.5), Inches(6.6), Inches(12), Inches(0.6), "React 18 + Vite · Flask (JWT) · MySQL/SQLite · every value computed live from the database, models and solvers", 13, MUTED)

# 19 reproducibility
s = slide("Reproducibility and testing")
bullets(s, ["bash run_all.sh: pinned, hash-verified downloads → data generation (seed 42) → Exps 1–5 → verification → figures → database → tests.",
            "Experiment registry (run.json per experiment): dataset hash, seeds, features, split counts, class balance, hyper-parameters, metrics, timestamps.",
            "verify_results.py: 61/61 checks — confusion matrices sum to test size, metrics recomputed from predictions, prevalence, CRN integrity, impact components.",
            "pytest: 35 tests pass (leakage, metrics, policy, simulator, MILP, validation, 20 API end-to-end); API suite also passes on MySQL (MariaDB 10.11).",
            "Browser check: all 12 pages rendered with live data and zero console errors (Playwright); deployed on Render.",
            "Gemini function calling executed with a live key; free-tier throttling falls back to the tested deterministic router."], size=18)

# 20 limitations
s = slide("Limitations and future work")
bullets(s, ["Supplier-level results rest on a synthetic generator we designed; real-world supplier-network performance is not established.",
            "Economic parameters are illustrative; conclusions are about relative policy performance.",
            "Link prediction quality is moderate (AP ≈ 2× base rate); predicted links need supplier verification.",
            "Optimisation over-hedges under mild shocks; future: risk-aware / stochastic programming policies.",
            "Future data: bills of lading, UN Comtrade cocoa flows, supplier surveys to label hidden links."], size=20)

# 21 conclusion
s = slide("Conclusion")
bullets(s, ["Calibrated prediction makes thresholds defensible but rarely justifies supplier-level action when buffers are healthy.",
            "The decisive value lies in exposing hidden dependencies and optimising recovery under severe network shocks (≈75 % lower simulated impact).",
            "Every recommendation is traceable to a model output, an exposure calculation and a solver result — delivered in a tested, working application."], size=22)
out = ROOT / "presentation" / "Supply_Chain_Black_Box.pptx"
prs.save(out)
print(out, len(prs.slides), "slides")
