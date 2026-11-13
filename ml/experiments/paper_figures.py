"""Publication figures for the IEEE paper (v2), generated ONLY from saved data, models and results.

Run: python -m ml.experiments.paper_figures   -> documentation/final_report/figures/fig_*.pdf
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

from ml.common.config import EXPDATA, EXPERIMENTS, FIGURES, MODELS, ROOT, SYN, TEST_START  # noqa: E402
from ml.data_engineering.features_supplier import FEATURES  # noqa: E402

NAVY, BLUE, TEAL, AMBER, RED, GREY, PURPLE, GREEN = "#1B2A4A", "#2F5BEA", "#1497A3", "#E0A21B", "#D64545", "#8A94A6", "#7A5AF8", "#2E9E6B"
POL = {"A: No intervention": RED, "B: Rule-based": AMBER, "C: Optimisation": GREEN}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": "#E6E9F0", "grid.linewidth": 0.5, "axes.axisbelow": True, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def tag(ax, letter, x=None, y=None):
    """Prefix the panel title with its letter (avoids collisions between tag and title)."""
    t = ax.get_title()
    ax.set_title(f"({letter}) {t}" if t else f"({letter})", fontsize=7.5, fontweight="bold" if not t else None, loc="center")


# ------------------------------------------------------------------ architecture
def architecture():
    fig, ax = plt.subplots(figsize=(7.16, 2.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    lanes = [("DATA", 30.5, "#EEF3FF"), ("MODELS", 20.8, "#E9F7F6"), ("DECIDE", 11.1, "#FFF6E5"), ("APP", 1.4, "#F2EEFF")]
    for name, y, col in lanes:
        ax.add_patch(FancyBboxPatch((0.3, y), 99.4, 8.6, boxstyle="round,pad=0,rounding_size=1.2", fc=col, ec="none"))
        ax.text(2.4, y + 4.3, name, rotation=90, ha="center", va="center", fontsize=5.8, fontweight="bold", color=NAVY)

    def box(x, y, w, t, sub, c):
        ax.add_patch(FancyBboxPatch((x, y + 1.1), w, 6.4, boxstyle="round,pad=0,rounding_size=0.9", fc="white", ec=c, lw=1.0))
        ax.add_patch(FancyBboxPatch((x, y + 6.0), w, 1.5, boxstyle="round,pad=0,rounding_size=0.6", fc=c, ec=c, lw=0))
        ax.text(x + w / 2, y + 6.75, t, ha="center", va="center", fontsize=6.3, color="white", fontweight="bold")
        ax.text(x + w / 2, y + 3.55, sub, ha="center", va="center", fontsize=5.7, color=NAVY, linespacing=1.15)

    data = [(4.5, "Synthetic FMCG ecosystem", "500 suppliers, 1,500 links\n(1,039 hidden), 14 tables"), (28.5, "DataCo (real)", "180,519 order lines\nshipment outcomes"),
            (52.5, "SupplyGraph (real FMCG)", "40 products, 684 edges\n221 days"), (76.5, "MySQL + validation", "22 tables, 178 checks\npoint-in-time features")]
    intel = [(4.5, "Calibrated risk (Exp 1)", "LR / DT / RF / XGBoost\n+ Platt / isotonic, SHAP"), (28.5, "Link discovery (Exp 2)", "CN, AA, node2vec, MLP,\nGCN, GraphSAGE, GAT"),
             (52.5, "Delay model (Exp 3)", "chronological split\nreal DataCo"), (76.5, "Cascade simulator (Exp 4)", "daily flows, buffers, lags\n500 CRN replications")]
    dec = [(4.5, "Network exposure", "reach x margin x\nuncovered days"),
           (28.5, "Cost-optimal action", "argmin cost + q(1-r)E\nover 4 actions"),
           (52.5, "Recovery MILP (Exp 5)", "PuLP/CBC: capacity, deadline,\nrisk, share, onboarding"), (76.5, "Nine-scenario evaluation", "3 severities x 3 policies\npaired CRN savings")]
    app = [(4.5, "React control tower", "12 pages, live charts\nnetwork explorer"), (28.5, "Flask REST API", "31 routes, JWT\naudit trail"),
           (52.5, "Gemini assistant", "tool calling over 9 tools\nevidence returned"), (76.5, "Render deployment", "Docker, gunicorn\n35 tests pass")]
    for row, y, c in [(data, 30.5, BLUE), (intel, 20.8, TEAL), (dec, 11.1, AMBER), (app, 1.4, PURPLE)]:
        for x, t, sub in row:
            box(x + 1.0, y, 19.5, t, sub, c)
    for xa, ya, xb, yb in [(cx, y0, cx, y0 - 2.6) for cx in (15.2, 39.2, 63.2, 87.2) for y0 in (31.7, 22.0, 12.3)] + \
            [(25.0, 15.4, 29.5, 15.4), (49.0, 15.4, 53.5, 15.4), (73.0, 15.4, 77.5, 15.4)]:
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=6, lw=0.7, color=NAVY, shrinkA=0, shrinkB=0))
    save(fig, "fig_architecture")


# ------------------------------------------------------------------ black-box gap
def blackbox():
    sup = pd.read_csv(SYN / "suppliers.csv").set_index("supplier_id")
    rel = pd.read_csv(SYN / "supplier_relationships.csv")
    hubs = list(sup[(sup.tier == 2) & (sup.primary_material_id == "M29")].sort_values("size_index", ascending=False).index[:5])
    fig = plt.figure(figsize=(7.16, 2.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1.05, 1.05], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    e = rel[(rel.supplier_id.isin(hubs)) | ((rel.buyer_id == "C001") & rel.material_id.isin(["M01", "M02", "M03", "M04"]))]
    t1 = set(e[e.buyer_id == "C001"].supplier_id) | set(e[e.supplier_id.isin(hubs)].buyer_id)
    e = pd.concat([e, rel[rel.buyer_id.isin(hubs) & (rel.visibility == "observed")].head(0)])
    G = nx.DiGraph()
    for _, r in e.iterrows():
        if r.buyer_id == "C001" and r.supplier_id not in t1:
            continue
        G.add_edge(r.supplier_id, r.buyer_id, vis=r.visibility)
    tier = {n: 0 if n == "C001" else int(sup.loc[n, "tier"]) for n in G.nodes}
    pos = {}
    for t, yv in [(0, 1.0), (1, 0.5), (2, 0.0)]:
        ns = sorted([n for n in G if tier[n] == t])
        for i, n in enumerate(ns):
            pos[n] = ((i + 0.5) / len(ns), yv)
    obs = [(u, v) for u, v, d in G.edges(data=True) if d["vis"] == "observed"]
    hid = [(u, v) for u, v, d in G.edges(data=True) if d["vis"] == "hidden"]
    nx.draw_networkx_edges(G, pos, edgelist=obs, edge_color=NAVY, width=0.75, arrows=False, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=hid, edge_color=RED, width=0.75, style=(0, (3, 2)), arrows=False, ax=ax)
    col = {0: NAVY, 1: BLUE, 2: TEAL}
    nx.draw_networkx_nodes(G, pos, node_color=[col[tier[n]] for n in G], node_size=[70 if tier[n] == 0 else (26 if n in hubs else 12) for n in G],
                           edgecolors="white", linewidths=0.4, ax=ax)
    for t, yv, lab in [(0, 1.0, "Manufacturer"), (1, 0.5, "Tier-1 cocoa\nprocessors"), (2, 0.0, "Tier-2 cocoa-bean\ntraders")]:
        ax.text(-0.03, yv, lab, ha="right", va="center", fontsize=6.3, color=NAVY)
    ax.plot([], [], color=NAVY, lw=0.9, label="observed link")
    ax.plot([], [], color=RED, lw=0.9, ls=(0, (3, 2)), label="hidden link (ground truth)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.28), ncol=2, frameon=False)
    ax.set_xlim(-0.33, 1.02)
    ax.set_ylim(-0.12, 1.1)
    ax.axis("off")
    ax.set_title("Cocoa sub-network, five largest Tier-2 traders (synthetic)", fontsize=7.5)
    tag(ax, "a")
    # exposure gap
    r1 = json.loads((EXPERIMENTS / "exp1_supplier_disruption" / "run.json").read_text())["exposure_summary"]
    ax2 = fig.add_subplot(gs[1])
    tiers = ["1", "2", "3"]
    o = [r1["observed_graph_mean_by_tier"][t] / 1e3 for t in tiers]
    tr = [r1["true_graph_mean_by_tier"][t] / 1e3 for t in tiers]
    x = np.arange(3)
    ax2.bar(x - 0.19, o, 0.36, color=BLUE, label="observed graph")
    ax2.bar(x + 0.19, tr, 0.36, color=RED, label="true graph (synthetic)")
    for i in (1, 2):
        ax2.annotate(f"{tr[i] / o[i]:.1f}x", (x[i] + 0.19, tr[i]), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.8, fontweight="bold", color=RED)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Tier 1", "Tier 2", "Tier 3"])
    ax2.set_ylabel("mean exposure $E_i$ (INR '000, simulated)")
    ax2.set_title("Exposure by tier", fontsize=7.5)
    ax2.set_ylim(0, 140)
    ax2.legend(frameon=False, loc="upper right", fontsize=5.8)
    tag(ax2, "b")
    # hidden share
    ax3 = fig.add_subplot(gs[2])
    st = sup.tier.to_dict()
    rel["src_tier"] = rel.supplier_id.map(st)
    share = rel.groupby("src_tier").visibility.apply(lambda s: (s == "hidden").mean() * 100)
    ax3.bar(["Tier 1", "Tier 2", "Tier 3"], share.values, color=[BLUE, TEAL, PURPLE], width=0.6)
    for i, v in enumerate(share.values):
        ax3.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=6.8)
    ax3.set_ylim(0, 105)
    ax3.set_ylabel("% of links hidden from buyer")
    ax3.set_title("Hidden links by tier", fontsize=7.5)
    ax3.set_xlabel("tier of supplying firm")
    tag(ax3, "c")
    save(fig, "fig_blackbox")


# ------------------------------------------------------------------ Exp 1 panel
def exp1():
    df = pd.read_csv(EXPDATA / "exp1_supplier_snapshots.csv", parse_dates=["snapshot_ts"])
    te = df[df.snapshot_ts >= TEST_START]
    y = te.disruption_next_30_days.values
    names = {"Logistic Regression": ("exp1_logistic_regression", BLUE), "Decision Tree": ("exp1_decision_tree", GREY),
             "Random Forest": ("exp1_random_forest", TEAL), "XGBoost": ("exp1_xgboost", RED)}
    preds, raw = {}, {}
    for n, (f, _) in names.items():
        b = joblib.load(MODELS / f"{f}.joblib")
        p = b["model"].predict_proba(te[b["features"]])[:, 1]
        raw[n] = p
        preds[n] = b["calibrator"].predict(p)
    r1 = json.loads((EXPERIMENTS / "exp1_supplier_disruption" / "run.json").read_text())
    assert abs(roc_auc_score(y, preds["XGBoost"]) - r1["results"]["XGBoost"][r1["primary_calibration"]]["roc_auc"]) < 1e-4, "figure must match logged results"
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.85), gridspec_kw={"width_ratios": [1, 1, 1, 1.05], "wspace": 0.45})
    for n, (_, c) in names.items():
        f, t, _ = roc_curve(y, preds[n])
        axs[0].plot(f, t, color=c, lw=1.0, label=f"{n.split()[0][:8]} AUC {roc_auc_score(y, preds[n]):.3f}")
        pr, rc, _ = precision_recall_curve(y, preds[n])
        axs[1].plot(rc, pr, color=c, lw=1.0, label=f"{n.split()[0][:8]} AP {average_precision_score(y, preds[n]):.3f}")
    axs[0].plot([0, 1], [0, 1], ls=":", c=GREY, lw=0.7)
    axs[1].axhline(y.mean(), ls=":", c=GREY, lw=0.7)
    axs[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC (test)")
    axs[1].set(xlabel="recall", ylabel="precision", title="PR curve (test)", ylim=(0, 0.6))
    axs[0].legend(frameon=False, loc="lower right", fontsize=5.6, handlelength=1.2)
    axs[1].legend(frameon=False, loc="upper right", fontsize=5.6, handlelength=1.2)
    for p, c, lab in [(raw["XGBoost"], RED, "raw"), (preds["XGBoost"], GREEN, "Platt")]:
        fr, mp = calibration_curve(y, p, n_bins=8, strategy="quantile")
        axs[2].plot(mp, fr, marker="o", ms=2.6, lw=1.0, color=c, label=lab)
    lim = 0.42
    axs[2].plot([0, lim], [0, lim], ls=":", c=GREY, lw=0.7)
    axs[2].set(xlim=(0, lim), ylim=(0, lim), xlabel="predicted probability", ylabel="observed frequency", title="XGBoost calibration")
    x = r1["results"]["XGBoost"]
    axs[2].legend(frameon=False, loc="upper left", title=f"ECE {x['none']['ece_uniform10']:.3f} → {x[r1['primary_calibration']]['ece_uniform10']:.3f}",
                  title_fontsize=6.0, alignment="left")
    imp = pd.read_csv(EXPERIMENTS / "exp1_supplier_disruption" / "shap_importance.csv", index_col=0).iloc[:, 0].head(7)[::-1]
    lab = {"capacity_utilisation": "capacity utilisation", "weather_risk_forecast": "weather forecast", "financial_risk_score": "financial risk",
           "reliability_score": "reliability", "port_congestion_index": "port congestion", "max_dependency_pct": "dependency share",
           "days_since_last_disruption": "days since event", "max_severity_365d": "max severity (1 y)", "alt_supplier_count": "alternatives",
           "delay_rate_90d": "delay rate (90 d)"}
    yy = np.arange(len(imp))
    axs[3].barh(yy, imp.values, color=BLUE, height=0.62)
    for yi, (k, v) in zip(yy, imp.items()):
        axs[3].text(v + 0.008, yi, lab.get(k, k), va="center", fontsize=6.0, color=NAVY)
    axs[3].set_yticks([])
    axs[3].set_xlim(0, imp.max() * 2.15)
    axs[3].set(xlabel="mean |SHAP| (log-odds)", title="Top drivers (XGBoost)")
    axs[3].grid(axis="y", visible=False)
    for a, l in zip(axs, "abcd"):
        tag(a, l)
    save(fig, "fig_exp1")


# ------------------------------------------------------------------ scenarios
def scenarios():
    """Fig. 4: (a) mean financial impact, (b) unmet demand, (c) minimum service level, (d) cheapest policy.
    Ablation and sensitivity results are reported in Tables VII and VIII."""
    s = pd.read_csv(EXPERIMENTS / "scenario_results.csv")
    reps = pd.read_csv(EXPERIMENTS / "exp4_5_scenarios" / "scenario_replications.csv")
    sev = ["Mild", "Moderate", "Severe"]
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.95), gridspec_kw={"wspace": 0.45})
    w = 0.26
    ek = dict(lw=0.6, capsize=1.2, ecolor="black")
    for j, (p, c) in enumerate(POL.items()):
        g = s[s.policy == p].set_index("severity").reindex(sev)
        xs = np.arange(3) + (j - 1) * w
        axs[0].bar(xs, g.financial_impact_inr / 1e7, w, color=c, label=p.split(": ")[1], yerr=1.96 * g.financial_impact_inr_se / 1e7, error_kw=ek)
        axs[1].bar(xs, g.unmet_demand_units / 1e3, w, color=c, yerr=1.96 * g.unmet_demand_units_se / 1e3, error_kw=ek)
        axs[2].bar(xs, 100 * g.min_service_level, w, color=c)
    axs[0].set(ylabel="INR crore (simulated)", title="Mean financial impact")
    axs[1].set(ylabel="thousand units", title="Unmet demand")
    axs[2].set(ylabel="% of demand met", title="Minimum service level", ylim=(50, 102))
    axs[0].legend(frameon=False, loc="upper left", fontsize=5.4, handlelength=0.9, borderaxespad=0.1)
    win = reps.loc[reps.groupby(["severity", "replication"]).financial_impact_inr.idxmin()].groupby("severity").policy.value_counts(normalize=True).unstack(fill_value=0).reindex(sev)
    bottom = np.zeros(3)
    for p, c in POL.items():
        v = 100 * win.get(p, pd.Series(0, index=sev)).values
        axs[3].bar(np.arange(3), v, 0.6, bottom=bottom, color=c)
        for i in range(3):
            if v[i] > 9:
                axs[3].text(i, bottom[i] + v[i] / 2, f"{v[i]:.0f}", ha="center", va="center", fontsize=6, color="white", fontweight="bold")
        bottom += v
    axs[3].set(ylabel="% of replications", title="Cheapest policy", ylim=(0, 100))
    for a, l in zip(axs, "abcd"):
        a.set_xticks(range(3))
        a.set_xticklabels(["Mild", "Mod.", "Severe"])
        a.set_xlabel("disruption severity", fontsize=6.5)
        tag(a, l)
    save(fig, "fig_scenarios")


# ------------------------------------------------------------------ application screenshots
def application():
    from PIL import Image
    sd = ROOT / "documentation" / "testing" / "screenshots"
    shots = [("01_dashboard.png", "Executive dashboard", (0, 0, 1440, 900)),
             ("03_network.png", "Network explorer", None),
             ("05_simulator.png", "Disruption simulator", None),
             ("09_assistant.png", "Gemini assistant", None)]
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.3), gridspec_kw={"wspace": 0.04})
    for a, (f, t, crop), l in zip(axs, shots, "abcd"):
        im = Image.open(sd / f).convert("RGB")
        if crop:
            im = im.crop(crop)
        w0, h0 = im.size
        target = 1.6
        if w0 / h0 > target:
            nw = int(h0 * target)
            im = im.crop(((w0 - nw) // 2, 0, (w0 + nw) // 2, h0))
        else:
            im = im.crop((0, 0, w0, int(w0 / target)))
        im = im.resize((800, 500), Image.LANCZOS)
        a.imshow(im)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_color("#C9D1E3")
        a.grid(False)
        a.set_title(f"({l}) {t}", fontsize=6.2, pad=2)
    fig.savefig(FIGURES / "fig_app.pdf", bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(FIGURES / "fig_app.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    blackbox()
    exp1()
    scenarios()
    application()
    print("figures written to", FIGURES)
