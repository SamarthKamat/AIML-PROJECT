"""Paper / presentation figures generated ONLY from saved experiment outputs.

Run: python -m ml.experiments.report_figures
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.common.config import EXPERIMENTS, FIGURES  # noqa: E402

POL = {"A: No intervention": "#e05d5d", "B: Rule-based": "#f2a541", "C: Optimisation": "#3fb68b"}
SEV = ["Mild", "Moderate", "Severe"]
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)


def scenarios():
    s = pd.read_csv(EXPERIMENTS / "scenario_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for ax, (col, lab, scale) in zip(axes, [("financial_impact_inr", "Financial impact (INR crore)", 1e7),
                                            ("unmet_demand_units", "Unmet demand ('000 units)", 1e3),
                                            ("recovery_time_days", "Recovery time (days)", 1)]):
        w = 0.26
        for j, (p, c) in enumerate(POL.items()):
            g = s[s.policy == p].set_index("severity").reindex(SEV)
            y = g[col] / scale
            err = None
            if col + "_p2.5" in g:
                err = np.vstack([(g[col] - g[col + "_p2.5"]) / scale, (g[col + "_p97.5"] - g[col]) / scale])
            ax.bar(np.arange(3) + (j - 1) * w, y, w, color=c, label=p, yerr=err, error_kw=dict(lw=0.6, capsize=1.5))
        ax.set_xticks(range(3))
        ax.set_xticklabels(SEV)
        ax.set_title(lab, fontsize=7.5)
    axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    save(fig, "scenarios_nine")
    # cost vs risk
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for p, c in POL.items():
        g = s[s.policy == p]
        ax.scatter(g.procurement_spend_inr / 1e5, g.unmet_demand_units / 1e3, s=g.financial_impact_inr / g.financial_impact_inr.max() * 160 + 10,
                   color=c, alpha=0.8, label=p)
        for _, r in g.iterrows():
            ax.annotate(r.scenario, (r.procurement_spend_inr / 1e5, r.unmet_demand_units / 1e3), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="Recovery procurement spend (INR lakh)", ylabel="Unmet demand ('000 units)", title="Cost vs. risk (bubble = impact)")
    ax.legend(fontsize=5.5, frameon=False)
    save(fig, "scenarios_cost_vs_risk")
    conv = pd.read_csv(EXPERIMENTS / "exp4_5_scenarios" / "mc_convergence.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.plot(conv.n, conv.running_mean / 1e7, color="#1f3b73", lw=1)
    ax.fill_between(conv.n, (conv.running_mean - 1.96 * conv.se) / 1e7, (conv.running_mean + 1.96 * conv.se) / 1e7, alpha=0.25, color="#5b8def")
    ax.set(xlabel="Replications", ylabel="Mean impact (INR crore)", title="Monte Carlo convergence (S7)")
    save(fig, "mc_convergence")


def links():
    r = json.loads((EXPERIMENTS / "exp2a_link_prediction" / "run.json").read_text())
    names = list(r["A2_results"])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    a1 = [r["A1_results"][k]["hard_compatible_negatives"]["ap"] for k in names]
    a1e = [r["A1_results"][k]["hard_compatible_negatives"].get("ap_sd", 0) for k in names]
    a2 = [r["A2_results"][k]["ap"] for k in names]
    a2e = [r["A2_results"][k].get("ap_sd", 0) for k in names]
    col = ["#9aa5b1"] * 4 + ["#3fb68b", "#5b8def", "#5b8def", "#5b8def", "#5b8def"]
    ax[0].bar(range(len(names)), a1, yerr=a1e, color=col, error_kw=dict(lw=0.6))
    ax[0].axhline(0.5, ls=":", c="grey", lw=0.8)
    ax[0].set_title("A1: held-out observed edges vs compatible negatives (AP)", fontsize=7)
    ax[1].bar(range(len(names)), a2, yerr=a2e, color=col, error_kw=dict(lw=0.6))
    ax[1].axhline(r["A2_candidates"]["base_rate"], ls=":", c="grey", lw=0.8)
    ax[1].set_title(f"A2: hidden-edge discovery among {r['A2_candidates']['n']:,} candidates (AP)", fontsize=7)
    for a in ax:
        a.set_xticks(range(len(names)))
        a.set_xticklabels([n.replace("Preferential Attachment", "Pref. Att.").replace("Common Neighbours", "CN").replace("Adamic-Adar", "AA")
                           .replace("GraphSAGE+skip", "SAGE+skip").replace("GraphSAGE", "SAGE") for n in names], rotation=35, ha="right", fontsize=6.5)
    save(fig, "exp2a_links")


if __name__ == "__main__":
    scenarios()
    links()
    print("figures written to", FIGURES)
