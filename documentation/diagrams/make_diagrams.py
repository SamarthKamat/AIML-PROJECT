"""Generate the 13 technical diagrams (PDF + PNG) into documentation/diagrams/.

Run: python documentation/diagrams/make_diagrams.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
C = dict(data="#dbe7ff", ml="#d9f7f0", sim="#fff1d6", opt="#fde2e2", app="#ece5ff", dec="#e3f4ff", edge="#1f3b73")
plt.rcParams.update({"font.family": "serif", "font.size": 8})


def box(ax, x, y, w, h, text, fc, fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=C["edge"], lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, a, b, text=None, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, lw=0.9, color=C["edge"], linestyle=ls,
                                 connectionstyle="arc3,rad=0"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.012, text, ha="center", fontsize=6.3, style="italic")


def canvas(w=7.0, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def d01_system():
    fig, ax = canvas(7.0, 3.3)
    box(ax, 0.01, 0.62, 0.17, 0.30, "Data layer\nSynthetic FMCG tables\nDataCo (real)\nSupplyGraph (real)\nMySQL schema", C["data"])
    box(ax, 0.22, 0.62, 0.17, 0.30, "Feature engineering\npoint-in-time\nfeatures (t)\nobserved graph only", C["data"])
    box(ax, 0.43, 0.77, 0.17, 0.15, "Exp 1: LR / DT / RF /\nXGBoost + calibration", C["ml"], 6.6)
    box(ax, 0.43, 0.60, 0.17, 0.14, "Exp 2: GCN / SAGE / GAT\nlink prediction", C["ml"], 6.6)
    box(ax, 0.64, 0.77, 0.16, 0.15, "Exposure model\nreach x margin x\nuncovered days", C["dec"])
    box(ax, 0.64, 0.60, 0.16, 0.14, "Cascade simulator\n(true network, MC)", C["sim"])
    box(ax, 0.83, 0.60, 0.16, 0.32, "Decision engine\nargmin cost + p(1-r)E\n\nMILP recovery\n(PuLP / CBC)", C["opt"])
    arrow(ax, (0.18, 0.77), (0.22, 0.77))
    arrow(ax, (0.39, 0.84), (0.43, 0.84))
    arrow(ax, (0.39, 0.67), (0.43, 0.67))
    arrow(ax, (0.60, 0.84), (0.64, 0.84))
    arrow(ax, (0.60, 0.67), (0.64, 0.67))
    arrow(ax, (0.80, 0.84), (0.83, 0.84))
    arrow(ax, (0.80, 0.67), (0.83, 0.67))
    box(ax, 0.06, 0.12, 0.26, 0.30, "React control tower (12 pages)\nDashboard, Network Explorer,\nRisk Centre, Simulator, Scenarios,\nRecovery Planner, AI Assistant, Data", C["app"])
    box(ax, 0.38, 0.12, 0.26, 0.30, "Flask REST API (JWT)\n/api/ml, /api/simulation,\n/api/recovery, /api/decision,\n/api/chat, /api/data", C["app"])
    box(ax, 0.70, 0.12, 0.27, 0.30, "AI assistant\nGemini function calling\nor deterministic router\n-> 9 evidence tools", C["app"])
    arrow(ax, (0.32, 0.27), (0.38, 0.27), style="<|-|>")
    arrow(ax, (0.64, 0.27), (0.70, 0.27), style="<|-|>")
    arrow(ax, (0.51, 0.42), (0.51, 0.60), style="<|-|>")
    arrow(ax, (0.91, 0.60), (0.85, 0.42), style="<|-|>")
    ax.text(0.5, 0.97, "Supply Chain Black Box - system architecture", ha="center", fontsize=9, weight="bold")
    save(fig, "01_system_architecture")


def flow(name, title, steps, colors, w=7.0, h=1.5, fs=6.8):
    fig, ax = canvas(w, h)
    n = len(steps)
    bw = 0.9 / n
    for i, (s, c) in enumerate(zip(steps, colors)):
        x = 0.02 + i * (bw + 0.08 / max(n - 1, 1))
        box(ax, x, 0.18, bw, 0.6, s, c, fs)
        if i < n - 1:
            arrow(ax, (x + bw, 0.48), (x + bw + 0.08 / max(n - 1, 1), 0.48))
    ax.text(0.5, 0.92, title, ha="center", fontsize=8.5, weight="bold")
    save(fig, name)


def d03_er():
    fig, ax = canvas(7.0, 3.6)
    ents = {
        "companies": (0.02, 0.70, "companies\nPK company_id"), "warehouses": (0.02, 0.38, "warehouses\nPK site_id\nFK company_id"),
        "products": (0.20, 0.38, "products\nPK product_id\nFK plant_id"), "bom": (0.20, 0.06, "bill_of_materials\nFK product_id\nFK material_id"),
        "materials": (0.40, 0.06, "materials\nPK material_id"), "suppliers": (0.40, 0.70, "suppliers\nPK supplier_id\nFK primary_material_id"),
        "rel": (0.60, 0.70, "supplier_relationships\nPK relationship_id\nFK supplier_id, material_id\nbuyer_id, visibility"),
        "proc": (0.60, 0.38, "procurement\nPK po_id\nFK supplier_id, material_id"), "ship": (0.80, 0.38, "shipments\nPK shipment_id\nFK po_id, supplier_id"),
        "dis": (0.80, 0.70, "disruptions\nPK disruption_id\nFK supplier_id"), "inv": (0.02, 0.06, "inventory\nPK inventory_id\nFK site_id, material_id"),
        "prodn": (0.40, 0.38, "production\nPK production_id\nFK product_id"), "pred": (0.80, 0.06, "predictions / decisions /\nsimulation_runs /\nrecovery_recommendations"),
        "ext": (0.60, 0.06, "external_risks\nregion, month"),
    }
    for k, (x, y, t) in ents.items():
        box(ax, x, y, 0.17, 0.22, t, C["data"] if k not in ("pred",) else C["app"], 6.3)
    links = [("companies", "warehouses"), ("warehouses", "products"), ("products", "bom"), ("bom", "materials"), ("suppliers", "rel"),
             ("suppliers", "dis"), ("suppliers", "proc"), ("proc", "ship"), ("materials", "suppliers"), ("products", "prodn"),
             ("warehouses", "inv"), ("suppliers", "pred")]
    for a, b in links:
        xa, ya, _ = ents[a]
        xb, yb, _ = ents[b]
        arrow(ax, (xa + 0.085, ya + 0.11), (xb + 0.085, yb + 0.11), style="-")
    ax.text(0.5, 0.97, "Entity-relationship diagram (MySQL schema, 22 tables)", ha="center", fontsize=8.5, weight="bold")
    save(fig, "03_er_diagram")


def d04_network():
    import networkx as nx
    import pandas as pd
    sup = pd.read_csv(ROOT / "datasets/synthetic/suppliers.csv").set_index("supplier_id")
    rel = pd.read_csv(ROOT / "datasets/synthetic/supplier_relationships.csv")
    G = nx.DiGraph()
    keep = set(rel[rel.supplier_id.isin(["S0122", "S0135", "S0136", "S0142", "S0123"])].buyer_id) | {"S0122", "S0135", "S0136", "S0142", "S0123"}
    sub = rel[rel.supplier_id.isin(keep) | rel.buyer_id.isin(keep)]
    sub = sub[sub.material_id.isin(["M29", "M01", "M02", "M03", "M04", "M43"])]
    for _, e in sub.iterrows():
        G.add_edge(e.supplier_id, e.buyer_id, vis=e.visibility)
    tier = {n: (0 if n == "C001" else int(sup.loc[n, "tier"])) for n in G.nodes}
    pos = {}
    for t in range(4):
        ns = sorted([n for n in G.nodes if tier[n] == t])
        for i, n in enumerate(ns):
            pos[n] = ((i + 0.5) / max(len(ns), 1), 1 - t / 3)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    obs = [(u, v) for u, v, d in G.edges(data=True) if d["vis"] == "observed"]
    hid = [(u, v) for u, v, d in G.edges(data=True) if d["vis"] == "hidden"]
    nx.draw_networkx_edges(G, pos, edgelist=obs, edge_color="#1f3b73", width=0.8, arrowsize=6, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=hid, edge_color="#e05d5d", style="dashed", width=0.7, arrowsize=6, ax=ax)
    col = {0: "#111111", 1: "#4f8dff", 2: "#22a7c4", 3: "#8b6fe0"}
    nx.draw_networkx_nodes(G, pos, node_color=[col[tier[n]] for n in G.nodes], node_size=[160 if tier[n] == 0 else 40 for n in G.nodes], ax=ax)
    for t, lab in [(0, "Manufacturer"), (1, "Tier-1 cocoa processors"), (2, "Tier-2 cocoa-bean traders"), (3, "Tier-3 agro-inputs")]:
        ax.text(-0.02, 1 - t / 3, lab, ha="right", va="center", fontsize=7)
    ax.plot([], [], c="#1f3b73", label="observed relationship")
    ax.plot([], [], c="#e05d5d", ls="--", label="hidden relationship (ground truth)")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, bbox_to_anchor=(0.0, 0.18))
    ax.set_xlim(-0.35, 1.02)
    ax.axis("off")
    ax.set_title("Supplier dependency network around the five largest Tier-2 cocoa-bean traders (synthetic)", fontsize=8)
    save(fig, "04_supplier_dependency_network")


def d06_graphsage():
    fig, ax = canvas(7.0, 2.0)
    box(ax, 0.01, 0.2, 0.16, 0.6, "Node features x_v\n(131-dim: tier, output &\ninput materials, region,\nsize, reliability, risk)", C["data"], 6.5)
    box(ax, 0.21, 0.2, 0.17, 0.6, "SAGEConv 1 (mean)\nh = ReLU(W1 x_v +\nW2 mean_{u in N(v)} x_u)\n64 units, dropout 0.3", C["ml"], 6.5)
    box(ax, 0.42, 0.2, 0.16, 0.6, "SAGEConv 2 (mean)\n64-dim embedding z_v", C["ml"], 6.5)
    box(ax, 0.62, 0.2, 0.17, 0.6, "Pair decoder\nMLP([z_u * z_v,\n|z_u - z_v|])", C["dec"], 6.5)
    box(ax, 0.83, 0.2, 0.16, 0.6, "sigma(score)\nP(edge u-v)\nBCE with random +\ncompatible negatives", C["opt"], 6.5)
    for a in [(0.17, 0.21), (0.38, 0.42), (0.58, 0.62), (0.79, 0.83)]:
        arrow(ax, (a[0], 0.5), (a[1], 0.5))
    ax.text(0.5, 0.93, "GraphSAGE link-prediction architecture", ha="center", fontsize=8.5, weight="bold")
    save(fig, "06_graphsage_architecture")


def d09_nine():
    fig, ax = canvas(7.0, 2.6)
    box(ax, 0.01, 0.35, 0.18, 0.35, "Common baseline\nsame network, demand,\ninventory, economics\nseed = base + r", C["data"], 6.5)
    for i, s in enumerate(["Mild 25 %", "Moderate 50 %", "Severe 100 %"]):
        box(ax, 0.24, 0.72 - i * 0.3, 0.14, 0.2, s, C["sim"], 6.8)
        arrow(ax, (0.19, 0.52), (0.24, 0.82 - i * 0.3))
        for j, p in enumerate(["A none", "B rule", "C MILP"]):
            box(ax, 0.43 + j * 0.13, 0.72 - i * 0.3, 0.11, 0.2, f"S{3 * i + j + 1}\n{p}", C["opt"] if j == 2 else C["app"], 6.3)
        arrow(ax, (0.38, 0.82 - i * 0.3), (0.43, 0.82 - i * 0.3))
    box(ax, 0.84, 0.25, 0.15, 0.55, "500 replications\nmean, 95 % PI, SE\npaired (CRN)\ndifferences\n13 metrics", C["dec"], 6.5)
    arrow(ax, (0.80, 0.52), (0.84, 0.52))
    ax.text(0.5, 0.97, "Nine-scenario evaluation architecture", ha="center", fontsize=8.5, weight="bold")
    save(fig, "09_nine_scenario_architecture")


def main():
    d01_system()
    flow("02_data_engineering", "Data engineering workflow",
         ["Raw downloads\n(DataCo, SupplyGraph)\nnever modified", "Synthetic generator\nseed 42", "Validation\n196 checks\nFK, dates, ranges", "Processing &\npoint-in-time\nfeatures", "MySQL import\n+ data dictionary"],
         [C["data"]] * 5)
    d03_er()
    d04_network()
    flow("05_supplier_risk_workflow", "Supplier disruption prediction workflow",
         ["Snapshot t\n(1st of month)", "Features from\nrecords < t\n25 features", "Chronological split\ntrain | embargo |\ncal | test", "LR / DT / RF / XGB", "Platt / isotonic\nfit on cal split", "Test metrics\n+ SHAP"],
         [C["data"], C["data"], C["data"], C["ml"], C["ml"], C["dec"]])
    d06_graphsage()
    flow("07_link_prediction_workflow", "Link prediction / hidden-dependency discovery workflow",
         ["Observed graph\n461 edges", "Edge split\n70/10/20\ntest edges removed", "Negatives: random\n+ material-\ncompatible", "Train GNN\nearly stop on val AP", "Rank compatible\nunobserved pairs", "Candidate links\n(unverified)"],
         [C["data"], C["data"], C["data"], C["ml"], C["dec"], C["opt"]])
    flow("08_cascade_simulation", "Cascading-disruption simulation",
         ["Disruption\n(supplier, loss,\nduration)", "Daily availability\na_s(t) bottleneck\nover inputs", "Buffers\n10 d per tier\nlead-time lags", "Manufacturer\nBOM fill & unmet\ndemand", "Policy orders\n(alternates)", "Metrics:\ncost, unmet,\nrecovery time"],
         [C["sim"]] * 4 + [C["opt"], C["dec"]])
    d09_nine()
    flow("10_recovery_optimisation", "Recovery optimisation workflow",
         ["Detected deficit\nper material", "Candidates: approved\n+ substitutes\n(compatibility)", "Risk screen\ncalibrated p30\n<= 0.35", "MILP: cost +\nholding + onboarding\n+ shortage penalty", "Allocation\nqty, mode, lead", "Simulated\ndelivery & yield"],
         [C["dec"], C["data"], C["ml"], C["opt"], C["opt"], C["sim"]])
    flow("11_frontend_backend", "Frontend-backend architecture",
         ["React 18 + Vite\nRecharts, force-graph\nframer-motion", "JWT bearer\n/api/*", "Flask blueprints\nauth, core, ml, chat", "Services &\nML modules", "SQLAlchemy\nMySQL / SQLite"],
         [C["app"], C["app"], C["app"], C["ml"], C["data"]])
    flow("12_ai_assistant_workflow", "AI assistant workflow",
         ["User question", "Query understanding\nGemini or\nintent router", "Tool selection\n(9 tools)", "DB retrieval /\nmodel inference /\nsimulation / MILP", "Evidence-backed\nanswer + tool log"],
         [C["app"], C["ml"], C["dec"], C["sim"], C["app"]])
    flow("13_end_to_end_decision", "End-to-end business decision workflow",
         ["Supplier &\ntimestamp t", "Features < t", "p raw ->\ncalibrated p", "Exposure E\n(observed +\npredicted links)", "Action =\nargmin C_a +\np(1-r_a)E", "Consequence\nsimulation", "MILP plan if\nHigh / Critical", "Decision record"],
         [C["data"], C["data"], C["ml"], C["dec"], C["dec"], C["sim"], C["opt"], C["app"]], h=1.6, fs=6.2)
    print("diagrams written to", OUT)


if __name__ == "__main__":
    main()
