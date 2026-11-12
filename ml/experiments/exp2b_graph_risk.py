"""EXPERIMENT 2B - Does graph structure improve risk classification over tabular XGBoost?

Part 1 (SYNTHETIC supplier network): same target, features, snapshots and chronological
  train/calibration/test split as Experiment 1.  Every snapshot is one graph with the same
  topology.  Graph variants:
     observed           - edges visible to the manufacturer
     observed+predicted - observed + top-K candidate links from the Exp 2A selected model (K = 500)
     true (oracle)      - full synthetic ground truth (upper bound, NOT available in practice)
  All probability outputs are Platt-calibrated on the calibration split for a fair Brier.

Part 2 (REAL SupplyGraph, FMCG): 7-day service-shortfall classification per product-day.
  Chronological split by date 60/20/20 with a 7-day embargo between splits (label horizon).

Run: python -m ml.experiments.exp2b_graph_risk
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from ml.common.config import (CAL_END, CAL_START, EXPDATA, EXPERIMENTS, FIGURES, GNN_SEEDS, PROC, SEED, TEST_START,
                              TRAIN_END)  # noqa: E402
from ml.common.tracking import log_run  # noqa: E402
from ml.data_engineering.features_supplier import FEATURES  # noqa: E402
from ml.graph.build import edge_pairs, load_graph_tables, node_index, to_undirected_index  # noqa: E402
from ml.graph.models import NodeClassifier  # noqa: E402

EXP_ID = "exp2b_graph_risk_classification"
TARGET = "disruption_next_30_days"
torch.set_num_threads(4)


def platt(p_cal, y_cal, p_te):
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lg(p_cal).reshape(-1, 1), y_cal)
    return m.predict_proba(lg(p_te).reshape(-1, 1))[:, 1]


def scores(y, p):
    return dict(roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)))


def agg(list_of_dicts):
    out = {}
    for k in list_of_dicts[0]:
        v = np.array([d[k] for d in list_of_dicts])
        out[k] = round(float(v.mean()), 4)
        out[k + "_sd"] = round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)
    return out


def train_node_gnn(kind, graphs_tr, graphs_ca, graphs_te, ei, in_dim, seed, epochs=150, pos_weight=None):
    torch.manual_seed(seed)
    model = NodeClassifier(kind, in_dim, 64, 0.3)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        for x, y, m in graphs_tr:
            out = model(x, ei)
            loss = F.binary_cross_entropy_with_logits(out[m], y[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pc = np.concatenate([torch.sigmoid(model(x, ei))[m].numpy() for x, y, m in graphs_ca])
        pt = np.concatenate([torch.sigmoid(model(x, ei))[m].numpy() for x, y, m in graphs_te])
    return pc, pt


# ===================================================================== Part 1 synthetic
def part1():
    df = pd.read_csv(EXPDATA / "exp1_supplier_snapshots.csv", parse_dates=["snapshot_ts"])
    sup, rel = load_graph_tables()
    idx, ids = node_index(sup)
    n = len(ids)
    tr = df[df.snapshot_ts <= TRAIN_END]
    med = tr[FEATURES].median()
    mu, sd = tr[FEATURES].fillna(med).mean(), tr[FEATURES].fillna(med).std().replace(0, 1)

    def graphs(part):
        out = []
        for ts, g in part.groupby("snapshot_ts"):
            g = g.set_index("supplier_id").reindex(ids[:-1])
            feats = ((g[FEATURES].fillna(med) - mu) / sd).values
            miss = g[["delay_rate_90d"]].isna().astype(float).values
            x = np.zeros((n, len(FEATURES) + 1), np.float32)
            x[:-1] = np.c_[feats, miss]
            y = np.zeros(n, np.float32)
            y[:-1] = g[TARGET].values
            m = np.zeros(n, bool)
            m[:-1] = True
            out.append((torch.tensor(x), torch.tensor(y), torch.tensor(m)))
        return out

    ca = df[(df.snapshot_ts >= CAL_START) & (df.snapshot_ts <= CAL_END)]
    te = df[df.snapshot_ts >= TEST_START]
    G_tr, G_ca, G_te = graphs(tr), graphs(ca), graphs(te)
    order = lambda part: part.sort_values(["snapshot_ts"]).groupby("snapshot_ts", group_keys=False).apply(  # noqa: E731
        lambda g: g.set_index("supplier_id").reindex(ids[:-1]).reset_index())
    y_ca, y_te = order(ca)[TARGET].values, order(te)[TARGET].values

    obs = edge_pairs(rel, idx, "observed")
    full = edge_pairs(rel, idx)
    cand = pd.read_csv(EXPERIMENTS / "exp2a_link_prediction" / "candidate_links.csv").sort_values("rank").head(500)
    pred = {tuple(sorted((idx[a], idx[b]))) for a, b in zip(cand.node_a, cand.node_b)}
    variants = {"observed": obs, "observed+predicted(top500)": obs | pred, "true graph (oracle)": full}

    res = {}
    xgb = XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=5, eval_metric="logloss", random_state=SEED, n_jobs=4)
    xgb.fit(tr[FEATURES], tr[TARGET])
    res["XGBoost (tabular)"] = agg([scores(y_te, platt(xgb.predict_proba(order(ca)[FEATURES])[:, 1], y_ca,
                                                      xgb.predict_proba(order(te)[FEATURES])[:, 1]))])
    for kind in ["MLP", "GCN", "GraphSAGE", "GAT"]:
        for vname, pairs in variants.items():
            if kind == "MLP" and vname != "observed":
                continue
            ei = to_undirected_index(pairs)
            runs = []
            for s in GNN_SEEDS:
                pc, pt = train_node_gnn(kind, G_tr, G_ca, G_te, ei, len(FEATURES) + 1, s)
                runs.append(scores(y_te, platt(pc, y_ca, pt)))
            res[f"{kind} [{vname}]" if kind != "MLP" else "MLP (tabular, no graph)"] = agg(runs)
            print(kind, vname, res[list(res)[-1]])
    return res, dict(train_rows=len(tr), cal_rows=len(ca), test_rows=len(te), test_prevalence=round(float(y_te.mean()), 4),
                     edges={k: len(v) for k, v in variants.items()})


# ===================================================================== Part 2 real SupplyGraph
def part2():
    df = pd.read_csv(PROC / "supplygraph_nodes_daily.csv", parse_dates=["date"])
    ed = pd.read_csv(PROC / "supplygraph_edges.csv")
    n = int(df.node_index.max()) + 1
    feats = [c for c in df.columns if c.endswith("d") and ("_sum_" in c or "fill_rate" in c or "prod_to" in c)] + ["dow"]
    feats = [c for c in feats if not c.startswith("future")]
    dates = np.sort(df.date.unique())
    d1, d2 = dates[int(0.6 * len(dates))], dates[int(0.8 * len(dates))]
    emb = np.timedelta64(7, "D")
    tr = df[df.date < d1 - emb]
    va = df[(df.date >= d1) & (df.date < d2 - emb)]
    te = df[df.date >= d2]
    med = tr[feats].median()
    mu, sd = tr[feats].fillna(med).mean(), tr[feats].fillna(med).std().replace(0, 1)
    T = "service_shortfall_7d"

    def graphs(part):
        out = []
        for d, g in part.groupby("date"):
            x = np.zeros((n, len(feats)), np.float32)
            y = np.zeros(n, np.float32)
            m = np.zeros(n, bool)
            x[g.node_index.values] = ((g[feats].fillna(med) - mu) / sd).values
            y[g.node_index.values] = g[T].values
            m[g.node_index.values] = True
            out.append((torch.tensor(x), torch.tensor(y), torch.tensor(m)))
        return out

    srt = lambda part: part.sort_values(["date", "node_index"])  # noqa: E731
    y_va, y_te = srt(va)[T].values, srt(te)[T].values
    res = {}
    lr = LogisticRegression(max_iter=3000).fit(((tr[feats].fillna(med) - mu) / sd), tr[T])
    f = lambda part, m: m.predict_proba(((srt(part)[feats].fillna(med) - mu) / sd))[:, 1]  # noqa: E731
    res["Logistic Regression"] = agg([scores(y_te, platt(f(va, lr), y_va, f(te, lr)))])
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=SEED, n_jobs=4).fit(tr[feats], tr[T])
    res["XGBoost (tabular)"] = agg([scores(y_te, platt(xgb.predict_proba(srt(va)[feats])[:, 1], y_va,
                                                      xgb.predict_proba(srt(te)[feats])[:, 1]))])
    pairs = {tuple(sorted(p)) for p in ed[["src", "dst"]].values.tolist()}
    ei = to_undirected_index(pairs)
    G_tr, G_va, G_te = graphs(tr), graphs(va), graphs(te)
    for kind in ["MLP", "GCN", "GraphSAGE", "GAT"]:
        runs = []
        for s in GNN_SEEDS:
            pc, pt = train_node_gnn(kind, G_tr, G_va, G_te, ei, len(feats), s, epochs=100)
            runs.append(scores(y_te, platt(pc, y_va, pt)))
        res[kind if kind != "MLP" else "MLP (no graph)"] = agg(runs)
        print("SupplyGraph", kind, res[list(res)[-1]])
    info = dict(nodes=n, edges=len(pairs), features=feats, train_rows=len(tr), val_rows=len(va), test_rows=len(te),
                train_dates=[str(tr.date.min().date()), str(tr.date.max().date())],
                val_dates=[str(va.date.min().date()), str(va.date.max().date())],
                test_dates=[str(te.date.min().date()), str(te.date.max().date())],
                prevalence={"train": round(float(tr[T].mean()), 4), "val": round(float(va[T].mean()), 4), "test": round(float(te[T].mean()), 4)})
    return res, info


def main():
    r1, i1 = part1()
    r2, i2 = part2()
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.6, 1]})
    for a, r, title in [(ax[0], r1, "Synthetic supplier network (PR-AUC, test)"), (ax[1], r2, "SupplyGraph real FMCG (PR-AUC, test)")]:
        names = list(r)
        v = [r[k]["pr_auc"] for k in names]
        e = [r[k]["pr_auc_sd"] for k in names]
        cols = ["#9aa5b1" if ("XGB" in k or "MLP" in k or "Logistic" in k) else ("#e05d5d" if "oracle" in k else "#5b8def") for k in names]
        a.barh(range(len(names)), v, xerr=e, color=cols, error_kw=dict(lw=0.6))
        a.set_yticks(range(len(names)))
        a.set_yticklabels([k.replace("observed+predicted(top500)", "obs+pred").replace("true graph (oracle)", "oracle") for k in names], fontsize=6)
        a.set_xlim(min(v) * 0.85, max(v) * 1.08)
        a.set_title(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "exp2b_graph_risk.pdf")
    fig.savefig(FIGURES / "exp2b_graph_risk.png", dpi=200)
    plt.close(fig)
    rec = dict(part1_synthetic=dict(info=i1, results=r1, note="SYNTHETIC labels: validates behaviour on a simulated network only"),
               part2_supplygraph_real=dict(info=i2, results=r2, data_label="real (SupplyGraph, FMCG Bangladesh)"),
               seeds=GNN_SEEDS, calibration="Platt on calibration/validation split for every model",
               summary={"synthetic_best": max(r1, key=lambda k: r1[k]["pr_auc"]), "real_best": max(r2, key=lambda k: r2[k]["pr_auc"])})
    log_run(EXP_ID, rec)
    print(json.dumps({"p1": {k: (v["pr_auc"], v["roc_auc"], v["brier"]) for k, v in r1.items()},
                      "p2": {k: (v["pr_auc"], v["roc_auc"], v["brier"]) for k, v in r2.items()}, "i2": i2}, indent=1, default=str))


if __name__ == "__main__":
    main()
