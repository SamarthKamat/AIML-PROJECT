"""EXPERIMENT 2A - Candidate supplier-relationship discovery (link prediction). Data: SYNTHETIC.

Two evaluations:
  (A1) Standard held-out link prediction on the OBSERVED graph: observed edges are split
       70/10/20 (train/val/test). Test edges are removed from the message-passing graph.
       Negatives: (i) random non-edges, (ii) 'hard' compatible non-edges (supplier makes an
       input the buyer needs). Ground-truth hidden edges are excluded from negative pools.
  (A2) Hidden-dependency discovery: train on ALL observed edges (10 % held as validation for
       early stopping), then rank every compatible but unobserved pair. Positives = synthetic
       ground-truth hidden edges. Outputs are CANDIDATE relationships, never confirmed ones.

Methods: Common Neighbours, Adamic-Adar, Preferential Attachment, Node2Vec (biased walks +
skip-gram), MLP (features only), GCN, GraphSAGE, GAT.  GNNs: 5 seeds (mean +- sd).
Run: python -m ml.experiments.exp2a_link_prediction
"""
from __future__ import annotations

import json
import math
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.common.config import EXPERIMENTS, GNN_SEEDS, MODELS, SEED
from ml.common.tracking import log_run
from ml.graph.build import compatible_pairs, edge_pairs, load_graph_tables, node_features, node_index, to_undirected_index
from ml.graph.models import Encoder, LinkDecoder

EXP_ID = "exp2a_link_prediction"
torch.set_num_threads(4)


def precision_recall_at_k(scores, labels, ks=(50, 100, 200, 500)):
    order = np.argsort(-scores)
    lab = np.asarray(labels)[order]
    tot = lab.sum()
    return {f"P@{k}": round(float(lab[:k].mean()), 4) for k in ks} | {f"R@{k}": round(float(lab[:k].sum() / tot), 4) for k in ks}


def eval_scores(scores, labels):
    return dict(roc_auc=round(float(roc_auc_score(labels, scores)), 4), ap=round(float(average_precision_score(labels, scores)), 4),
                **precision_recall_at_k(scores, labels))


# ------------------------------------------------------------------ heuristics
def heuristic_scores(G: nx.Graph, pairs, kind):
    if kind == "Common Neighbours":
        return np.array([len(list(nx.common_neighbors(G, u, v))) for u, v in pairs], float)
    if kind == "Adamic-Adar":
        return np.array([sum(1 / math.log(G.degree(w)) for w in nx.common_neighbors(G, u, v) if G.degree(w) > 1)
                         for u, v in pairs], float)
    if kind == "Preferential Attachment":
        return np.array([G.degree(u) * G.degree(v) for u, v in pairs], float)
    raise ValueError(kind)


# ------------------------------------------------------------------ node2vec
def node2vec_embed(G: nx.Graph, n, dim=64, walks=20, length=20, p=1.0, q=0.5, window=5, epochs=5, seed=SEED):
    rnd = random.Random(seed)
    nbrs = {u: list(G.neighbors(u)) for u in G.nodes}
    corpus = []
    for _ in range(walks):
        for s in G.nodes:
            if not nbrs[s]:
                continue
            w = [s, rnd.choice(nbrs[s])]
            while len(w) < length:
                cur, prev = w[-1], w[-2]
                cand = nbrs[cur]
                wts = [(1 / p) if x == prev else (1.0 if G.has_edge(x, prev) else 1 / q) for x in cand]
                w.append(rnd.choices(cand, weights=wts)[0])
            corpus.append(w)
    pairs = [(w[i], w[j]) for w in corpus for i in range(len(w)) for j in range(max(0, i - window), min(len(w), i + window + 1)) if i != j]
    pairs = torch.tensor(pairs, dtype=torch.long)
    torch.manual_seed(seed)
    emb_in = torch.nn.Embedding(n, dim)
    emb_out = torch.nn.Embedding(n, dim)
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=0.01)
    deg = torch.tensor([max(G.degree(i), 1) if i in G else 1 for i in range(n)], dtype=torch.float) ** 0.75
    for _ in range(epochs):
        perm = torch.randperm(len(pairs))
        for b in range(0, len(pairs), 4096):
            bt = pairs[perm[b:b + 4096]]
            neg = torch.multinomial(deg, len(bt) * 5, replacement=True).view(len(bt), 5)
            c = emb_in(bt[:, 0])
            pos = (c * emb_out(bt[:, 1])).sum(-1)
            ng = torch.bmm(emb_out(neg), c.unsqueeze(-1)).squeeze(-1)
            loss = -(F.logsigmoid(pos).mean() + F.logsigmoid(-ng).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
    return F.normalize(emb_in.weight.detach(), dim=1).numpy()


# ------------------------------------------------------------------ GNN
def train_gnn(kind, x, train_pairs, val_pos, val_neg, neg_pool, n, seed, epochs=300, patience=40):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = Encoder(kind, x.shape[1], 64, 64, 0.3)
    dec = LinkDecoder(64)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.01, weight_decay=5e-4)
    ei = to_undirected_index(train_pairs)
    tp = torch.tensor(sorted(train_pairs), dtype=torch.long).T
    pool = np.array(sorted(neg_pool))
    vp = torch.tensor(np.array(list(val_pos) + list(val_neg)), dtype=torch.long).T
    vy = np.r_[np.ones(len(val_pos)), np.zeros(len(val_neg))]
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        enc.train()
        dec.train()
        k = tp.shape[1]
        rnd = torch.randint(0, n, (2, k))
        hard = torch.tensor(pool[np.random.randint(0, len(pool), k)], dtype=torch.long).T
        neg = torch.cat([rnd, hard], 1)
        h = enc(x, ei)
        lp, ln = dec(h, tp), dec(h, neg)
        loss = F.binary_cross_entropy_with_logits(torch.cat([lp, ln]), torch.cat([torch.ones_like(lp), torch.zeros_like(ln)]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        enc.eval()
        dec.eval()
        with torch.no_grad():
            s = dec(enc(x, ei), vp).numpy()
        ap = average_precision_score(vy, s)
        if ap > best:
            best, bad = ap, 0
            best_state = ({k: v.clone() for k, v in enc.state_dict().items()}, {k: v.clone() for k, v in dec.state_dict().items()})
        else:
            bad += 1
            if bad > patience:
                break
    enc.load_state_dict(best_state[0])
    dec.load_state_dict(best_state[1])
    enc.eval()
    dec.eval()
    enc.best_val_ap = float(best)
    return enc, dec, ei


def score_pairs(enc, dec, x, ei, pairs):
    with torch.no_grad():
        return dec(enc(x, ei), torch.tensor(np.array(pairs), dtype=torch.long).T).numpy()


def run_eval(name_set, x, n, train_pairs, val_pos, val_neg, neg_pool, test_pos, test_negs: dict, seeds):
    """Returns {method: {neg_type: metrics}} ; GNN metrics aggregated over seeds."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(train_pairs | set(val_pos))
    out = {}
    for neg_type, tneg in test_negs.items():
        pairs = list(test_pos) + list(tneg)
        y = np.r_[np.ones(len(test_pos)), np.zeros(len(tneg))]
        for h in ["Common Neighbours", "Adamic-Adar", "Preferential Attachment"]:
            out.setdefault(h, {})[neg_type] = eval_scores(heuristic_scores(G, pairs, h), y)
    emb = node2vec_embed(G, n)
    for neg_type, tneg in test_negs.items():
        pairs = list(test_pos) + list(tneg)
        y = np.r_[np.ones(len(test_pos)), np.zeros(len(tneg))]
        out.setdefault("Node2Vec", {})[neg_type] = eval_scores(np.array([emb[u] @ emb[v] for u, v in pairs]), y)
    xt = torch.tensor(x)
    for kind in name_set:
        per_seed = {nt: [] for nt in test_negs}
        for sd in seeds:
            enc, dec, ei = train_gnn(kind, xt, train_pairs, val_pos, val_neg, neg_pool, n, sd)
            for neg_type, tneg in test_negs.items():
                pairs = list(test_pos) + list(tneg)
                y = np.r_[np.ones(len(test_pos)), np.zeros(len(tneg))]
                per_seed[neg_type].append(eval_scores(score_pairs(enc, dec, xt, ei, pairs), y))
        out[kind] = {}
        for nt, lst in per_seed.items():
            agg = {}
            for key in lst[0]:
                v = np.array([d[key] for d in lst])
                agg[key] = round(float(v.mean()), 4)
                agg[key + "_sd"] = round(float(v.std(ddof=1)), 4)
            out[kind][nt] = agg
    return out


def main():
    sup, rel = load_graph_tables()
    idx, ids = node_index(sup)
    n = len(ids)
    x = node_features(sup)
    obs = edge_pairs(rel, idx, "observed")
    hid = edge_pairs(rel, idx, "hidden") - obs
    comp = compatible_pairs(sup, idx)
    assert obs <= comp and hid <= comp, "every true edge must be material-compatible"
    rng = np.random.default_rng(SEED)

    # ---------------- A1: standard split on observed graph
    o = np.array(sorted(obs))
    perm = rng.permutation(len(o))
    n_te, n_va = int(0.2 * len(o)), int(0.1 * len(o))
    te_pos = {tuple(p) for p in o[perm[:n_te]]}
    va_pos = {tuple(p) for p in o[perm[n_te:n_te + n_va]]}
    tr_pos = obs - te_pos - va_pos
    hard_pool = comp - obs - hid           # compatible true negatives (ground-truth)
    hard_list = sorted(hard_pool)
    sel = rng.permutation(len(hard_list))
    te_hard = [hard_list[i] for i in sel[:len(te_pos)]]
    va_hard = [hard_list[i] for i in sel[len(te_pos):len(te_pos) + len(va_pos)]]
    allpos = obs | hid
    te_rand = set()
    while len(te_rand) < len(te_pos):
        a, b = sorted(rng.integers(0, n, 2))
        if a != b and (a, b) not in allpos:
            te_rand.add((int(a), int(b)))
    train_neg_pool = comp - tr_pos - set(va_pos) - set(te_pos) - set(te_hard) - set(va_hard)  # includes hidden (unknown) pairs: PU setting
    a1 = run_eval(["MLP", "GCN", "GraphSAGE", "GAT", "GraphSAGE+skip"], x, n, tr_pos, va_pos, va_hard, train_neg_pool, te_pos,
                  {"random_negatives": te_rand, "hard_compatible_negatives": te_hard}, GNN_SEEDS)

    # ---------------- A2: hidden-dependency discovery
    o2 = np.array(sorted(obs))
    perm2 = rng.permutation(len(o2))
    va2 = {tuple(p) for p in o2[perm2[:int(0.1 * len(o2))]]}
    tr2 = obs - va2
    cand = sorted(comp - obs)              # every compatible unobserved pair
    y_cand = np.array([1 if c in hid else 0 for c in cand])
    va2_neg = [hard_list[i] for i in sel[:len(va2)]]
    pool2 = comp - obs
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(obs)
    a2 = {}
    for h in ["Common Neighbours", "Adamic-Adar", "Preferential Attachment"]:
        a2[h] = eval_scores(heuristic_scores(G, cand, h), y_cand)
    emb = node2vec_embed(G, n)
    a2["Node2Vec"] = eval_scores(np.array([emb[u] @ emb[v] for u, v in cand]), y_cand)
    xt = torch.tensor(x)
    scores_seed0, val_ap = {}, {}
    for kind in ["MLP", "GCN", "GraphSAGE", "GAT", "GraphSAGE+skip"]:
        res, vals = [], []
        for sd in GNN_SEEDS:
            enc, dec, ei = train_gnn(kind, xt, tr2, va2, va2_neg, pool2, n, sd)
            s = score_pairs(enc, dec, xt, ei, cand)
            res.append(eval_scores(s, y_cand))
            vals.append(enc.best_val_ap)
            if sd == GNN_SEEDS[0]:
                scores_seed0[kind] = s
                torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(), "in_dim": x.shape[1], "kind": kind},
                           MODELS / f"exp2a_{kind.replace('+', '_').lower()}_link.pt")
        val_ap[kind] = round(float(np.mean(vals)), 4)
        a2[kind] = {k: round(float(np.mean([r[k] for r in res])), 4) for k in res[0]} | \
                   {k + "_sd": round(float(np.std([r[k] for r in res], ddof=1)), 4) for k in res[0]} | {"val_ap": val_ap[kind]}
    selected = max(val_ap, key=val_ap.get)        # model selection on VALIDATION AP only (observed held-out edges)
    best_scores = scores_seed0[selected]

    # candidate relationship table for the application (GraphSAGE seed 0)
    cdf = pd.DataFrame(cand, columns=["u", "v"])
    cdf["node_a"] = [ids[u] for u in cdf.u]
    cdf["node_b"] = [ids[v] for v in cdf.v]
    cdf["score"] = best_scores
    cdf["rank"] = cdf.score.rank(ascending=False, method="first").astype(int)
    cdf["ground_truth_hidden_edge"] = y_cand  # evaluation only (synthetic)
    cdf["status"] = "candidate (predicted, unverified)"
    cdf["model"] = selected
    (EXPERIMENTS / EXP_ID).mkdir(parents=True, exist_ok=True)
    cdf.sort_values("rank").drop(columns=["u", "v"]).to_csv(EXPERIMENTS / EXP_ID / "candidate_links.csv", index=False)

    rec = dict(data_label="synthetic", nodes=n, observed_edges=len(obs), hidden_edges=len(hid), compatible_pairs=len(comp),
               A1_split=dict(train=len(tr_pos), val=len(va_pos), test=len(te_pos), test_random_neg=len(te_rand), test_hard_neg=len(te_hard)),
               A2_candidates=dict(n=len(cand), positives=int(y_cand.sum()), base_rate=round(float(y_cand.mean()), 4)),
               leakage_controls=["test edges removed from message-passing graph", "val edges used only for early stopping",
                                 "hidden edges never used in training (A1 excluded from negatives; A2 positives only at evaluation)"],
               seeds=GNN_SEEDS, hyperparameters=dict(hidden=64, layers=2, dropout=0.3, lr=0.01, weight_decay=5e-4,
                                                   epochs=300, patience=40, negatives="1 random + 1 compatible per positive"),
               A1_results=a1, A2_results=a2, candidate_model=selected, candidate_model_selection="highest mean validation AP (A2)",
               summary={"candidate_model": selected, "A2_selected_AP": a2[selected]["ap"], "A2_base_rate": round(float(y_cand.mean()), 4)})
    log_run(EXP_ID, rec)
    print(json.dumps({"A1": {k: {nt: (v["roc_auc"], v["ap"]) for nt, v in d.items()} for k, d in a1.items()},
                      "A2": {k: (v["roc_auc"], v["ap"], v["P@100"], v["R@500"]) for k, v in a2.items()},
                      "cand": rec["A2_candidates"]}, indent=1))


if __name__ == "__main__":
    main()
