"""Build graph tensors for the synthetic supplier network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.common.config import SYN
from ml.data_generation.catalog import MATERIALS, REGIONS

MANU = "C001"


def load_graph_tables(syn_dir=SYN):
    sup = pd.read_csv(syn_dir / "suppliers.csv")
    rel = pd.read_csv(syn_dir / "supplier_relationships.csv")
    return sup, rel


def node_index(sup):
    ids = sup.supplier_id.tolist() + [MANU]
    return {s: i for i, s in enumerate(ids)}, ids


def node_features(sup) -> np.ndarray:
    """Observable node attributes only (no latent variables, no disruption labels)."""
    mats = list(MATERIALS)
    regs = list(REGIONS)
    n = len(sup) + 1
    X = []
    for _, s in sup.iterrows():
        tier = np.eye(4)[s.tier]
        out_m = np.zeros(len(mats))
        out_m[mats.index(s.primary_material_id)] = 1
        need = np.zeros(len(mats))
        for im in MATERIALS[s.primary_material_id][6]:
            need[mats.index(im)] = 1
        reg = np.zeros(len(regs))
        reg[regs.index(s.region)] = 1
        num = [s.latitude / 90, s.longitude / 180, np.log(s.size_index), s.reliability_score, s.financial_risk_score,
               s.capacity_utilisation]
        X.append(np.concatenate([tier, out_m, need, reg, num]))
    # manufacturer: tier 0, needs all level-1 materials, located in Pune
    tier = np.eye(4)[0]
    need = np.array([1.0 if MATERIALS[m][2] == 1 else 0.0 for m in mats])
    reg = np.zeros(len(regs))
    reg[regs.index("Maharashtra")] = 1
    X.append(np.concatenate([tier, np.zeros(len(mats)), need, reg, [18.52 / 90, 73.86 / 180, np.log(5.0), 0.95, 0.1, 0.8]]))
    X = np.array(X, dtype=np.float32)
    assert X.shape[0] == n
    return X


def compatible_pairs(sup, idx) -> set[tuple[int, int]]:
    """Supplier u can plausibly supply buyer v iff u's output material is an input of v (Tier adjacency)."""
    by_mat = sup.groupby("primary_material_id").supplier_id.apply(list).to_dict()
    pairs = set()
    for _, b in sup.iterrows():
        for im in MATERIALS[b.primary_material_id][6]:
            for u in by_mat.get(im, []):
                pairs.add(tuple(sorted((idx[u], idx[b.supplier_id]))))
    for _, s in sup[sup.tier == 1].iterrows():
        pairs.add(tuple(sorted((idx[s.supplier_id], idx[MANU]))))
    return pairs


def edge_pairs(rel, idx, visibility=None) -> set[tuple[int, int]]:
    r = rel if visibility is None else rel[rel.visibility == visibility]
    return {tuple(sorted((idx[a], idx[b]))) for a, b in zip(r.supplier_id, r.buyer_id)}


def to_undirected_index(pairs):
    import torch
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    a = np.array(sorted(pairs)).T
    return torch.tensor(np.concatenate([a, a[::-1]], axis=1), dtype=torch.long)
