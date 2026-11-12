"""GNN encoders (GCN, GraphSAGE, GAT), link decoder and node classifier (PyTorch Geometric)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class Encoder(nn.Module):
    def __init__(self, kind: str, in_dim: int, hid: int = 64, out: int = 64, dropout: float = 0.3, heads: int = 4):
        super().__init__()
        self.residual = kind.endswith("+skip")
        kind = kind.replace("+skip", "")
        self.kind = kind
        if self.residual:  # input skip path: z = GNN(x) + W x  (keeps node-feature signal on sparse graphs)
            self.skip = nn.Linear(in_dim, out)
        self.dropout = dropout
        if kind == "GCN":
            self.c1, self.c2 = GCNConv(in_dim, hid), GCNConv(hid, out)
        elif kind == "GraphSAGE":
            self.c1, self.c2 = SAGEConv(in_dim, hid, aggr="mean"), SAGEConv(hid, out, aggr="mean")
        elif kind == "GAT":
            self.c1 = GATConv(in_dim, hid // heads, heads=heads, dropout=dropout)
            self.c2 = GATConv(hid, out, heads=1, dropout=dropout)
        elif kind == "MLP":
            self.c1, self.c2 = nn.Linear(in_dim, hid), nn.Linear(hid, out)
        else:
            raise ValueError(kind)

    def forward(self, x, edge_index):
        if self.kind == "MLP":
            h = F.relu(self.c1(x))
            h = F.dropout(h, self.dropout, self.training)
            return self.c2(h)
        h = F.relu(self.c1(x, edge_index))
        h = F.dropout(h, self.dropout, self.training)
        z = self.c2(h, edge_index)
        return z + self.skip(x) if self.residual else z


class LinkDecoder(nn.Module):
    """Symmetric pair scorer: MLP([h_u * h_v, |h_u - h_v|])."""

    def __init__(self, dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, h, pairs):
        a, b = h[pairs[0]], h[pairs[1]]
        return self.mlp(torch.cat([a * b, (a - b).abs()], dim=-1)).squeeze(-1)


class NodeClassifier(nn.Module):
    def __init__(self, kind, in_dim, hid=64, dropout=0.3):
        super().__init__()
        self.enc = Encoder(kind, in_dim, hid, hid, dropout)
        self.skip = nn.Linear(in_dim, hid)  # residual path keeps own-node signal
        self.head = nn.Linear(hid, 1)

    def forward(self, x, edge_index):
        return self.head(F.relu(self.enc(x, edge_index) + self.skip(x))).squeeze(-1)
