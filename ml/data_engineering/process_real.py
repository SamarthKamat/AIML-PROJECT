"""Process the two REAL public datasets into experiment-ready tables.

1. DataCo Smart Supply Chain (Constante, Silva & Pereira, 2019, Mendeley Data V5,
   doi:10.17632/8gx2fvg2k6.5, CC BY 4.0; obtained via GitHub mirror
   ashishpatel26/DataCo-SMART-SUPPLY-CHAIN-FOR-BIG-DATA-ANALYSIS @7a466e7).
   -> datasets/processed/dataco_shipments.csv  (order-time features + outcomes)
   Leakage control: 'Days for shipping (real)', 'Delivery Status', shipping date,
   'Order Status' and benefit/profit fields are outcome or post-order fields and are
   kept only as *targets* / audit columns, never as features. PII-like columns
   (names, e-mail, password, street, zip) are dropped.

2. SupplyGraph (Wasi, Islam & Akib, 2024, arXiv:2401.15299; GitHub
   CIOL-SUST/SupplyGraph @0e6d010, LGPL-2.1). Real FMCG company (Bangladesh).
   -> datasets/processed/supplygraph_nodes_daily.csv, supplygraph_edges.csv
   Target: 7-day service shortfall: sum of deliveries to distributors over
   (t, t+7] < 80 % of sales orders over (t, t+7]. Features use days <= t only.

Run: python -I -m ml.data_engineering.process_real   (from repo root)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.common.config import PROC, RAW  # noqa: E402

DATACO_FEATURES_CAT = ["Type", "Shipping Mode", "Market", "Order Region", "Customer Segment", "Department Name",
                       "Category Name"]
DATACO_FEATURES_NUM = ["Days for shipment (scheduled)", "Order Item Quantity", "Order Item Product Price",
                       "Order Item Discount Rate", "Sales", "Latitude", "Longitude"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_dataco():
    src = RAW / "dataco" / "DataCoSupplyChainDataset.csv"
    d = pd.read_csv(src, encoding="latin-1")
    raw_shape = d.shape
    d["order_ts"] = pd.to_datetime(d["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
    cancelled = d["Delivery Status"].eq("Shipping canceled")
    out = d.loc[~cancelled].copy()
    out["delay_days"] = out["Days for shipping (real)"] - out["Days for shipment (scheduled)"]
    out["is_late"] = (out["delay_days"] > 0).astype(int)
    assert (out["is_late"] == out["Late_delivery_risk"]).all(), "is_late must reproduce Late_delivery_risk"
    out["order_month"] = out.order_ts.dt.month
    out["order_dow"] = out.order_ts.dt.dayofweek
    out["order_hour"] = out.order_ts.dt.hour
    keep = ["Order Item Id", "Order Id", "order_ts"] + DATACO_FEATURES_CAT + DATACO_FEATURES_NUM + \
        ["order_month", "order_dow", "order_hour", "delay_days", "is_late"]
    out = out[keep].sort_values("order_ts").reset_index(drop=True)
    out.to_csv(PROC / "dataco_shipments.csv", index=False)
    return dict(source=str(src.relative_to(ROOT)), sha256=sha256(src), raw_rows=raw_shape[0], raw_cols=raw_shape[1],
                cancelled_removed=int(cancelled.sum()), processed_rows=len(out),
                date_min=str(out.order_ts.min()), date_max=str(out.order_ts.max()),
                late_rate=round(float(out.is_late.mean()), 4))


def process_supplygraph():
    base = RAW / "supplygraph" / "Raw Dataset" / "Homogenoeus"
    tdir = base / "Temporal Data" / "Unit"
    series = {
        "sales_order": pd.read_csv(tdir / "Sales Order.csv", index_col=0, parse_dates=True),
        "delivery": pd.read_csv(tdir / "Delivery To distributor.csv", index_col=0, parse_dates=True),
        "production": pd.read_csv(tdir / "Production .csv", index_col=0, parse_dates=True),
        "factory_issue": pd.read_csv(tdir / "Factory Issue.csv", index_col=0, parse_dates=True),
    }
    nodes = pd.read_csv(base / "Nodes" / "NodesIndex.csv")
    # DATA QUALITY: 'POP001L12P' appears twice in NodesIndex.csv (both with index 12) and the temporal
    # files carry a duplicate column 'POP001L12P.1'. We keep the first occurrence and drop the duplicate.
    n_dup = int(nodes.Node.duplicated().sum())
    nodes = nodes.drop_duplicates("Node").reset_index(drop=True)
    nodes["NodeIndex"] = range(len(nodes))
    for k in series:
        series[k] = series[k].loc[:, ~series[k].columns.str.endswith(".1")]
        series[k] = series[k].reindex(columns=nodes.Node).fillna(0.0).clip(lower=0)
    so, de, pr, fi = (series[k] for k in ["sales_order", "delivery", "production", "factory_issue"])
    fut_so = so[::-1].rolling(7, min_periods=7).sum()[::-1].shift(-1)
    fut_de = de[::-1].rolling(7, min_periods=7).sum()[::-1].shift(-1)
    rows = []
    for i, node in enumerate(nodes.Node):
        f = pd.DataFrame(index=so.index)
        f["node"] = node
        f["node_index"] = i
        for name, s in series.items():
            for w in (7, 14, 28):
                f[f"{name}_sum_{w}d"] = np.log1p(s[node].rolling(w, min_periods=1).sum())
        f["fill_rate_7d"] = (de[node].rolling(7, min_periods=1).sum() / so[node].rolling(7, min_periods=1).sum().replace(0, np.nan)).clip(0, 3)
        f["fill_rate_28d"] = (de[node].rolling(28, min_periods=1).sum() / so[node].rolling(28, min_periods=1).sum().replace(0, np.nan)).clip(0, 3)
        f["prod_to_issue_28d"] = (pr[node].rolling(28, min_periods=1).sum() / fi[node].rolling(28, min_periods=1).sum().replace(0, np.nan)).clip(0, 5)
        f["dow"] = f.index.dayofweek
        f["future_orders_7d"] = fut_so[node]
        f["future_delivery_7d"] = fut_de[node]
        rows.append(f.reset_index().rename(columns={"index": "date", "Date": "date"}))
    df = pd.concat(rows, ignore_index=True)
    df = df[df.date >= df.date.min() + pd.Timedelta(days=27)]  # 28 days of history
    df = df.dropna(subset=["future_orders_7d"])
    df = df[df.future_orders_7d > 0].copy()
    df["service_shortfall_7d"] = (df.future_delivery_7d < 0.8 * df.future_orders_7d).astype(int)
    df.to_csv(PROC / "supplygraph_nodes_daily.csv", index=False)

    idx = dict(zip(nodes.Node, nodes.NodeIndex))
    edges = []
    for fname, rel in [("Edges (Plant).csv", "same_plant"), ("Edges (Storage Location).csv", "same_storage"),
                       ("Edges (Product Group).csv", "same_group"), ("Edges (Product Sub-Group).csv", "same_subgroup")]:
        e = pd.read_csv(base / "Edges" / fname)
        for _, r in e.iterrows():
            a, b = idx.get(r.node1), idx.get(r.node2)
            if a is None or b is None or a == b:
                continue
            edges.append((min(a, b), max(a, b), rel))
    ed = pd.DataFrame(edges, columns=["src", "dst", "relation"]).drop_duplicates()
    ed.to_csv(PROC / "supplygraph_edges.csv", index=False)
    return dict(source="datasets/raw/supplygraph (CIOL-SUST/SupplyGraph @0e6d010)", nodes=len(nodes),
                duplicate_nodes_removed=n_dup,
                days=len(so), date_min=str(so.index.min().date()), date_max=str(so.index.max().date()),
                processed_rows=len(df), shortfall_rate=round(float(df.service_shortfall_7d.mean()), 4),
                unique_edges=int(ed[["src", "dst"]].drop_duplicates().shape[0]),
                edges_by_relation=ed.relation.value_counts().to_dict())


if __name__ == "__main__":
    summary = {"dataco": process_dataco(), "supplygraph": process_supplygraph()}
    (PROC / "processing_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
