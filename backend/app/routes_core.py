"""Dashboard, suppliers, graph and data-management endpoints."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from flask import Blueprint, g, jsonify, request
from sqlalchemy import insert

from backend.app import db, services as S
from backend.app.auth import login_required
from backend.data_validation import SCHEMAS, validate_frame

bp = Blueprint("core", __name__, url_prefix="/api")


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


@bp.get("/dashboard/summary")
@login_required
def dashboard():
    """Executive dashboard KPIs computed from the database and the calibrated model."""
    return jsonify(_clean(S.dashboard_summary()))


@bp.get("/suppliers")
@login_required
def suppliers():
    """List suppliers. Query: tier, region, material, search, risk_band, sort, page, page_size."""
    sup = S.suppliers_df()
    pred = S.ensure_predictions()[["supplier_id", "probability_calibrated", "risk_category", "recommended_action", "exposure_inr"]]
    df = sup.merge(pred, on="supplier_id", how="left")
    df["risk_band"] = df.probability_calibrated.map(S.risk_band)
    a = request.args
    if a.get("tier"):
        df = df[df.tier == int(a["tier"])]
    if a.get("region"):
        df = df[df.region == a["region"]]
    if a.get("material"):
        df = df[df.primary_material_id == a["material"]]
    if a.get("risk_band"):
        df = df[df.risk_band == a["risk_band"]]
    if a.get("search"):
        s = a["search"].lower()
        df = df[df.supplier_name.str.lower().str.contains(s, regex=False) | df.supplier_id.str.lower().str.contains(s, regex=False)]
    sort = a.get("sort", "-probability_calibrated")
    df = df.sort_values(sort.lstrip("-"), ascending=not sort.startswith("-"))
    page, size = max(1, int(a.get("page", 1))), min(200, max(1, int(a.get("page_size", 25))))
    total = len(df)
    df = df.iloc[(page - 1) * size: page * size]
    return jsonify(_clean(dict(total=total, page=page, page_size=size, items=df.to_dict(orient="records"),
                               regions=sorted(S.suppliers_df().region.unique().tolist()))))


@bp.get("/suppliers/<sid>")
@login_required
def supplier(sid):
    """Supplier detail: attributes, risk, centrality, upstream/downstream, candidate links, alternatives, history."""
    return jsonify(_clean(S.supplier_detail(sid)))


@bp.get("/materials")
@login_required
def materials():
    """Materials catalogue."""
    m = S.q("SELECT * FROM materials")
    return jsonify(_clean(m.astype(object).where(pd.notna(m), None).to_dict(orient="records")))


@bp.post("/graph/build")
@login_required
def graph_build():
    """(Re)compute graph centrality metrics on the observed network and store them."""
    m = S.compute_graph_metrics()
    top = m.sort_values("pagerank", ascending=False).head(10)
    return jsonify(_clean(dict(nodes=len(m) + 1, metrics_stored=len(m), top_pagerank=top.to_dict(orient="records"))))


@bp.get("/graph/network")
@login_required
def graph_network():
    """Directed supplier network. Query: include_predicted=0/1, max_tier=1..3."""
    return jsonify(_clean(S.network_json(request.args.get("include_predicted", "0") == "1", int(request.args.get("max_tier", 3)))))


# ------------------------------------------------------------------ data management
def _read_upload(f) -> pd.DataFrame:
    name = (f.filename or "").lower()
    raw = f.read()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    raise ValueError("only .csv, .xlsx or .xls files are accepted")


def _suggest_mapping(cols, table):
    target = [c.name for c in db.metadata.tables[table].columns]
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())  # noqa: E731
    tn = {norm(t): t for t in target}
    out = {}
    for c in cols:
        n = norm(c)
        if n in tn:
            out[c] = tn[n]
        else:
            hit = [t for k, t in tn.items() if k and (k in n or n in k)]
            out[c] = hit[0] if hit else None
    return out


@bp.post("/data/upload")
@login_required
def upload():
    """Upload CSV/Excel (multipart: file, target_table) -> preview, column-mapping suggestion and validation report.
    Files are validated and logged, not merged into the master tables."""
    if "file" not in request.files:
        raise ValueError("multipart field 'file' is required")
    table = request.form.get("target_table", "suppliers")
    if table not in SCHEMAS:
        raise ValueError(f"target_table must be one of {sorted(SCHEMAS)}")
    f = request.files["file"]
    df = _read_upload(f)
    mapping = _suggest_mapping(df.columns, table)
    mapped = df.rename(columns={k: v for k, v in mapping.items() if v})
    refs = {t: S.q(f"SELECT * FROM {t}") for t in ("materials", "suppliers", "warehouses", "products", "procurement")}
    rep = validate_frame(mapped, table, refs)
    rep["duplicates_on_key"] = int(mapped[SCHEMAS[table]["pk"]].duplicated().sum()) if SCHEMAS[table].get("pk") in mapped else None
    with db.get_engine().begin() as c:
        r = c.execute(insert(db.upload_history).values(user_id=g.user.get("user_id"), filename=f.filename, target_table=table,
                                                        row_count=len(df), column_count=df.shape[1], status=rep["status"], report=_clean(rep)))
    return jsonify(_clean(dict(upload_id=r.inserted_primary_key[0], filename=f.filename, target_table=table, rows=len(df), columns=list(df.columns),
                               dtypes={k: str(v) for k, v in df.dtypes.items()}, preview=df.head(10).astype(object).where(pd.notna(df.head(10)), None).to_dict(orient="records"),
                               suggested_mapping=mapping, validation=rep)))


@bp.post("/data/map-columns")
@login_required
def map_columns():
    """Apply an explicit column mapping to an uploaded file and re-validate (multipart: file, target_table, mapping JSON)."""
    import json
    table = request.form.get("target_table", "suppliers")
    if table not in SCHEMAS:
        raise ValueError("unknown target_table")
    mapping = json.loads(request.form.get("mapping", "{}"))
    df = _read_upload(request.files["file"]).rename(columns={k: v for k, v in mapping.items() if v})
    refs = {t: S.q(f"SELECT * FROM {t}") for t in ("materials", "suppliers", "warehouses", "products", "procurement")}
    return jsonify(_clean(dict(target_table=table, columns=list(df.columns), validation=validate_frame(df, table, refs))))


@bp.get("/data/validation")
@login_required
def data_validation():
    """Validation report for every master table currently in the database."""
    frames = {t: S.q(f"SELECT * FROM {t}") for t in db.DATA_TABLES}
    return jsonify(_clean({t: validate_frame(df, t, frames) for t, df in frames.items() if t in SCHEMAS}))


@bp.get("/data/tables")
@login_required
def data_tables():
    """Schema of the master tables with row counts."""
    out = []
    for t in db.DATA_TABLES:
        n = int(S.q(f"SELECT COUNT(*) AS n FROM {t}").n.iloc[0])
        out.append(dict(table=t, rows=n, columns=[dict(name=c.name, type=str(c.type), nullable=c.nullable, primary_key=c.primary_key)
                                                  for c in db.metadata.tables[t].columns]))
    return jsonify(out)


@bp.get("/data/preview/<table>")
@login_required
def data_preview(table):
    """First rows of a master table."""
    if table not in db.DATA_TABLES:
        raise ValueError("unknown table")
    return jsonify(_clean(S.q(f"SELECT * FROM {table} LIMIT 25").astype(object).to_dict(orient="records")))


@bp.get("/data/history")
@login_required
def data_history():
    """Upload processing history."""
    return jsonify(_clean(S.q("SELECT upload_id, filename, target_table, row_count, column_count, status, created_at FROM upload_history "
                              "ORDER BY upload_id DESC LIMIT 50").astype(object).to_dict(orient="records")))
