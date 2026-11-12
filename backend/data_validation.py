"""Data validation for source tables and uploaded files.

Checks: required columns, data types, primary-key duplicates, missing values, invalid dates,
logical quantity rules, referential integrity, and IQR outlier counts (investigated, not removed).
Run: python -m backend.data_validation   -> documentation/testing/data_quality_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMAS = {
    "suppliers": dict(pk="supplier_id", required=["supplier_id", "supplier_name", "tier", "primary_material_id", "region"],
                      numeric=["tier", "capacity_utilisation", "reliability_score", "financial_risk_score", "latitude", "longitude"],
                      ranges={"capacity_utilisation": (0, 1), "reliability_score": (0, 1), "financial_risk_score": (0, 1), "tier": (1, 3)},
                      fk={"primary_material_id": ("materials", "material_id")}),
    "supplier_relationships": dict(pk="relationship_id", required=["relationship_id", "supplier_id", "buyer_id", "material_id", "dependency_pct", "visibility"],
                                   numeric=["dependency_pct", "lead_time_days"], ranges={"dependency_pct": (0, 1)},
                                   fk={"supplier_id": ("suppliers", "supplier_id"), "material_id": ("materials", "material_id")},
                                   categories={"visibility": ["observed", "hidden"]}),
    "materials": dict(pk="material_id", required=["material_id", "material_name", "bom_level", "unit_price_inr", "criticality"],
                      numeric=["unit_price_inr", "bom_level"], ranges={"unit_price_inr": (0, 1e7)}, categories={"criticality": ["A", "B", "C"]}),
    "products": dict(pk="product_id", required=["product_id", "product_name", "price_inr", "plant_id"], numeric=["price_inr", "contribution_margin_pct"],
                     ranges={"contribution_margin_pct": (0, 1)}, fk={"plant_id": ("warehouses", "site_id")}),
    "inventory": dict(pk="inventory_id", required=["inventory_id", "site_id", "material_id", "week_start"], dates=["week_start"],
                      numeric=["opening_qty", "receipts_qty", "consumption_qty", "closing_qty"],
                      nonneg=["opening_qty", "receipts_qty", "consumption_qty", "closing_qty", "material_shortage_qty"],
                      fk={"material_id": ("materials", "material_id"), "site_id": ("warehouses", "site_id")}),
    "shipments": dict(pk="shipment_id", required=["shipment_id", "supplier_id", "material_id", "promised_date", "actual_delivery_date"],
                      dates=["ship_date", "promised_date", "actual_delivery_date"], numeric=["delay_days", "shipped_qty", "delivered_qty"],
                      nonneg=["delay_days", "shipped_qty", "delivered_qty", "freight_cost_inr"],
                      rules=[("delivered_qty <= shipped_qty * 1.0001", "delivered quantity cannot exceed shipped quantity"),
                             ("actual_delivery_date >= promised_date", "delay_days >= 0 implies delivery on/after promise")],
                      fk={"supplier_id": ("suppliers", "supplier_id"), "po_id": ("procurement", "po_id")}),
    "disruptions": dict(pk="disruption_id", required=["disruption_id", "supplier_id", "start_date", "end_date", "capacity_loss_pct"],
                        dates=["start_date", "end_date", "reported_date"], numeric=["duration_days", "capacity_loss_pct"],
                        ranges={"capacity_loss_pct": (0, 1)}, rules=[("end_date >= start_date", "end date on/after start"),
                                                                    ("reported_date >= start_date", "reported on/after start")],
                        fk={"supplier_id": ("suppliers", "supplier_id")}),
    "procurement": dict(pk="po_id", required=["po_id", "supplier_id", "material_id", "order_date", "ordered_qty"], dates=["order_date", "promised_date"],
                        numeric=["ordered_qty", "unit_price_inr"], nonneg=["ordered_qty", "unit_price_inr", "po_value_inr"],
                        rules=[("promised_date >= order_date", "promise on/after order")], fk={"supplier_id": ("suppliers", "supplier_id")}),
    "production": dict(pk="production_id", required=["production_id", "product_id", "week_start", "planned_qty"], dates=["week_start"],
                       nonneg=["planned_qty", "actual_qty", "shortfall_qty"], rules=[("actual_qty <= planned_qty + 1e-6", "actual <= planned")],
                       fk={"product_id": ("products", "product_id")}),
    "external_risks": dict(pk=None, required=["region", "month"], dates=["month"], numeric=["weather_risk_forecast"]),
    "warehouses": dict(pk="site_id", required=["site_id", "site_name", "site_type"]),
    "companies": dict(pk="company_id", required=["company_id", "company_name"]),
    "bill_of_materials": dict(pk=None, required=["product_id", "material_id", "qty_per_unit"], nonneg=["qty_per_unit"],
                              fk={"product_id": ("products", "product_id"), "material_id": ("materials", "material_id")}),
    "material_substitutes": dict(pk=None, required=["material_id", "substitute_material_id", "compatibility_score"], ranges={"compatibility_score": (0, 1)}),
}


def validate_frame(df: pd.DataFrame, table: str, refs: dict | None = None) -> dict:
    sch = SCHEMAS.get(table, {})
    rep = dict(table=table, rows=int(len(df)), columns=int(df.shape[1]), errors=[], warnings=[], checks_passed=0, checks_run=0)

    def check(ok, msg, level="errors"):
        rep["checks_run"] += 1
        if ok:
            rep["checks_passed"] += 1
        else:
            rep[level].append(msg)

    missing_cols = [c for c in sch.get("required", []) if c not in df.columns]
    check(not missing_cols, f"missing required columns: {missing_cols}")
    pk = sch.get("pk")
    if pk and pk in df.columns:
        d = int(df[pk].duplicated().sum())
        check(d == 0, f"{d} duplicate primary keys in {pk}")
    dup_rows = int(df.duplicated().sum())
    check(dup_rows == 0, f"{dup_rows} fully duplicated rows", "warnings")
    na = df.isna().sum()
    rep["missing_values"] = {k: int(v) for k, v in na[na > 0].items()}
    for c in sch.get("required", []):
        if c in df.columns:
            check(int(df[c].isna().sum()) == 0, f"{int(df[c].isna().sum())} missing values in required column {c}")
    parsed = df.copy()
    for c in sch.get("dates", []):
        if c in df.columns:
            p = pd.to_datetime(df[c], errors="coerce")
            bad = int((p.isna() & df[c].notna()).sum())
            check(bad == 0, f"{bad} invalid dates in {c}")
            parsed[c] = p
    for c in sch.get("numeric", []):
        if c in df.columns:
            p = pd.to_numeric(df[c], errors="coerce")
            bad = int((p.isna() & df[c].notna()).sum())
            check(bad == 0, f"{bad} non-numeric values in {c}")
            parsed[c] = p
    for c, (lo, hi) in sch.get("ranges", {}).items():
        if c in parsed.columns:
            v = pd.to_numeric(parsed[c], errors="coerce")
            bad = int(((v < lo) | (v > hi)).sum())
            check(bad == 0, f"{bad} values of {c} outside [{lo}, {hi}]")
    for c in sch.get("nonneg", []):
        if c in parsed.columns:
            bad = int((pd.to_numeric(parsed[c], errors="coerce") < 0).sum())
            check(bad == 0, f"{bad} negative values in {c}")
    for c, allowed in sch.get("categories", {}).items():
        if c in df.columns:
            bad = int((~df[c].isin(allowed)).sum())
            check(bad == 0, f"{bad} values of {c} not in {allowed}")
    for expr, desc in sch.get("rules", []):
        try:
            ok = parsed.eval(expr)
            bad = int((~ok).sum())
            check(bad == 0, f"rule violated ({desc}): {bad} rows")
        except Exception as e:  # noqa: BLE001
            rep["warnings"].append(f"rule {expr} not evaluated: {e}")
    if refs:
        for c, (rt, rc) in sch.get("fk", {}).items():
            if c in df.columns and rt in refs:
                vals = df[c].dropna()
                bad = int((~vals.isin(refs[rt][rc])).sum())
                check(bad == 0, f"{bad} {c} values without matching {rt}.{rc}")
    outl = {}
    for c in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            n = int(((df[c] < q1 - 3 * iqr) | (df[c] > q3 + 3 * iqr)).sum())
            if n:
                outl[c] = n
    rep["outliers_3iqr_investigated_not_removed"] = outl
    rep["status"] = "pass" if not rep["errors"] else "fail"
    return rep


def validate_directory(d: Path) -> dict:
    frames = {t: pd.read_csv(d / f"{t}.csv") for t in SCHEMAS if (d / f"{t}.csv").exists()}
    reports = {t: validate_frame(df, t, frames) for t, df in frames.items()}
    # cross-table logical checks
    rel = frames["supplier_relationships"]
    sup = frames["suppliers"]
    tier = dict(zip(sup.supplier_id, sup.tier))
    tier["C001"] = 0
    bad_tier = int(sum(tier[s] != tier.get(b, -9) + 1 for s, b in zip(rel.supplier_id, rel.buyer_id)))
    reports["_cross_table"] = dict(relationship_tier_order_violations=bad_tier,
                                   buyer_ids_unknown=int((~rel.buyer_id.isin(list(sup.supplier_id) + ["C001"])).sum()),
                                   status="pass" if bad_tier == 0 else "fail")
    return reports


if __name__ == "__main__":
    from ml.common.config import SYN
    rep = validate_directory(SYN)
    out = ROOT / "documentation" / "testing" / "data_quality_report.json"
    out.write_text(json.dumps(rep, indent=2, default=str))
    for t, r in rep.items():
        print(t, r.get("status"), r.get("checks_passed"), "/", r.get("checks_run"), r.get("errors"), r.get("outliers_3iqr_investigated_not_removed", ""))
