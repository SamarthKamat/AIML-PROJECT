"""Generate datasets/data_dictionary/DATA_DICTIONARY.md from the actual CSV files.

Run: python -m ml.data_engineering.data_dictionary
"""
from __future__ import annotations

import pandas as pd

from ml.common.config import DATA, PROC, SYN

DESC = {
    "supplier_id": "Supplier key S0001-S0500 (S0001-S0120 Tier-1, S0121-S0320 Tier-2, S0321-S0500 Tier-3)",
    "supplier_name": "Fictional generated name, suffixed '(Synthetic)'", "tier": "Supply tier (1 = delivers to manufacturer)",
    "primary_material_id": "Material the supplier produces (FK materials)", "region": "State / country region (21 values)",
    "latitude": "Approximate latitude (region centroid + noise)", "longitude": "Approximate longitude",
    "size_index": "Relative size (log-normal); drives preferential attachment", "capacity_utilisation": "Share of capacity in use [0.45, 0.97]",
    "reliability_score": "On-time reliability score [0.70, 0.995]", "financial_risk_score": "Observed financial-risk rating = latent risk + N(0, 0.08) noise",
    "geo_risk_index": "Static regional geo-risk [0, 1]", "certified_fssc22000": "Food-safety certification flag",
    "import_lead_extra_days": "Extra sea lead time for imports (days)", "capacity_units_per_month": "Capacity (Tier-1 sized from manufacturer demand)",
    "data_label": "Provenance label ('synthetic')", "relationship_id": "Relationship key", "buyer_id": "Buyer: supplier_id or C001 (manufacturer)",
    "material_id": "Material key M01-M50", "dependency_pct": "Share of the buyer's need of this material from this supplier (sums to 1 per buyer-material)",
    "lead_time_days": "Nominal lead time (days)", "visibility": "'observed' = known to manufacturer; 'hidden' = ground truth only (never a model feature)",
    "buyer_tier": "Tier of buyer (0 = manufacturer)", "disruption_id": "Disruption key", "disruption_type": "weather, labour_strike, quality_failure, equipment_failure, financial_distress, logistics_port, regulatory, upstream_material_shortage",
    "start_date": "Actual start (used for labels)", "end_date": "Actual end", "duration_days": "Duration (days)", "capacity_loss_pct": "Capacity lost [0, 1]",
    "reported_date": "Date the event became known (start + Poisson(1.5) days); features use reported_date < t",
    "origin_supplier_id": "Upstream supplier that caused an upstream_material_shortage", "is_qualifying": "1 if capacity_loss >= 0.25 and duration >= 3",
    "shipment_id": "Shipment key", "po_id": "Purchase order key", "destination_site_id": "Receiving plant", "transport_mode": "road / sea",
    "ship_date": "Dispatch date", "promised_date": "Promised delivery date", "actual_delivery_date": "Actual delivery date", "delay_days": "actual - promised (>= 0)",
    "is_delayed": "1 if delay_days >= 2", "shipped_qty": "Quantity shipped", "delivered_qty": "Quantity received in good condition", "freight_cost_inr": "Freight (2.5 % road, 4 % sea of value)",
    "order_date": "PO date", "ordered_qty": "PO quantity", "unit_price_inr": "Unit price (illustrative INR)", "po_value_inr": "ordered_qty x unit_price",
    "site_id": "Plant / DC key", "week_start": "Monday of the week", "opening_qty": "Opening stock", "receipts_qty": "Receipts in week",
    "consumption_qty": "Consumption by production", "required_qty": "Requirement from production plan x BOM", "closing_qty": "Closing stock (capped at 6 weeks cover)",
    "material_shortage_qty": "max(0, required - (opening + receipts))", "days_of_cover": "7 x closing / weekly requirement",
    "product_id": "Product key P001-P100", "planned_qty": "Planned weekly production", "actual_qty": "Achieved production (limited by scarcest BOM input)",
    "shortfall_qty": "planned - actual", "month": "First day of month", "weather_risk_forecast": "Seasonal weather-risk forecast for the month (synthetic)",
    "port_congestion_index": "Port congestion index (synthetic)", "geopolitical_index": "Geopolitical index (synthetic)",
    "material_name": "Material name", "bom_level": "1 = delivered to manufacturer, 2 = Tier-1 input, 3 = Tier-2 input", "criticality": "A (critical) / B / C",
    "input_materials": "Upstream input materials (;-separated)", "qty_per_unit": "Material per product unit (kg or units)",
    "price_inr": "Selling price per pack (illustrative)", "contribution_margin_pct": "Contribution margin share of price",
}


def profile(df: pd.DataFrame, name: str, label: str) -> str:
    lines = [f"### {name} ({label}) — {len(df):,} rows × {df.shape[1]} columns", "",
             "| Column | Type | Missing | Range / example | Description |", "|---|---|---:|---|---|"]
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            rng = f"{s.min():.4g} – {s.max():.4g}" if s.notna().any() else "—"
        else:
            rng = str(s.dropna().iloc[0])[:40] if s.notna().any() else "—"
        lines.append(f"| `{c}` | {s.dtype} | {int(s.isna().sum())} | {rng} | {DESC.get(c, '')} |")
    return "\n".join(lines) + "\n"


def main():
    out = ["# Data Dictionary", "", "Generated from the actual files by `python -m ml.data_engineering.data_dictionary`.",
           "SYNTHETIC tables describe a fictional ecosystem; REAL tables are processed public datasets.", ""]
    out.append("## Synthetic FMCG ecosystem (`datasets/synthetic/`)\n")
    for t in ["companies", "suppliers", "supplier_relationships", "materials", "products", "bill_of_materials", "inventory", "shipments",
              "disruptions", "procurement", "external_risks", "warehouses", "production", "material_substitutes"]:
        out.append(profile(pd.read_csv(SYN / f"{t}.csv"), t, "synthetic"))
    out.append("## Real processed datasets (`datasets/processed/`)\n")
    for t in ["dataco_shipments", "supplygraph_nodes_daily", "supplygraph_edges"]:
        out.append(profile(pd.read_csv(PROC / f"{t}.csv"), t, "real"))
    out.append("## Experiment dataset (`datasets/experiment/`)\n")
    out.append(profile(pd.read_csv(DATA / "experiment" / "exp1_supplier_snapshots.csv"), "exp1_supplier_snapshots", "derived from synthetic"))
    p = DATA / "data_dictionary" / "DATA_DICTIONARY.md"
    p.write_text("\n".join(out))
    print(p)


if __name__ == "__main__":
    main()
