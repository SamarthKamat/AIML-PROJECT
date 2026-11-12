# Data Dictionary

Generated from the actual files by `python -m ml.data_engineering.data_dictionary`.
SYNTHETIC tables describe a fictional ecosystem; REAL tables are processed public datasets.

## Synthetic FMCG ecosystem (`datasets/synthetic/`)

### companies (synthetic) — 1 rows × 6 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `company_id` | str | 0 | C001 |  |
| `company_name` | str | 0 | Cocoa & Malt Foods Ltd (fictional) |  |
| `industry` | str | 0 | FMCG - chocolate and malt-based beverage |  |
| `country` | str | 0 | India |  |
| `headquarters` | str | 0 | Pune, Maharashtra |  |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |

### suppliers (synthetic) — 500 rows × 20 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `supplier_id` | str | 0 | S0001 | Supplier key S0001-S0500 (S0001-S0120 Tier-1, S0121-S0320 Tier-2, S0321-S0500 Tier-3) |
| `supplier_name` | str | 0 | Vindhya Cocoa Processors (Synthetic) | Fictional generated name, suffixed '(Synthetic)' |
| `tier` | int64 | 0 | 1 – 3 | Supply tier (1 = delivers to manufacturer) |
| `primary_material_id` | str | 0 | M01 | Material the supplier produces (FK materials) |
| `category` | str | 0 | Cocoa |  |
| `region` | str | 0 | Indonesia | State / country region (21 values) |
| `country` | str | 0 | Indonesia |  |
| `latitude` | float64 | 0 | -40.43 – 53.66 | Approximate latitude (region centroid + noise) |
| `longitude` | float64 | 0 | -6.844 – 176 | Approximate longitude |
| `size_index` | float64 | 0 | 0.1606 – 7.858 | Relative size (log-normal); drives preferential attachment |
| `capacity_utilisation` | float64 | 0 | 0.45 – 0.969 | Share of capacity in use [0.45, 0.97] |
| `reliability_score` | float64 | 0 | 0.7644 – 0.9787 | On-time reliability score [0.70, 0.995] |
| `financial_risk_score` | float64 | 0 | 0 – 0.8013 | Observed financial-risk rating = latent risk + N(0, 0.08) noise |
| `geo_risk_index` | float64 | 0 | 0.15 – 0.55 | Static regional geo-risk [0, 1] |
| `certified_fssc22000` | bool | 0 | 0 – 1 | Food-safety certification flag |
| `onboarding_year` | int64 | 0 | 2008 – 2022 |  |
| `import_lead_extra_days` | int64 | 0 | 0 – 35 | Extra sea lead time for imports (days) |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |
| `capacity_units_per_month` | float64 | 0 | 15.2 – 2.088e+06 | Capacity (Tier-1 sized from manufacturer demand) |
| `capacity_unit` | str | 0 | kg |  |

### supplier_relationships (synthetic) — 1,500 rows × 10 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `relationship_id` | str | 0 | R00001 | Relationship key |
| `supplier_id` | str | 0 | S0001 | Supplier key S0001-S0500 (S0001-S0120 Tier-1, S0121-S0320 Tier-2, S0321-S0500 Tier-3) |
| `buyer_id` | str | 0 | C001 | Buyer: supplier_id or C001 (manufacturer) |
| `material_id` | str | 0 | M01 | Material key M01-M50 |
| `dependency_pct` | float64 | 0 | 0.0015 – 1 | Share of the buyer's need of this material from this supplier (sums to 1 per buyer-material) |
| `lead_time_days` | int64 | 0 | 5 – 49 | Nominal lead time (days) |
| `relationship_status` | str | 0 | active |  |
| `visibility` | str | 0 | observed | 'observed' = known to manufacturer; 'hidden' = ground truth only (never a model feature) |
| `buyer_tier` | int64 | 0 | 0 – 2 | Tier of buyer (0 = manufacturer) |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |

### materials (synthetic) — 50 rows × 10 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `material_id` | str | 0 | M01 | Material key M01-M50 |
| `material_name` | str | 0 | Cocoa powder (natural) | Material name |
| `category` | str | 0 | Cocoa |  |
| `bom_level` | int64 | 0 | 1 – 3 | 1 = delivered to manufacturer, 2 = Tier-1 input, 3 = Tier-2 input |
| `unit` | str | 0 | kg |  |
| `unit_price_inr` | float64 | 0 | 1.2 – 3800 | Unit price (illustrative INR) |
| `criticality` | str | 0 | A | A (critical) / B / C |
| `input_materials` | str | 8 | M29 | Upstream input materials (;-separated) |
| `typical_regions` | str | 0 | Kerala;Karnataka;Indonesia;Malaysia;Ghan |  |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |

### products (synthetic) — 100 rows × 10 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `product_id` | str | 0 | P001 | Product key P001-P100 |
| `product_name` | str | 0 | Milk chocolate bar - Classic |  |
| `family` | str | 0 | Milk chocolate bar |  |
| `pack_scale` | float64 | 0 | 0.5 – 2 |  |
| `price_inr` | float64 | 0 | 12.63 – 959.1 | Selling price per pack (illustrative) |
| `contribution_margin_pct` | float64 | 0 | 0.214 – 0.417 | Contribution margin share of price |
| `plant_id` | str | 0 | W02 |  |
| `base_weekly_demand_units` | float64 | 0 | 337 – 2.563e+04 |  |
| `seasonality` | str | 0 | festive_peak |  |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |

### bill_of_materials (synthetic) — 960 rows × 3 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `product_id` | str | 0 | P001 | Product key P001-P100 |
| `material_id` | str | 0 | M01 | Material key M01-M50 |
| `qty_per_unit` | float64 | 0 | 1e-05 – 2 | Material per product unit (kg or units) |

### inventory (synthetic) — 5,824 rows × 12 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `inventory_id` | str | 0 | I00001 |  |
| `site_id` | str | 0 | W01 | Plant / DC key |
| `material_id` | str | 0 | M01 | Material key M01-M50 |
| `week_start` | str | 0 | 2024-01-01 | Monday of the week |
| `opening_qty` | float64 | 0 | 0 – 1.938e+06 | Opening stock |
| `receipts_qty` | float64 | 0 | 0 – 1.059e+06 | Receipts in week |
| `consumption_qty` | float64 | 0 | 0 – 3.432e+05 | Consumption by production |
| `required_qty` | float64 | 0 | 0.91 – 3.432e+05 | Requirement from production plan x BOM |
| `closing_qty` | float64 | 0 | 0 – 1.938e+06 | Closing stock (capped at 6 weeks cover) |
| `material_shortage_qty` | float64 | 0 | 0 – 2861 | max(0, required - (opening + receipts)) |
| `days_of_cover` | float64 | 0 | 0 – 42 | 7 x closing / weekly requirement |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |

### shipments (synthetic) — 9,579 rows × 15 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `shipment_id` | str | 0 | SH00001 | Shipment key |
| `po_id` | str | 0 | PO00001 | Purchase order key |
| `supplier_id` | str | 0 | S0001 | Supplier key S0001-S0500 (S0001-S0120 Tier-1, S0121-S0320 Tier-2, S0321-S0500 Tier-3) |
| `material_id` | str | 0 | M01 | Material key M01-M50 |
| `destination_site_id` | str | 0 | W01 | Receiving plant |
| `transport_mode` | str | 0 | sea | road / sea |
| `ship_date` | str | 0 | 2023-11-30 | Dispatch date |
| `promised_date` | str | 0 | 2024-01-01 | Promised delivery date |
| `actual_delivery_date` | str | 0 | 2024-01-04 | Actual delivery date |
| `delay_days` | int64 | 0 | 0 – 17 | actual - promised (>= 0) |
| `is_delayed` | int64 | 0 | 0 – 1 | 1 if delay_days >= 2 |
| `shipped_qty` | float64 | 0 | 0.38 – 5.715e+05 | Quantity shipped |
| `delivered_qty` | float64 | 0 | 0.38 – 5.636e+05 | Quantity received in good condition |
| `freight_cost_inr` | float64 | 0 | 12 – 5.096e+05 | Freight (2.5 % road, 4 % sea of value) |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |

### disruptions (synthetic) — 1,048 rows × 11 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `disruption_id` | str | 0 | D00001 | Disruption key |
| `supplier_id` | str | 0 | S0082 | Supplier key S0001-S0500 (S0001-S0120 Tier-1, S0121-S0320 Tier-2, S0321-S0500 Tier-3) |
| `disruption_type` | str | 0 | logistics_port | weather, labour_strike, quality_failure, equipment_failure, financial_distress, logistics_port, regulatory, upstream_material_shortage |
| `start_date` | str | 0 | 2024-01-01 | Actual start (used for labels) |
| `end_date` | str | 0 | 2024-01-08 | Actual end |
| `duration_days` | int64 | 0 | 2 – 138 | Duration (days) |
| `capacity_loss_pct` | float64 | 0 | 0.05 – 0.9739 | Capacity lost [0, 1] |
| `reported_date` | str | 0 | 2024-01-02 | Date the event became known (start + Poisson(1.5) days); features use reported_date < t |
| `origin_supplier_id` | str | 921 | S0433 | Upstream supplier that caused an upstream_material_shortage |
| `is_qualifying` | int64 | 0 | 0 – 1 | 1 if capacity_loss >= 0.25 and duration >= 3 |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |

### procurement (synthetic) — 4,929 rows × 11 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `po_id` | str | 0 | PO00001 | Purchase order key |
| `supplier_id` | str | 0 | S0001 | Supplier key S0001-S0500 (S0001-S0120 Tier-1, S0121-S0320 Tier-2, S0321-S0500 Tier-3) |
| `material_id` | str | 0 | M01 | Material key M01-M50 |
| `relationship_id` | str | 0 | R00001 | Relationship key |
| `order_date` | str | 0 | 2023-11-30 | PO date |
| `promised_date` | str | 0 | 2024-01-01 | Promised delivery date |
| `ordered_qty` | float64 | 0 | 1.15 – 6.145e+05 | PO quantity |
| `unit_price_inr` | float64 | 0 | 1.14 – 3633 | Unit price (illustrative INR) |
| `po_value_inr` | float64 | 0 | 1439 – 1.274e+07 | ordered_qty x unit_price |
| `currency` | str | 0 | INR |  |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |

### external_risks (synthetic) — 504 rows × 7 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `region` | str | 0 | Maharashtra | State / country region (21 values) |
| `country` | str | 0 | India |  |
| `month` | str | 0 | 2024-01-01 | First day of month |
| `weather_risk_forecast` | float64 | 0 | 0 – 1.049 | Seasonal weather-risk forecast for the month (synthetic) |
| `port_congestion_index` | float64 | 0 | 0 – 0.502 | Port congestion index (synthetic) |
| `geopolitical_index` | float64 | 0 | 0.085 – 0.682 | Geopolitical index (synthetic) |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |

### warehouses (synthetic) — 4 rows × 10 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `site_id` | str | 0 | W01 | Plant / DC key |
| `site_name` | str | 0 | Plant P1 - Pune (synthetic) |  |
| `site_type` | str | 0 | plant |  |
| `region` | str | 0 | Maharashtra | State / country region (21 values) |
| `city` | str | 0 | Pune |  |
| `latitude` | float64 | 0 | 18.52 – 22.59 | Approximate latitude (region centroid + noise) |
| `longitude` | float64 | 0 | 73.06 – 88.36 | Approximate longitude |
| `company_id` | str | 0 | C001 |  |
| `storage_capacity_tonnes` | int64 | 0 | 6000 – 1.2e+04 |  |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |

### production (synthetic) — 10,400 rows × 8 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `production_id` | str | 0 | PR00001 |  |
| `product_id` | str | 0 | P001 | Product key P001-P100 |
| `plant_id` | str | 0 | W02 |  |
| `week_start` | str | 0 | 2024-01-01 | Monday of the week |
| `planned_qty` | float64 | 0 | 228 – 3.438e+04 | Planned weekly production |
| `actual_qty` | float64 | 0 | 0 – 3.438e+04 | Achieved production (limited by scarcest BOM input) |
| `shortfall_qty` | float64 | 0 | 0 – 1.263e+04 | planned - actual |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |

### material_substitutes (synthetic) — 10 rows × 5 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `material_id` | str | 0 | M01 | Material key M01-M50 |
| `substitute_material_id` | str | 0 | M02 |  |
| `compatibility_score` | float64 | 0 | 0.5 – 0.9 |  |
| `note` | str | 0 | Spec approval needed (alkalisation chang |  |
| `data_label` | str | 0 | synthetic | Provenance label ('synthetic') |

## Real processed datasets (`datasets/processed/`)

### dataco_shipments (real) — 172,765 rows × 22 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `Order Item Id` | int64 | 0 | 1 – 1.805e+05 |  |
| `Order Id` | int64 | 0 | 1 – 7.72e+04 |  |
| `order_ts` | str | 0 | 2015-01-01 00:00:00 |  |
| `Type` | str | 0 | CASH |  |
| `Shipping Mode` | str | 0 | Standard Class |  |
| `Market` | str | 0 | LATAM |  |
| `Order Region` | str | 0 | Central America |  |
| `Customer Segment` | str | 0 | Consumer |  |
| `Department Name` | str | 0 | Fan Shop |  |
| `Category Name` | str | 0 | Camping & Hiking |  |
| `Days for shipment (scheduled)` | int64 | 0 | 0 – 4 |  |
| `Order Item Quantity` | int64 | 0 | 1 – 5 |  |
| `Order Item Product Price` | float64 | 0 | 9.99 – 2000 |  |
| `Order Item Discount Rate` | float64 | 0 | 0 – 0.25 |  |
| `Sales` | float64 | 0 | 9.99 – 2000 |  |
| `Latitude` | float64 | 0 | -33.94 – 48.78 |  |
| `Longitude` | float64 | 0 | -158 – 115.3 |  |
| `order_month` | int64 | 0 | 1 – 12 |  |
| `order_dow` | int64 | 0 | 0 – 6 |  |
| `order_hour` | int64 | 0 | 0 – 23 |  |
| `delay_days` | int64 | 0 | -2 – 4 | actual - promised (>= 0) |
| `is_late` | int64 | 0 | 0 – 1 |  |

### supplygraph_nodes_daily (real) — 5,466 rows × 22 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `date` | str | 0 | 2023-01-28 |  |
| `node` | str | 0 | SOS008L02P |  |
| `node_index` | int64 | 0 | 0 – 39 |  |
| `sales_order_sum_7d` | float64 | 0 | 0 – 11.51 |  |
| `sales_order_sum_14d` | float64 | 0 | 0 – 12.13 |  |
| `sales_order_sum_28d` | float64 | 0 | 0 – 12.72 |  |
| `delivery_sum_7d` | float64 | 0 | 0 – 11.53 |  |
| `delivery_sum_14d` | float64 | 0 | 0 – 12.15 |  |
| `delivery_sum_28d` | float64 | 0 | 0 – 12.72 |  |
| `production_sum_7d` | float64 | 0 | 0 – 11.51 |  |
| `production_sum_14d` | float64 | 0 | 0 – 12.14 |  |
| `production_sum_28d` | float64 | 0 | 0 – 12.76 |  |
| `factory_issue_sum_7d` | float64 | 0 | 0 – 11.55 |  |
| `factory_issue_sum_14d` | float64 | 0 | 0 – 12.15 |  |
| `factory_issue_sum_28d` | float64 | 0 | 0 – 12.76 |  |
| `fill_rate_7d` | float64 | 162 | 0 – 3 |  |
| `fill_rate_28d` | float64 | 74 | 0 – 3 |  |
| `prod_to_issue_28d` | float64 | 145 | 0 – 5 |  |
| `dow` | int64 | 0 | 0 – 6 |  |
| `future_orders_7d` | float64 | 0 | 1 – 9.937e+04 |  |
| `future_delivery_7d` | float64 | 0 | 0 – 1.02e+05 |  |
| `service_shortfall_7d` | int64 | 0 | 0 – 1 |  |

### supplygraph_edges (real) — 1,252 rows × 3 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `src` | int64 | 0 | 0 – 38 |  |
| `dst` | int64 | 0 | 1 – 39 |  |
| `relation` | str | 0 | same_plant |  |

## Experiment dataset (`datasets/experiment/`)

### exp1_supplier_snapshots (derived from synthetic) — 10,500 rows × 28 columns

| Column | Type | Missing | Range / example | Description |
|---|---|---:|---|---|
| `supplier_id` | str | 0 | S0001 | Supplier key S0001-S0500 (S0001-S0120 Tier-1, S0121-S0320 Tier-2, S0321-S0500 Tier-3) |
| `snapshot_ts` | str | 0 | 2024-04-01 |  |
| `tier` | int64 | 0 | 1 – 3 | Supply tier (1 = delivers to manufacturer) |
| `reliability_score` | float64 | 0 | 0.7644 – 0.9787 | On-time reliability score [0.70, 0.995] |
| `financial_risk_score` | float64 | 0 | 0 – 0.8013 | Observed financial-risk rating = latent risk + N(0, 0.08) noise |
| `capacity_utilisation` | float64 | 0 | 0.45 – 0.969 | Share of capacity in use [0.45, 0.97] |
| `geo_risk_index` | float64 | 0 | 0.15 – 0.55 | Static regional geo-risk [0, 1] |
| `weather_risk_forecast` | float64 | 0 | 0 – 1.049 | Seasonal weather-risk forecast for the month (synthetic) |
| `port_congestion_index` | float64 | 0 | 0 – 0.502 | Port congestion index (synthetic) |
| `material_criticality` | int64 | 0 | 1 – 3 |  |
| `max_dependency_pct` | float64 | 0 | 0 – 1 |  |
| `n_buyers_observed` | float64 | 0 | 0 – 8 |  |
| `n_sources_observed` | float64 | 0 | 0 – 6 |  |
| `single_sourced_input` | int64 | 0 | 0 – 1 |  |
| `alt_supplier_count` | int64 | 0 | 0 – 39 |  |
| `disruptions_90d` | float64 | 0 | 0 – 4 |  |
| `disruptions_365d` | float64 | 0 | 0 – 7 |  |
| `days_since_last_disruption` | float64 | 0 | 1 – 730 |  |
| `max_severity_365d` | float64 | 0 | 0 – 0.9739 |  |
| `currently_disrupted_reported` | int64 | 0 | 0 – 1 |  |
| `upstream_disruptions_90d` | float64 | 0 | 0 – 7 |  |
| `upstream_disrupted_now_share` | float64 | 0 | 0 – 1 |  |
| `has_shipment_history` | int64 | 0 | 0 – 1 |  |
| `delay_rate_90d` | float64 | 7980 | 0 – 1 |  |
| `mean_delay_days_90d` | float64 | 7980 | 0 – 8.3 |  |
| `fill_rate_90d` | float64 | 7980 | 0.7395 – 0.995 |  |
| `import_supplier` | int64 | 0 | 0 – 1 |  |
| `disruption_next_30_days` | int64 | 0 | 0 – 1 |  |
