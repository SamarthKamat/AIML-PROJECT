"""Database schema (SQLAlchemy Core) - single source of truth for MySQL DDL and the application.

`python -m backend.database_setup` creates the tables; database_schema.sql is generated from
this metadata with the MySQL dialect.  DATABASE_URL selects the engine, e.g.
    mysql+pymysql://scbb:scbb@localhost:3306/supply_chain_black_box
    sqlite:///backend/instance/scbb.db   (default, zero-config fallback)
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import (JSON, Boolean, Column, Date, DateTime, Float, ForeignKey, Index, Integer, MetaData, String, Table,
                        Text, create_engine, func)

ROOT = Path(__file__).resolve().parents[2]
metadata = MetaData()

companies = Table("companies", metadata,
                  Column("company_id", String(10), primary_key=True), Column("company_name", String(120), nullable=False),
                  Column("industry", String(120)), Column("country", String(60)), Column("headquarters", String(120)),
                  Column("data_label", String(20), nullable=False))

materials = Table("materials", metadata,
                  Column("material_id", String(10), primary_key=True), Column("material_name", String(120), nullable=False),
                  Column("category", String(60)), Column("bom_level", Integer, nullable=False), Column("unit", String(20)),
                  Column("unit_price_inr", Float, nullable=False), Column("criticality", String(1), nullable=False),
                  Column("input_materials", String(120)), Column("typical_regions", String(255)), Column("data_label", String(20)))

suppliers = Table("suppliers", metadata,
                  Column("supplier_id", String(10), primary_key=True), Column("supplier_name", String(160), nullable=False),
                  Column("tier", Integer, nullable=False), Column("primary_material_id", String(10), ForeignKey("materials.material_id"), nullable=False),
                  Column("category", String(60)), Column("region", String(60)), Column("country", String(60)),
                  Column("latitude", Float), Column("longitude", Float), Column("size_index", Float),
                  Column("capacity_utilisation", Float), Column("reliability_score", Float), Column("financial_risk_score", Float),
                  Column("geo_risk_index", Float), Column("certified_fssc22000", Boolean), Column("onboarding_year", Integer),
                  Column("import_lead_extra_days", Integer), Column("capacity_units_per_month", Float), Column("capacity_unit", String(20)),
                  Column("data_label", String(20)), Index("ix_sup_tier", "tier"), Index("ix_sup_region", "region"))

supplier_relationships = Table("supplier_relationships", metadata,
                               Column("relationship_id", String(10), primary_key=True),
                               Column("supplier_id", String(10), ForeignKey("suppliers.supplier_id"), nullable=False),
                               Column("buyer_id", String(10), nullable=False),  # supplier_id or company_id (C001)
                               Column("material_id", String(10), ForeignKey("materials.material_id"), nullable=False),
                               Column("dependency_pct", Float, nullable=False), Column("lead_time_days", Integer),
                               Column("relationship_status", String(20)), Column("visibility", String(10), nullable=False),
                               Column("buyer_tier", Integer), Column("data_label", String(20)),
                               Index("ix_rel_buyer", "buyer_id"), Index("ix_rel_vis", "visibility"))

products = Table("products", metadata,
                 Column("product_id", String(10), primary_key=True), Column("product_name", String(160), nullable=False),
                 Column("family", String(80)), Column("pack_scale", Float), Column("price_inr", Float), Column("contribution_margin_pct", Float),
                 Column("plant_id", String(10), ForeignKey("warehouses.site_id")), Column("base_weekly_demand_units", Float),
                 Column("seasonality", String(20)), Column("data_label", String(20)))

bill_of_materials = Table("bill_of_materials", metadata,
                          Column("id", Integer, primary_key=True, autoincrement=True),
                          Column("product_id", String(10), ForeignKey("products.product_id"), nullable=False),
                          Column("material_id", String(10), ForeignKey("materials.material_id"), nullable=False),
                          Column("qty_per_unit", Float, nullable=False))

warehouses = Table("warehouses", metadata,
                   Column("site_id", String(10), primary_key=True), Column("site_name", String(120)), Column("site_type", String(30)),
                   Column("region", String(60)), Column("city", String(60)), Column("latitude", Float), Column("longitude", Float),
                   Column("company_id", String(10), ForeignKey("companies.company_id")), Column("storage_capacity_tonnes", Float),
                   Column("data_label", String(20)))

inventory = Table("inventory", metadata,
                  Column("inventory_id", String(10), primary_key=True),
                  Column("site_id", String(10), ForeignKey("warehouses.site_id"), nullable=False),
                  Column("material_id", String(10), ForeignKey("materials.material_id"), nullable=False),
                  Column("week_start", Date, nullable=False), Column("opening_qty", Float), Column("receipts_qty", Float),
                  Column("consumption_qty", Float), Column("required_qty", Float), Column("closing_qty", Float),
                  Column("material_shortage_qty", Float), Column("days_of_cover", Float), Column("data_label", String(20)),
                  Index("ix_inv_week", "week_start"))

procurement = Table("procurement", metadata,
                    Column("po_id", String(10), primary_key=True),
                    Column("supplier_id", String(10), ForeignKey("suppliers.supplier_id"), nullable=False),
                    Column("material_id", String(10), ForeignKey("materials.material_id"), nullable=False),
                    Column("relationship_id", String(10), ForeignKey("supplier_relationships.relationship_id")),
                    Column("order_date", Date), Column("promised_date", Date), Column("ordered_qty", Float),
                    Column("unit_price_inr", Float), Column("po_value_inr", Float), Column("currency", String(3)), Column("data_label", String(20)))

shipments = Table("shipments", metadata,
                  Column("shipment_id", String(10), primary_key=True), Column("po_id", String(10), ForeignKey("procurement.po_id")),
                  Column("supplier_id", String(10), ForeignKey("suppliers.supplier_id"), nullable=False),
                  Column("material_id", String(10), ForeignKey("materials.material_id"), nullable=False),
                  Column("destination_site_id", String(10), ForeignKey("warehouses.site_id")), Column("transport_mode", String(10)),
                  Column("ship_date", Date), Column("promised_date", Date), Column("actual_delivery_date", Date),
                  Column("delay_days", Integer), Column("is_delayed", Integer), Column("shipped_qty", Float), Column("delivered_qty", Float),
                  Column("freight_cost_inr", Float), Column("data_label", String(20)), Index("ix_ship_supplier", "supplier_id"))

disruptions = Table("disruptions", metadata,
                    Column("disruption_id", String(10), primary_key=True),
                    Column("supplier_id", String(10), ForeignKey("suppliers.supplier_id"), nullable=False),
                    Column("disruption_type", String(40)), Column("start_date", Date), Column("end_date", Date), Column("duration_days", Integer),
                    Column("capacity_loss_pct", Float), Column("reported_date", Date), Column("origin_supplier_id", String(10)),
                    Column("is_qualifying", Integer), Column("data_label", String(20)), Index("ix_dis_start", "start_date"))

external_risks = Table("external_risks", metadata,
                       Column("id", Integer, primary_key=True, autoincrement=True), Column("region", String(60), nullable=False),
                       Column("country", String(60)), Column("month", Date, nullable=False), Column("weather_risk_forecast", Float),
                       Column("port_congestion_index", Float), Column("geopolitical_index", Float), Column("data_label", String(20)))

production = Table("production", metadata,
                   Column("production_id", String(10), primary_key=True),
                   Column("product_id", String(10), ForeignKey("products.product_id"), nullable=False),
                   Column("plant_id", String(10), ForeignKey("warehouses.site_id")), Column("week_start", Date),
                   Column("planned_qty", Float), Column("actual_qty", Float), Column("shortfall_qty", Float), Column("data_label", String(20)))

material_substitutes = Table("material_substitutes", metadata,
                             Column("id", Integer, primary_key=True, autoincrement=True),
                             Column("material_id", String(10), ForeignKey("materials.material_id")),
                             Column("substitute_material_id", String(10), ForeignKey("materials.material_id")),
                             Column("compatibility_score", Float), Column("note", String(255)), Column("data_label", String(20)))

# ------------------------------------------------------------------ application tables
users = Table("users", metadata,
              Column("user_id", Integer, primary_key=True, autoincrement=True), Column("email", String(160), unique=True, nullable=False),
              Column("full_name", String(120)), Column("password_hash", String(255), nullable=False),
              Column("role", String(20), default="analyst"), Column("created_at", DateTime, server_default=func.now()))

predictions = Table("predictions", metadata,
                    Column("prediction_id", Integer, primary_key=True, autoincrement=True),
                    Column("supplier_id", String(10), ForeignKey("suppliers.supplier_id"), nullable=False),
                    Column("prediction_timestamp", DateTime, nullable=False), Column("model_version", String(60)),
                    Column("calibration_method", String(20)), Column("probability_raw", Float), Column("probability_calibrated", Float),
                    Column("risk_category", String(20)), Column("recommended_action", String(40)), Column("exposure_inr", Float),
                    Column("created_at", DateTime, server_default=func.now()), Index("ix_pred_ts", "prediction_timestamp"))

graph_metrics = Table("graph_metrics", metadata,
                      Column("supplier_id", String(10), primary_key=True), Column("degree_centrality", Float),
                      Column("betweenness_centrality", Float), Column("pagerank", Float), Column("downstream_reach", Integer),
                      Column("graph_view", String(30)), Column("computed_at", DateTime, server_default=func.now()))

simulation_runs = Table("simulation_runs", metadata,
                        Column("run_id", Integer, primary_key=True, autoincrement=True), Column("user_id", Integer),
                        Column("run_type", String(30)), Column("parameters", JSON), Column("results", JSON),
                        Column("created_at", DateTime, server_default=func.now()))

recovery_recommendations = Table("recovery_recommendations", metadata,
                                 Column("recommendation_id", Integer, primary_key=True, autoincrement=True), Column("user_id", Integer),
                                 Column("supplier_id", String(10)), Column("material_id", String(10)), Column("solver_status", String(30)),
                                 Column("allocation", JSON), Column("total_cost_inr", Float), Column("created_at", DateTime, server_default=func.now()))

decisions = Table("decisions", metadata,
                  Column("decision_id", Integer, primary_key=True, autoincrement=True), Column("user_id", Integer),
                  Column("supplier_id", String(10)), Column("prediction_timestamp", DateTime), Column("calibrated_probability", Float),
                  Column("risk_category", String(20)), Column("recommended_action", String(40)), Column("rationale", Text),
                  Column("payload", JSON), Column("created_at", DateTime, server_default=func.now()))

upload_history = Table("upload_history", metadata,
                       Column("upload_id", Integer, primary_key=True, autoincrement=True), Column("user_id", Integer),
                       Column("filename", String(255)), Column("target_table", String(60)), Column("row_count", Integer),
                       Column("column_count", Integer), Column("status", String(30)), Column("report", JSON),
                       Column("created_at", DateTime, server_default=func.now()))

chat_logs = Table("chat_logs", metadata,
                  Column("chat_id", Integer, primary_key=True, autoincrement=True), Column("user_id", Integer),
                  Column("question", Text), Column("engine", String(30)), Column("tools_called", JSON), Column("answer", Text),
                  Column("created_at", DateTime, server_default=func.now()))

DATA_TABLES = ["companies", "materials", "warehouses", "suppliers", "supplier_relationships", "products", "bill_of_materials",
               "procurement", "shipments", "inventory", "disruptions", "external_risks", "production", "material_substitutes"]


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    inst = ROOT / "backend" / "instance"
    inst.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{inst / 'scbb.db'}"


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        url = database_url()
        kw = {"pool_pre_ping": True} if url.startswith("mysql") else {"connect_args": {"check_same_thread": False}}
        _engine = create_engine(url, future=True, **kw)
    return _engine


def reset_engine():
    global _engine
    _engine = None
