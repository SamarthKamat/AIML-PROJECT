-- Supply Chain Black Box - MySQL 8 schema (generated from backend/app/db.py)
CREATE DATABASE IF NOT EXISTS supply_chain_black_box CHARACTER SET utf8mb4;
USE supply_chain_black_box;

CREATE TABLE chat_logs (
	chat_id INTEGER NOT NULL AUTO_INCREMENT, 
	user_id INTEGER, 
	question TEXT, 
	engine VARCHAR(30), 
	tools_called JSON, 
	answer TEXT, 
	created_at DATETIME DEFAULT (now()), 
	PRIMARY KEY (chat_id)
) ENGINE=InnoDB;


CREATE TABLE companies (
	company_id VARCHAR(10) NOT NULL, 
	company_name VARCHAR(120) NOT NULL, 
	industry VARCHAR(120), 
	country VARCHAR(60), 
	headquarters VARCHAR(120), 
	data_label VARCHAR(20) NOT NULL, 
	PRIMARY KEY (company_id)
) ENGINE=InnoDB;


CREATE TABLE decisions (
	decision_id INTEGER NOT NULL AUTO_INCREMENT, 
	user_id INTEGER, 
	supplier_id VARCHAR(10), 
	prediction_timestamp DATETIME, 
	calibrated_probability FLOAT, 
	risk_category VARCHAR(20), 
	recommended_action VARCHAR(40), 
	rationale TEXT, 
	payload JSON, 
	created_at DATETIME DEFAULT (now()), 
	PRIMARY KEY (decision_id)
) ENGINE=InnoDB;


CREATE TABLE external_risks (
	id INTEGER NOT NULL AUTO_INCREMENT, 
	region VARCHAR(60) NOT NULL, 
	country VARCHAR(60), 
	month DATE NOT NULL, 
	weather_risk_forecast FLOAT, 
	port_congestion_index FLOAT, 
	geopolitical_index FLOAT, 
	data_label VARCHAR(20), 
	PRIMARY KEY (id)
) ENGINE=InnoDB;


CREATE TABLE graph_metrics (
	supplier_id VARCHAR(10) NOT NULL, 
	degree_centrality FLOAT, 
	betweenness_centrality FLOAT, 
	pagerank FLOAT, 
	downstream_reach INTEGER, 
	graph_view VARCHAR(30), 
	computed_at DATETIME DEFAULT (now()), 
	PRIMARY KEY (supplier_id)
) ENGINE=InnoDB;


CREATE TABLE materials (
	material_id VARCHAR(10) NOT NULL, 
	material_name VARCHAR(120) NOT NULL, 
	category VARCHAR(60), 
	bom_level INTEGER NOT NULL, 
	unit VARCHAR(20), 
	unit_price_inr FLOAT NOT NULL, 
	criticality VARCHAR(1) NOT NULL, 
	input_materials VARCHAR(120), 
	typical_regions VARCHAR(255), 
	data_label VARCHAR(20), 
	PRIMARY KEY (material_id)
) ENGINE=InnoDB;


CREATE TABLE recovery_recommendations (
	recommendation_id INTEGER NOT NULL AUTO_INCREMENT, 
	user_id INTEGER, 
	supplier_id VARCHAR(10), 
	material_id VARCHAR(10), 
	solver_status VARCHAR(30), 
	allocation JSON, 
	total_cost_inr FLOAT, 
	created_at DATETIME DEFAULT (now()), 
	PRIMARY KEY (recommendation_id)
) ENGINE=InnoDB;


CREATE TABLE simulation_runs (
	run_id INTEGER NOT NULL AUTO_INCREMENT, 
	user_id INTEGER, 
	run_type VARCHAR(30), 
	parameters JSON, 
	results JSON, 
	created_at DATETIME DEFAULT (now()), 
	PRIMARY KEY (run_id)
) ENGINE=InnoDB;


CREATE TABLE upload_history (
	upload_id INTEGER NOT NULL AUTO_INCREMENT, 
	user_id INTEGER, 
	filename VARCHAR(255), 
	target_table VARCHAR(60), 
	row_count INTEGER, 
	column_count INTEGER, 
	status VARCHAR(30), 
	report JSON, 
	created_at DATETIME DEFAULT (now()), 
	PRIMARY KEY (upload_id)
) ENGINE=InnoDB;


CREATE TABLE users (
	user_id INTEGER NOT NULL AUTO_INCREMENT, 
	email VARCHAR(160) NOT NULL, 
	full_name VARCHAR(120), 
	password_hash VARCHAR(255) NOT NULL, 
	`role` VARCHAR(20), 
	created_at DATETIME DEFAULT (now()), 
	PRIMARY KEY (user_id), 
	UNIQUE (email)
) ENGINE=InnoDB;


CREATE TABLE material_substitutes (
	id INTEGER NOT NULL AUTO_INCREMENT, 
	material_id VARCHAR(10), 
	substitute_material_id VARCHAR(10), 
	compatibility_score FLOAT, 
	note VARCHAR(255), 
	data_label VARCHAR(20), 
	PRIMARY KEY (id), 
	FOREIGN KEY(material_id) REFERENCES materials (material_id), 
	FOREIGN KEY(substitute_material_id) REFERENCES materials (material_id)
) ENGINE=InnoDB;


CREATE TABLE suppliers (
	supplier_id VARCHAR(10) NOT NULL, 
	supplier_name VARCHAR(160) NOT NULL, 
	tier INTEGER NOT NULL, 
	primary_material_id VARCHAR(10) NOT NULL, 
	category VARCHAR(60), 
	region VARCHAR(60), 
	country VARCHAR(60), 
	latitude FLOAT, 
	longitude FLOAT, 
	size_index FLOAT, 
	capacity_utilisation FLOAT, 
	reliability_score FLOAT, 
	financial_risk_score FLOAT, 
	geo_risk_index FLOAT, 
	certified_fssc22000 BOOL, 
	onboarding_year INTEGER, 
	import_lead_extra_days INTEGER, 
	capacity_units_per_month FLOAT, 
	capacity_unit VARCHAR(20), 
	data_label VARCHAR(20), 
	PRIMARY KEY (supplier_id), 
	FOREIGN KEY(primary_material_id) REFERENCES materials (material_id)
) ENGINE=InnoDB;

CREATE INDEX ix_sup_tier ON suppliers (tier);
CREATE INDEX ix_sup_region ON suppliers (region);

CREATE TABLE warehouses (
	site_id VARCHAR(10) NOT NULL, 
	site_name VARCHAR(120), 
	site_type VARCHAR(30), 
	region VARCHAR(60), 
	city VARCHAR(60), 
	latitude FLOAT, 
	longitude FLOAT, 
	company_id VARCHAR(10), 
	storage_capacity_tonnes FLOAT, 
	data_label VARCHAR(20), 
	PRIMARY KEY (site_id), 
	FOREIGN KEY(company_id) REFERENCES companies (company_id)
) ENGINE=InnoDB;


CREATE TABLE disruptions (
	disruption_id VARCHAR(10) NOT NULL, 
	supplier_id VARCHAR(10) NOT NULL, 
	disruption_type VARCHAR(40), 
	start_date DATE, 
	end_date DATE, 
	duration_days INTEGER, 
	capacity_loss_pct FLOAT, 
	reported_date DATE, 
	origin_supplier_id VARCHAR(10), 
	is_qualifying INTEGER, 
	data_label VARCHAR(20), 
	PRIMARY KEY (disruption_id), 
	FOREIGN KEY(supplier_id) REFERENCES suppliers (supplier_id)
) ENGINE=InnoDB;

CREATE INDEX ix_dis_start ON disruptions (start_date);

CREATE TABLE inventory (
	inventory_id VARCHAR(10) NOT NULL, 
	site_id VARCHAR(10) NOT NULL, 
	material_id VARCHAR(10) NOT NULL, 
	week_start DATE NOT NULL, 
	opening_qty FLOAT, 
	receipts_qty FLOAT, 
	consumption_qty FLOAT, 
	required_qty FLOAT, 
	closing_qty FLOAT, 
	material_shortage_qty FLOAT, 
	days_of_cover FLOAT, 
	data_label VARCHAR(20), 
	PRIMARY KEY (inventory_id), 
	FOREIGN KEY(site_id) REFERENCES warehouses (site_id), 
	FOREIGN KEY(material_id) REFERENCES materials (material_id)
) ENGINE=InnoDB;

CREATE INDEX ix_inv_week ON inventory (week_start);

CREATE TABLE predictions (
	prediction_id INTEGER NOT NULL AUTO_INCREMENT, 
	supplier_id VARCHAR(10) NOT NULL, 
	prediction_timestamp DATETIME NOT NULL, 
	model_version VARCHAR(60), 
	calibration_method VARCHAR(20), 
	probability_raw FLOAT, 
	probability_calibrated FLOAT, 
	risk_category VARCHAR(20), 
	recommended_action VARCHAR(40), 
	exposure_inr FLOAT, 
	created_at DATETIME DEFAULT (now()), 
	PRIMARY KEY (prediction_id), 
	FOREIGN KEY(supplier_id) REFERENCES suppliers (supplier_id)
) ENGINE=InnoDB;

CREATE INDEX ix_pred_ts ON predictions (prediction_timestamp);

CREATE TABLE products (
	product_id VARCHAR(10) NOT NULL, 
	product_name VARCHAR(160) NOT NULL, 
	family VARCHAR(80), 
	pack_scale FLOAT, 
	price_inr FLOAT, 
	contribution_margin_pct FLOAT, 
	plant_id VARCHAR(10), 
	base_weekly_demand_units FLOAT, 
	seasonality VARCHAR(20), 
	data_label VARCHAR(20), 
	PRIMARY KEY (product_id), 
	FOREIGN KEY(plant_id) REFERENCES warehouses (site_id)
) ENGINE=InnoDB;


CREATE TABLE supplier_relationships (
	relationship_id VARCHAR(10) NOT NULL, 
	supplier_id VARCHAR(10) NOT NULL, 
	buyer_id VARCHAR(10) NOT NULL, 
	material_id VARCHAR(10) NOT NULL, 
	dependency_pct FLOAT NOT NULL, 
	lead_time_days INTEGER, 
	relationship_status VARCHAR(20), 
	visibility VARCHAR(10) NOT NULL, 
	buyer_tier INTEGER, 
	data_label VARCHAR(20), 
	PRIMARY KEY (relationship_id), 
	FOREIGN KEY(supplier_id) REFERENCES suppliers (supplier_id), 
	FOREIGN KEY(material_id) REFERENCES materials (material_id)
) ENGINE=InnoDB;

CREATE INDEX ix_rel_buyer ON supplier_relationships (buyer_id);
CREATE INDEX ix_rel_vis ON supplier_relationships (visibility);

CREATE TABLE bill_of_materials (
	id INTEGER NOT NULL AUTO_INCREMENT, 
	product_id VARCHAR(10) NOT NULL, 
	material_id VARCHAR(10) NOT NULL, 
	qty_per_unit FLOAT NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(product_id) REFERENCES products (product_id), 
	FOREIGN KEY(material_id) REFERENCES materials (material_id)
) ENGINE=InnoDB;


CREATE TABLE procurement (
	po_id VARCHAR(10) NOT NULL, 
	supplier_id VARCHAR(10) NOT NULL, 
	material_id VARCHAR(10) NOT NULL, 
	relationship_id VARCHAR(10), 
	order_date DATE, 
	promised_date DATE, 
	ordered_qty FLOAT, 
	unit_price_inr FLOAT, 
	po_value_inr FLOAT, 
	currency VARCHAR(3), 
	data_label VARCHAR(20), 
	PRIMARY KEY (po_id), 
	FOREIGN KEY(supplier_id) REFERENCES suppliers (supplier_id), 
	FOREIGN KEY(material_id) REFERENCES materials (material_id), 
	FOREIGN KEY(relationship_id) REFERENCES supplier_relationships (relationship_id)
) ENGINE=InnoDB;


CREATE TABLE production (
	production_id VARCHAR(10) NOT NULL, 
	product_id VARCHAR(10) NOT NULL, 
	plant_id VARCHAR(10), 
	week_start DATE, 
	planned_qty FLOAT, 
	actual_qty FLOAT, 
	shortfall_qty FLOAT, 
	data_label VARCHAR(20), 
	PRIMARY KEY (production_id), 
	FOREIGN KEY(product_id) REFERENCES products (product_id), 
	FOREIGN KEY(plant_id) REFERENCES warehouses (site_id)
) ENGINE=InnoDB;


CREATE TABLE shipments (
	shipment_id VARCHAR(10) NOT NULL, 
	po_id VARCHAR(10), 
	supplier_id VARCHAR(10) NOT NULL, 
	material_id VARCHAR(10) NOT NULL, 
	destination_site_id VARCHAR(10), 
	transport_mode VARCHAR(10), 
	ship_date DATE, 
	promised_date DATE, 
	actual_delivery_date DATE, 
	delay_days INTEGER, 
	is_delayed INTEGER, 
	shipped_qty FLOAT, 
	delivered_qty FLOAT, 
	freight_cost_inr FLOAT, 
	data_label VARCHAR(20), 
	PRIMARY KEY (shipment_id), 
	FOREIGN KEY(po_id) REFERENCES procurement (po_id), 
	FOREIGN KEY(supplier_id) REFERENCES suppliers (supplier_id), 
	FOREIGN KEY(material_id) REFERENCES materials (material_id), 
	FOREIGN KEY(destination_site_id) REFERENCES warehouses (site_id)
) ENGINE=InnoDB;

CREATE INDEX ix_ship_supplier ON shipments (supplier_id);
