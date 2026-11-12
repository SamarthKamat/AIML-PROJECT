"""End-to-end API tests against a fresh SQLite database populated by data_import."""
import io

import pytest

from backend.app import db


@pytest.fixture(scope="module")
def client():
    db.reset_engine()
    from backend.data_import import import_all
    import_all(drop=True, verbose=False)
    from backend.app import create_app
    app = create_app({"TESTING": True})
    c = app.test_client()
    r = c.post("/api/auth/register", json=dict(email="tester@example.com", password="secret123", full_name="Tester"))
    assert r.status_code == 201
    c.environ_base["HTTP_AUTHORIZATION"] = "Bearer " + r.json["token"]
    return c


def test_auth_rules():
    from backend.app import create_app
    c = create_app().test_client()
    assert c.get("/api/dashboard/summary").status_code == 401
    assert c.post("/api/auth/register", json=dict(email="bad", password="x")).status_code == 400
    assert c.post("/api/auth/login", json=dict(email="nobody@example.com", password="wrongpass")).status_code == 401
    assert c.get("/api/health").json["status"] == "ok"


def test_duplicate_registration(client):
    r = client.post("/api/auth/register", json=dict(email="tester@example.com", password="secret123"))
    assert r.status_code == 409
    assert client.post("/api/auth/login", json=dict(email="tester@example.com", password="secret123")).status_code == 200


def test_dashboard_is_computed(client):
    d = client.get("/api/dashboard/summary").json
    t = d["totals"]
    assert t["suppliers"] == 500 and t["tier1"] + t["tier2"] + t["tier3"] == 500
    assert sum(d["risk_distribution"].values()) == 500
    assert t["total_exposure_inr"] > 0 and len(d["top_risk_suppliers"]) == 10


def test_suppliers_and_detail(client):
    r = client.get("/api/suppliers?tier=2&page_size=5").json
    assert r["total"] == 200 and len(r["items"]) == 5 and all(i["tier"] == 2 for i in r["items"])
    d = client.get("/api/suppliers/S0122").json
    assert d["supplier"]["tier"] == 2 and "pagerank" in d["centrality"]
    assert client.get("/api/suppliers/S9999").status_code == 400


def test_graph(client):
    assert client.post("/api/graph/build").json["metrics_stored"] == 500
    g = client.get("/api/graph/network").json
    assert len(g["nodes"]) == 501 and all(l["status"] == "observed" for l in g["links"])
    gp = client.get("/api/graph/network?include_predicted=1").json
    assert any(l["status"] == "predicted" for l in gp["links"])


def test_risk_and_link_prediction(client):
    r = client.post("/api/ml/supplier-risk", json=dict(timestamp="2025-10-01", supplier_ids=["S0122"], with_explanation=True)).json
    p = r["predictions"][0]
    assert 0 <= p["probability_calibrated"] <= 1 and len(r["explanations"]["S0122"]) == 6
    assert client.post("/api/ml/supplier-risk", json=dict(timestamp="2023-01-01")).status_code == 400
    lp = client.post("/api/ml/link-prediction", json=dict(top=10)).json
    assert len(lp["candidates"]) == 10 and all("unverified" in c["status"] for c in lp["candidates"])


def test_delivery_delay(client):
    r = client.post("/api/ml/delivery-delay", json={"Shipping Mode": "First Class", "Days for shipment (scheduled)": 1}).json
    s = client.post("/api/ml/delivery-delay", json={"Shipping Mode": "Standard Class", "Days for shipment (scheduled)": 4}).json
    assert r["late_probability"] > s["late_probability"]


def test_simulation_run_and_validation(client):
    r = client.post("/api/simulation/run", json=dict(supplier_id="S0122", capacity_reduction=1.0, duration_days=60, policy="B", replications=3)).json
    assert r["summary"]["affected_suppliers"] >= 1 and len(r["series"]["service_level"]) == 120 and r["run_id"] >= 1
    w = client.post("/api/simulation/run", json=dict(region="Kerala", disruption_type="weather", capacity_reduction=0.7, duration_days=21, replications=2)).json
    assert len(w["parameters"]["targets"]) > 1
    assert client.post("/api/simulation/run", json=dict(supplier_id="S0122", capacity_reduction=1.5)).status_code == 400
    assert client.post("/api/simulation/run", json=dict()).status_code == 400


def test_scenarios_and_impact(client):
    s = client.get("/api/simulation/scenarios").json
    assert s["available"] and len(s["results"]) == 9
    assert {r["scenario"] for r in s["results"]} == {f"S{i}" for i in range(1, 10)}
    assert len(client.get("/api/simulation/impact-types").json) == 7


def test_recovery(client):
    r = client.post("/api/recovery/optimise", json=dict(material_id="M01", quantity=6000, deadline_days=30, exclude_supplier_ids=["S0001"])).json
    assert r["solver"]["status"] == "Optimal" and r["total_cost_inr"] > 0 and "rule_based" in r
    assert all(a["supplier_id"] != "S0001" for a in r["allocation"])
    assert client.post("/api/recovery/optimise", json=dict(material_id="M40", quantity=10)).status_code == 400


def test_decision_engine(client):
    r = client.post("/api/decision/evaluate", json=dict(supplier_id="S0005", timestamp="2025-11-01")).json
    for k in ["supplier_id", "prediction_timestamp", "disruption_probability", "calibrated_probability", "risk_category", "material_exposure",
              "inventory_coverage_days", "consequence_simulation", "recommended_action", "alternative_suppliers", "decision_rationale"]:
        assert k in r
    rep = client.get("/api/reports").json
    assert any(d["supplier_id"] == "S0005" for d in rep["decisions"])


@pytest.mark.parametrize("q,tool", [
    ("Which supplier is most critical?", "most_critical_suppliers"),
    ("What happens if S0122 fails for 45 days?", "supplier_failure_impact"),
    ("Which raw materials are at risk?", "materials_at_risk"),
    ("What is the current inventory coverage?", "inventory_coverage"),
    ("Which alternative supplier is suitable for S0003?", "alternative_suppliers"),
    ("Compare mild and severe disruption scenarios", "compare_scenarios"),
    ("Why was S0122 classified as high risk?", "explain_supplier_risk"),
])
def test_assistant_routes_to_tools(client, q, tool):
    r = client.post("/api/chat/query", json=dict(question=q)).json
    assert r["tool_calls"] and r["tool_calls"][0]["tool"] == tool
    assert len(r["answer"]) > 40


def test_assistant_help_and_empty(client):
    assert "Try:" in client.post("/api/chat/query", json=dict(question="hello there")).json["answer"]
    assert client.post("/api/chat/query", json=dict(question="")).status_code == 400


def test_upload_validation(client):
    csv = b"Supplier ID,supplier name,tier,primary_material_id,region,reliability_score\nS0001,A,1,M01,Goa,0.9\nS0001,B,5,M99,Goa,1.4\n"
    r = client.post("/api/data/upload", data={"target_table": "suppliers", "file": (io.BytesIO(csv), "s.csv")}, content_type="multipart/form-data").json
    assert r["suggested_mapping"]["Supplier ID"] == "supplier_id"
    v = r["validation"]
    assert v["status"] == "fail" and any("duplicate" in e for e in v["errors"]) and any("outside" in e for e in v["errors"])
    assert client.get("/api/data/history").json[0]["filename"] == "s.csv"
    bad = client.post("/api/data/upload", data={"target_table": "suppliers", "file": (io.BytesIO(b"x"), "s.txt")}, content_type="multipart/form-data")
    assert bad.status_code == 400
    assert len(client.get("/api/data/tables").json) == 14
