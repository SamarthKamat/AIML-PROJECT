import pandas as pd

from backend.data_validation import validate_frame


def test_detects_errors():
    df = pd.DataFrame(dict(shipment_id=["A", "A", "B"], supplier_id=["S0001"] * 3, material_id=["M01"] * 3,
                           promised_date=["2025-01-01", "2025-01-02", "not-a-date"], actual_delivery_date=["2025-01-03", "2025-01-01", "2025-01-05"],
                           delay_days=[2, -1, 3], shipped_qty=[10, 10, 10], delivered_qty=[10, 12, 9]))
    r = validate_frame(df, "shipments")
    txt = " ".join(r["errors"])
    assert r["status"] == "fail"
    assert "duplicate primary keys" in txt and "invalid dates" in txt and "negative values in delay_days" in txt
    assert "delivered quantity cannot exceed" in txt


def test_clean_synthetic_tables_pass():
    from backend.data_validation import validate_directory
    from ml.common.config import SYN
    rep = validate_directory(SYN)
    assert all(r["status"] == "pass" for r in rep.values())
