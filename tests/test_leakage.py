"""Point-in-time correctness of Experiment 1 features and splits."""
import pandas as pd
import pytest

from ml.common.config import CAL_END, CAL_START, TEST_START, TRAIN_END
from ml.data_engineering.features_supplier import build_snapshot_features, load_tables

TS = pd.Timestamp("2025-03-01")


@pytest.fixture(scope="module")
def tables():
    return load_tables()


def test_future_records_do_not_change_features(tables):
    base = build_snapshot_features(tables, [TS]).set_index("supplier_id")
    t2 = {k: v.copy() for k, v in tables.items()}
    d = t2["disruptions"]
    fut = d.reported_date >= TS
    d.loc[fut, "capacity_loss_pct"] = 1.0          # tamper with information not yet known at TS
    d.loc[fut, "duration_days"] = 99
    s = t2["shipments"]
    s.loc[s.actual_delivery_date >= TS, "delay_days"] = 50
    s.loc[s.actual_delivery_date >= TS, "is_delayed"] = 1
    after = build_snapshot_features(t2, [TS]).set_index("supplier_id")
    feats = [c for c in base.columns if c not in ("snapshot_ts", "disruption_next_30_days")]
    pd.testing.assert_frame_equal(base[feats], after[feats])


def test_hidden_edges_do_not_change_features(tables):
    base = build_snapshot_features(tables, [TS])
    t2 = {k: v.copy() for k, v in tables.items()}
    t2["supplier_relationships"] = t2["supplier_relationships"][t2["supplier_relationships"].visibility == "observed"]
    after = build_snapshot_features(t2, [TS])
    pd.testing.assert_frame_equal(base, after)


def test_label_uses_only_window(tables):
    f = build_snapshot_features(tables, [TS]).set_index("supplier_id")
    d = tables["disruptions"]
    q = d[(d.capacity_loss_pct >= 0.25) & (d.duration_days >= 3) & (d.start_date > TS) & (d.start_date <= TS + pd.Timedelta(days=30))]
    assert set(f.index[f.disruption_next_30_days == 1]) == set(q.supplier_id)


def test_splits_are_chronological_and_embargoed():
    assert pd.Timestamp(TRAIN_END) + pd.Timedelta(days=30) < pd.Timestamp(CAL_START)
    assert pd.Timestamp(CAL_END) + pd.Timedelta(days=30) <= pd.Timestamp(TEST_START)
