"""EXPERIMENT 3 - Shipment delay prediction on the REAL DataCo dataset.

Prediction timestamp: the order timestamp ('order date (DateOrders)').  Only
attributes known when the order is placed are used (shipping mode, scheduled
days, market/region, category, segment, payment type, quantity, price,
discount, sales value, calendar fields).  Outcome columns are excluded.

Split: chronological by order timestamp - first 60 % train, next 20 % validation
(calibration), final 20 % test.  Cancelled orders (no delivery outcome) removed.
Tasks: (a) is_late classification; (b) delay_days regression (real - scheduled).

Run: python -m ml.experiments.exp3_delivery_delay
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402

from ml.common.config import FIGURES, MODELS, PROC, SEED  # noqa: E402
from ml.common.tracking import binary_metrics, file_hash, log_run, regression_metrics, save_predictions, verify_binary  # noqa: E402
from ml.data_engineering.process_real import DATACO_FEATURES_CAT, DATACO_FEATURES_NUM  # noqa: E402

EXP_ID = "exp3_delivery_delay_dataco"
NUM = DATACO_FEATURES_NUM + ["order_month", "order_dow", "order_hour"]
CAT = DATACO_FEATURES_CAT


def prep(onehot=True):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CAT),
        ("num", StandardScaler(), NUM)])


def main():
    path = PROC / "dataco_shipments.csv"
    df = pd.read_csv(path, parse_dates=["order_ts"]).sort_values("order_ts").reset_index(drop=True)
    n = len(df)
    c1, c2 = df.order_ts.iloc[int(0.6 * n)], df.order_ts.iloc[int(0.8 * n)]
    tr, va, te = df[df.order_ts < c1], df[(df.order_ts >= c1) & (df.order_ts < c2)], df[df.order_ts >= c2]
    assert len(tr) + len(va) + len(te) == n
    X = lambda d: d[CAT + NUM]  # noqa: E731

    clf = {
        "Logistic Regression": make_pipeline(prep(), LogisticRegression(max_iter=3000, C=1.0)),
        "Random Forest": make_pipeline(prep(), RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1,
                                                                      random_state=SEED)),
        "XGBoost": make_pipeline(prep(), XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                                                       colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=4)),
    }
    cls_res, preds = {}, {}
    for name, m in clf.items():
        m.fit(X(tr), tr.is_late)
        pv = m.predict_proba(X(va))[:, 1]
        pt = m.predict_proba(X(te))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(pv, va.is_late)
        for meth, p in [("none", pt), ("isotonic", iso.predict(pt))]:
            r = binary_metrics(te.is_late, p, 0.5)
            assert not verify_binary(te.is_late, p, r)
            cls_res.setdefault(name, {})[meth] = r
        preds[name] = pt
    # rule baseline: "late if First Class or Second Class" learned from training late rates per mode
    mode_rate = tr.groupby("Shipping Mode").is_late.mean()
    rule_p = te["Shipping Mode"].map(mode_rate).values
    cls_res["Baseline: shipping-mode late rate"] = {"none": binary_metrics(te.is_late, rule_p, 0.5)}

    reg = {
        "Linear Regression": make_pipeline(prep(), LinearRegression()),
        "Random Forest": make_pipeline(prep(), RandomForestRegressor(n_estimators=200, min_samples_leaf=20, n_jobs=-1,
                                                                     random_state=SEED)),
        "XGBoost": make_pipeline(prep(), XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                                                      colsample_bytree=0.8, random_state=SEED, n_jobs=4)),
    }
    reg_res = {"Baseline: training mean": regression_metrics(te.delay_days, np.full(len(te), tr.delay_days.mean()))}
    mode_mean = tr.groupby("Shipping Mode").delay_days.mean()
    reg_res["Baseline: shipping-mode mean"] = regression_metrics(te.delay_days, te["Shipping Mode"].map(mode_mean))
    for name, m in reg.items():
        m.fit(X(tr), tr.delay_days)
        reg_res[name] = regression_metrics(te.delay_days, m.predict(X(te)))

    import joblib
    joblib.dump({"classifier": clf["XGBoost"], "regressor": reg["XGBoost"], "features_cat": CAT, "features_num": NUM,
                 "trained_on": "DataCo (real) train split", "train_end": str(tr.order_ts.max())}, MODELS / "exp3_dataco_xgb.joblib")
    out = te[["Order Item Id", "order_ts", "Shipping Mode", "is_late", "delay_days"]].copy()
    for k, v in preds.items():
        out[f"p_{k.replace(' ', '_').lower()}"] = v
    save_predictions(EXP_ID, out)

    # late rate by shipping mode (descriptive, training split)
    desc = tr.groupby("Shipping Mode").agg(late_rate=("is_late", "mean"), mean_delay=("delay_days", "mean"),
                                           n=("is_late", "size")).round(4)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    names = list(cls_res)
    vals = [cls_res[k]["none"]["pr_auc"] for k in names]
    ax.barh([k.replace("Baseline: ", "Base: ") for k in names], vals, color=["#5b8def", "#3fb68b", "#e05d5d", "#9aa5b1"])
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set(xlim=(0.5, 0.9), xlabel="PR-AUC (test)", title="DataCo late-delivery prediction (real data)")
    fig.tight_layout()
    fig.savefig(FIGURES / "exp3_dataco.pdf")
    fig.savefig(FIGURES / "exp3_dataco.png", dpi=200)
    plt.close(fig)

    rec = dict(dataset="datasets/processed/dataco_shipments.csv (REAL: DataCo, Mendeley Data V5, CC BY 4.0)",
               dataset_hash=file_hash(path), data_label="real", random_seed=SEED,
               prediction_timestamp="order timestamp; only order-time attributes used",
               excluded_leakage_columns=["Days for shipping (real)", "Delivery Status", "shipping date (DateOrders)",
                                         "Order Status", "Late_delivery_risk", "Benefit per order", "Order Profit Per Order"],
               features=CAT + NUM,
               splits={"train": dict(rows=len(tr), start=str(tr.order_ts.min()), end=str(tr.order_ts.max()), late_rate=round(tr.is_late.mean(), 4)),
                       "validation": dict(rows=len(va), start=str(va.order_ts.min()), end=str(va.order_ts.max()), late_rate=round(va.is_late.mean(), 4)),
                       "test": dict(rows=len(te), start=str(te.order_ts.min()), end=str(te.order_ts.max()), late_rate=round(te.is_late.mean(), 4))},
               classification=cls_res, regression=reg_res, late_rate_by_mode_train=desc.to_dict(orient="index"),
               summary={"xgb_test_pr_auc": cls_res["XGBoost"]["none"]["pr_auc"], "xgb_mae": reg_res["XGBoost"]["mae"]})
    log_run(EXP_ID, rec)
    print(json.dumps({"splits": rec["splits"], "cls": {k: {m: (v["roc_auc"], v["pr_auc"], v["f1"], v["brier"]) for m, v in d.items()} for k, d in cls_res.items()},
                      "reg": reg_res, "mode": rec["late_rate_by_mode_train"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
