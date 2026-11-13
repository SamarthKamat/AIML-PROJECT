"""Statistical transparency: bootstrap confidence intervals and paired significance tests.

* Exp 1: cluster bootstrap over SUPPLIERS (supplier-months of one supplier are not independent),
  B = 1,000 resamples, percentile 95 % CIs for ROC-AUC / PR-AUC / Brier / ECE, and paired
  differences (LR - XGB; raw - calibrated).
* Exp 3: bootstrap over order lines (B = 1,000) for XGBoost - shipping-mode baseline.
* Scenarios: Wilcoxon signed-rank tests on the 500 paired (CRN) replications, Holm-adjusted
  over the nine policy comparisons.
Writes experiments/statistics.json.  Run: python -m ml.experiments.statistics
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ml.common.config import EXPDATA, EXPERIMENTS, MODELS, SEED, TEST_START
from ml.common.tracking import ece

B = 1000


def ci(a):
    a = np.asarray(a)
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def exp1():
    df = pd.read_csv(EXPDATA / "exp1_supplier_snapshots.csv", parse_dates=["snapshot_ts"])
    te = df[df.snapshot_ts >= TEST_START].reset_index(drop=True)
    y = te.disruption_next_30_days.values
    preds = {}
    for name, f in [("LR", "exp1_logistic_regression"), ("DT", "exp1_decision_tree"), ("RF", "exp1_random_forest"), ("XGB", "exp1_xgboost")]:
        b = joblib.load(MODELS / f"{f}.joblib")
        raw = b["model"].predict_proba(te[b["features"]])[:, 1]
        preds[name] = b["calibrator"].predict(raw)
        if name == "XGB":
            preds["XGB_raw"] = raw
    groups = te.supplier_id.values
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(SEED)
    stats = {k: {"roc": [], "pr": [], "brier": [], "ece": []} for k in preds}
    d_pr, d_roc, d_brier, d_ece = [], [], [], []
    for _ in range(B):
        samp = rng.choice(uniq, len(uniq), replace=True)
        ix = np.concatenate([idx_by[g] for g in samp])
        yy = y[ix]
        if yy.min() == yy.max():
            continue
        vals = {}
        for k, p in preds.items():
            pp = p[ix]
            vals[k] = (roc_auc_score(yy, pp), average_precision_score(yy, pp), brier_score_loss(yy, pp), ece(yy, pp))
            for j, m in enumerate(["roc", "pr", "brier", "ece"]):
                stats[k][m].append(vals[k][j])
        d_pr.append(vals["LR"][1] - vals["XGB"][1])
        d_roc.append(vals["LR"][0] - vals["XGB"][0])
        d_brier.append(vals["XGB_raw"][2] - vals["XGB"][2])
        d_ece.append(vals["XGB_raw"][3] - vals["XGB"][3])
    out = {k: {m: ci(v) for m, v in s.items()} for k, s in stats.items()}
    p_two = lambda d: float(min(1.0, 2 * min(np.mean(np.asarray(d) <= 0), np.mean(np.asarray(d) >= 0))))  # noqa: E731
    out["paired"] = {
        "PR-AUC LR - XGB": dict(mean=round(float(np.mean(d_pr)), 4), ci=ci(d_pr), p=p_two(d_pr)),
        "ROC-AUC LR - XGB": dict(mean=round(float(np.mean(d_roc)), 4), ci=ci(d_roc), p=p_two(d_roc)),
        "Brier raw - Platt (XGB)": dict(mean=round(float(np.mean(d_brier)), 5), ci=ci(d_brier), p=p_two(d_brier)),
        "ECE raw - Platt (XGB)": dict(mean=round(float(np.mean(d_ece)), 4), ci=ci(d_ece), p=p_two(d_ece)),
    }
    out["design"] = dict(bootstrap="cluster by supplier", B=B, n_test=int(len(te)), n_suppliers=int(len(uniq)), positives=int(y.sum()))
    return out


def exp3():
    r = json.loads((EXPERIMENTS / "exp3_delivery_delay_dataco" / "run.json").read_text())
    p = pd.read_csv(EXPERIMENTS / "exp3_delivery_delay_dataco" / "test_predictions.csv")
    rate = {k: v["late_rate"] for k, v in r["late_rate_by_mode_train"].items()}
    base = p["Shipping Mode"].map(rate).values
    y = p.is_late.values
    xgb = p.p_xgboost.values
    rng = np.random.default_rng(SEED)
    d, a = [], []
    n = len(y)
    for _ in range(B):
        ix = rng.integers(0, n, n)
        ax = roc_auc_score(y[ix], xgb[ix])
        a.append(ax)
        d.append(ax - roc_auc_score(y[ix], base[ix]))
    return dict(xgb_auc_ci=ci(a), delta_auc_xgb_minus_mode=dict(mean=round(float(np.mean(d)), 4), ci=ci(d),
                                                                     p=float(min(1.0, 2 * np.mean(np.asarray(d) <= 0)))), B=B, n=n)


def scenarios():
    reps = pd.read_csv(EXPERIMENTS / "exp4_5_scenarios" / "scenario_replications.csv")
    rows = []
    for sev, g in reps.groupby("severity", sort=False):
        piv = g.pivot(index="replication", columns="policy", values="financial_impact_inr")
        for a, b in [("A: No intervention", "B: Rule-based"), ("A: No intervention", "C: Optimisation"), ("B: Rule-based", "C: Optimisation")]:
            diff = (piv[a] - piv[b]).values
            if np.allclose(diff, 0):
                pval = 1.0
            else:
                pval = float(wilcoxon(piv[a], piv[b], zero_method="wilcox").pvalue)
            rows.append(dict(severity=sev, comparison=f"{a[0]} vs {b[0]}", mean_saving=float(diff.mean()), p_raw=pval,
                             share_second_cheaper=float((diff > 0).mean())))
    # Holm adjustment over all comparisons
    order = np.argsort([r["p_raw"] for r in rows])
    m = len(rows)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i]["p_raw"]))
        rows[i]["p_holm"] = running
    return rows


def main():
    res = dict(exp1=exp1(), exp3=exp3(), scenarios=scenarios())
    (EXPERIMENTS / "statistics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"exp1_paired": res["exp1"]["paired"], "exp1_ci": {k: v for k, v in res["exp1"].items() if k in ("LR", "XGB", "XGB_raw")},
                      "exp3": res["exp3"], "scen": [(r["severity"], r["comparison"], round(r["mean_saving"] / 1e5, 2), f"{r['p_holm']:.2g}") for r in res["scenarios"]]}, indent=1))


if __name__ == "__main__":
    main()
