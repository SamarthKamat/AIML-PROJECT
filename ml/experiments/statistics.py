"""Statistical transparency: bootstrap confidence intervals and paired significance tests.

* Exp 1: cluster bootstrap over SUPPLIERS (supplier-months of one supplier are not independent),
  B = 1,000 resamples, percentile 95 % CIs for ROC-AUC / PR-AUC / Brier / ECE, and paired
  differences (LR - XGB; raw - calibrated).
* Exp 3: cluster bootstrap over ORDERS (B = 1,000) for XGBoost - shipping-mode baseline.
* Scenarios: two-sided Wilcoxon signed-rank tests on the 500 paired (CRN) replications, Holm-adjusted
  over the nine policy comparisons; bootstrap CI of the mean paired saving; quantiles of paired savings.
* Ablation (robustness.py): same tests, separate Holm family of 12 comparisons.
Writes experiments/statistics.json.  Run: python -m ml.experiments.statistics
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ml.common.config import EXPDATA, EXPERIMENTS, MODELS, PROC, SEED, TEST_START
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
    """Cluster bootstrap over ORDERS (order lines of one order share date, customer and mode)."""
    r = json.loads((EXPERIMENTS / "exp3_delivery_delay_dataco" / "run.json").read_text())
    p = pd.read_csv(EXPERIMENTS / "exp3_delivery_delay_dataco" / "test_predictions.csv")
    oid = pd.read_csv(PROC / "dataco_shipments.csv", usecols=["Order Item Id", "Order Id"])
    p = p.merge(oid, on="Order Item Id", how="left")
    assert p["Order Id"].notna().all()
    rate = {k: v["late_rate"] for k, v in r["late_rate_by_mode_train"].items()}
    base = p["Shipping Mode"].map(rate).values
    y = p.is_late.values
    xgb = p.p_xgboost.values
    groups = p["Order Id"].values
    uniq, inv = np.unique(groups, return_inverse=True)
    members = pd.Series(np.arange(len(p))).groupby(inv).apply(np.array).values
    rng = np.random.default_rng(SEED)
    d, a = [], []
    for _ in range(B):
        ix = np.concatenate(members[rng.integers(0, len(uniq), len(uniq))])
        ax = roc_auc_score(y[ix], xgb[ix])
        a.append(ax)
        d.append(ax - roc_auc_score(y[ix], base[ix]))
    return dict(xgb_auc_ci=ci(a), delta_auc_xgb_minus_mode=dict(mean=round(float(np.mean(d)), 4), ci=ci(d),
                                                                     p=float(min(1.0, 2 * np.mean(np.asarray(d) <= 0)))),
                B=B, n=int(len(y)), n_orders=int(len(uniq)), bootstrap="cluster by order")


def _paired_rows(piv, pairs, sev, rng):
    rows = []
    for a, b in pairs:
        diff = (piv[a] - piv[b]).values          # > 0  <=>  second policy cheaper (saving of b over a)
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(B)]
        pval = 1.0 if np.allclose(diff, 0) else float(wilcoxon(piv[a], piv[b], zero_method="wilcox").pvalue)
        rows.append(dict(severity=sev, first=a, second=b, mean_saving=float(diff.mean()), ci=ci(boot),
                         median=float(np.median(diff)), p10=float(np.quantile(diff, .1)), p90=float(np.quantile(diff, .9)),
                         share_second_cheaper=float((diff > 0).mean()), share_tied=float((np.abs(diff) < 1).mean()), p_raw=pval))
    return rows


def _holm(rows):
    order = np.argsort([r["p_raw"] for r in rows])
    m, running = len(rows), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i]["p_raw"]))
        rows[i]["p_holm"] = running
    return rows


def scenarios():
    """Two-sided Wilcoxon signed-rank on 500 CRN pairs; Holm over the nine comparisons (one family).
    Saving = impact(first) - impact(second); positive => second policy cheaper.  CI = bootstrap of the mean."""
    reps = pd.read_csv(EXPERIMENTS / "exp4_5_scenarios" / "scenario_replications.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for sev, g in reps.groupby("severity", sort=False):
        piv = g.pivot(index="replication", columns="policy", values="financial_impact_inr")
        rows += _paired_rows(piv, [("A: No intervention", "B: Rule-based"), ("A: No intervention", "C: Optimisation"),
                                   ("B: Rule-based", "C: Optimisation")], sev, rng)
    for r in rows:
        r["comparison"] = f"{r['first'][0]} vs {r['second'][0]}"
    return _holm(rows)


def ablation():
    """Second family (Holm over its 12 comparisons): value of predicted links holding the decision logic fixed."""
    f = EXPERIMENTS / "robustness" / "ablation_replications.csv"
    if not f.exists():
        return []
    reps = pd.read_csv(f)
    rng = np.random.default_rng(SEED + 1)
    rows = []
    pairs = [("C-obs: MILP, observed graph", "C-pred: MILP, obs.+predicted"),
             ("B-obs: rule, observed graph", "B-pred: rule, obs.+predicted"),
             ("B-obs: rule, observed graph", "C-obs: MILP, observed graph"),
             ("C-pred: MILP, obs.+predicted", "C-oracle: MILP, true graph")]
    for sev, g in reps.groupby("severity", sort=False):
        piv = g.pivot(index="replication", columns="policy", values="financial_impact_inr")
        rows += _paired_rows(piv, pairs, sev, rng)
    return _holm(rows)


def main():
    res = dict(exp1=exp1(), exp3=exp3(), scenarios=scenarios(), ablation=ablation())
    (EXPERIMENTS / "statistics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"exp1_paired": res["exp1"]["paired"], "exp1_ci": {k: v for k, v in res["exp1"].items() if k in ("LR", "XGB", "XGB_raw")},
                      "exp3": res["exp3"], "scen": [(r["severity"], r["comparison"], round(r["mean_saving"] / 1e5, 2), round(r["median"] / 1e5, 2), f"{r['p_holm']:.2g}") for r in res["scenarios"]],
                      "abl": [(r["severity"], r["first"][:6], r["second"][:8], round(r["mean_saving"] / 1e5, 2), round(r["median"] / 1e5, 2), round(r["share_second_cheaper"], 2), f"{r['p_holm']:.2g}") for r in res["ablation"]]}, indent=1))


if __name__ == "__main__":
    main()
