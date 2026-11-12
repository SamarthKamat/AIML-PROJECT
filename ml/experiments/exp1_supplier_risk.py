"""EXPERIMENT 1 - Supplier disruption prediction (next 30 days) + probability calibration
+ cost-based business decision policy.  Data: SYNTHETIC ecosystem.

Split (chronological, by prediction timestamp):
    train       : snapshots 2024-04-01 .. 2025-02-01   (fit models)
    embargo     : 2025-03-01 (dropped; its label window overlaps calibration features)
    calibration : snapshots 2025-04-01 .. 2025-06-01   (fit Platt / isotonic, choose method)
    test        : snapshots 2025-07-01 .. 2025-12-01   (untouched until final evaluation)

Run: python -m ml.experiments.exp1_supplier_risk
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from ml.common.config import (CAL_END, CAL_START, EXPDATA, FIGURES, MODELS, SEED, TEST_START, TRAIN_END)  # noqa: E402
from ml.common.tracking import binary_metrics, file_hash, log_run, save_predictions, verify_binary  # noqa: E402
from ml.data_engineering.features_supplier import FEATURES, build_snapshot_features, load_tables  # noqa: E402
from ml.decision.calibration import Calibrator  # noqa: E402
from ml.decision.policy import ACTIONS, choose_action  # noqa: E402
from ml.decision.exposure import ExposureModel  # noqa: E402

EXP_ID = "exp1_supplier_disruption"
TARGET = "disruption_next_30_days"
PALETTE = {"Logistic Regression": "#5b8def", "Decision Tree": "#f2a541", "Random Forest": "#3fb68b", "XGBoost": "#e05d5d"}


def make_models(pos_weight):
    return {
        "Logistic Regression": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                             LogisticRegression(C=0.5, max_iter=3000)),
        "Decision Tree": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                       DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, random_state=SEED)),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                       RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features="sqrt",
                                                              n_jobs=-1, random_state=SEED)),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=5, reg_lambda=1.0, eval_metric="logloss", random_state=SEED, n_jobs=4),
    }


def cv_calibration_choice(p_cal, y_cal):
    """Choose calibration method by 5-fold CV Brier score *within* the calibration split."""
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scores = {}
    for meth in ["none", "platt", "isotonic"]:
        s = []
        for a, b in skf.split(p_cal, y_cal):
            c = Calibrator(meth).fit(p_cal[a], y_cal[a]) if meth != "none" else Calibrator("none")
            s.append(brier_score_loss(y_cal[b], c.predict(p_cal[b])))
        scores[meth] = float(np.mean(s))
    return min(scores, key=scores.get), scores


def main():
    path = EXPDATA / "exp1_supplier_snapshots.csv"
    if not path.exists():
        build_snapshot_features(load_tables()).to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["snapshot_ts"])
    tr = df[df.snapshot_ts <= TRAIN_END]
    ca = df[(df.snapshot_ts >= CAL_START) & (df.snapshot_ts <= CAL_END)]
    te = df[df.snapshot_ts >= TEST_START]
    assert tr.snapshot_ts.max() < ca.snapshot_ts.min() and ca.snapshot_ts.max() < te.snapshot_ts.min()
    Xtr, ytr = tr[FEATURES], tr[TARGET].values
    Xca, yca = ca[FEATURES], ca[TARGET].values
    Xte, yte = te[FEATURES], te[TARGET].values

    models = make_models((1 - ytr.mean()) / ytr.mean())
    results, preds, cal_choice = {}, {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        p_ca = mdl.predict_proba(Xca)[:, 1]
        p_te = mdl.predict_proba(Xte)[:, 1]
        best, cv_scores = cv_calibration_choice(p_ca, yca)
        cal_choice[name] = dict(selected=best, cv_brier_on_calibration_split=cv_scores)
        results[name] = {}
        for meth in ["none", "platt", "isotonic"]:
            cal = Calibrator(meth).fit(p_ca, yca) if meth != "none" else Calibrator("none")
            pt = cal.predict(p_te)
            m = binary_metrics(yte, pt, 0.5)
            fails = verify_binary(yte, pt, m)
            assert not fails, fails
            results[name][meth] = m
            preds[(name, meth)] = pt
        joblib.dump({"model": mdl, "calibrator": Calibrator(best).fit(p_ca, yca) if best != "none" else Calibrator("none"),
                     "features": FEATURES, "calibration_method": best}, MODELS / f"exp1_{name.replace(' ', '_').lower()}.joblib")

    # ----- Business decision policy on the test split (primary model: XGBoost + selected calibration) -----
    primary = "XGBoost"
    meth = cal_choice[primary]["selected"]
    p_primary = preds[(primary, meth)]
    em_obs = ExposureModel(graph="observed")
    em_true = ExposureModel(graph="full")
    E_obs = em_obs.all_exposures()
    E_true = em_true.all_exposures()
    te = te.copy()
    te["p_raw"] = models[primary].predict_proba(Xte)[:, 1]
    te["p_cal"] = p_primary
    te["exposure_obs_inr"] = te.supplier_id.map(E_obs)
    te["exposure_true_inr"] = te.supplier_id.map(E_true)
    te["action"] = [choose_action(p, e)["action"] for p, e in zip(te.p_cal, te.exposure_obs_inr)]
    te["alert"] = te.action.isin(["contingency_review", "escalate_recovery"]).astype(int)
    policy_metrics = binary_metrics(yte, te.alert.values, 0.5)
    policy_metrics = {k: v for k, v in policy_metrics.items() if k in
                      ("n", "prevalence", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "specificity", "f1", "alerts")}
    assert not verify_binary(yte, te.alert.values, {**policy_metrics, "threshold": 0.5})

    def realised_cost(actions):
        cost = 0.0
        for a, y, e in zip(actions, yte, te.exposure_true_inr):
            c, r = ACTIONS[a]["cost_inr"], ACTIONS[a]["mitigation"]
            cost += c + y * (1 - r) * e
        return cost

    business = {
        "no_action_monitor_only": realised_cost(["routine_monitoring"] * len(te)),
        "contingency_review_for_all": realised_cost(["contingency_review"] * len(te)),
        "threshold_0.5_contingency": realised_cost(np.where(te.p_cal >= 0.5, "contingency_review", "routine_monitoring")),
        "expected_cost_policy (calibrated)": realised_cost(te.action),
        "expected_cost_policy (uncalibrated)": realised_cost([choose_action(p, e)["action"] for p, e in zip(te.p_raw, te.exposure_obs_inr)]),
        "oracle (perfect foresight, upper bound)": realised_cost(
            [min(ACTIONS, key=lambda a: ACTIONS[a]["cost_inr"] + y * (1 - ACTIONS[a]["mitigation"]) * e)
             for y, e in zip(yte, te.exposure_true_inr)]),
    }
    business = {k: round(v, 0) for k, v in business.items()}
    action_counts = te.action.value_counts().reindex(list(ACTIONS)).fillna(0).astype(int).to_dict()
    action_prec = te.groupby("action")[TARGET].mean().round(4).to_dict()

    save_predictions(EXP_ID, te[["supplier_id", "snapshot_ts", TARGET, "p_raw", "p_cal", "exposure_obs_inr",
                                 "exposure_true_inr", "action", "alert"]])

    # ----- SHAP -----
    import shap
    explainer = shap.TreeExplainer(models[primary])
    sv = explainer.shap_values(Xte)
    imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
    imp.round(5).to_csv(EXPERIMENTS_DIR() / "shap_importance.csv", header=["mean_abs_shap"])
    latest = te.snapshot_ts.max()
    m_latest = (te.snapshot_ts == latest).values
    shap_latest = pd.DataFrame(sv[m_latest], columns=FEATURES)
    shap_latest.insert(0, "supplier_id", te.supplier_id.values[m_latest])
    shap_latest.insert(1, "base_value", float(np.ravel(explainer.expected_value)[0]))
    shap_latest.to_csv(EXPERIMENTS_DIR() / "shap_latest_snapshot.csv", index=False)

    # ----- Figures -----
    figs(results, preds, yte, imp, te, business, cal_choice)

    split_counts = {k: dict(rows=int(len(v)), positives=int(v[TARGET].sum()), prevalence=round(float(v[TARGET].mean()), 4),
                            snapshots=[str(v.snapshot_ts.min().date()), str(v.snapshot_ts.max().date())])
                    for k, v in {"train": tr, "calibration": ca, "test": te}.items()}
    assert sum(v["rows"] for v in split_counts.values()) + int((df.snapshot_ts == "2025-03-01").sum()) == len(df)
    record = dict(
        dataset="datasets/experiment/exp1_supplier_snapshots.csv (derived from SYNTHETIC tables)",
        dataset_hash=file_hash(path), data_label="synthetic", target=TARGET,
        target_definition="qualifying disruption (capacity loss >= 25% and duration >= 3 days) starting in (t, t+30d]",
        prediction_timestamp="00:00 on the 1st of each month; features use records with reported/delivery date < t",
        features=FEATURES, splits=split_counts, embargo_snapshot="2025-03-01", random_seed=SEED,
        hyperparameters={k: str(v) for k, v in models.items()},
        calibration={"fit_on": "calibration split only", "selection": cal_choice,
                     "ece_binning": "10 equal-width bins (also 10 equal-mass bins reported)"},
        results=results, primary_model=primary, primary_calibration=meth,
        decision_policy={"actions": ACTIONS, "rule": "argmin_a cost_a + p_cal * (1 - mitigation_a) * exposure_obs",
                         "alert_definition": "action in {contingency_review, escalate_recovery}",
                         "policy_confusion": policy_metrics, "action_counts": action_counts,
                         "observed_disruption_rate_by_action": action_prec},
        business_outcome_test_inr=business,
        exposure_summary={"observed_graph_mean_by_tier": E_obs.groupby(em_obs.sup.tier).mean().round(0).to_dict(),
                          "true_graph_mean_by_tier": E_true.groupby(em_true.sup.tier).mean().round(0).to_dict()},
        shap_top10=imp.head(10).round(5).to_dict(),
        summary={"xgb_test_pr_auc": results[primary][meth]["pr_auc"], "xgb_test_brier": results[primary][meth]["brier"]},
    )
    log_run(EXP_ID, record)
    print(json.dumps({k: record[k] for k in ["splits", "primary_calibration", "business_outcome_test_inr"]}, indent=2))
    for n in results:
        print(n, {m: (results[n][m]["roc_auc"], results[n][m]["pr_auc"], results[n][m]["brier"], results[n][m]["ece_uniform10"]) for m in results[n]})
    print(policy_metrics, action_counts)


def EXPERIMENTS_DIR():
    from ml.common.config import EXPERIMENTS
    d = EXPERIMENTS / EXP_ID
    d.mkdir(parents=True, exist_ok=True)
    return d


def figs(results, preds, yte, imp, te, business, cal_choice):
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    # ROC + PR for selected calibration of each model
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for name in results:
        meth = cal_choice[name]["selected"]
        p = preds[(name, meth)]
        f, t, _ = roc_curve(yte, p)
        ax[0].plot(f, t, color=PALETTE[name], lw=1.4, label=f"{name} ({results[name][meth]['roc_auc']:.3f})")
        pr, rc, _ = precision_recall_curve(yte, p)
        ax[1].plot(rc, pr, color=PALETTE[name], lw=1.4, label=f"{name} ({results[name][meth]['pr_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
    ax[1].axhline(yte.mean(), ls=":", c="grey", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    ax[0].legend(fontsize=6, frameon=False)
    ax[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "exp1_roc_pr.pdf")
    fig.savefig(FIGURES / "exp1_roc_pr.png", dpi=200)
    plt.close(fig)
    # Reliability diagram for XGBoost
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for meth, c, lab in [("none", "#e05d5d", "Uncalibrated"), ("platt", "#5b8def", "Platt"), ("isotonic", "#3fb68b", "Isotonic")]:
        fr, mp = calibration_curve(yte, preds[("XGBoost", meth)], n_bins=10, strategy="quantile")
        ax.plot(mp, fr, marker="o", ms=3, c=c, lw=1.2,
                label=f"{lab} (Brier {results['XGBoost'][meth]['brier']:.4f}, ECE {results['XGBoost'][meth]['ece_uniform10']:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
    lim = max(0.5, float(np.quantile(preds[("XGBoost", "none")], 0.99)))
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted probability", ylabel="Observed frequency",
           title="XGBoost reliability (test, equal-mass bins)")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "exp1_reliability.pdf")
    fig.savefig(FIGURES / "exp1_reliability.png", dpi=200)
    plt.close(fig)
    # SHAP importance
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    top = imp.head(10)[::-1]
    ax.barh(top.index, top.values, color="#5b8def")
    ax.set(xlabel="mean |SHAP| (log-odds)", title="XGBoost feature attribution (test)")
    fig.tight_layout()
    fig.savefig(FIGURES / "exp1_shap.pdf")
    fig.savefig(FIGURES / "exp1_shap.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
