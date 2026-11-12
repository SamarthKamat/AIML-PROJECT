"""Experiment tracking, metric computation and independent verification.

Every experiment calls ``log_run`` which appends one JSON line to
experiments/registry.jsonl and writes experiments/<exp_id>/run.json.  Binary
classification metrics are computed from the confusion matrix with explicit
formulas and cross-checked against scikit-learn (``verify_binary``).
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from ml.common.config import EXPERIMENTS


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def ece(y, p, n_bins=10, strategy="uniform"):
    """Expected Calibration Error.

    uniform  : 10 equal-width bins on [0,1]; ECE = sum_b (n_b/N) |mean(y_b) - mean(p_b)|
    quantile : equal-mass bins (adaptive ECE)
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    ids = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = ids == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def binary_metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    yhat = (p >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = dict(threshold=float(threshold), n=int(len(y)), prevalence=float(y.mean()), tp=tp, fp=fp, tn=tn, fn=fn,
               accuracy=(tp + tn) / len(y), precision=prec, recall=rec, specificity=spec, f1=f1,
               alerts=tp + fp)
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(skm.roc_auc_score(y, p))
        out["pr_auc"] = float(skm.average_precision_score(y, p))
    out["brier"] = float(skm.brier_score_loss(y, p))
    out["ece_uniform10"] = ece(y, p, 10, "uniform")
    out["ece_quantile10"] = ece(y, p, 10, "quantile")
    out["log_loss"] = float(skm.log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


def verify_binary(y, p, m: dict) -> list[str]:
    """Independent reconciliation checks. Returns list of failures (empty = pass)."""
    fails = []
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(p) >= m["threshold"]).astype(int)
    cm = skm.confusion_matrix(y, yhat, labels=[0, 1])
    if [[m["tn"], m["fp"]], [m["fn"], m["tp"]]] != cm.tolist():
        fails.append("confusion matrix mismatch")
    if m["tp"] + m["fp"] + m["tn"] + m["fn"] != len(y):
        fails.append("confusion matrix does not sum to test size")
    if abs(m["prevalence"] - y.mean()) > 1e-6:
        fails.append("prevalence mismatch")
    checks = {"precision": skm.precision_score(y, yhat, zero_division=0), "recall": skm.recall_score(y, yhat),
              "f1": skm.f1_score(y, yhat, zero_division=0), "accuracy": skm.accuracy_score(y, yhat)}
    for k, v in checks.items():
        if abs(m[k] - v) > 1e-5:
            fails.append(f"{k} mismatch {m[k]} vs {v}")
    return fails


def regression_metrics(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return dict(n=int(len(y)), mae=round(float(skm.mean_absolute_error(y, yhat)), 6),
                rmse=round(float(np.sqrt(skm.mean_squared_error(y, yhat))), 6),
                r2=round(float(skm.r2_score(y, yhat)), 6))


def log_run(exp_id: str, record: dict) -> Path:
    d = EXPERIMENTS / exp_id
    d.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("experiment_id", exp_id)
    record["execution_timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["python"] = platform.python_version()
    (d / "run.json").write_text(json.dumps(record, indent=2, default=str))
    with open(EXPERIMENTS / "registry.jsonl", "a") as f:
        f.write(json.dumps({"experiment_id": exp_id, "execution_timestamp": record["execution_timestamp"],
                            "summary": record.get("summary", {})}, default=str) + "\n")
    return d


def save_predictions(exp_id: str, df: pd.DataFrame, name="test_predictions.csv"):
    d = EXPERIMENTS / exp_id
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / name, index=False)
