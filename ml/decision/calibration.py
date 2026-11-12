"""Probability calibrators (fitted on the calibration split only)."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    """method in {'none', 'platt', 'isotonic'}.

    Platt: q = sigmoid(a * logit(p) + b), (a, b) by maximum likelihood on the calibration split.
    Isotonic: q = g(p) with g monotone non-decreasing, fitted by pool-adjacent-violators.
    """

    def __init__(self, method: str):
        self.method = method

    def fit(self, p, y):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        if self.method == "platt":
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def predict(self, p):
        p = np.asarray(p, float)
        if self.method == "none":
            return p
        p = np.clip(p, 1e-6, 1 - 1e-6)
        if self.method == "platt":
            return self.m.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
        return self.m.predict(p)
