import numpy as np

from ml.common.tracking import binary_metrics, ece, verify_binary


def test_metrics_reconcile_with_confusion_matrix():
    rng = np.random.default_rng(0)
    y = rng.random(1000) < 0.1
    p = np.clip(y * 0.3 + rng.random(1000) * 0.5, 0, 1)
    m = binary_metrics(y, p, 0.4)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == 1000
    assert abs(m["precision"] - m["tp"] / (m["tp"] + m["fp"])) < 1e-6
    assert abs(m["recall"] - m["tp"] / (m["tp"] + m["fn"])) < 1e-6
    assert abs(m["specificity"] - m["tn"] / (m["tn"] + m["fp"])) < 1e-6
    assert abs(m["f1"] - 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])) < 1e-6
    assert verify_binary(y, p, m) == []


def test_ece_perfect_and_biased():
    y = np.array([0] * 90 + [1] * 10)
    assert ece(y, np.full(100, 0.1)) < 1e-9
    assert abs(ece(y, np.full(100, 0.3)) - 0.2) < 1e-9
