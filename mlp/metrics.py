import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def cross_entropy(y_true, probs, eps=1e-12):
    y_true = np.asarray(y_true)
    p = probs[np.arange(len(y_true)), y_true]
    return float(-np.log(np.clip(p, eps, 1.0)).mean())


def confusion_matrix(y_true, y_pred, n_classes=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def bootstrap_ci(y_true, y_pred, n_boot=2000, alpha=0.05, seed=0):
    """Resample the test set to get a confidence interval for accuracy."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[i] = accuracy(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return accuracy(y_true, y_pred), float(lo), float(hi)


def mcnemar(y_true, pred_a, pred_b):
    """Test whether two classifiers differ, using only the cases they disagree on.

    Returns (n01, n10, chi2, p) where n01 counts cases A got right and B got wrong.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(pred_a) == y_true
    b = np.asarray(pred_b) == y_true
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return n01, n10, 0.0, 1.0
    chi2 = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    return n01, n10, float(chi2), chi2_pvalue(chi2)


def chi2_pvalue(x):
    """Upper tail of a chi-squared with one degree of freedom."""
    from math import erfc, sqrt

    return float(erfc(sqrt(x / 2.0)))
