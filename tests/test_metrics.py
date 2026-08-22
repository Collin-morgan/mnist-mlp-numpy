import numpy as np

from mlp import metrics


def test_accuracy_matches_the_confusion_matrix():
    y = np.array([0, 1, 2, 2, 1, 0])
    pred = np.array([0, 1, 2, 1, 1, 2])
    cm = metrics.confusion_matrix(y, pred)
    assert cm.sum() == len(y)
    assert np.trace(cm) / cm.sum() == metrics.accuracy(y, pred)


def test_cross_entropy_matches_hand_calculation():
    y = np.array([0, 1])
    probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    expected = -(np.log(0.7) + np.log(0.6)) / 2
    assert abs(metrics.cross_entropy(y, probs) - expected) < 1e-12


def test_bootstrap_interval_contains_the_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 1000)
    pred = np.where(rng.random(1000) < 0.8, y, 1 - y)
    point, lo, hi = metrics.bootstrap_ci(y, pred, n_boot=500, seed=3)
    assert lo < point < hi
    assert hi - lo < 0.1


def test_mcnemar_ignores_mistakes_both_models_make():
    y = np.arange(100) % 2
    a, b = y.copy(), y.copy()
    a[:10] = 1 - a[:10]
    b[:10] = 1 - b[:10]
    n01, n10, chi2, p = metrics.mcnemar(y, a, b)
    assert (n01, n10) == (0, 0)
    assert p == 1.0


def test_mcnemar_finds_a_one_sided_difference():
    y = np.zeros(200, dtype=int)
    a, b = y.copy(), y.copy()
    b[:30] = 1
    n01, n10, chi2, p = metrics.mcnemar(y, a, b)
    assert (n01, n10) == (30, 0)
    assert p < 1e-6
