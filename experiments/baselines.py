"""Compare the network with standard scikit-learn models on the same data."""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import (BOOTSTRAP_SEED, COLORS, N_BOOT, RESULTS, heading,
                                mnist_splits, save_fig, save_table, set_style)
from mlp import metrics

NETWORK = "neural network (784-256-128-10)"


def baseline_models():
    """Each model gets one setting tuned on the validation set, same as the network."""
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC

    return [
        ("majority class", lambda _: DummyClassifier(strategy="most_frequent"), [None]),
        ("logistic regression", lambda C: LogisticRegression(C=C, max_iter=2000),
         [0.01, 0.1, 1.0]),
        ("linear SVM", lambda C: LinearSVC(C=C, dual="auto"), [0.001, 0.01, 0.1]),
        ("k nearest neighbours", lambda k: KNeighborsClassifier(n_neighbors=k, n_jobs=-1),
         [1, 3, 5]),
        ("random forest", lambda n: RandomForestClassifier(n_estimators=n, n_jobs=-1,
                                                           random_state=0), [100, 300]),
    ]


def tune_and_fit(factory, values, X_tr, y_tr, X_val, y_val):
    best = None
    for value in values:
        model = factory(value)
        model.fit(X_tr, y_tr)
        score = metrics.accuracy(y_val, model.predict(X_val))
        if best is None or score > best[0]:
            best = (score, value, model)
    setting = "default" if best[1] is None else str(best[1])
    return best[2], setting, best[0]


def load_network_predictions():
    path = os.path.join(RESULTS, "mnist_predictions.npz")
    if not os.path.exists(path):
        raise SystemExit("run experiments.train_mnist first")
    stored = np.load(path)
    return stored["y_pred"], stored["y_test"]


def forest_plot(table, xmin=0.88):
    ordered = table.sort_values("accuracy")
    y = np.arange(len(ordered))
    centre = ordered["accuracy"].values
    lower = centre - ordered["ci_low"].values
    upper = ordered["ci_high"].values - centre
    colors = [COLORS[1] if name == NETWORK else COLORS[0] for name in ordered["model"]]

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    for i in range(len(ordered)):
        if centre[i] < xmin:
            ax.plot(xmin + 0.004, y[i], marker="<", markersize=6, color=COLORS[5])
            ax.annotate("%.4f, off the chart" % centre[i], xy=(xmin + 0.012, y[i]),
                        va="center", fontsize=7.5, color=COLORS[5])
            continue
        ax.errorbar(centre[i], y[i], xerr=[[lower[i]], [upper[i]]], fmt="o", markersize=5.5,
                    color=colors[i], ecolor=colors[i], elinewidth=1.3, capsize=3)
        ax.annotate("%.4f" % centre[i], xy=(centre[i], y[i]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model"])
    ax.set_ylim(-0.75, len(ordered) - 0.4)
    ax.set_xlim(xmin, 1.0)
    ax.set_xlabel("test accuracy with a 95% bootstrap interval")
    ax.set_title("MNIST test accuracy, all models trained on the same 50,000 images")
    return fig


def main():
    set_style()
    X_tr, y_tr, X_val, y_val, X_test, y_test = mnist_splits()
    net_pred, y_stored = load_network_predictions()
    assert np.array_equal(y_stored, y_test)

    predictions = {NETWORK: net_pred}
    rows = [{"model": NETWORK, "setting": "see train_mnist", "fit_seconds": np.nan}]

    heading("baselines, tuned on the validation set and fitted on %d images" % len(X_tr))
    for name, factory, values in baseline_models():
        start = time.time()
        model, setting, val_score = tune_and_fit(factory, values, X_tr, y_tr, X_val, y_val)
        elapsed = time.time() - start
        predictions[name] = model.predict(X_test)
        print("  %-22s %-8s validation %.4f  test %.4f  (%.0f seconds)"
              % (name, setting, val_score, metrics.accuracy(y_test, predictions[name]), elapsed))
        rows.append({"model": name, "setting": setting, "fit_seconds": elapsed})

    table = pd.DataFrame(rows)
    stats = []
    for name in table["model"]:
        point, lo, hi = metrics.bootstrap_ci(y_test, predictions[name], n_boot=N_BOOT,
                                             seed=BOOTSTRAP_SEED)
        stats.append({"model": name, "accuracy": point, "ci_low": lo, "ci_high": hi})
    table = table.merge(pd.DataFrame(stats), on="model")

    heading("McNemar test against the network")
    mcnemar_rows = []
    for name, pred in predictions.items():
        if name == NETWORK:
            continue
        n01, n10, chi2, p = metrics.mcnemar(y_test, net_pred, pred)
        mcnemar_rows.append({"model": name, "network_only": n01, "other_only": n10,
                             "chi2": chi2, "p_value": p})
        print("  vs %-22s network right %4d, other right %4d, chi2 %8.1f, p %s"
              % (name, n01, n10, chi2, "%.2g" % p if p > 1e-300 else "< 1e-300"))

    save_table(table, "baselines.csv")
    save_table(pd.DataFrame(mcnemar_rows), "baselines_mcnemar.csv")
    save_fig(forest_plot(table), "baselines.png")

    heading("summary")
    print(table.sort_values("accuracy", ascending=False)[["model", "accuracy", "ci_low",
                                                          "ci_high"]]
          .to_string(index=False, float_format=lambda v: "%.4f" % v))


if __name__ == "__main__":
    main()
