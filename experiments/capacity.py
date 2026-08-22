"""How the number of hidden units and the amount of regularization affect overfitting."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mlp
from experiments.common import COLORS, heading, mnist_splits, save_fig, save_table, set_style
from mlp import metrics

TRAIN_SIZE = 10000
WIDTHS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
SEEDS = [0, 1, 2]
EPOCHS = 40
BATCH_SIZE = 128
LR = 1e-3
ABLATION_WIDTH = 512


def train_once(X_tr, y_tr, X_val, y_val, width, seed, dropout=0.0, weight_decay=0.0):
    net = mlp.mlp([X_tr.shape[1], width, 10], dropout=dropout, seed=seed)
    optimizer = mlp.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)
    net.fit(X_tr, y_tr, optimizer, epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val), seed=seed, verbose=False)
    train_error = 1 - metrics.accuracy(y_tr, net.predict(X_tr))
    val_error = 1 - metrics.accuracy(y_val, net.predict(X_val))
    return net, train_error, val_error


def width_sweep(X_tr, y_tr, X_val, y_val):
    rows = []
    for width in WIDTHS:
        for seed in SEEDS:
            net, train_error, val_error = train_once(X_tr, y_tr, X_val, y_val, width, seed)
            rows.append({"width": width, "seed": seed, "parameters": net.n_parameters(),
                         "train_error": train_error, "val_error": val_error})
        recent = [r for r in rows if r["width"] == width]
        print("  width %5d  train %.4f  validation %.4f"
              % (width, np.mean([r["train_error"] for r in recent]),
                 np.mean([r["val_error"] for r in recent])))
    return pd.DataFrame(rows)


def regularization_ablation(X_tr, y_tr, X_val, y_val):
    settings = [
        ("none", {}),
        ("L2 (1e-4)", {"weight_decay": 1e-4}),
        ("L2 (1e-3)", {"weight_decay": 1e-3}),
        ("dropout (0.3)", {"dropout": 0.3}),
        ("dropout (0.5)", {"dropout": 0.5}),
        ("dropout (0.3) + L2 (1e-4)", {"dropout": 0.3, "weight_decay": 1e-4}),
    ]
    rows = []
    for name, kwargs in settings:
        for seed in SEEDS:
            _, train_error, val_error = train_once(X_tr, y_tr, X_val, y_val, ABLATION_WIDTH,
                                                   seed, **kwargs)
            rows.append({"setting": name, "seed": seed,
                         "train_error": train_error, "val_error": val_error})
        recent = [r for r in rows if r["setting"] == name]
        print("  %-26s train %.4f  validation %.4f (sd %.4f)"
              % (name, np.mean([r["train_error"] for r in recent]),
                 np.mean([r["val_error"] for r in recent]),
                 np.std([r["val_error"] for r in recent])))
    return pd.DataFrame(rows)


def width_figure(frame):
    grouped = frame.groupby("width").agg(["mean", "std"])
    widths = grouped.index.values
    floor = 1.0 / TRAIN_SIZE

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for key, label, color in [("train_error", "train", COLORS[0]),
                              ("val_error", "validation", COLORS[1])]:
        mean = np.maximum(grouped[(key, "mean")].values, floor / 2)
        sd = grouped[(key, "std")].values
        axes[0].plot(widths, mean, marker="o", markersize=4, color=color, label=label)
        axes[0].fill_between(widths, np.maximum(mean - sd, floor / 2), mean + sd,
                             color=color, alpha=0.18, linewidth=0)
    axes[0].axhline(floor, color=COLORS[5], linestyle="--", linewidth=0.9)
    axes[0].text(widths[0], floor * 1.15, "one training image", fontsize=7.5, color=COLORS[5])
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("hidden units")
    axes[0].set_ylabel("error rate")
    axes[0].set_title("Error against network size (%d training images)" % TRAIN_SIZE)
    axes[0].legend()

    gap = grouped[("val_error", "mean")].values - grouped[("train_error", "mean")].values
    axes[1].plot(widths, gap, marker="o", markersize=4, color=COLORS[2])
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("hidden units")
    axes[1].set_ylabel("validation error - train error")
    axes[1].set_title("Gap between training and validation error")
    return fig


def ablation_figure(frame):
    summary = frame.groupby("setting").agg(["mean", "std"]).sort_values(("val_error", "mean"))
    labels = list(summary.index)
    y = np.arange(len(labels))
    mean = summary[("val_error", "mean")].values
    sd = summary[("val_error", "std")].values

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.axvline(summary.loc["none", ("val_error", "mean")], color=COLORS[5],
               linestyle="--", linewidth=0.9, zorder=0)
    ax.errorbar(mean, y, xerr=sd, fmt="o", markersize=5.5, color=COLORS[0],
                ecolor=COLORS[0], elinewidth=1.2, capsize=3)
    for i, value in enumerate(mean):
        ax.annotate("%.4f" % value, xy=(value, i), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.7)
    ax.set_xlim(mean.min() - sd.max() - 0.002, mean.max() + sd.max() + 0.002)
    ax.set_xlabel("validation error (average of %d seeds, bars are one sd)" % len(SEEDS))
    ax.set_title("Regularization at %d hidden units" % ABLATION_WIDTH)
    ax.text(summary.loc["none", ("val_error", "mean")], len(labels) - 0.62,
            " no regularization", fontsize=7.5, color=COLORS[5], va="center")
    return fig


def main():
    set_style()
    X_tr, y_tr, X_val, y_val, _, _ = mnist_splits()
    X_tr, y_tr = X_tr[:TRAIN_SIZE], y_tr[:TRAIN_SIZE]

    heading("hidden units, averaged over %d seeds" % len(SEEDS))
    widths = width_sweep(X_tr, y_tr, X_val, y_val)
    save_table(widths, "capacity_width_sweep.csv")
    save_fig(width_figure(widths), "capacity_width_sweep.png")

    summary = widths.groupby("width")[["train_error", "val_error"]].mean()
    memorized = summary.index[summary["train_error"] < 1e-9]
    if len(memorized):
        first = memorized.min()
        widest = summary.index.max()
        print("\n  the network fits the training set perfectly from %d units on, but "
              "validation error still\n  drops from %.4f to %.4f by %d units"
              % (first, summary.loc[first, "val_error"],
                 summary.loc[widest, "val_error"], widest))

    heading("regularization at %d hidden units" % ABLATION_WIDTH)
    ablation = regularization_ablation(X_tr, y_tr, X_val, y_val)
    save_table(ablation, "capacity_regularization.csv")
    save_fig(ablation_figure(ablation), "capacity_regularization.png")


if __name__ == "__main__":
    main()
