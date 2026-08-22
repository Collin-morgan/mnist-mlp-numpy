"""Compare optimizers, tuning the learning rate separately for each one."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mlp
from experiments.common import COLORS, heading, mnist_splits, save_fig, save_table, set_style

TRAIN_SIZE = 10000
WIDTH = 256
EPOCHS = 25
BATCH_SIZE = 128
SEED = 0

SETTINGS = [
    ("SGD", mlp.SGD, {}, [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]),
    ("SGD + momentum", mlp.SGD, {"momentum": 0.9}, [0.001, 0.003, 0.01, 0.03, 0.1]),
    ("RMSProp", mlp.RMSProp, {}, [0.0001, 0.0003, 0.001, 0.003, 0.01]),
    ("Adam", mlp.Adam, {}, [0.0001, 0.0003, 0.001, 0.003, 0.01]),
]


def run(optimizer_cls, kwargs, lr, X_tr, y_tr, X_val, y_val):
    net = mlp.mlp([X_tr.shape[1], WIDTH, 10], seed=SEED)
    optimizer = optimizer_cls(net.parameters(), lr=lr, **kwargs)
    return net.fit(X_tr, y_tr, optimizer, epochs=EPOCHS, batch_size=BATCH_SIZE,
                   validation_data=(X_val, y_val), seed=SEED, verbose=False)


def sweep(X_tr, y_tr, X_val, y_val):
    rows = []
    best_curves = {}
    for name, cls, kwargs, learning_rates in SETTINGS:
        best = None
        for lr in learning_rates:
            history = run(cls, kwargs, lr, X_tr, y_tr, X_val, y_val)
            final = history["val_loss"][-1]
            rows.append({"optimizer": name, "lr": lr, "final_val_loss": final,
                         "final_val_error": 1 - history["val_acc"][-1]})
            if best is None or final < best[0]:
                best = (final, lr, history)
        best_curves[name] = best[2]
        print("  %-16s best learning rate %-8g validation loss %.4f  error %.4f"
              % (name, best[1], best[0], 1 - best[2]["val_acc"][-1]))
    return pd.DataFrame(rows), best_curves


def curve_figure(curves):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for i, (name, history) in enumerate(curves.items()):
        color = COLORS[i % len(COLORS)]
        axes[0].plot(history["epoch"], history["train_loss"], color=color, label=name)
        axes[1].plot(history["epoch"], history["val_loss"], color=color, label=name)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("training cross entropy")
    axes[0].set_title("Training loss at each optimizer's best learning rate")
    axes[1].set_ylabel("validation cross entropy")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    return fig


def sensitivity_figure(table):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, name in enumerate(table["optimizer"].unique()):
        sub = table[table["optimizer"] == name].sort_values("lr")
        color = COLORS[i % len(COLORS)]
        ax.plot(sub["lr"], sub["final_val_loss"], marker="o", markersize=4,
                color=color, label=name)
        best = sub.loc[sub["final_val_loss"].idxmin()]
        ax.scatter([best["lr"]], [best["final_val_loss"]], s=90, facecolors="none",
                   edgecolors=color, linewidths=1.2, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("validation cross entropy after %d epochs" % EPOCHS)
    ax.set_title("Which learning rates actually work")
    ax.legend(fontsize=8)
    return fig


def main():
    set_style()
    X_tr, y_tr, X_val, y_val, _, _ = mnist_splits()
    X_tr, y_tr = X_tr[:TRAIN_SIZE], y_tr[:TRAIN_SIZE]

    heading("learning rate sweep on %d training images" % TRAIN_SIZE)
    table, curves = sweep(X_tr, y_tr, X_val, y_val)
    save_table(table, "optimizer_sweep.csv")
    save_fig(curve_figure(curves), "optimizer_curves.png")
    save_fig(sensitivity_figure(table), "optimizer_sensitivity.png")


if __name__ == "__main__":
    main()
