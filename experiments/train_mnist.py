"""Train the main network on MNIST and save its test set predictions."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import mlp
from experiments.common import (BOOTSTRAP_SEED, COLORS, INK, N_BOOT, RESULTS, heading,
                                mnist_splits, save_array, save_fig, set_style)
from mlp import metrics

HIDDEN = [256, 128]
DROPOUT = 0.2
EPOCHS = 60
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
SEED = 0


def build(n_features, n_classes):
    return mlp.mlp([n_features] + HIDDEN + [n_classes], dropout=DROPOUT, seed=SEED)


def learning_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    epochs = history["epoch"]
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_ylabel("cross entropy")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, 1 - np.array(history["train_acc"]), label="train")
    axes[1].plot(epochs, 1 - np.array(history["val_acc"]), label="validation")
    axes[1].set_ylabel("error rate")
    axes[1].set_yscale("log")
    axes[1].set_title("Error")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle("784-%s-10, dropout %.1f, Adam" % ("-".join(str(h) for h in HIDDEN), DROPOUT),
                 y=1.02)
    return fig


def confusion_figure(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred, n_classes=10)
    shares = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.grid(False)
    im = ax.imshow(shares, cmap="Blues", vmin=0, vmax=0.02)
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color=INK if shares[i, j] < 0.012 else "white")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Test set mistakes by digit")
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    return fig


def error_gallery(X, y_true, y_pred, probs, n=24):
    wrong = np.flatnonzero(y_true != y_pred)
    order = wrong[np.argsort(-probs[wrong].max(axis=1))][:n]
    fig, axes = plt.subplots(3, 8, figsize=(9.0, 3.9))
    for ax, idx in zip(axes.ravel(), order):
        ax.imshow(X[idx].reshape(28, 28), cmap="gray_r")
        ax.set_title("%d called %d" % (y_true[idx], y_pred[idx]), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("The mistakes the network was most sure about", y=1.03)
    return fig


def main():
    set_style()
    X_tr, y_tr, X_val, y_val, X_test, y_test = mnist_splits()
    heading("data")
    print("  train %s  validation %s  test %s" % (X_tr.shape, X_val.shape, X_test.shape))

    net = build(X_tr.shape[1], 10)
    print("  %s" % net)
    print("  %d parameters" % net.n_parameters())

    optimizer = mlp.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    heading("training")
    history = net.fit(X_tr, y_tr, optimizer, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_data=(X_val, y_val),
                      lr_schedule=mlp.optimizers.step_decay(LR, drop=0.5, every=20),
                      early_stopping=PATIENCE, seed=SEED)

    probs = net.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)
    point, lo, hi = metrics.bootstrap_ci(y_test, y_pred, n_boot=N_BOOT, seed=BOOTSTRAP_SEED)

    heading("test set")
    print("  accuracy       %.4f  (95%% interval %.4f to %.4f)" % (point, lo, hi))
    print("  error rate     %.4f" % (1 - point))
    print("  cross entropy  %.4f" % metrics.cross_entropy(y_test, probs))
    print("  epochs run     %d" % len(history["epoch"]))
    print("  total time     %.0f seconds" % sum(history["seconds"]))

    save_array("mnist_predictions.npz", probs=probs, y_pred=y_pred, y_test=y_test)
    with open(os.path.join(RESULTS, "mnist_summary.json"), "w") as handle:
        json.dump({
            "architecture": [X_tr.shape[1]] + HIDDEN + [10],
            "parameters": net.n_parameters(),
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "epochs_run": len(history["epoch"]),
            "test_accuracy": point,
            "test_accuracy_interval": [lo, hi],
            "test_cross_entropy": metrics.cross_entropy(y_test, probs),
            "history": {k: list(map(float, v)) for k, v in history.items()},
        }, handle, indent=2)
    print("wrote results/mnist_summary.json")

    save_fig(learning_curves(history), "mnist_learning_curves.png")
    save_fig(confusion_figure(y_test, y_pred), "mnist_confusion.png")
    save_fig(error_gallery(X_test, y_test, y_pred, probs), "mnist_errors.png")


if __name__ == "__main__":
    main()
