import os
import warnings

import matplotlib
import numpy as np

# numpy's matmul sometimes reports these on macOS even though the results are fine
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")

N_BOOT = 2000
BOOTSTRAP_SEED = 2

INK = "#1b1b1b"
COLORS = ["#3d5a80", "#ee6c4d", "#5c946e", "#a15ea1", "#c9a227", "#7f8c8d"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.prop_cycle": plt.cycler(color=COLORS),
    })


def save_fig(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote figures/%s" % name)


def save_table(frame, name):
    os.makedirs(RESULTS, exist_ok=True)
    frame.to_csv(os.path.join(RESULTS, name), index=False)
    print("wrote results/%s" % name)


def save_array(name, **arrays):
    os.makedirs(RESULTS, exist_ok=True)
    np.savez_compressed(os.path.join(RESULTS, name), **arrays)
    print("wrote results/%s" % name)


def mnist_splits(val_size=10000, seed=0):
    from mlp.data import load_mnist, train_val_split

    X_train, y_train, X_test, y_test = load_mnist()
    X_tr, y_tr, X_val, y_val = train_val_split(X_train, y_train, val_size=val_size, seed=seed)
    return X_tr, y_tr, X_val, y_val, X_test, y_test


def heading(text):
    print("\n" + text)
    print("-" * len(text))
