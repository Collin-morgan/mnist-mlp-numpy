"""Check the hand-written gradients against numerical ones."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mlp
from experiments.common import COLORS, heading, save_fig, save_table, set_style
from mlp.gradcheck import check_layer, check_network, numeric_gradient, relative_error

TOL = 1e-8


def rng(seed):
    return np.random.default_rng(seed)


def check_everything():
    x = rng(0).normal(size=(32, 6))
    y = rng(1).integers(0, 3, 32)
    rows = []

    layers = [
        ("Dense", mlp.Dense(6, 4, rng=rng(2))),
        ("ReLU", mlp.ReLU()),
        ("Tanh", mlp.Tanh()),
        ("Sigmoid", mlp.Sigmoid()),
        ("Dropout(0.4)", mlp.Dropout(0.4, rng=rng(3))),
    ]
    for name, layer in layers:
        rows.append({"component": name, "error": max(check_layer(layer, x).values())})

    for name, loss, pred, target in [
        ("SoftmaxCrossEntropy", mlp.SoftmaxCrossEntropy(), rng(4).normal(size=(32, 3)), y),
        ("MeanSquaredError", mlp.MeanSquaredError(), rng(5).normal(size=(32, 3)),
         rng(6).normal(size=(32, 3))),
    ]:
        def total():
            return loss(pred, target)

        total()
        rows.append({"component": name,
                     "error": relative_error(loss.backward(), numeric_gradient(total, pred))})

    for name, kwargs in [("MLP 6-8-5-3", {}), ("MLP with dropout", {"dropout": 0.3}),
                         ("MLP with tanh", {"activation": mlp.Tanh})]:
        net = mlp.mlp([6, 8, 5, 3], seed=7, **kwargs)
        # ReLU has a corner at zero and biases start there, so nudge them off first
        gen = rng(8)
        for layer in net.layers:
            if isinstance(layer, mlp.Dense):
                layer.b.value = gen.normal(scale=0.1, size=layer.b.value.shape)
        rows.append({"component": name, "error": max(check_network(net, x, y).values())})

    return pd.DataFrame(rows)


def step_size_sweep():
    """The step size cannot be too big or too small, so there is a best value in between."""
    net = mlp.mlp([6, 12, 3], activation=mlp.Tanh, seed=20)
    x = rng(21).normal(size=(32, 6))
    y = rng(22).integers(0, 3, 32)

    def total():
        return net.loss(net.forward(x, training=True), y)

    total()
    net.backward(net.loss.backward())
    analytic = net.layers[0].W.grad.copy()

    steps = np.logspace(-1, -11, 21)
    errors = np.array([relative_error(analytic, numeric_gradient(total, net.layers[0].W.value, h))
                       for h in steps])
    return steps, errors


def main():
    set_style()
    table = check_everything()
    table["passed"] = table["error"] < TOL

    heading("gradient check (tolerance %.0e)" % TOL)
    for _, row in table.iterrows():
        print("  %-22s %.2e  %s" % (row["component"], row["error"],
                                    "ok" if row["passed"] else "FAILED"))
    print("\n%d of %d passed" % (table["passed"].sum(), len(table)))
    save_table(table, "gradient_check.csv")

    steps, errors = step_size_sweep()
    best = steps[np.argmin(errors)]
    print("\nsmallest error %.2e at step size %.0e" % (errors.min(), best))

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.loglog(steps, errors, marker="o", markersize=3.5, color=COLORS[0])
    ax.axvline(best, color=COLORS[5], linestyle="--", linewidth=0.9)
    ax.annotate("best step size\n%.0e" % best, xy=(best, errors.min()),
                xytext=(best * 6, errors.min() * 0.3), fontsize=8)
    ax.set_ylim(errors.min() * 0.05, errors.max() * 20)
    ax.set_xlabel("step size h")
    ax.set_ylabel("error against the analytic gradient")
    ax.set_title("Too large truncates, too small loses precision")
    save_fig(fig, "gradient_check_step_size.png")


if __name__ == "__main__":
    main()
