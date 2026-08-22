import time

import numpy as np

from .activations import ReLU, log_softmax
from .layers import Dense, Dropout
from .losses import SoftmaxCrossEntropy


class Network:
    def __init__(self, layers, loss=None):
        self.layers = list(layers)
        self.loss = loss if loss is not None else SoftmaxCrossEntropy()
        self.history = {}

    def forward(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def n_parameters(self):
        return int(sum(p.value.size for p in self.parameters()))

    def predict_logits(self, x, batch_size=1024):
        out = [self.forward(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]
        return np.concatenate(out)

    def predict_proba(self, x, batch_size=1024):
        return np.exp(log_softmax(self.predict_logits(x, batch_size)))

    def predict(self, x, batch_size=1024):
        return self.predict_logits(x, batch_size).argmax(axis=1)

    def evaluate(self, x, y, batch_size=1024):
        logits = self.predict_logits(x, batch_size)
        loss = self.loss(logits, y)
        acc = float((logits.argmax(axis=1) == np.asarray(y)).mean())
        return loss, acc

    def get_weights(self):
        return [p.value.copy() for p in self.parameters()]

    def set_weights(self, weights):
        for p, w in zip(self.parameters(), weights):
            p.value = w.copy()

    def fit(self, X, y, optimizer, epochs=20, batch_size=128, validation_data=None,
            lr_schedule=None, early_stopping=None, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        n = len(X)
        history = {"epoch": [], "train_loss": [], "train_acc": [], "seconds": []}
        if validation_data is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        best_loss = np.inf
        best_weights = None
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(epochs):
            if lr_schedule is not None:
                optimizer.lr = float(lr_schedule(epoch))
            start = time.time()
            order = rng.permutation(n)
            total_loss = 0.0
            correct = 0

            for first in range(0, n, batch_size):
                idx = order[first:first + batch_size]
                xb, yb = X[idx], y[idx]
                logits = self.forward(xb, training=True)
                total_loss += self.loss(logits, yb) * len(idx)
                correct += int((logits.argmax(axis=1) == yb).sum())
                self.backward(self.loss.backward())
                optimizer.step()
                optimizer.zero_grad()

            history["epoch"].append(epoch)
            history["train_loss"].append(total_loss / n)
            history["train_acc"].append(correct / n)
            history["seconds"].append(time.time() - start)

            message = "epoch %3d  loss %.4f  acc %.4f" % (epoch, total_loss / n, correct / n)
            if validation_data is not None:
                val_loss, val_acc = self.evaluate(*validation_data)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                message += "  val_loss %.4f  val_acc %.4f" % (val_loss, val_acc)
                if val_loss < best_loss:
                    best_loss, best_weights, best_epoch = val_loss, self.get_weights(), epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            if verbose:
                print(message)

            if early_stopping is not None and epochs_without_improvement >= early_stopping:
                if verbose:
                    print("stopping early, going back to epoch %d" % best_epoch)
                self.set_weights(best_weights)
                break

        self.history = history
        return history

    def __repr__(self):
        return "Network([%s])" % ", ".join(repr(l) for l in self.layers)


def mlp(sizes, activation=ReLU, dropout=0.0, weight_init="he_normal", seed=0):
    """Build a plain feed-forward network, e.g. mlp([784, 256, 10])."""
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Dense(sizes[i], sizes[i + 1], weight_init=weight_init, rng=rng))
        if i == len(sizes) - 2:
            break
        layers.append(activation())
        if dropout:
            layers.append(Dropout(dropout, rng=rng))
    return Network(layers)
