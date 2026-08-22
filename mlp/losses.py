import numpy as np

from .activations import log_softmax


class Loss:
    def forward(self, pred, y):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, pred, y):
        return self.forward(pred, y)


class SoftmaxCrossEntropy(Loss):
    """Softmax and cross entropy together, which makes the gradient just p - y."""

    def forward(self, logits, y):
        y = np.asarray(y)
        n = logits.shape[0]
        log_p = log_softmax(logits)
        self._probs = np.exp(log_p)
        self._y = y
        self._n = n
        return float(-log_p[np.arange(n), y].mean())

    def backward(self):
        grad = self._probs.copy()
        grad[np.arange(self._n), self._y] -= 1.0
        return grad / self._n


class MeanSquaredError(Loss):
    def forward(self, pred, y):
        y = np.asarray(y, dtype=float).reshape(pred.shape)
        self._diff = pred - y
        self._n = pred.shape[0]
        return float(0.5 * (self._diff ** 2).sum() / self._n)

    def backward(self):
        return self._diff / self._n
