import numpy as np

from .layers import Layer


class ReLU(Layer):
    def forward(self, x, training=False):
        self._positive = x > 0
        return x * self._positive

    def backward(self, grad):
        return grad * self._positive


class Tanh(Layer):
    def forward(self, x, training=False):
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad):
        return grad * (1.0 - self._out ** 2)


class Sigmoid(Layer):
    def forward(self, x, training=False):
        self._out = sigmoid(x)
        return self._out

    def backward(self, grad):
        return grad * self._out * (1.0 - self._out)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def log_softmax(z):
    # subtracting the row max keeps exp() from overflowing on large logits
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def softmax(z):
    return np.exp(log_softmax(z))
