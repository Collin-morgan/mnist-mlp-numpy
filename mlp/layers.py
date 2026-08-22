import numpy as np

from . import initializers


class Parameter:
    """A weight array and the gradient the optimizer will use to update it."""

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class Layer:
    def forward(self, x, training=False):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        return []

    def __repr__(self):
        return "%s()" % type(self).__name__


class Dense(Layer):
    def __init__(self, n_in, n_out, weight_init="he_normal", rng=None):
        rng = np.random.default_rng() if rng is None else rng
        self.W = Parameter(initializers.get(weight_init)((n_in, n_out), rng))
        self.b = Parameter(np.zeros(n_out))
        self._x = None

    def forward(self, x, training=False):
        self._x = x
        return x @ self.W.value + self.b.value

    def backward(self, grad):
        self.W.grad = self._x.T @ grad
        self.b.grad = grad.sum(axis=0)
        return grad @ self.W.value.T

    def parameters(self):
        return [self.W, self.b]

    def __repr__(self):
        return "Dense(%d, %d)" % self.W.value.shape


class Dropout(Layer):
    def __init__(self, p=0.5, rng=None):
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p = p
        self.rng = np.random.default_rng() if rng is None else rng
        self._mask = None

    def forward(self, x, training=False):
        if not training or self.p == 0.0:
            self._mask = None
            return x
        # scale here so that prediction is just a normal forward pass
        self._mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self._mask

    def backward(self, grad):
        if self._mask is None:
            return grad
        return grad * self._mask

    def __repr__(self):
        return "Dropout(%.2f)" % self.p
