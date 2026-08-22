import numpy as np


class Optimizer:
    def __init__(self, params, lr, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def gradient(self, p):
        if self.weight_decay:
            return p.grad + self.weight_decay * p.value
        return p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.value)

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.value) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            g = self.gradient(p)
            if self.momentum:
                self.velocity[i] = self.momentum * self.velocity[i] + g
                g = self.velocity[i]
            p.value -= self.lr * g


class RMSProp(Optimizer):
    def __init__(self, params, lr=0.001, rho=0.9, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.rho = rho
        self.eps = eps
        self.squared = [np.zeros_like(p.value) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            g = self.gradient(p)
            self.squared[i] = self.rho * self.squared[i] + (1.0 - self.rho) * g ** 2
            p.value -= self.lr * g / (np.sqrt(self.squared[i]) + self.eps)


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.value) for p in self.params]
        self.v = [np.zeros_like(p.value) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = self.gradient(p)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g ** 2
            # correct for the fact that m and v start at zero
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            p.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def step_decay(lr, drop=0.5, every=10):
    return lambda epoch: lr * drop ** (epoch // every)
