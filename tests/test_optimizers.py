import numpy as np
import pytest

import mlp
from mlp.layers import Parameter


def minimize_quadratic(optimizer_cls, steps=200, start=5.0, **kwargs):
    """Minimize f(w) = 0.5 * w^2, so the gradient is just w."""
    p = Parameter(np.array([start]))
    opt = optimizer_cls([p], **kwargs)
    for _ in range(steps):
        p.grad = p.value.copy()
        opt.step()
        opt.zero_grad()
    return float(p.value[0])


@pytest.mark.parametrize("cls,kwargs", [
    (mlp.SGD, {"lr": 0.1}),
    (mlp.SGD, {"lr": 0.1, "momentum": 0.9}),
    (mlp.RMSProp, {"lr": 0.05}),
    (mlp.Adam, {"lr": 0.1}),
])
def test_optimizers_find_the_minimum(cls, kwargs):
    assert abs(minimize_quadratic(cls, **kwargs)) < 1e-3


def test_momentum_gets_there_faster():
    plain = abs(minimize_quadratic(mlp.SGD, steps=25, lr=0.05))
    with_momentum = abs(minimize_quadratic(mlp.SGD, steps=25, lr=0.05, momentum=0.9))
    assert with_momentum < plain


def test_adam_first_step_is_about_the_learning_rate():
    p = Parameter(np.array([1.0]))
    opt = mlp.Adam([p], lr=0.01)
    p.grad = np.array([1e6])
    opt.step()
    # after bias correction the first step does not depend on how big the gradient is
    assert abs(1.0 - p.value[0] - 0.01) < 1e-6


def test_weight_decay_shrinks_the_weights():
    p = Parameter(np.array([1.0]))
    opt = mlp.SGD([p], lr=0.1, weight_decay=0.5)
    opt.zero_grad()
    opt.step()
    assert p.value[0] < 1.0


def test_zero_grad_clears_the_gradients():
    p = Parameter(np.ones(3))
    opt = mlp.SGD([p], lr=0.1)
    p.grad = np.ones(3)
    opt.zero_grad()
    assert np.abs(p.grad).max() == 0.0


def test_step_decay_halves_the_rate():
    schedule = mlp.optimizers.step_decay(0.2, drop=0.5, every=4)
    assert schedule(0) == 0.2
    assert schedule(3) == 0.2
    assert abs(schedule(4) - 0.1) < 1e-12
