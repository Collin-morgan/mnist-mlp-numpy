import numpy as np
import pytest

import mlp
from mlp.gradcheck import check_layer, check_network, numeric_gradient, relative_error

TOL = 1e-8


def rng(seed=0):
    return np.random.default_rng(seed)


LAYERS = [
    ("dense", mlp.Dense(6, 4, rng=rng(1))),
    ("dense_xavier", mlp.Dense(6, 4, weight_init="xavier_uniform", rng=rng(2))),
    ("relu", mlp.ReLU()),
    ("tanh", mlp.Tanh()),
    ("sigmoid", mlp.Sigmoid()),
    ("dropout", mlp.Dropout(0.4, rng=rng(3))),
]


@pytest.mark.parametrize("name,layer", LAYERS, ids=[c[0] for c in LAYERS])
def test_layer_gradients(name, layer):
    x = rng(7).normal(size=(32, 6))
    for key, error in check_layer(layer, x).items():
        assert error < TOL, "%s %s: %.3e" % (name, key, error)


def test_softmax_cross_entropy_gradient():
    logits = rng(8).normal(size=(20, 5))
    y = rng(9).integers(0, 5, 20)
    loss = mlp.SoftmaxCrossEntropy()

    def total():
        return loss(logits, y)

    total()
    assert relative_error(loss.backward(), numeric_gradient(total, logits)) < TOL


def test_mse_gradient():
    pred = rng(10).normal(size=(20, 3))
    y = rng(11).normal(size=(20, 3))
    loss = mlp.MeanSquaredError()

    def total():
        return loss(pred, y)

    total()
    assert relative_error(loss.backward(), numeric_gradient(total, pred)) < TOL


def spread_biases(net, seed=20):
    """ReLU has a corner at zero and biases start at zero, so a dead unit sits right on
    the corner where the numerical gradient is not defined. Move them off zero first."""
    gen = rng(seed)
    for layer in net.layers:
        if isinstance(layer, mlp.Dense):
            layer.b.value = gen.normal(scale=0.1, size=layer.b.value.shape)


@pytest.mark.parametrize("kwargs", [
    {},
    {"dropout": 0.3},
    {"activation": mlp.Tanh},
])
def test_network_gradients(kwargs):
    net = mlp.mlp([6, 8, 5, 3], seed=14, **kwargs)
    spread_biases(net)
    x = rng(15).normal(size=(32, 6))
    y = rng(16).integers(0, 3, 32)
    for key, error in check_network(net, x, y).items():
        assert error < TOL, "%s: %.3e" % (key, error)


def test_big_logits_do_not_overflow():
    logits = np.array([[900.0, -900.0, 0.0], [-1e4, -1e4, -1e4]])
    loss = mlp.SoftmaxCrossEntropy()
    value = loss(logits, np.array([0, 1]))
    assert np.isfinite(value)
    assert np.isfinite(loss.backward()).all()
