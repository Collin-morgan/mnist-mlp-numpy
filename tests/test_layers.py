import numpy as np
import pytest

import mlp


def test_dense_forward_matches_manual_calculation():
    rng = np.random.default_rng(0)
    layer = mlp.Dense(6, 4, rng=rng)
    x = rng.normal(size=(20, 6))
    expected = np.einsum("ij,jk->ik", x, layer.W.value) + layer.b.value
    assert np.abs(layer.forward(x) - expected).max() < 1e-12


def test_dense_parameter_shapes():
    layer = mlp.Dense(5, 3)
    assert layer.W.value.shape == (5, 3)
    assert layer.b.value.shape == (3,)
    assert len(layer.parameters()) == 2


def test_he_initialization_keeps_variance_stable():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4096, 256))
    h = x
    for _ in range(6):
        dense = mlp.Dense(256, 256, weight_init="he_normal", rng=rng)
        h = mlp.ReLU().forward(dense.forward(h))
    assert 0.5 < h.var() / x.var() < 2.0


def test_tiny_weights_make_the_signal_vanish():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2048, 256))
    h = x
    for _ in range(6):
        dense = mlp.Dense(256, 256, weight_init="normal", rng=rng)
        h = mlp.ReLU().forward(dense.forward(h))
    assert h.var() / x.var() < 1e-6


def test_dropout_does_nothing_when_predicting():
    x = np.random.default_rng(3).normal(size=(50, 8))
    layer = mlp.Dropout(0.5, rng=np.random.default_rng(4))
    assert np.array_equal(layer.forward(x, training=False), x)


def test_dropout_keeps_the_average_the_same():
    x = np.ones((20000, 4))
    out = mlp.Dropout(0.4, rng=np.random.default_rng(5)).forward(x, training=True)
    assert abs(out.mean() - 1.0) < 0.01


def test_dropout_rejects_bad_probability():
    with pytest.raises(ValueError):
        mlp.Dropout(1.0)


def test_softmax_rows_are_a_probability_distribution():
    from mlp.activations import softmax

    p = softmax(np.random.default_rng(8).normal(scale=50.0, size=(30, 6)))
    assert np.abs(p.sum(axis=1) - 1.0).max() < 1e-12
    assert (p >= 0).all()


def test_relu_zeroes_negatives():
    x = np.array([[-2.0, -0.5, 0.0, 1.5]])
    assert np.array_equal(mlp.ReLU().forward(x), np.array([[0.0, 0.0, 0.0, 1.5]]))
