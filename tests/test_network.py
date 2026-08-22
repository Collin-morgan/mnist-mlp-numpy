import numpy as np

import mlp


def spiral(n_per_class=200, n_classes=3, noise=0.2, seed=0):
    """Spiral dataset, a standard toy problem that a straight line cannot separate."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * n_classes, 2))
    y = np.zeros(n_per_class * n_classes, dtype=int)
    for c in range(n_classes):
        idx = slice(n_per_class * c, n_per_class * (c + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class)
        t = t + rng.normal(scale=noise, size=n_per_class)
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = c
    return X, y


def test_network_learns_the_spiral():
    X, y = spiral()
    net = mlp.mlp([2, 64, 64, 3], seed=1)
    net.fit(X, y, mlp.Adam(net.parameters(), lr=0.02), epochs=120, batch_size=64, verbose=False)
    assert mlp.metrics.accuracy(y, net.predict(X)) > 0.97


def test_one_layer_cannot_learn_the_spiral():
    X, y = spiral()
    net = mlp.Network([mlp.Dense(2, 3, rng=np.random.default_rng(2))])
    net.fit(X, y, mlp.Adam(net.parameters(), lr=0.02), epochs=120, batch_size=64, verbose=False)
    assert mlp.metrics.accuracy(y, net.predict(X)) < 0.6


def test_predicted_probabilities_add_up_to_one():
    X, _ = spiral(n_per_class=50)
    probs = mlp.mlp([2, 16, 3], seed=3).predict_proba(X)
    assert probs.shape == (len(X), 3)
    assert np.abs(probs.sum(axis=1) - 1.0).max() < 1e-12


def test_training_loss_goes_down():
    X, y = spiral(n_per_class=50)
    net = mlp.mlp([2, 16, 3], seed=4)
    history = net.fit(X, y, mlp.SGD(net.parameters(), lr=0.1), epochs=7, batch_size=32,
                      validation_data=(X, y), verbose=False)
    assert len(history["train_loss"]) == 7
    assert len(history["val_acc"]) == 7
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_early_stopping_keeps_the_best_weights():
    X, y = spiral(n_per_class=60)
    X_train, y_train, X_val, y_val = X[::2], y[::2], X[1::2], y[1::2]
    net = mlp.mlp([2, 128, 128, 3], seed=5)
    net.fit(X_train, y_train, mlp.Adam(net.parameters(), lr=0.05), epochs=200, batch_size=16,
            validation_data=(X_val, y_val), early_stopping=5, verbose=False)
    assert abs(net.evaluate(X_val, y_val)[0] - min(net.history["val_loss"])) < 1e-9
    assert len(net.history["epoch"]) < 200


def test_dropout_is_off_when_predicting():
    X, _ = spiral(n_per_class=40)
    net = mlp.mlp([2, 32, 3], dropout=0.5, seed=6)
    assert np.array_equal(net.predict_proba(X), net.predict_proba(X))


def test_parameter_count():
    net = mlp.mlp([4, 10, 10, 3], seed=8)
    assert net.n_parameters() == (4 * 10 + 10) + (10 * 10 + 10) + (10 * 3 + 3)
