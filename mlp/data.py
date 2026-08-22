import numpy as np


def load_mnist(flatten=True, scale=True, cache_dir=None):
    """Standard 60k/10k MNIST split, fetched from OpenML and cached on disk."""
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False,
                        data_home=cache_dir)
    X = X.astype(np.float64)
    y = np.asarray(y).astype(np.int64)
    if scale:
        X /= 255.0
    if not flatten:
        X = X.reshape(-1, 28, 28)
    return X[:60000], y[:60000], X[60000:], y[60000:]


def train_val_split(X, y, val_size=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    val_idx, train_idx = idx[:val_size], idx[val_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def standardize(X_train, *others):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    out = [(X_train - mean) / std]
    out.extend((X - mean) / std for X in others)
    return out if len(out) > 1 else out[0]


def one_hot(y, n_classes=None):
    y = np.asarray(y)
    n_classes = int(y.max()) + 1 if n_classes is None else n_classes
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out
