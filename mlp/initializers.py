import numpy as np


def he_normal(shape, rng):
    """Scaling that keeps the activation variance stable through ReLU layers."""
    fan_in = shape[0]
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)


def xavier_uniform(shape, rng):
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


def normal(shape, rng, scale=0.01):
    return rng.normal(0.0, scale, size=shape)


def zeros(shape, rng=None):
    return np.zeros(shape)


INITIALIZERS = {
    "he_normal": he_normal,
    "xavier_uniform": xavier_uniform,
    "normal": normal,
    "zeros": zeros,
}


def get(name):
    if callable(name):
        return name
    if name not in INITIALIZERS:
        raise ValueError("unknown initializer %r, options are %s" % (name, sorted(INITIALIZERS)))
    return INITIALIZERS[name]
