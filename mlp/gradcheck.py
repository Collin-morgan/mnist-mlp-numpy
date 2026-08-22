import numpy as np


def relative_error(analytic, numeric):
    """Compare two gradients on a scale that does not depend on their size."""
    a = np.asarray(analytic, dtype=float).ravel()
    b = np.asarray(numeric, dtype=float).ravel()
    denominator = np.linalg.norm(a) + np.linalg.norm(b)
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(a - b) / denominator)


def numeric_gradient(f, x, h=1e-5):
    """Estimate df/dx with a central difference, one entry of x at a time."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        i = it.multi_index
        original = x[i]
        x[i] = original + h
        f_plus = f()
        x[i] = original - h
        f_minus = f()
        x[i] = original
        grad[i] = (f_plus - f_minus) / (2.0 * h)
        it.iternext()
    return grad


def reseed(layers, seed=0):
    """Dropout picks a new random mask every forward pass, which would make the two
    finite difference evaluations use different masks. Reseeding keeps the mask fixed."""
    for layer in layers:
        if hasattr(layer, "rng"):
            layer.rng = np.random.default_rng(seed)


def check_layer(layer, x, h=1e-5):
    """Check one layer's gradients. Returns the error for the input and each parameter."""
    x = np.array(x, dtype=float)
    out = layer.forward(x, training=True)
    upstream = np.random.default_rng(1234).normal(size=out.shape)

    def total():
        reseed([layer])
        return float((layer.forward(x, training=True) * upstream).sum())

    reseed([layer])
    layer.forward(x, training=True)
    analytic_dx = layer.backward(upstream)
    analytic_params = [p.grad.copy() for p in layer.parameters()]

    errors = {"input": relative_error(analytic_dx, numeric_gradient(total, x, h))}
    for i, p in enumerate(layer.parameters()):
        errors["param_%d" % i] = relative_error(analytic_params[i],
                                                numeric_gradient(total, p.value, h))
    return errors


def check_network(net, x, y, h=1e-5):
    """Check every weight in a network against the numerical gradient of the loss."""
    x = np.array(x, dtype=float)

    def total():
        reseed(net.layers)
        return net.loss(net.forward(x, training=True), y)

    total()
    net.backward(net.loss.backward())
    analytic = [p.grad.copy() for p in net.parameters()]

    errors = {}
    for i, p in enumerate(net.parameters()):
        errors["param_%d" % i] = relative_error(analytic[i], numeric_gradient(total, p.value, h))
    return errors
