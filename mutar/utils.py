"""Utils for OT."""
import numpy as np

import numbers


def residual(X, coef_, y):
    """Compute X dot theta."""
    R = y - np.array([x.dot(th) for x, th in zip(X, coef_.T)])
    return R


def get_unsigned(x):
    x1 = np.clip(x, 0., None)
    x2 = - np.clip(x, None, 0.)
    return x1, x2


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
