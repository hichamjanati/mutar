"""Utils functions."""
import numpy as np


def residual(X, coef_, y):
    """Compute X dot theta."""
    R = y - np.array([x.dot(th) for x, th in zip(X, coef_.T)])
    return R
