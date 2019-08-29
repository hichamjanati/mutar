import pytest
import numpy as np


@pytest.fixture(scope='session', autouse=True)
def data(request):
    rng = np.random.RandomState(42)
    n_tasks, n_samples, n_features = 2, 10, 20
    X = rng.randn(n_tasks, n_samples, n_features)
    coef = np.zeros((n_features, n_tasks))
    coef[:3] = 10.
    coef[4, 0] = - 10.
    coef[5, 1] = - 10.
    y = np.array([x.dot(c) for x, c in zip(X, coef.T)])
    y += 0.2 * np.std(y) + rng.randn(n_tasks, n_samples)

    return X, y
