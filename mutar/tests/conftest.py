import pytest
import numpy as np


@pytest.fixture(scope='session', autouse=True)
def gaussian_data(request):
    rng = np.random.RandomState(42)
    n_tasks, n_samples, n_features = 2, 10, 25
    X = rng.randn(n_tasks, n_samples, n_features)
    coef = np.zeros((n_features, n_tasks))
    coef[:3] = 10.
    coef[:3, 0] = 10
    coef[4, 0] = - 10
    coef[5, 1] = - 10.
    y = np.array([x.dot(c) for x, c in zip(X, coef.T)])
    y += 0.5 * np.std(y) + rng.randn(n_tasks, n_samples)

    return X, y


@pytest.fixture(scope='session', autouse=True)
def identity_data(request):
    n_tasks, n_features = 2, 100
    X = n_tasks * [np.eye(n_features)]
    coef = np.zeros((n_features, n_tasks))
    coef[0, 0] = 10
    coef[50, 1] = -10

    y = np.array([x.dot(c) for x, c in zip(X, coef.T)])

    return X, y
