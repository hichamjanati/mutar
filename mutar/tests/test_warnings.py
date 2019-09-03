import pytest

from mutar import (IndLasso, IndRewLasso, GroupLasso, DirtyModel,
                   MultiLevelLasso, ReMTW, MTW)
from sklearn.exceptions import ConvergenceWarning


def test_zero_alpha(gaussian_data):
    X, y = gaussian_data
    est0 = IndRewLasso(alpha=len(X) * [0.])
    est1 = IndLasso(alpha=len(X) * [0.])
    est2 = GroupLasso(alpha=0.)
    est3 = DirtyModel(alpha=0., beta=0.)
    est4 = MultiLevelLasso(alpha=0.)
    est5 = ReMTW(alpha=0., beta=0.)
    est6 = MTW(alpha=0., beta=0.)

    models = [est0, est1, est2, est3, est4, est5, est6]
    for model in models:
        with pytest.warns(UserWarning, match='LinearRegression'):
            model.fit(X, y)


def test_slow_convergence(gaussian_data):
    X, y = gaussian_data
    max_iter = 1
    est0 = IndRewLasso(alpha=len(X) * [0.01], max_iter=max_iter)
    est1 = IndLasso(alpha=len(X) * [0.01], max_iter=max_iter)
    est2 = GroupLasso(alpha=0.01, max_iter=max_iter)
    est3 = DirtyModel(alpha=0.01, beta=0.01, max_iter=max_iter)
    est4 = MultiLevelLasso(alpha=0.01, max_iter=max_iter)
    est5 = ReMTW(alpha=0.01, beta=0.01, max_iter=max_iter)
    est6 = MTW(alpha=0.01, beta=0.01, max_iter=max_iter)

    models = [est0, est1, est2, est3, est4, est5, est6]
    for model in models:
        with pytest.warns(ConvergenceWarning, match='number of iterations'):
            model.fit(X, y)


def test_not_converged_reweighting(gaussian_data):
    X, y = gaussian_data
    m = 2
    est0 = IndRewLasso(alpha=len(X) * [0.1], max_iter_reweighting=m)
    est1 = ReMTW(alpha=0., beta=0.1, max_iter_reweighting=m)

    models = [est0, est1]
    for model in models:
        with pytest.warns(UserWarning, match='reweighting'):
            model.fit(X, y)


def test_log_domain(identity_data):
    X, y = identity_data
    _, n_features = X[0].shape

    # catch log-domain stabilization with tiny epsilon
    est1 = ReMTW(alpha=0.01, beta=0.01, max_iter=20, gamma=1., epsilon=1e-5)

    models = [est1]
    for model in models:
        with pytest.warns(UserWarning, match='log-domain'):
            model.fit(X, y)
