import pytest

from mutar import (IndLasso, IndRewLasso, GroupLasso, DirtyModel,
                   MultiLevelLasso)


def test_zero_alpha(data):
    X, y = data
    est0 = IndRewLasso(alpha=len(X) * [0.])
    est1 = IndLasso(alpha=len(X) * [0.])
    est2 = GroupLasso(alpha=0.)
    est3 = DirtyModel(alpha=0., beta=0.)
    est4 = MultiLevelLasso(alpha=0.)

    models = [est0, est1, est2, est3, est4]
    for model in models:
        with pytest.warns(UserWarning, match='LinearRegression'):
            model.fit(X, y)


def test_slow_convergence(data):
    X, y = data
    max_iter = 5
    est0 = IndRewLasso(alpha=len(X) * [0.01], max_iter=max_iter)
    est1 = IndLasso(alpha=len(X) * [0.01], max_iter=max_iter)
    est2 = GroupLasso(alpha=0.01, max_iter=max_iter)
    est3 = DirtyModel(alpha=0.01, beta=0.01, max_iter=max_iter)
    est4 = MultiLevelLasso(alpha=0.01, max_iter=max_iter)

    models = [est0, est1, est2, est3, est4]
    for model in models:
        with pytest.warns(UserWarning, match='number of iterations'):
            model.fit(X, y)
