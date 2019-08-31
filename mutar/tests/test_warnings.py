import pytest

from mutar import IndLasso, IndRewLasso, GroupLasso, DirtyModel


def test_indrewlasso(data):
    X, y = data
    est0 = IndRewLasso(alpha=len(X) * [0.])
    est1 = IndLasso(alpha=len(X) * [0.])
    est2 = GroupLasso(alpha=0.)
    est3 = DirtyModel(alpha=0., beta=0.)
    models = [est0, est1, est2, est3]
    for model in models:
        with pytest.warns(UserWarning, match='LinearRegression'):
            model.fit(X, y)
