import pytest
import numpy as np

# from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from mutar import IndLasso, IndRewLasso

from itertools import product


@pytest.mark.parametrize("fit_intercept, normalize", product([False, True],
                         [False, True]))
def test_indlasso(data, fit_intercept, normalize):

    X, y = data
    n_samples = y.shape[1]
    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    alpha_max = abs(Xty).max() / n_samples
    alpha = alpha_max * np.ones(len(X)) * 0.2

    est = IndLasso(alpha=alpha, fit_intercept=fit_intercept,
                   normalize=normalize)
    est.fit(X, y)
    assert hasattr(est, 'is_fitted_')

    var_y = np.var(y, axis=1)
    res = y - est.predict(X)
    r2 = 1 - (res ** 2).mean(1) / var_y
    scores = est.score(X, y)
    assert_allclose(r2, scores)


@pytest.mark.parametrize("fit_intercept, normalize",
                         product([False, True], [False, True]))
def test_indrewlasso(data, fit_intercept, normalize):

    X, y = data
    n_samples = y.shape[1]
    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    alpha_max = abs(Xty).max() / n_samples
    alpha = alpha_max * np.ones(len(X)) * 0.1

    est = IndRewLasso(alpha=alpha, fit_intercept=fit_intercept,
                      normalize=normalize)
    est.fit(X, y)
    assert hasattr(est, 'is_fitted_')

    var_y = np.var(y, axis=1)
    res = y - est.predict(X)
    r2 = 1 - (res ** 2).mean(1) / var_y
    scores = est.score(X, y)
    assert_allclose(r2, scores)

    # Make sure the loss function decreases
    objs = []
    for rw_step in [1, 2, 4, 6, 10]:
        est = IndRewLasso(alpha=alpha, fit_intercept=fit_intercept,
                          normalize=normalize, max_reweighting_iter=rw_step)
        est.coef_ = est.fit(X, y).coef_
        obj = 0.5 * ((y - est.predict(X)) ** 2).sum() / n_samples
        obj += (alpha[None, :] * abs(est.coef_) ** 0.5).sum()
        objs.append(obj)
    # assert objective decreases at every step
    assert np.diff(objs).max() < 1e-5
