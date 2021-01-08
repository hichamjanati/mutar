import pytest
import numpy as np

from itertools import product

# from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from sklearn.linear_model import MultiTaskLasso

from mutar import DirtyModel, GroupLasso


@pytest.mark.parametrize("fit_intercept, normalize, alpha, beta, positive",
                         product([False, True], [False, True],
                                 [0.1, 0.5], [0.1, 0.5], [False, True]))
def test_dirty(gaussian_data, fit_intercept, normalize, alpha, beta, positive):

    X, y = gaussian_data
    n_samples = y.shape[1]
    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    alpha_max = np.linalg.norm(Xty, axis=0).max()
    beta_max = abs(Xty).max()
    alpha *= alpha_max / n_samples
    beta *= beta_max / n_samples
    est = DirtyModel(alpha=alpha, beta=beta,
                     fit_intercept=fit_intercept, normalize=normalize,
                     positive=positive)
    est.fit(X, y)
    assert hasattr(est, 'is_fitted_')
    if est.alpha <= est.beta:
        assert_allclose(est.coef_specific_, 0.)
    elif est.alpha > len(X) ** 0.5 * est.beta:
        assert_allclose(est.coef_shared_, 0.)

    if positive:
        assert est.coef_.min() >= 0.
    var_y = np.var(y, axis=1)
    r2 = 1 - (est.residuals_ ** 2).mean(1) / var_y
    scores = est.score(X, y)
    assert_allclose(r2, scores)


@pytest.mark.parametrize("fit_intercept, normalize, alpha, positive",
                         product([False, True], [False, True],
                                 [0.1, 0.5, 0.95], [False, True]))
def test_grouplasso(gaussian_data, fit_intercept, normalize, alpha, positive):

    X, y = gaussian_data
    n_samples = y.shape[1]

    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    alpha_max = np.linalg.norm(Xty, axis=0).max()
    alpha *= alpha_max / n_samples
    est = GroupLasso(alpha=alpha, fit_intercept=fit_intercept,
                     normalize=normalize, positive=positive)
    est.fit(X, y)
    assert hasattr(est, 'is_fitted_')
    assert_allclose(est.coef_specific_, 0.)
    if positive:
        assert est.coef_.min() >= 0.
    var_y = np.var(y, axis=1)
    r2 = 1 - (est.residuals_ ** 2).mean(1) / var_y
    scores = est.score(X, y)
    assert_allclose(r2, scores)


@pytest.mark.parametrize("fit_intercept, normalize, alpha",
                         product([False, True], [False, True],
                                 [0.1, 0.5, 0.95]))
def test_multitasklasso(gaussian_data, fit_intercept, normalize, alpha):

    X, y = gaussian_data
    X = [X[0], X[0]]
    n_samples = y.shape[1]

    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    alpha_max = np.linalg.norm(Xty, axis=0).max()
    alpha *= alpha_max / n_samples
    est = GroupLasso(alpha=alpha, fit_intercept=fit_intercept,
                     normalize=normalize)
    est.fit(X, y)
    assert hasattr(est, 'is_fitted_')

    mtlasso = MultiTaskLasso(alpha=alpha, fit_intercept=fit_intercept,
                             normalize=normalize)
    mtlasso.fit(X[0], y.T)
    assert_allclose(est.coef_, mtlasso.coef_.T, rtol=1e-2)


def test_warmstart(gaussian_data):

    X, y = gaussian_data
    n_samples = y.shape[1]

    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    alpha_max = np.linalg.norm(Xty, axis=0).max()
    alpha1 = 0.6 * alpha_max / n_samples
    alpha2 = 0.05 * alpha_max / n_samples

    est = DirtyModel(alpha=alpha1, beta=alpha1, warm_start=True, tol=1e-5)
    est.fit(X, y)
    est.alpha = alpha2
    est.fit(X, y)
    assert hasattr(est, 'is_fitted_')
    assert_allclose(est.coef_specific_, 0.)
    coef1 = est.coef_.copy()

    est = DirtyModel(alpha=alpha2, beta=alpha1, tol=1e-5)
    est.fit(X, y)
    coef2 = est.coef_

    assert_allclose(coef1, coef2, 1e-4)
