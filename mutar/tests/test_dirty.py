import pytest
import numpy as np

from itertools import product

# from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from mutar import DirtyModel, GroupLasso


@pytest.mark.parametrize("fit_intercept, alpha, beta",
                         product([False, True], [0.1, 0.5], [0.1, 0.5]))
def test_dirty(data, fit_intercept, alpha, beta):

    X, y = data

    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    alpha_max = np.linalg.norm(Xty, axis=0).max()
    beta_max = abs(Xty).max()
    alpha *= alpha_max
    beta *= beta_max
    est = DirtyModel(alpha=alpha, beta=beta,
                     fit_intercept=fit_intercept)
    est.fit(X, y)
    assert hasattr(est, 'is_fitted_')
    if est.alpha <= est.beta:
        assert_allclose(est.coef_specific_, 0.)
    elif est.alpha > len(X) ** 0.5 * est.beta:
        assert_allclose(est.coef_shared_, 0.)


@pytest.mark.parametrize("fit_intercept, alpha",
                         product([False, True], [0.1, 0.5, 0.95]))
def test_grouplasso(data, fit_intercept, alpha):

    X, y = data

    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    alpha_max = np.linalg.norm(Xty, axis=0).max()
    alpha *= alpha_max
    est = GroupLasso(alpha=alpha, fit_intercept=fit_intercept)
    est.fit(X, y)
    assert hasattr(est, 'is_fitted_')
    assert_allclose(est.coef_specific_, 0.)
