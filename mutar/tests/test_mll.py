import pytest
import numpy as np

from itertools import product

# from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from mutar import MultiLevelLasso


@pytest.mark.parametrize("fit_intercept, normalize, alpha, positive",
                         product([False, True], [False, True],
                                 [0.1, 0.5, 1.], [False, True]))
def test_mll(gaussian_data, fit_intercept, normalize, alpha, positive):

    X, y = gaussian_data
    n_samples = y.shape[1]
    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    alpha_max = abs(Xty).max()
    alpha *= alpha_max / n_samples
    est = MultiLevelLasso(alpha=alpha, fit_intercept=fit_intercept,
                          normalize=normalize, positive=positive)
    est.fit(X, y)
    assert hasattr(est, 'is_fitted_')
    if positive:
        assert est.coef_.min() >= 0.
    var_y = np.var(y, axis=1)
    res = y - est.predict(X)
    r2 = 1 - (res ** 2).mean(1) / var_y
    scores = est.score(X, y)
    assert_allclose(r2, scores)
