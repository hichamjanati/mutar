import pytest
import numpy as np

from itertools import product

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from mutar import MTW, ReMTW, utils, IndRewLasso, IndLasso


@pytest.mark.parametrize("fit_intercept, normalize, alpha, beta, "
                         "concomitant, positive",
                         product([False, True], [False, False],
                                 [0.1, 0.5], [0.1, 0.5], [False, True],
                                 [False, True]))
def test_mtw(gaussian_data, fit_intercept, normalize, alpha, beta,
             concomitant, positive):

    X, y = gaussian_data
    n_samples = y.shape[1]
    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    beta_max = abs(Xty).max()
    beta *= beta_max / n_samples
    est = MTW(alpha=alpha, beta=beta, concomitant=concomitant,
              fit_intercept=fit_intercept, normalize=normalize,
              positive=positive)
    est.fit(X, y)
    assert hasattr(est, 'is_fitted_')

    if positive:
        assert est.coef_.min() >= 0
        assert est.coef2_.max() == 0.
        if not normalize:
            assert_array_equal(est.coef_, est.coef1_)

    else:
        var_y = np.var(y, axis=1)
        r2 = 1 - (est.residuals_ ** 2).mean(1) / var_y
        scores = est.score(X, y)
        assert_allclose(r2, scores)


@pytest.mark.parametrize("max_iter_reweighting", [1, 10])
def test_mtw_convolutions(identity_data, max_iter_reweighting):

    X, y = identity_data
    n_samples, n_features = X[0].shape
    width = int(n_features ** 0.5)
    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    beta_max = abs(Xty).max()
    beta = 0.2 * beta_max / n_samples
    M2d = utils.groundmetric2d(width, p=2, normed=True)
    Mimg = utils.groundmetric_img(width, p=2, normed=True)

    est_img = ReMTW(alpha=0.5, beta=beta, M=Mimg, gamma=1)
    est_img.fit(X, y)
    coef_img = est_img.coef_.flatten()

    est = ReMTW(alpha=0.5, beta=beta, M=M2d, gamma=1)
    est.fit(X, y)
    coef = est.coef_.flatten()

    assert_allclose(coef, coef_img, atol=1e-2)

    assert hasattr(est_img, 'is_fitted_')


@pytest.mark.parametrize("positive", [True, False])
def test_mtw_stabilization(gaussian_data, positive):

    X, y = gaussian_data
    n_samples, n_features = X[0].shape
    width = int(n_features ** 0.5)
    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    beta_max = abs(Xty).max()
    beta = 0.15 * beta_max / n_samples
    M2d = utils.groundmetric2d(width, p=2, normed=True)
    Mimg = utils.groundmetric_img(width, p=2, normed=True)

    est = ReMTW(alpha=0.2, beta=beta, M=Mimg, gamma=0.1, stable=True,
                positive=positive)
    est.fit(X, y)
    coef_img = est.coef_.flatten()

    est = ReMTW(alpha=0.2, beta=beta, M=M2d, gamma=0.1, positive=positive)
    est.fit(X, y)
    coef = est.coef_.flatten()

    est = ReMTW(alpha=0.2, beta=beta, M=M2d, gamma=0.1, stable=True,
                positive=positive)
    est.fit(X, y)
    coef_stable = est.coef_.flatten()

    assert_allclose(coef, coef_img, atol=1e-2)
    assert_allclose(coef_stable, coef_img, atol=1e-2)


@pytest.mark.parametrize("beta, positive", product([0.3], [True, False]))
def test_mtw_vs_lasso(gaussian_data, beta, positive):
    X, y = gaussian_data
    n_samples = y.shape[1]
    Xty = np.array([xx.T.dot(yy) for xx, yy in zip(X, y)])
    beta_max = abs(Xty).max()
    beta *= beta_max / n_samples
    remtw = ReMTW(alpha=0., beta=beta, concomitant=False, positive=positive)
    remtw.fit(X, y)
    relasso = IndRewLasso(alpha=len(X) * [beta], positive=positive)
    relasso.fit(X, y)

    loss1 = remtw.loss_
    loss2 = relasso.loss_
    n = min(len(loss1), len(loss2))
    assert_allclose(remtw.coef_, relasso.coef_, atol=1e-2)
    assert_allclose(loss1[:n], loss2[:n], atol=1e-2)

    mtw = MTW(alpha=0., beta=beta, concomitant=False, positive=positive)
    mtw.fit(X, y)

    lasso = IndLasso(alpha=len(X) * [beta], positive=positive)
    lasso.fit(X, y)

    if positive:
        assert remtw.coef_.min() >= 0.
        assert relasso.coef_.min() >= 0.
        assert mtw.coef_.min() >= 0.
        assert lasso.coef_.min() >= 0.

    assert_allclose(mtw.coef_, lasso.coef_, atol=1e-2)
