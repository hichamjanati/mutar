"""Solvers for multitask regression models."""
import warnings
import numpy as np
import numba as nb
from numba import (jit, float64, int64)

from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning

from . import utils


def solver_dirty(X, y, coef_shared_, coef_specific_, alpha=1., beta=1.,
                 max_iter=10000, tol=1e-4):
    """BCD in numba."""
    R = utils.residual(X, coef_shared_ + coef_specific_, y)
    X = np.asfortranarray(X)

    Ls = (X ** 2).sum(axis=1).max(0)
    Ls[Ls == 0.] = Ls[Ls != 0].min()

    coef_shared_, coef_specific_, R, n_iter = \
        _solver_dirty(X, R, coef_shared_, coef_specific_, Ls, alpha, beta,
                      max_iter, tol)
    if n_iter == max_iter - 1:
        warnings.warn('Objective did not converge.' +
                      ' You might want' +
                      ' to increase the number of iterations.' +
                      ' Fitting data with very small alpha' +
                      ' may cause precision problems.',
                      ConvergenceWarning)
    return coef_shared_, coef_specific_, R, n_iter


output_type = nb.types.Tuple((float64[::1, :], float64[::1, :],
                              float64[:, :], int64))


@jit(output_type(float64[::1, :, :], float64[:, :],
                 float64[::1, :], float64[::1, :],
                 float64[:], float64, float64, int64,
                 float64),
     nopython=True, cache=True)
def _solver_dirty(X, R, coef_shared_, coef_specific_, Ls, alpha, beta,
                  max_iter, tol):
    """ Coordinate solver of DirtyModels in numba."""
    n_tasks = len(X)
    n_samples, n_features = X[0].shape
    theta = coef_shared_ + coef_specific_
    alpha *= n_samples
    beta *= n_samples

    # dg = 1.
    for i in range(max_iter):
        w_max = 0.0
        d_w_max = 0.0
        for j in range(n_features):
            if Ls[j] == 0.:
                continue
            # compute residual
            grad = np.zeros(n_tasks)
            tmp1 = np.zeros(n_tasks)
            tmp2 = np.zeros(n_tasks)

            normtmp = 0.
            for t in range(n_tasks):
                for n in range(n_samples):
                    grad[t] += X[t, n, j] * R[t, n]
                grad[t] /= Ls[j]
                tmp1[t] = grad[t] + coef_shared_[j, t]
                tmp2[t] = grad[t] + coef_specific_[j, t]

                normtmp += tmp1[t] ** 2

            normtmp = np.sqrt(normtmp)

            # l2 thresholding

            thresholdl2 = 0.
            if normtmp:
                thresholdl2 = max(1. - alpha / (Ls[j] * normtmp), 0.)
            tmp1 *= thresholdl2
            thresholdl1 = beta / Ls[j]
            tmp2 = np.sign(tmp2) * np.maximum(np.abs(tmp2) - thresholdl1, 0.)
            new_theta = tmp1 + tmp2
            if theta[j].any():
                for t in range(n_tasks):
                    R[t] += X[t, :, j] * theta[j, t]

            d_w_j = np.abs(theta[j] - new_theta).max()
            d_w_max = max(d_w_max, d_w_j)
            w_max = max(w_max, np.abs(tmp1 + tmp2).max())
            coef_shared_[j] = tmp1
            coef_specific_[j] = tmp2
            theta[j] = new_theta

            if theta[j].any():
                for t in range(n_tasks):
                    R[t] -= X[t, :, j] * theta[j, t]

        if (w_max == 0.0 or d_w_max / w_max < tol):
            break

    return coef_shared_, coef_specific_, R, i


def solver_lasso(X, y, alpha=None, max_iter=3000, tol=1e-4):
    """Solver for Independent Lasso."""

    n_tasks, n_samples, n_features = X.shape
    theta = np.zeros((n_features, n_tasks))

    if alpha is None:
        alpha = np.ones(n_tasks)
    alpha = np.asarray(alpha).reshape(n_tasks)
    for k in range(n_tasks):
        lasso = Lasso(alpha=alpha[k], tol=tol, max_iter=max_iter,
                      fit_intercept=False)
        lasso.fit(X[k], y[k])
        theta[:, k] = lasso.coef_

    return theta


def solver_mll(X, y, C, S, alpha=0.1, max_iter=1000, tol=1e-4):
    """Perform Lasso alternating to solve Multi-level lasso"""
    n_tasks, n_samples, n_features = X.shape
    lasso = Lasso(alpha=alpha, fit_intercept=False)
    lasso_p = Lasso(alpha=alpha / n_tasks, fit_intercept=False,
                    positive=True)
    old_theta = C[:, None] * S

    for i in range(max_iter):
        W = X * C[None, None, :]
        for k in range(n_tasks):
            lasso.fit(W[k], y[k])
            S[:, k] = lasso.coef_
        Z = S.T[:, None, :] * X
        Z = Z.reshape(n_tasks * n_samples, n_features)
        lasso_p.fit(Z, y.flatten())
        C = lasso_p.coef_
        theta = C[:, None] * S
        dll = abs(theta - old_theta).max()
        dll /= max(abs(theta).max(), abs(old_theta).max(), 1.)
        old_theta = theta.copy()

        if dll < tol:
            break

    if i == max_iter - 1:
        warnings.warn('Objective did not converge.' +
                      ' You might want' +
                      ' to increase the number of iterations.' +
                      ' Fitting data with very small alpha' +
                      ' may cause precision problems.',
                      ConvergenceWarning)
    return C, S, i
