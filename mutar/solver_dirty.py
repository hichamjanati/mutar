"""Solvers for multitask regression models."""
import numpy as np
import numba as nb
from numba import (jit, float64, int64, boolean)
from . import utils


@jit(float64(float64[::1, :, :], float64[::1, :], float64[::1, :],
             float64[:, :], float64[:, :], float64, float64),
     nopython=True, cache=True)
def dualgap(X, coef_shared_, coef_specific_, R, y, alpha, beta):
    """Compute dual gap for multi task group lasso."""
    n_samples, n_tasks = R.T.shape
    n_features = X.shape[-1]
    xtr = np.zeros_like(coef_shared_)
    dualnorm = 0.
    dualmax = 0.
    for k in range(n_tasks):
        xtr[:, k] = X[k].T.dot(R[k])
    for j in range(n_features):
        dn = np.linalg.norm(xtr[j])
        dm = np.abs(xtr[j]).max()
        if dn > dualnorm:
            dualnorm = dn
        if dm > dualmax:
            dualmax = dm
    R_norm = np.linalg.norm(R)
    const = 1.
    if dualnorm < alpha and dualmax < beta:
        dg = R_norm ** 2
    else:
        const = min(alpha / dualnorm, beta / dualmax)
        A_norm = R_norm * const
        dg = 0.5 * (R_norm ** 2 + A_norm ** 2)
    dg += - const * (R * y).sum()
    for j in range(n_features):
        dg += alpha * np.linalg.norm(coef_shared_[j])
    dg += beta * np.abs(coef_specific_).sum()
    return dg


def solver_dirty(X, y, coef_shared_, coef_specific_, alpha=1., beta=1.,
                 maxiter=2000, tol=1e-4, verbose=False):
    """BCD in numba."""
    X = np.asfortranarray(X)

    # normalize dual gap threshold
    tol = tol * np.linalg.norm(y) ** 2

    Ls = (X ** 2).sum(axis=1).max(0)
    Ls[Ls == 0.] = Ls[Ls != 0].min()

    R = utils.residual(X, coef_shared_ + coef_specific_, y)

    coef_shared_, coef_specific_, R, dual_gap, n_iter = \
        _solver_dirty(X, y, R, coef_shared_, coef_specific_, Ls, alpha, beta,
                      maxiter, tol, verbose)
    coef_shared_ = np.ascontiguousarray(coef_shared_)
    coef_specific_ = np.ascontiguousarray(coef_specific_)
    return coef_shared_, coef_specific_, R, dual_gap, n_iter


output_type = nb.types.Tuple((float64[::1, :], float64[::1, :],
                              float64[:, :], float64, int64))


@jit(output_type(float64[::1, :, :], float64[:, :], float64[:, :],
                 float64[::1, :], float64[::1, :],
                 float64[:], float64, float64, int64,
                 float64, boolean),
     nopython=True, cache=True)
def _solver_dirty(X, y, R, coef_shared_, coef_specific_, Ls, alpha, beta,
                  maxiter, tol, verbose):
    """Perform GFB with BCD to solve Multi-task Dirty group lasso."""
    n_tasks = len(X)
    n_samples, n_features = X[0].shape
    theta = coef_shared_ + coef_specific_

    # dg = 1.
    for i in range(maxiter):
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

        if (w_max == 0.0 or d_w_max / w_max < tol or
                i == maxiter - 1):
            dg = dualgap(X, coef_shared_, coef_specific_, R, y, alpha, beta)
            if dg < tol:
                break
    if i == maxiter - 1:
        print("**************************************\n"
              "******** WARNING: Stopped early. *****\n"
              "\n"
              "You may want to increase maxiter.")
    return coef_shared_, coef_specific_, R, dg, i
