"""Solvers for multitask regression models."""
import warnings
import numpy as np
import numba as nb
from numba import (jit, float64, int64, boolean)

from joblib import Parallel, delayed

from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning

from . import utils
from .solver_mtw_cd import cython_wrapper
from .otfunctions import (barycenterkl, barycenterkl_img, barycenterkl_log,
                          barycenterkl_img_log)


def solver_dirty(X, y, coef_shared_, coef_specific_, alpha=1., beta=1.,
                 max_iter=10000, tol=1e-4, positive=False):
    """BCD in numba."""
    R = utils.residual(X, coef_shared_ + coef_specific_, y)
    X = np.asfortranarray(X)

    Ls = (X ** 2).sum(axis=1).max(0)
    Ls[Ls == 0.] = Ls[Ls != 0].min()

    coef_shared_, coef_specific_, R, n_iter = \
        _solver_dirty(X, R, coef_shared_, coef_specific_, Ls, alpha, beta,
                      max_iter, tol, positive)
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
                 float64, boolean),
     nopython=True, cache=True)
def _solver_dirty(X, R, coef_shared_, coef_specific_, Ls, alpha, beta,
                  max_iter, tol, positive):
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
            if positive:
                tmp2 = np.maximum(tmp2, 0.)
                tmp1 = np.maximum(tmp1, 0.)
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


def solver_lasso(X, y, alpha=None, max_iter=3000, tol=1e-4, positive=False):
    """Solver for Independent Lasso."""

    n_tasks, n_samples, n_features = X.shape
    theta = np.zeros((n_features, n_tasks))

    if alpha is None:
        alpha = np.ones(n_tasks)
    alpha = np.asarray(alpha).reshape(n_tasks)
    for k in range(n_tasks):
        lasso = Lasso(alpha=alpha[k], tol=tol, max_iter=max_iter,
                      fit_intercept=False, positive=positive)
        lasso.fit(X[k], y[k])
        theta[:, k] = lasso.coef_

    return theta


def solver_mll(X, y, C, S, alpha=0.1, max_iter=1000, tol=1e-4, positive=False):
    """Perform Lasso alternating to solve Multi-level lasso"""
    n_tasks, n_samples, n_features = X.shape
    lasso = Lasso(alpha=alpha, fit_intercept=False, positive=positive)
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


def solver_mtw(Xf, mXf, Ls, y, M, alpha, beta1, beta2, epsilon, gamma, coef1,
               coef2, R, b1, b2, sigmas, concomitant=False,
               stable=False, tol=1e-4, max_iter=1000, tol_ot=1e-5,
               max_iter_ot=20, max_iter_cd=2000,
               positive=False, n_jobs=1, tol_cd=1e-5, gpu=False,
               ot_threshold=0):
    """Perform Alternating Optimization of the MTW problem.
    """
    log = {'loss': [], 'dloss': [], 'log_sinkhorn1': [], 'log_sinkhorn2': [],
           'stable': stable, "objcd": [], "fot1": [0.], "fot2": [0.],
           'reweighting_loss': [], "n_coords": [], "obj": 0.}
    n_samples, n_features = Xf[0].shape
    n_tasks = len(Xf)
    if n_jobs == -1:
        n_jobs = n_tasks
    marginals1, marginals2 = np.ones((2, n_tasks, n_features)) / n_features

    theta1 = coef1.copy()
    theta2 = coef2.copy()
    theta = theta1 - theta2

    thetaold = theta.copy()

    ot_img = True
    if len(M) == n_features:
        ot_img = False

    update_ot_1 = set_ot_func(stable, ot_img)
    update_ot_2 = set_ot_func(stable, ot_img)

    xp = utils.set_module(gpu)
    M = xp.asarray(- M / epsilon)
    if b1 is not None:
        b1 = xp.asarray(b1)
    if b2 is not None:
        b2 = xp.asarray(b2)

    thetabar1 = np.ones_like(coef1).mean(axis=-1)
    thetabar2 = np.ones_like(coef2).mean(axis=-1)

    if positive:
        theta2 *= 0.
        thetabar2 *= 0.
        theta = theta1
    a = n_samples * alpha * gamma
    beta1 = n_samples * beta1
    beta2 = n_samples * beta2

    if concomitant:
        sigma0 = 0.01 * np.linalg.norm(y, axis=1).min() / (n_samples ** 0.5)
    else:
        sigma0 = 0.
    with Parallel(n_jobs=n_jobs, backend="threading") as pll:
        if alpha == 0.:
            theta, R, sigmas, mx = update_coefs(pll, Xf, y, theta, R,
                                                Ls, marginals1,
                                                sigmas, a, beta1,
                                                sigma0,
                                                tol=tol_cd,
                                                max_iter=max_iter_cd,
                                                positive=positive)
            obj = 0.5 * (R ** 2).sum(axis=1).dot(1 / sigmas) / n_samples
            obj += beta1 * abs(theta).sum() + 0.5 * sigmas.sum()
            theta1, theta2 = utils.get_unsigned(theta)
            log['loss'].append(obj)
        else:
            for i in range(max_iter):
                obj = 0.
                if not positive:
                    Y1 = utils.residual(Xf, - theta2, y)
                else:
                    Y1 = y
                theta1, R, sigmas, mxp = update_coefs(pll, Xf, Y1, theta1, R,
                                                      Ls,
                                                      marginals1,
                                                      sigmas,
                                                      a, beta1,
                                                      sigma0,
                                                      tol=tol_cd,
                                                      max_iter=max_iter_cd)
                if not positive:
                    Y2 = utils.residual(Xf, theta1, y)
                    theta2, R, sigmas, mx = update_coefs(pll, mXf, Y2, theta2,
                                                         R,
                                                         Ls,
                                                         marginals2,
                                                         sigmas,
                                                         a, beta2,
                                                         sigma0,
                                                         tol=tol_cd,
                                                         max_iter=max_iter_cd)
                    theta = theta1 - theta2
                else:
                    theta = theta1

                dx = abs(theta - thetaold) / max(1, thetaold.max(),
                                                 theta.max())
                dx = dx.max()
                thetaold = theta.copy()
                if alpha:
                    if (theta1 > ot_threshold).any(0).all():
                        fot1, log_ot1, marginals1, b1, q1 = \
                            update_ot_1(theta1, M, epsilon, gamma,
                                        b=b1, tol=tol_ot,
                                        max_iter=max_iter_ot,
                                        threshold=ot_threshold)
                        if fot1 is None or not theta1.max(0).all():
                            warnings.warn("Numerical errors. Moving in "
                                          "log-domain.")
                            b1 = xp.log(b1 + 1e-100, out=b1)
                            stable = True
                            update_ot_1 = set_ot_func(True, ot_img)
                            fot1, log_ot1, marginals1, b1, q1 = \
                                update_ot_1(theta1, M, epsilon, gamma, b=b1,
                                            tol=tol_ot, max_iter=max_iter_ot,
                                            threshold=ot_threshold)

                        log["log_sinkhorn1"].append(log_ot1)
                        thetabar1 = q1
                        log["fot1"].append(fot1)
                        obj += alpha * fot1
                    if not positive and (theta2 > ot_threshold).any(0).all():
                        fot2, log_ot2, marginals2, b2, q2 = \
                            update_ot_2(theta2, M, epsilon, gamma,
                                        b=b2, tol=tol_ot, max_iter=max_iter_ot)

                        if fot2 is None or not theta2.max(0).all():
                            warnings.warn("Numerical errors. Moving in "
                                          "log-domain.")
                            b2 = xp.log(b2 + 1e-100, out=b2)
                            stable = True
                            update_ot_2 = set_ot_func(True, ot_img)
                            fot2, log_ot2, marginals2, b2, q2 = \
                                update_ot_2(theta2, M, epsilon, gamma,
                                            b=b2, tol=tol_ot,
                                            max_iter=max_iter_ot)

                        log["log_sinkhorn2"].append(log_ot2)
                        thetabar2 = q2
                        log["fot2"].append(fot2)
                        obj += alpha * fot2

                log['loss'].append(obj)
                log['dloss'].append(dx)

                if dx < tol:
                    break
            if i == max_iter - 1:
                warnings.warn('Objective did not converge.' +
                              ' You might want' +
                              ' to increase the number of iterations.' +
                              ' Fitting data with very small alpha and' +
                              ' beta may cause precision problems.',
                              ConvergenceWarning)
        log['stable'] = stable

    if positive:
        theta2 *= 0.
        thetabar2 = xp.zeros_like(thetabar1)
        try:
            thetabar2 = thetabar2.get()
        except AttributeError:
            pass
    return (theta1, theta2, thetabar1, thetabar2, log, sigmas, b1, b2, R)


def set_ot_func(stable, ot_img):
    """Set barycenter function."""
    if stable:
        update_ot = barycenterkl_log
    else:
        update_ot = barycenterkl

    if ot_img:
        if stable:
            update_ot = barycenterkl_img_log
        else:
            update_ot = barycenterkl_img
    else:
        if stable:
            update_ot = barycenterkl_log
        else:
            update_ot = barycenterkl
    return update_ot


def update_coefs(pll, X, y, coefs0, R, Ls, marginals, sigmas, a, b, sigma0,
                 max_iter=20000, tol=1e-4, positive=False):
    """BCD in numba."""
    n_tasks, n_samples = y.shape
    n_features = Ls.shape[1]

    dell = delayed(cython_wrapper)
    it = (dell(n_samples, n_features,
               X[k],
               y[k], Ls[k], marginals[k], coefs0[:, k],
               R[k], sigmas[k:k + 1], a, b[:, k], sigma0, tol, max_iter,
               positive)
          for k in range(n_tasks))
    output = pll(it)
    thetas, R, sigmas, n_iter_cd = list(zip(*output))

    thetas = np.stack(thetas, axis=1)
    R = np.stack(R, axis=0)

    sigmas = np.r_[sigmas]
    return thetas, R, sigmas, n_iter_cd
