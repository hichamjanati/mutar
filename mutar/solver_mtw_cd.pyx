cimport cython
cimport numpy as np
import numpy as np
from cython cimport floating
from libc.math cimport fabs, sqrt, fmax
from scipy.linalg.cython_blas cimport ddot, dasum, daxpy, dnrm2, dcopy, dscal
from scipy.linalg.cython_blas cimport sdot, sasum, saxpy, snrm2, scopy, sscal


cdef floating fsign(floating f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0

cdef floating fdot(int * n, floating * x, int * inc1, floating * y,
                        int * inc2) nogil:
    if floating is double:
        return ddot(n, x, inc1, y, inc2)
    else:
        return sdot(n, x, inc1, y, inc2)

cdef void faxpy(int * n, floating * alpha, floating * x, int * incx,
                 floating * y, int * incy) nogil:
    if floating is double:
        daxpy(n, alpha, x, incx, y, incy)
    else:
        saxpy(n, alpha, x, incx, y, incy)

cdef floating fnrm2(int * n, floating * x, int * inc) nogil:
    if floating is double:
        return dnrm2(n, x, inc)
    else:
        return snrm2(n, x, inc)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating proximal(floating theta, floating marginal,
                       floating a, floating b) nogil:
    """Compute proximal operator of MTW penalty
       (b * Lasso + a * KL(.|marginal))"""
    cdef:
        floating z
        floating delta
    z = theta - a - b
    delta = sqrt(z ** 2 + 4 * a * marginal)
    theta = (z + delta) / 2
    return theta


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating cd_update(int n_samples, int n_features,
                        floating[::1, :] X, floating[:] y,
                        floating[:] Ls, floating[:] marginals,
                        floating[:] theta, floating[:] R,
                        floating[:] sigma,
                        floating a, floating[:] b,
                        int j,
                        floating sigma0=0.,
                        bint positive=0,
                        floating maxw=0.) nogil:

   cdef:
       int inc = 1
       floating tmp
       floating old_j
       floating mthetaj
   # tmp is the prox argument
   if theta[j] != 0.:
       faxpy(&n_samples, &theta[j], &X[0, j], &inc, &R[0], &inc)
       # R += X[:, j] * theta[j]

   # tmp = X[:, j].dot(R)
   tmp = fdot(&n_samples, &R[0], &inc, &X[0, j], &inc)
   # l1-log thresholding
   old_j = theta[j]
   if a > 0. or positive:
       theta[j] = proximal(tmp, marginals[j] * Ls[j], a,
                           b[j] * sigma[0]) / Ls[j]
   else:
       theta[j] = fsign(tmp) * fmax(fabs(tmp) - b[j] * sigma[0], 0.) / Ls[j]
   dw = fabs(old_j - theta[j]) / max(1, fabs(theta[j]),
                                     fabs(old_j))
   if dw > maxw:
       maxw = dw

   if theta[j] != 0.:
       mthetaj = - theta[j]
       faxpy(&n_samples, &mthetaj, &X[0, j], &inc, &R[0], &inc)
       # R += - theta[j] * X[:, j]
       if sigma0:
           sigma[0] = fmax(fnrm2(&n_samples, &R[0], &inc) / sqrt(n_samples),
                           sigma0)
   return maxw

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int cython_solver(int n_samples, int n_features,
                               floating[::1, :] X, floating[:] y,
                               floating[:] Ls, floating[:] marginals,
                               floating[:] theta, floating[:] R,
                               floating[:] sigma,
                               floating a, floating[:] b,
                               int ws_size,
                               floating sigma0=0.,
                               floating tol=1e-4, int maxiter=10000,
                               bint positive=0) nogil:
    """CD solver for l2 reg kl interpolation."""

    cdef:
        floating maxw
        int inc = 1
        int i
        int jj
        floating tmp
        floating old_j
        floating mthetaj

    if sigma0:
        sigma[0] = fmax(fnrm2(&n_samples, &R[0], &inc), sigma0)
        sigma[0] = sigma[0] / sqrt(n_samples)

    for i in range(maxiter):
        maxw = 0.
        for kk in range(ws_size):
            # jj = working_set[kk]
            jj = kk
            maxw = cd_update(n_samples, n_features, X, y, Ls, marginals,
                             theta, R, sigma, a, b, jj, sigma0, positive,
                             maxw)
        if maxw < tol:
            break

    return i



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cython_wrapper(int n_samples, int n_features, floating[::1, :] X,
                   floating[:] y, floating[:] Ls, floating[:] marginals,
                   floating[:] theta, floating[:] R, floating[:] sigma,
                   floating a, floating[:] b,
                   floating sigma0=0.,
                   floating tol=1e-4, int maxiter=10000,
                   bint positive=0):
    cdef:
        floating[:] sol
        int ws_size = n_features
    with nogil:
        i = cython_solver(n_samples, n_features, X, y, Ls,
                            marginals, theta, R, sigma, a, b,
                            ws_size, sigma0, tol, maxiter, positive)
    return np.asarray(theta), np.asarray(R), np.asarray(sigma), i
