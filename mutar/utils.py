"""Util functions."""
import numpy as np
try:
    import cupy as cp
    get_module = cp.get_array_module
except ImportError:
    def get_module(x):
        return np


def residual(X, coef_, y):
    """Compute y - X @ coef_."""
    R = y - np.array([x.dot(th) for x, th in zip(X, coef_.T)])
    return R


def groundmetric(width, height=None, p=2, normed=False):
    """Compute ground metric matrix on the 1D grid 0:`n_features`.

    Parameters
    ----------
    width : int
        width of the matrix
    height : int (optional, default equal to width)
        height of the matrix
    p: int > 0.
        Power to raise the pairwise distance metrix. Quadratic by default.
    normed: boolean (default True)
        If True, the matrix is divided by its median.

    Returns
    -------
    M: 2D array (width, height).

    """
    if height is None:
        height = width
    x = np.arange(0, width).reshape(-1, 1).astype(float)
    y = np.arange(0, height).reshape(-1, 1).astype(float)
    xx, yy = np.meshgrid(x, y)
    M = abs(xx - yy) ** p
    if normed:
        M /= np.median(M)
    return M


def groundmetric_img(width, height=None, p=2, normed=False):
    """Compute ground metric for convolutional Wasserstein.

    Parameters
    ----------

    width, height : int,
        shape of images
    p: int, optional (default 2)
        Power to raise the pairwise distance metrix. Quadratic by default.
    normed: boolean (default True)
        If True, the matrix is divided by its median.
    Returns
    -------
    M: 2D array (width, height).
    """
    if height is None:
        height = width
    M = groundmetric(width, height, p=2, normed=False)
    if normed:
        Mlarge = groundmetric2d(width, height, p=2, normed=False)
        median = np.median(Mlarge)
        M /= median
    return M


def groundmetric2d(width, height=None, p=1, normed=False):
    """Compute ground metric matrix on the 2D grid (width, height).

    Parameters
    ----------
    n_features: int > 0.
    p: int > 0.
        Power to raise the pairwise distance metrix. Quadratic by default.
    normed: boolean (default True)
        If True, the matrix is divided by its median.

    Returns
    -------
    M: 2D array (n_features, n_features).

    """
    if height is None:
        height = width
    n_features = width * height
    M = groundmetric(width, height, p=2, normed=False)
    M = M[:, np.newaxis, :, np.newaxis] + M[np.newaxis, :, np.newaxis, :]
    M = M.reshape(n_features, n_features) ** (p / 2)

    if normed:
        M /= np.median(M)
    return M


def compute_gamma(tau, M):
    """Compute the sufficient KL weight for a full mass minimum."""
    xp = get_module(M)
    return max(0., - M.max() / (2 * xp.log(tau)))


def get_unsigned(x):
    x1 = np.clip(x, 0., None)
    x2 = - np.clip(x, None, 0.)
    return x1, x2


def set_module(gpu):
    try:
        import cupy as cp
        if gpu:
            return cp
        return np
    except ImportError:
        return np


def logsumexp(a, axis=None):
    """Compute the log of the sum of exponentials of input elements."""
    xp = get_module(a)
    a_max = xp.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~xp.isfinite(a_max)] = 0
    elif not xp.isfinite(a_max):
        a_max = 0

    out = a - a_max
    out = xp.exp(out, out=out)
    a_max = xp.squeeze(a_max, axis=axis)

    # suppress warnings about log of zero
    out = xp.sum(out, axis=axis, keepdims=False)
    out = xp.log(out, out=out)

    out += a_max
    free_gpu_memory(xp)
    return out


# for lists, vectorized:
def kls(img, C):
    """Compute log separable kernal application."""
    xp = get_module(C)
    x = (logsumexp(C[xp.newaxis, :, :, xp.newaxis] +
         img[:, xp.newaxis], axis=-2))
    x = logsumexp(C.T[:, :, xp.newaxis, xp.newaxis] + x[:, xp.newaxis], axis=0)
    return x


def klconv1d(img, K):
    """Compute separable kernel application with convolutions."""
    X = K.dot(K.dot(img).T).T
    return X


def klconv1d_list(imgs, K):
    """Compute separable kernel application with convolutions."""
    w, w, m = imgs.shape
    convs = imgs.copy()
    for k in range(m):
        convs[:, :, k] = klconv1d(imgs[:, :, k], K)
    return convs


def free_gpu_memory(module):
    try:
        module.get_default_memory_pool().free_all_blocks()
    except AttributeError:
        pass
    return 0.
