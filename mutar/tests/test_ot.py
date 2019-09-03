import numpy as np
from mutar import utils, otfunctions
from numpy.testing import assert_allclose


def test_scaling_match():
    rnd = np.random.RandomState(42)
    width = 16
    n_tasks = 1
    n_features = width ** 2
    coefs = rnd.rand(width, width, n_tasks)
    coefs_flat = coefs.reshape(n_features, -1)
    M = utils.groundmetric2d(width, p=2, normed=False)
    K = np.exp(- M)
    scaling = K.dot(coefs_flat).flatten()
    scaling_log = np.exp(utils.logsumexp(np.log(coefs_flat)[None, :] - M,
                                         axis=1))

    M = utils.groundmetric(width, p=2, normed=False)
    K = np.exp(- M)
    scaling_conv = utils.klconv1d_list(coefs, K).flatten()
    scaling_conv_log = np.exp(utils.kls(np.log(coefs), - M))

    assert_allclose(scaling, scaling_log.flatten())
    assert_allclose(scaling, scaling_conv)
    assert_allclose(scaling_conv_log.flatten(), scaling_conv)


def test_barycenter_match():
    rnd = np.random.RandomState(42)
    width = 16
    n_tasks = 2
    n_features = width ** 2
    coefs_flat = rnd.rand(n_features, n_tasks)
    coefs_flat[n_features // 2] = 0.
    M = utils.groundmetric2d(width, p=2, normed=True)
    epsilon = 5. / n_features
    gamma = 1.
    max_iter = 50
    K = - M / epsilon
    options = dict(P=coefs_flat, M=K, epsilon=epsilon, gamma=gamma,
                   max_iter=max_iter, tol=0.)
    f, log, ms, b, q = otfunctions.barycenterkl(**options)
    fl, logl, msl, bl, ql = otfunctions.barycenterkl_log(**options)

    M = utils.groundmetric_img(width, p=2, normed=True)
    K = - M / epsilon
    options["M"] = K
    fc, logc, msc, bc, qc = otfunctions.barycenterkl_img(**options)
    fcl, logcl, mscl, bcl, qcl = otfunctions.barycenterkl_img_log(**options)

    assert_allclose(q, ql, rtol=1e-5, atol=1e-5)
    assert_allclose(ms, msl, rtol=1e-5, atol=1e-5)
    assert_allclose(b, np.exp(bl), rtol=1e-5, atol=1e-5)

    assert_allclose(qc, qcl, rtol=1e-5, atol=1e-5)
    assert_allclose(msc, mscl, rtol=1e-5, atol=1e-5)
    assert_allclose(bc, np.exp(bcl), rtol=1e-5, atol=1e-5)

    assert_allclose(q, qc.reshape(n_features), rtol=1e-5, atol=1e-5)
    assert_allclose(ms, msc.reshape(-1, n_features), rtol=1e-5, atol=1e-5)
