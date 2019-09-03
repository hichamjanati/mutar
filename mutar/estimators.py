"""
Mutar estimators module.
"""
import numpy as np
import warnings

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning

from .solvers import solver_dirty, solver_lasso, solver_mll, solver_mtw
from . import utils


class MultitaskRegression(BaseEstimator, RegressorMixin):
    """ Multitask regression abstract class.

    Parameters
    ----------
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.
    max_iter : int, optional
        The maximum number of iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    positive : boolean, optional (default False)
        If True, coefficients are constrained to be non-negative.
    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features, n_tasks)
        Parameter vector (W in the cost function formula).
    intercept_ : array, shape (n_tasks,)
        independent term in decision function.
    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    """
    def __init__(self, fit_intercept=True, normalize=False, positive=False,
                 max_iter=2000, tol=1e-4, warm_start=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.is_fitted_ = False
        self.positive = positive

    def _pre_fit(self, X, y):
        """ Normalize and scale the data."""
        n_tasks = len(X)
        n_samples, n_features = X[0].shape
        X, y = check_X_y(X, y, accept_sparse=False, allow_nd=True,
                         multi_output=True)

        if y.shape != (n_tasks, n_samples):
            raise ValueError("Data shape not understood. X must be "
                             "(n_tasks, n_samples, n_features) and y must be "
                             "(n_tasks, n_samples)")
        X_scale = np.ones((n_tasks, n_features))
        if self.fit_intercept:
            X = X.copy()
            y = y.copy()
            X_offset = X.mean(axis=1)
            y_offset = y.mean(axis=1)
            X -= X_offset[:, None, :]
            y -= y_offset[:, None]
            if self.normalize:
                X_scale = np.linalg.norm(X, axis=1)
                X /= X_scale[:, None, :]
        else:
            X_offset = np.zeros((n_tasks, 1, n_features))
            y_offset = np.zeros((n_tasks, 1))
        return X, y, X_offset, y_offset, X_scale

    def _fit(self, X, y):
        """Generic fit method to be specified for each model."""
        pass

    def _post_fit(self, X_offset, y_offset, X_scale):
        """Set intercept and scale after fit."""
        assert self.is_fitted_
        if self.fit_intercept:
            if self.normalize:
                self.coef_ /= X_scale.T
            self.intercept_ = y_offset
            self.intercept_ -= np.einsum("kj,jk->k", X_offset, self.coef_)
        else:
            self.intercept_ = np.zeros_like(y_offset).flatten()

    def fit(self, X, y):
        X, y, X_offset, y_offset, X_scale = self._pre_fit(X, y)
        self._fit(X, y)
        self.is_fitted_ = True
        self._post_fit(X_offset, y_offset, X_scale)
        return self

    def predict(self, X):
        """ Predict target given unseen data samples.

        Parameters
        ----------
        X : {array-like}, shape (n_tasks, n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_tasks, n_samples)
            Returns the predicted targets.
        """
        X = check_array(X, accept_sparse=False, allow_nd=True)
        check_is_fitted(self, 'is_fitted_')
        y_pred = [x.dot(c) for x, c in zip(X, self.coef_.T)]
        y_pred = np.array(y_pred) + self.intercept_[:, None]
        return y_pred

    def score(self, X, y, sample_weight=None):
        """ Returns the coefficient of determination R^2 of the prediction.

        Computes a score for each regression task.
        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum
        of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible
        score is 1.0 and it can be negative (because the model can be
        arbitrarily worse). A constant model that always predicts the expected
        value of y, disregarding the input features, would get a R^2 score of
        0.0.

        Parameters
        Xarray-like, shape = (n_tasks, n_samples, n_features)
        Test samples.

        yarray-like, shape = (n_tasks, n_samples)
        True values for y.

        sample_weightarray-like, shape = [n_tasks, n_samples], optional
        Sample weights.

        Returns
        -------
        array-like, shape = (n_tasks)
        R^2 of self.predict(X) wrt. y for each task.
        """
        y_pred = self.predict(X)
        if sample_weight is None:
            scores = [r2_score(y_i, y_pred_i, multioutput='variance_weighted')
                      for y_i, y_pred_i in zip(y, y_pred)]
        else:
            scores = [r2_score(y_i, y_pred_i, sample_weight=w_i,
                      multioutput='variance_weighted')
                      for y_i, y_pred_i, w_i in zip(y, y_pred, sample_weight)]
        return np.array(scores)


class DirtyModel(MultitaskRegression):
    """ DirtyModel estimator with L1 and L1/L2 mixed-norm as regularizers.

    The optimization objective for Dirty models is::

        (1 / (2 * n_samples)) * ||Y - X(W_1 + W_2)||^2_Fro + alpha * ||W_1||_21
        + beta * ||W_2||_1

    Where::

        ||W||_21 = sum_i sqrt{sum_j w_ij^2}

    i.e. the sum of norm of each row.

    and::

        ||W||_1 = sum_i sum_j |w_ij|

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1/L2 term. Defaults to 1.0
    beta : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.
    max_iter : int, optional
        The maximum number of iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    positive : boolean, optional (default False)
        If True, coefficients are constrained to be non-negative.
    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features, n_tasks)
        Parameter vector (W in the cost function formula).
    intercept_ : array, shape (n_tasks,)
        independent term in decision function.
    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Examples
    --------
    >>> from mutar import DirtyModel
    >>> import numpy as np
    >>> X = np.array([[[3, 1], [2, 0], [1, 0]],\
                     [[0, 2], [-1, 3], [1, -2]]], dtype=float)
    >>> coef = np.array([[1., 1.], [0., -1]])
    >>> y = np.array([x.dot(c) for x, c in zip(X, coef.T)])
    >>> y += 0.1
    >>> dirty = DirtyModel(alpha=0.15, beta=0.12).fit(X, y)
    >>> print(dirty.coef_shared_)
    [[ 0.4652447  0.3465437]
     [ 0.        -0.       ]]
    >>> print(dirty.coef_specific_)
    [[ 0.35453532  0.        ]
     [ 0.         -1.20766296]]

    """
    def __init__(self, alpha=1., beta=1., fit_intercept=True, normalize=False,
                 max_iter=2000, tol=1e-4, positive=False, warm_start=False):

        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         max_iter=max_iter, tol=tol, positive=positive,
                         warm_start=warm_start)
        self.alpha = alpha
        self.beta = beta

    def _fit(self, X, y):
        n_tasks = len(X)
        n_samples, n_features = X[0].shape

        if self.alpha == 0. and self.beta == 0.:
            warnings.warn("With alpha=beta=0, this algorithm does not converge"
                          " well. You are advised to use LinearRegression "
                          "estimator", stacklevel=2)

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_shared_ = np.zeros((n_features, n_tasks), dtype=X.dtype,
                                    order='F')
            coef_specific_ = np.zeros((n_features, n_tasks), dtype=X.dtype,
                                      order='F')
        coef_shared_, coef_specific_, R, n_iter = \
            solver_dirty(X, y, coef_shared_, coef_specific_,
                         self.alpha, self.beta, self.max_iter, self.tol,
                         self.positive)
        self.coef_ = coef_shared_ + coef_specific_
        self.coef_shared_ = coef_shared_
        self.coef_specific_ = coef_specific_
        self.residuals_ = R
        self.n_iter_ = n_iter


class GroupLasso(DirtyModel):
    """ GroupLasso estimator with L1/L2 mixed-norm as regularizer.

    The optimization objective for Group Lasso is::

        (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21

    Where::

        ||W||_21 = sum_i sqrt{sum_j w_ij^2}

    i.e. the sum of norm of each row.

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1/L2 term. Defaults to 1.
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.
    max_iter : int, optional
        The maximum number of iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    positive : boolean, optional (default False)
        If True, coefficients are constrained to be non-negative.
    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features, n_tasks)
        Parameter vector (W in the cost function formula).
    intercept_ : array, shape (n_tasks,)
        independent term in decision function.
    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Examples
    --------
    >>> from mutar import GroupLasso
    >>> import numpy as np
    >>> X = np.array([[[3, 1], [1, 0], [1, 0]],\
                     [[0, 2], [2, 3], [2, 3]]], dtype=float)
    >>> y = X.sum(axis=2) + 2
    >>> grouplasso = GroupLasso().fit(X, y)
    >>> print(grouplasso.coef_shared_)
    [[1.42045049 1.42045049]
     [0.         0.        ]]
    >>> print(grouplasso.coef_specific_)
    [[0. 0.]
     [0. 0.]]
    """
    def __init__(self, alpha=0.1, fit_intercept=True, normalize=False,
                 max_iter=2000, tol=1e-4, positive=False, warm_start=False):

        # if beta > alpha, Dirty models are equivalent to a Group Lasso
        beta = 10 * alpha
        super().__init__(alpha=alpha, beta=beta, fit_intercept=fit_intercept,
                         normalize=normalize, max_iter=max_iter, tol=tol,
                         positive=positive, warm_start=warm_start)


class IndRewLasso(MultitaskRegression):
    """ Independent Reweighted Lasso estimator with L1 regularizer.

    The optimization objective for IndRewLasso is::

        (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_0.5

    Where::

        ||W||_0.5 = sum_i sum_j sqrt|w_ij|

    Parameters
    ----------
    alpha : (float or array-like), shape (n_tasks)
        Optional, default ones(n_tasks)
        Constant that multiplies the L0.5 term. Defaults to 1.0
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.
    max_iter : int, optional
        The maximum number of inner loop iterations
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    positive : boolean, optional (default False)
        If True, coefficients are constrained to be non-negative.
    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features, n_tasks)
        Parameter vector (W in the cost function formula).
    intercept_ : array, shape (n_tasks,)
        independent term in decision function.
    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Examples
    --------
    >>> from mutar import IndRewLasso
    >>> import numpy as np
    >>> X = np.array([[[3, 1], [2, 0], [1, 0]],\
                     [[0, 2], [-1, 3], [1, -2]]], dtype=float)
    >>> coef = np.array([[1., 1.], [0., -1]])
    >>> y = np.array([x.dot(c) for x, c in zip(X, coef.T)])
    >>> y += 0.1
    >>> alpha = [0.1, 0.2]
    >>> relasso = IndRewLasso(alpha=alpha).fit(X, y)
    >>> print(relasso.coef_)
    [[ 0.92188134  0.        ]
     [ 0.         -1.33862186]]

    """
    def __init__(self, alpha=1., fit_intercept=True, normalize=False,
                 max_iter=2000, max_iter_reweighting=100, tol=1e-4,
                 positive=False, warm_start=False):

        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         max_iter=max_iter, tol=tol, positive=positive,
                         warm_start=warm_start)
        self.alpha = np.asarray(alpha)
        self.max_iter_reweighting = max_iter_reweighting

    def _fit(self, X, y):
        n_tasks = len(X)
        n_samples, n_features = X[0].shape

        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = np.zeros((n_features, n_tasks))
        weights = np.ones_like(self.coef_)
        coef_old = self.coef_.copy()
        self.loss_ = []
        for i in range(self.max_iter_reweighting):
            Xw = X * weights.T[:, None, :]
            coef_ = solver_lasso(Xw, y, self.alpha, self.max_iter, self.tol,
                                 positive=self.positive)
            coef_ = coef_ * weights
            err = abs(coef_ - coef_old).max()
            err /= max(abs(coef_).max(), abs(coef_old).max(), 1.)
            coef_old = coef_.copy()
            weights = 2 * (abs(coef_) ** 0.5 + 1e-10)
            obj = 0.5 * (utils.residual(X, coef_, y) ** 2).sum() / n_samples
            obj += (self.alpha[None, :] * abs(coef_) ** 0.5).sum()
            self.loss_.append(obj)
            if err < self.tol and i:
                break

        if i == self.max_iter_reweighting - 1 and i:
            warnings.warn('Reweighted objective did not converge.' +
                          ' You might want to increase ' +
                          'the number of iterations of reweighting.' +
                          ' Fitting data with very small alpha' +
                          ' may cause precision problems.',
                          ConvergenceWarning)
        self.coef_ = coef_


class IndLasso(IndRewLasso):
    """ Independent Lasso estimator with L1 regularizer.

    The optimization objective for IndLasso is::

        (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_1

    Where::

        ||W||_0.5 = sum_i sum_j |w_ij|

    Parameters
    ----------
    alpha : (float or array-like), shape (n_tasks)
        Optional, default ones(n_tasks)
        Constant that multiplies the L1 term. Defaults to 1.0
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.
    max_iter : int, optional
        The maximum number of inner loop iterations
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
    positive : boolean, optional (default False)
        If True, coefficients are constrained to be non-negative.
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features, n_tasks)
        Parameter vector (W in the cost function formula).
    intercept_ : array, shape (n_tasks,)
        independent term in decision function.
    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Examples
    --------
    >>> from mutar import IndLasso
    >>> import numpy as np
    >>> X = np.array([[[3, 1], [2, 0], [1, 0]],\
                     [[0, 2], [-1, 3], [1, -2]]], dtype=float)
    >>> coef = np.array([[1., 1.], [0., -1]])
    >>> y = np.array([x.dot(c) for x, c in zip(X, coef.T)])
    >>> y += 0.1
    >>> alpha = [0.1, 0.2]
    >>> relasso = IndLasso(alpha=alpha).fit(X, y)
    >>> print(relasso.coef_)
    [[ 0.85        0.        ]
     [ 0.         -1.31428571]]

    """
    def __init__(self, alpha=1., fit_intercept=True, normalize=False,
                 max_iter=2000, tol=1e-4, positive=False, warm_start=False):

        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                         normalize=normalize, max_iter=max_iter,
                         positive=positive, tol=tol, warm_start=warm_start)
        self.max_iter_reweighting = 1


class MultiLevelLasso(MultitaskRegression):
    """ MultiLevelLasso estimator with a non-convex product decomposition.

    The optimization objective for the Multilevel Lasso is::


        (1 / (2 * n_samples)) * ||Y - XW||^2_{Fro} + alpha ||W||_{1 1/2}}


    Where:

    .. math::

        \\|W\\|_{1 \\frac{1}{2}} = \\sum_j \\sqrt{\\|W_j\\|_1}


    Which is equivelent to::

        (1 / (2 * n_samples)) * ||Y - X(C[:, None] * S)||^2_Fro
        + beta * ||C||_1 + gamma * ||S||_1

    Where:

    .. math::

        C \\in \\R^{n\\_features}

        S \\in \\R^{n\\_features, n\\_tasks}

        \\alpha = 2  \\sqrt{\\beta * \\gamma}

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the common L{1 0.5} term. Defaults to 1.0
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.
    max_iter : int, optional
        The maximum number of iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    positive : boolean, optional (default False)
        If True, coefficients are constrained to be non-negative.
    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features, n_tasks)
        Parameter vector (W in the cost function formula).
    intercept_ : array, shape (n_tasks,)
        independent term in decision function.
    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Examples
    --------
    >>> from mutar import MultiLevelLasso
    >>> import numpy as np
    >>> X = np.array([[[3, 1], [2, 0], [1, 0]],\
                     [[0, 2], [-1, 3], [1, -2]]], dtype=float)
    >>> coef = np.array([[1., 1.], [0., -1]])
    >>> y = np.array([x.dot(c) for x, c in zip(X, coef.T)])
    >>> y += 0.1
    >>> mll = MultiLevelLasso(alpha=0.1).fit(X, y)
    >>> print(mll.coef_shared_)
    [0.91502387 1.04852402]
    >>> print(mll.coef_specific_)
    [[ 0.91339402  0.        ]
     [ 0.         -1.27834952]]
    >>> print(mll.coef_)
    [[ 0.83577734  0.        ]
     [ 0.         -1.34038017]]
    """
    def __init__(self, alpha=1., fit_intercept=True, normalize=False,
                 max_iter=2000, tol=1e-4, positive=False, warm_start=False):

        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         max_iter=max_iter, tol=tol, positive=positive,
                         warm_start=warm_start)
        self.alpha = alpha

    def _fit(self, X, y):
        n_tasks = len(X)
        n_samples, n_features = X[0].shape

        if self.alpha == 0.:
            warnings.warn("With alpha=0, this algorithm does not converge"
                          " well. You are advised to use LinearRegression "
                          "estimator", stacklevel=2)

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_shared_ = np.ones(n_features)
            coef_specific_ = np.zeros((n_features, n_tasks))
        coef_shared_, coef_specific_, n_iter = \
            solver_mll(X, y, C=coef_shared_, S=coef_specific_,
                       alpha=self.alpha, max_iter=self.max_iter, tol=self.tol,
                       positive=self.positive)
        self.coef_ = coef_shared_[:, None] * coef_specific_
        self.coef_shared_ = coef_shared_
        self.coef_specific_ = coef_specific_
        self.n_iter_ = n_iter


class ReMTW(MultitaskRegression):
    """A class for reweighted Multitask Wasserstein regularization.

    The optimization objective for Reweighted-MTW is::

        (1 / (2 * n_samples)) * ||Y - X(W+ - W-)||^2_Fro +
        alpha(sum_k OT(W+_k, Wb+) + sum_k OT(W-_k, Wb-)) +
        beta * (||W+||_{0.5} + ||W-||_{0.5})

    Where::

        OT is the Unbalanced Wasserstein distance with Kullback-Leibler
        marginal relaxation.

    and::

        ||W||_{0.5} = sum_i sum_j sqrt|w_ij|

    if `concomitant` is set to `True`, ReMTW also infers the standard deviation
    of each task. This allows to scale `beta` adaptively for each task
    according to the level of noise.

    The optimization objective for Concomitant Reweighted-MTW is::

        (1 / (2 * n_samples)) * sum||Y_k - X_k(W+k - W-k)||^2 / sigma_k +
        alpha(sum_k OT(W+_k, Wb+) + sum_k OT(W-_k, Wb-)) +
        beta * (||W+||_{0.5} + ||W-||_{0.5}) + sum sigma_k / 2

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the OT term. Defaults to 0.1
    beta : float, optional
        Constant that multiplies the L0.5 term. Defaults to 0.1
    M : array, shape (n_features, n_features)
        Ground metric matrix defining the Wasserstein distance.
    epsilon : float > 0, optional
        OT parameter. Weight of the entropy regularization.
    gamma : float > 0, optional
        OT parameter. Weight of the Kullback-Leibler marginal relaxation.
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.
    concomitant : boolean, optional (default False)
        If True, the concomittant version of MTW is used where the l1
        penalty is adaptively scaled to the noise std estimation.
    stable : boolean. optional (default False)
        if True, use log-domain Sinhorn stabilization from the first iter.
        if False, the solver will automatically switch to log-domain if
        numerical errors are encountered.
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
    callback : boolean. optional.
        if True, set a printing callback function to the solver.
    max_iter_ot : int, optional (default, 20)
        maximum Sinkhorn iterations
    max_iter_cd : int, optional (default, 10000)
        maximum coordinate descent iterations
    tol_ot : float, optional (default 1e-4)
        relative maximum change of the Wasserstein barycenter.
    tol_cd : float, optional (default 1e-4)
        relative maximum change of the coefficients in coordinate descent.
    n_jobs: int > 1, default 1
        number of threads used in coordinate descents
    gpu: boolean, optional (default False)
        if True, Sinkhorn iterations are performed on gpus using cupy.
    positive : boolean.
        if True, coefficients must be positive.
    max_iter : int, optional
        The maximum number of iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    ot_threshold : float, optional (default 0.)
        OT barycenters are computed on the support of coefs > ot_threshold.
        1e-7 recommended to speed up Sinkhorn with high dimensional data.
    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features, n_tasks)
        Parameter vector (W in the cost function formula).
    intercept_ : array, shape (n_tasks,)
        independent term in decision function.
    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Example
    -------
    >>> from mutar import ReMTW
    >>> import numpy as np
    >>> X = np.array([[[3, 1], [2, 0], [1, 0]],\
                     [[0, 2], [-1, 3], [1, -2]]], dtype=float)
    >>> coef = np.array([[0., 1.], [0., 2.]])
    >>> y = np.array([x.dot(c) for x, c in zip(X, coef.T)])
    >>> y += 0.1
    >>> remtw = ReMTW(alpha=0.2, beta=0.1).fit(X, y)
    >>> print(remtw.coef_)
    [[-0.01064798 -0.12868387]
     [ 0.10112546  1.55361523]]
    >>> print(remtw.barycenter_)
    [0.13216306 0.34976005]
    """

    def __init__(self, alpha=0.1, beta=0.1, M=None, epsilon=None,
                 gamma=None, fit_intercept=True, normalize=False,
                 concomitant=False, max_iter=2000, tol=1e-4,
                 warm_start=False, max_iter_reweighting=100,
                 stable=False, max_iter_ot=20, max_iter_cd=1000, tol_ot=1e-5,
                 tol_cd=1e-5, n_jobs=1, gpu=False, positive=False,
                 ot_threshold=0., tol_reweighting=0.001):

        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         max_iter=max_iter, tol=tol, positive=positive,
                         warm_start=warm_start)
        self.n_jobs = n_jobs
        self.M = M
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.stable = stable
        self.max_iter_ot = max_iter_ot
        self.tol_ot = tol_ot
        self.tol = tol
        self.n_jobs = n_jobs
        self.tol_cd = tol_cd
        self.tol_reweighting = tol_reweighting
        self.concomitant = concomitant
        self.max_iter_reweighting = max_iter_reweighting
        self.gpu = gpu
        self.ot_threshold = ot_threshold
        self.b1_ = None
        self.b2_ = None
        self.coef1_ = None
        self.coef2_ = None

    def _set_ot_params(self, n_features):
        """Set OT hyperparams if not specified."""
        if self.M is None:
            self.M = utils.groundmetric(n_features, p=2, normed=True)

        if self.epsilon is None:
            self.epsilon = 5. / n_features

        if self.gamma is None:
            self.gamma = utils.compute_gamma(0.8, self.M)

    def _fit(self, X, y):
        if self.alpha == 0. and self.beta == 0.:
            warnings.warn("With alpha=beta0, this algorithm does not converge"
                          " well. You are advised to use LinearRegression "
                          "estimator", stacklevel=2)
        n_tasks, n_samples, n_features = X.shape
        Xf = [np.asfortranarray(X[k]) for k in range(n_tasks)]
        mXf = [- np.asfortranarray(X[k]) for k in range(n_tasks)]
        Ls = (X ** 2).sum(axis=1)
        Ls[Ls == 0.] = Ls[Ls != 0].min()
        self._set_ot_params(n_features)
        self.loss_ = []
        weights1 = np.ones((n_features, n_tasks))
        weights2 = weights1.copy()
        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef1_ = np.ones((n_features, n_tasks)) / n_features
            self.coef2_ = np.ones((n_features, n_tasks)) / n_features
            if self.alpha == 0.:
                self.coef1_ *= 0.
                self.coef2_ *= 0.
            self.b1_ = None
            self.b2_ = None
            self.residuals_ = y.copy()
            self.sigma_ = np.ones(n_tasks)
        if self.positive:
            self.coef2_ *= 0.
        coef_old = np.zeros((n_features, n_tasks))
        beta1 = self.beta * weights1
        beta2 = self.beta * weights2
        for i in range(self.max_iter_reweighting):
            coef1, coef2, bar1, bar2, log, sigma, b1, b2, R = \
                solver_mtw(Xf, mXf, Ls, y, M=self.M, alpha=self.alpha,
                           beta1=beta1,
                           beta2=beta2,
                           epsilon=self.epsilon, gamma=self.gamma,
                           coef1=self.coef1_, coef2=self.coef2_,
                           R=self.residuals_,
                           b1=self.b1_, b2=self.b2_, sigmas=self.sigma_,
                           concomitant=self.concomitant,
                           stable=self.stable, tol=self.tol,
                           max_iter=self.max_iter, tol_ot=self.tol_ot,
                           max_iter_ot=self.max_iter_ot,
                           positive=self.positive,
                           n_jobs=self.n_jobs, tol_cd=self.tol_cd,
                           gpu=self.gpu, ot_threshold=self.ot_threshold)
            coef_ = coef1 - coef2
            self.coef1_ = coef1
            self.coef2_ = coef2
            self.coef_ = coef_
            self.sigma_ = sigma
            self.b1_ = b1
            self.b2_ = b2
            self.residuals_ = R
            err = abs(coef_ - coef_old).max()
            err /= max(abs(coef_).max(), abs(coef_old).max(), 1.)
            coef_old = coef_.copy()
            obj = self.alpha * (log["fot1"][-1] + log["fot2"][-1])
            obj += 0.5 * (R ** 2).sum(axis=1).dot(1 / self.sigma_) / n_samples
            obj += self.beta * (coef1 ** 0.5 + coef2 ** 0.5).sum()
            self.loss_.append(obj)
            if err < self.tol_reweighting and i:
                break

            if self.alpha == 0.:
                weights1 = 0.5 / (abs(coef_) ** 0.5 + 1e-10)
                weights2 = weights1
            else:
                weights1 = 0.5 / (coef1 ** 0.5 + 1e-10)
                weights2 = 0.5 / (coef2 ** 0.5 + 1e-10)

            beta1 = self.beta * weights1
            beta2 = self.beta * weights2

        if i == self.max_iter_reweighting - 1 and i:
            warnings.warn('Reweighted objective did not converge.' +
                          ' You might want to increase ' +
                          'the number of iterations of reweighting.' +
                          ' Fitting data with very small alpha and beta' +
                          ' may cause precision problems.',
                          ConvergenceWarning)
        self.barycenter1_ = bar1
        self.barycenter2_ = bar2
        self.barycenter_ = bar1 - bar2

        self.log_ = log

        return self

    def reset(self):
        if hasattr(self, 'coef_'):
            del self.coef_
            self.coef1_ = None
            self.coef2_ = None
            del self.barycenter2_
            del self.barycenter1_
            del self.barycenter_
            del self.log_
            self.residuals_ = None
            self.b1_ = None
            self.b2_ = None

            self.stable = False


class MTW(ReMTW):
    """A class for Multitask Wasserstein regularization.

    The optimization objective for Reweighted-MTW is::

        (1 / (2 * n_samples)) * ||Y - X(W+ - W-)||^2_Fro +
        alpha(sum_k OT(W+_k, Wb+) + sum_k OT(W-_k, Wb-)) +
        beta * (||W+||_1 + ||W-||_1)

    Where::

        OT is the Unbalanced Wasserstein distance with Kullback-Leibler
        marginal relaxation.

    and::

        ||W||_1 = sum_i sum_j |w_ij|

    if `concomitant` is set to `True`, ReMTW also infers the standard deviation
    of each task. This allows to scale `beta` adaptively for each task
    according to the level of noise.

    The optimization objective for Concomitant MTW is::

        (1 / (2 * n_samples)) * sum||Y_k - X_k(W+k - W-k)||^2 / sigma_k +
        alpha(sum_k OT(W+_k, Wb+) + sum_k OT(W-_k, Wb-)) +
        beta * (||W+||_1 + ||W-||_1) + sum sigma_k / 2

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the OT term. Defaults to 0.1
    beta : float, optional
        Constant that multiplies the L1 term. Defaults to 0.1
    M : array, shape (n_features, n_features)
        Ground metric matrix defining the Wasserstein distance.
    epsilon : float > 0, optional
        OT parameter. Weight of the entropy regularization.
    gamma : float > 0, optional
        OT parameter. Weight of the Kullback-Leibler marginal relaxation.
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.
    concomitant : boolean, optional (default False)
        If True, the concomittant version of MTW is used where the l1
        penalty is adaptively scaled to the noise std estimation.
    stable : boolean. optional (default False)
        if True, use log-domain Sinhorn stabilization from the first iter.
        if False, the solver will automatically switch to log-domain if
        numerical errors are encountered.
    callback : boolean. optional.
        if True, set a printing callback function to the solver.
    max_iter_ot : int, optional (default, 20)
        maximum Sinkhorn iterations
    max_iter_cd : int, optional (default, 10000)
        maximum coordinate descent iterations
    tol_ot : float, optional (default 1e-4)
        relative maximum change of the Wasserstein barycenter.
    tol_cd : float, optional (default 1e-4)
        relative maximum change of the coefficients in coordinate descent.
    n_jobs: int > 1, default 1
        number of threads used in coordinate descents
    gpu: boolean, optional (default False)
        if True, Sinkhorn iterations are performed on gpus using cupy.
    positive : boolean.
        if True, coefficients must be positive.
    max_iter : int, optional
        The maximum number of iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
    ot_threshold : float, optional (default 0.)
        OT barycenters are computed on the support of coefs > ot_threshold.
        1e-7 recommended to speed up Sinkhorn with high dimensional data.

    Attributes
    ----------
    coef_ : array, shape (n_features, n_tasks)
        Parameter vector (W in the cost function formula).
    intercept_ : array, shape (n_tasks,)
        independent term in decision function.
    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Example
    -------
    >>> from mutar import MTW
    >>> import numpy as np
    >>> X = np.array([[[3, 1], [2, 0], [1, 0]],\
                     [[0, 2], [-1, 3], [1, -2]]], dtype=float)
    >>> coef = np.array([[1., 1.], [0., -1]])
    >>> y = np.array([x.dot(c) for x, c in zip(X, coef.T)])
    >>> y += 0.1
    >>> mtw = MTW(alpha=1., beta=1.).fit(X, y)
    >>> print(mtw.coef_)
    [[ 0.28727587  0.49165381]
     [ 0.04883213 -0.93324644]]
    >>> print(mtw.barycenter_)
    [ 0.10630429 -0.13313143]
    """

    def __init__(self, alpha=0.1, beta=0.1, M=None, epsilon=None,
                 gamma=None, fit_intercept=True, normalize=False,
                 concomitant=False, max_iter=2000, tol=1e-4,
                 warm_start=False, stable=False, max_iter_ot=20,
                 max_iter_cd=1000, tol_ot=1e-5, tol_cd=1e-5, n_jobs=1,
                 gpu=False, positive=False, ot_threshold=0.):

        super().__init__(alpha=alpha, beta=beta, M=M, epsilon=epsilon,
                         gamma=gamma, max_iter_ot=max_iter_ot,
                         max_iter_cd=max_iter_cd, tol_cd=tol_cd, n_jobs=n_jobs,
                         fit_intercept=fit_intercept, normalize=normalize,
                         gpu=gpu, positive=positive,
                         max_iter=max_iter, tol=tol,
                         warm_start=warm_start, ot_threshold=ot_threshold)
        self.max_iter_reweighting = 1
