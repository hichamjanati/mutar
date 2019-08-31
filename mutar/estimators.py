"""
Mutar estimators module.
"""
import numpy as np
import warnings

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score

from .solvers import solver_dirty, solver_lasso


class MultitaskRegression(BaseEstimator, RegressorMixin):
    """ Multitask regression abstract class.

    Parameters
    ----------
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
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
    def __init__(self, fit_intercept=True,
                 max_iter=2000, tol=1e-4, warm_start=False):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start

    def _get_offset(self, X, y):
        n_tasks = len(X)
        n_samples, n_features = X[0].shape
        X, y = check_X_y(X, y, accept_sparse=False, allow_nd=True,
                         multi_output=True)

        if y.shape != (n_tasks, n_samples):
            raise ValueError("Data shape not understood. X must be "
                             "(n_tasks, n_samples, n_features) and y must be "
                             "(n_tasks, n_samples)")

        if self.fit_intercept:
            X = X.copy()
            y = y.copy()
            X_offset = X.mean(axis=1)
            y_offset = y.mean(axis=1)
            X -= X_offset[:, None, :]
            y -= y_offset[:, None]
        else:
            X_offset = np.zeros((n_tasks, 1, n_features))
            y_offset = np.zeros((n_tasks, 1))
        return X, y, X_offset, y_offset

    def _set_intercept(self, X_offset, y_offset):
        """Set intercept after fit."""
        assert self.is_fitted_
        if self.fit_intercept:
            self.intercept_ = y_offset
            self.intercept_ -= np.einsum("kj,jk->k", X_offset, self.coef_)
        else:
            self.intercept_ = np.zeros_like(y_offset).flatten()

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
    def __init__(self, alpha=1., beta=1., fit_intercept=True,
                 max_iter=2000, tol=1e-4, warm_start=False):

        super().__init__(fit_intercept=fit_intercept, max_iter=max_iter,
                         tol=tol, warm_start=warm_start)
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, y):
        n_tasks = len(X)
        n_samples, n_features = X[0].shape

        if self.alpha == 0. and self.beta == 0.:
            warnings.warn("With alpha=beta=0, this algorithm does not converge"
                          " well. You are advised to use LinearRegression "
                          "estimator", stacklevel=2)
        X, y, X_offset, y_offset = self._get_offset(X, y)

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_shared_ = np.zeros((n_features, n_tasks), dtype=X.dtype,
                                    order='F')
            coef_specific_ = np.zeros((n_features, n_tasks), dtype=X.dtype,
                                      order='F')
        coef_shared_, coef_specific_, R, n_iter = \
            solver_dirty(X, y, coef_shared_, coef_specific_,
                         self.alpha, self.beta, self.max_iter, self.tol,
                         )
        self.coef_ = coef_shared_ + coef_specific_
        self.coef_shared_ = coef_shared_
        self.coef_specific_ = coef_specific_
        self.residuals_ = R
        self.n_iter_ = n_iter
        self.is_fitted_ = True

        self._set_intercept(X_offset, y_offset)
        return self


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
    def __init__(self, alpha=0.1, fit_intercept=True,
                 max_iter=2000, tol=1e-4, warm_start=False):

        # if beta > alpha, Dirty models are equivalent to a Group Lasso
        beta = 10 * alpha
        super().__init__(alpha=alpha, beta=beta, fit_intercept=fit_intercept,
                         max_iter=max_iter, tol=tol, warm_start=warm_start)


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
    max_iter : int, optional
        The maximum number of inner loop iterations
    max_reweighting_iter : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
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
    def __init__(self, alpha=1., fit_intercept=True, max_iter=2000,
                 max_reweighting_iter=100, tol=1e-4, warm_start=False):

        super().__init__(fit_intercept=fit_intercept, max_iter=max_iter,
                         tol=tol, warm_start=warm_start)
        self.alpha = alpha
        self.max_reweighting_iter = max_reweighting_iter

    def fit(self, X, y):
        n_tasks = len(X)
        n_samples, n_features = X[0].shape

        X, y, X_offset, y_offset = self._get_offset(X, y)

        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = np.zeros((n_features, n_tasks))
        weights = np.ones_like(self.coef_)
        coef_old = self.coef_.copy()
        self.all_coefs = []
        for i in range(self.max_reweighting_iter):
            Xw = X * weights.T[:, None, :]
            coef_ = solver_lasso(Xw, y, self.alpha, self.max_iter, self.tol)
            coef_ = coef_ * weights
            self.all_coefs.append(coef_)
            err = abs(coef_ - coef_old).max()
            err /= max(abs(coef_).max(), abs(coef_old).max(), 1.)
            coef_old = coef_.copy()
            weights = 2 * abs(coef_) ** 0.5

            if err < self.tol and i:
                break
        self.coef_ = coef_
        self.is_fitted_ = True

        self._set_intercept(X_offset, y_offset)
        return self


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
    max_iter : int, optional
        The maximum number of inner loop iterations
    max_reweighting_iter : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
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
    def __init__(self, alpha=1., fit_intercept=True, max_iter=2000,
                 tol=1e-4, warm_start=False):

        super().__init__(fit_intercept=fit_intercept, max_iter=max_iter,
                         tol=tol, warm_start=warm_start)
        self.alpha = alpha
        self.max_reweighting_iter = 1
