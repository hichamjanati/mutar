"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import warnings

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .solver_dirty import solver_dirty


class DirtyModel(BaseEstimator, RegressorMixin):
    """ DirtyModel estimator with L1 and L1/L2 mixed-norm as regularizers.

    The optimization objective for Dirty models is::

        (1 / (2 * n_samples)) * ||Y - X(W_1 + W_2)||^2_Fro + alpha * ||W_1||_21
        + beta * ||W_2||_1

    Where::

        ||W||_21 = \\sum_i \\sqrt{\\sum_j w_ij^2}

    i.e. the sum of norm of each row.

    and::

        ||W||_1 = \\sum_i \\sum_j |w_ij|

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
        See :term:`the Glossary <warm_start>`.
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
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start

    def fit(self, X, y):
        n_tasks = len(X)
        n_samples, n_features = X[0].shape

        if self.alpha == 0 and self.beta == 0.:
            warnings.warn("With alpha=beta=0, this algorithm does not converge"
                          " well. You are advised to use LinearRegression "
                          "estimator", stacklevel=2)
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

        if self.fit_intercept:
            self.intercept_ = y_offset
            self.intercept_ -= np.einsum("kj,jk->k", X_offset, self.coef_)
        else:
            self.intercept_ = np.zeros(n_tasks)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like}, shape (n_tasks, n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_tasks, n_samples)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=False, allow_nd=True)
        check_is_fitted(self, 'is_fitted_')
        y_pred = [x.dot(c) for x, c in zip(X, self.coef_.T)]
        return np.array(y_pred)


class GroupLasso(DirtyModel):
    """ GroupLasso estimator with L1/L2 mixed-norm as regularizer.

    The optimization objective for Dirty models is::

        (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21

    Where::

        ||W||_21 = \\sum_i \\sqrt{\\sum_j w_ij^2}

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
        See :term:`the Glossary <warm_start>`.
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
