.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_

.. |Travis| image:: https://travis-ci.com/hichamjanati/mutar.svg?branch=master
.. _Travis: https://travis-ci.com/hichamjanati/mutar

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/hichamjanati/mutar

.. |Codecov| image:: https://codecov.io/gh/hichamjanati/mutar/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/hichamjanati/mutar


MuTaR: Multi-task Regression in Python
======================================

MuTaR is a collection of sparse models for multi-task regression. Mutar models
fit regularized regression models on a sequence of related linear
models (X_1, y_1) ... (X_k, y_k) and follows `scikit-learn's <http://scikit-learn.org>`_ API.
Compared with scikit-learn's MultiTaskLasso, MuTaR allows for a different design
data X for each task.

Estimators
----------

Mutar models include:

1. Mixed-norms multi-task linear models:

* GroupLasso: l1/l2 regularized regression with identical feature supports across tasks. `(Yuan and Lin, J. R Statistical Society 2006) <http://pages.stat.wisc.edu/~myuan/papers/glasso.final.pdf>`_
* DirtyModel Generalization of the Group Lasso with a partial overlap of features using a composite l1/l2 and l1 regularization `(Jalali et al., NeurIPS 2010) <https://papers.nips.cc/paper/4125-a-dirty-model-for-multi-task-learning?>`_.

2. Independent linear models:

* Independent Lasso estimator
* Independent Re-weighted (Adaptive) Lasso estimator

Installation
------------

On a miniconda environment:

::

    git clone https://github.com/hichamjanati/mutar
    cd mutar
    python setup.py develop

Otherwise, we recommend creating this minimal `conda env <https://raw.githubusercontent.com/hichamjanati/mutar/master/environment.yml>`_

::

    conda env create --file environment.yml
    conda activate mutar-env
    git clone https://github.com/hichamjanati/mutar
    cd mutar
    python setup.py develop

Example
-------

.. code:: python

    >>> import numpy as np
    >>> from mutar import GroupLasso
    >>> # create some X (n_tasks, n_samples, n_features)
    >>> X = np.array([[[3., 1.], [2., 0.]], [[0., 2.], [-1., 3.]]])
    >>> print(X.shape)
    (2, 2, 2)
    >>> # and target y (n_tasks, n_samples)
    >>> y = np.array([[-3., 1.], [1., -2.]])
    >>> print(y.shape)
    (2, 2)
    >>> gl = GroupLasso(alpha=1.)
    >>> coef = gl.fit(X, y).coef_
    >>> print(coef.shape)
    (2, 2)
    >>> # coefficients (n_features, n_tasks)
    >>> # share the same support
    >>> print(coef)
    [[-0.8  0.6]
     [-0.  -0. ]]



Documentation
-------------

See the doc and use examples at the `MuTaR webpage <https://hichamjanati.github.io/mutar>`_.
