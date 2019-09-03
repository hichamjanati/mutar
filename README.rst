MuTaR: Multi-task Regression in Python
======================================

-----------
Description
-----------

|Travis|_ |AppVeyor|_ |Codecov|_

.. |Travis| image:: https://travis-ci.com/hichamjanati/mutar.svg?branch=master
.. _Travis: https://travis-ci.com/hichamjanati/mutar

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/hichamjanati/mutar

.. |Codecov| image:: https://codecov.io/gh/hichamjanati/mutar/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/hichamjanati/mutar


MuTaR is a collection of sparse models for multi-task regression. Mutar models
fit regularized regression on a sequence of related linear
models (X_1, y_1) ... (X_k, y_k) and follows `scikit-learn's <http://scikit-learn.org>`_ API.
Compared with scikit-learn's MultiTaskLasso, MuTaR allows for a different design
data X for each task.

Mutar models include:

* Independent linear models:
    * Independent Lasso estimator
    * Independent Re-weighted (Adaptive) Lasso estimator

* Group-norms multi-task linear models:
    * `GroupLasso`: The Group Lasso is an l1/l2 regularized regression with identical feature supports across tasks `(Yuan and Lin, J. R Statistical Society 2006) <http://pages.stat.wisc.edu/~myuan/papers/glasso.final.pdf>`_.
    * `DirtyModel`: Dirty models are a generalization of the Group Lasso with a partial overlap of features. They are defined using a composite l1/l2 and l1 regularization `(Jalali et al., NeurIPS 2010) <https://papers.nips.cc/paper/4125-a-dirty-model-for-multi-task-learning?>`_.
    * `MultiLevelLasso` : Multilevel Lasso is a non-convex model that enhances further sparsity and encourages partial overlap with a product decomposition `(Lozano and Swirszcz, ICML 2012) <https://icml.cc/2012/papers/207.pdf>`_.

* Optimal transport regularized models:
    * `MTW`: Multi-task Wasserstein is a sparse regression model where relevant features across tasks are close according to some defined geometry. `(Janati et al., AISTATS 2019) <http://proceedings.mlr.press/v89/janati19a.html>`_.
    * `ReMTW`: Reweighted MTW is a non-convex variant of MTW that promotes even more sparsity and reduces the amplitude bias caused by the L1 norm. Both models are implemented with a `concomitant` argument for inferring the standard deviation of each task and adapting the amount of regularization accordingly.


Installation
------------

To install the last release of MuTaR:

::

    pip install -U mutar


To get the current development version:
::

    git clone https://github.com/hichamjanati/mutar
    cd mutar
    python setup.py develop

We recommend creating this minimal `conda env <https://raw.githubusercontent.com/hichamjanati/mutar/master/environment.yml>`_

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
