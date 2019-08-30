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

This is a collection of sparse models for multi-task regression. Mutar models
fit regularized regression models on a sequence of related data (X_1, y_1) ...
(X_k, y_k).

* GroupLasso: l1/l2 regularized regression with identical feature supports across tasks. `(Yuan and Lin, J. R Statistical Society 2006) <http://pages.stat.wisc.edu/~myuan/papers/glasso.final.pdf>`_
* DirtyModel Generalization of the Group Lasso with a partial overlap of features using a composite l1/l2 and l1 regularization `(Jalali et al., NeurIPS 2010) <https://papers.nips.cc/paper/4125-a-dirty-model-for-multi-task-learning?>`_.

Installation
------------

On a miniconda environment:

::

    git clone https://github.com/hichamjanati/mutar
    cd groupmne
    python setup.py develop

Otherwise, we recommend creating this minimal `conda env <https://raw.githubusercontent.com/hichamjanati/mutar/master/environment.yml>`_

::

    conda env create --file environment.yml
    conda activate mutar-env
    git clone https://github.com/hichamjanati/mutar
    cd mutar
    python setup.py develop


Examples
--------

See `the examples <https://hichamjanati.github.io/mutar/auto_examples>`_.
