.. mutar documentation master file, created by
   sphinx-quickstart on Mon Jun 24 00:32:12 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Documentation

  api

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Tutorial - Examples

  auto_examples/index


MuTaR: Multi-task Regression in Python
======================================

This is a collection of sparse models for multi-task regression. Mutar models
fit regularized regression models on a sequence of related data (X_1, y_1) ...
(X_k, y_k).


.. _DirtyModel: https://papers.nips.cc/paper/4125-a-dirty-model-for-multi-task-learning
.. _GroupLasso: http://pages.stat.wisc.edu/~myuan/papers/glasso.final.pdf

* GroupLasso_: l1/l2 regularized regression with identical feature supports across tasks.
(Yuan and Lin, J. R Statistical Society 2006)

* DirtyModel_ Generalization of the Group Lasso with a partial overlap of features using a
composite l1/l2 and l1 regularization (Jalali et al., NeurIPS 2010).

Examples
--------

For examples, see `the tutorials <auto_examples/index>`_.

Contact:
--------
Please contact hicham.janati@inria.fr for any bug encountered / any further information.
