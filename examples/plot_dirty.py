"""
Joint feature selection with Dirty models
=========================================

`DirtyModel` estimates a set of sparse coefficients for multiple regression
models that share a fraction of non-zero features. It is a generalization of
The `GroupLasso` estimator. It also takes a 3D `X (n_tasks, n_samples,
n_features)` and a 2D `y (n_tasks, n_samples)`.

DirtyModel solves the optimization problem::


        (1 / (2 * n_samples)) * ||Y - X(W_1 + W_2)||^2_Fro + alpha * ||W_1||_21
        + beta * ||W_2||_1

Where::

        ||W||_21 = \\sum_i \\sqrt{\\sum_j w_ij^2}

i.e. the sum of norm of each row. and::

        ||W||_1 = \\sum_i \\sum_j |w_ij|

"""

# Author: Hicham Janati (hicham.janati@inria.fr)
#
# License: BSD (3-clause)

import numpy as np
from matplotlib import pyplot as plt

from mutar import DirtyModel

##########################################################
# Generate multi-task data
#

rng = np.random.RandomState(42)
n_tasks, n_samples, n_features = 10, 100, 30
X = rng.randn(n_tasks, n_samples, n_features)

# generate random coefficients and make it sparse
# select support
support = rng.rand(n_features, n_tasks) > 0.95
coef = support * rng.randn(n_features, n_tasks)

# make features 0, 2, 4 and 6 shared
coef[:7:2] = rng.randn(4, n_tasks)

y = np.array([x.dot(c) for x, c in zip(X, coef.T)])

# add noise
y += 0.2 * np.std(y) + rng.randn(n_tasks, n_samples)

##########################################################
# Dirty models fit

alpha = 0.5
beta = 0.25
dirty = DirtyModel(alpha=alpha, beta=beta)
dirty.fit(X, y)


##########################################################
# Plot the supports of the true and obtained coefficients.

f, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, coef, name in zip(axes, [coef, dirty.coef_], ["True", "DirtyModel"]):
    ax.imshow(coef != 0)
    ax.set_title(name)
    ax.set_xlabel("Tasks")
    ax.set_ylabel("Features")

plt.show()
