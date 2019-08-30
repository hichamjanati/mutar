"""
Joint feature selection with the Group Lasso
============================================

The GroupLasso estimates a set of sparse coefficients for multiple regression
models that share the same non-zero features. All features (variable) are
either zero for all tasks or for None of them. The `GroupLasso` takes a 3D
`X (n_tasks, n_samples, n_features)` and a 2D `y (n_tasks, n_samples)`. If the
design matrix `X` of the data is the same for all tasks, we recommand using
scikit-learn's MultiTaskLasso.

The Group Lasso solves the optimization problem::


        (1 / (2 * n_samples)) * ||Y - XW||_Fro^2
        + alpha * ||W||_21


"""

# Author: Hicham Janati (hicham.janati@inria.fr)
#
# License: BSD (3-clause)

import numpy as np
from matplotlib import pyplot as plt

from mutar import GroupLasso

##########################################################
# Generate multi-task data
#

rng = np.random.RandomState(42)
n_tasks, n_samples, n_features = 10, 100, 30
X = rng.randn(n_tasks, n_samples, n_features)
coef = np.zeros((n_features, n_tasks))

# The features 0, 2, 4, 6 are shared for all 10 tasks
coef[:7:2] = rng.randn(4, n_tasks)

# We pick some additional features for tasks 1 and 3
coef[[10, 20], [1, 3]] = rng.randn(2)

y = np.array([x.dot(c) for x, c in zip(X, coef.T)])

# add noise
y += 0.2 * np.std(y) + rng.randn(n_tasks, n_samples)

##########################################################
# Group Lasso fit

alpha = 0.5
gl = GroupLasso(alpha=alpha)
gl.fit(X, y)


##########################################################
# Plot the supports of the true and obtained coefficients.

f, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, coef, name in zip(axes, [coef, gl.coef_], ["True", "GroupLasso"]):
    ax.imshow(coef != 0)
    ax.set_title(name)
    ax.set_xlabel("Tasks")
    ax.set_ylabel("Features")

plt.show()
