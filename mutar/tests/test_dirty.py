import pytest
# import numpy as np

from sklearn.datasets import load_boston
# from sklearn.utils.testing import assert_array_equal
# from sklearn.utils.testing import assert_allclose

from mutar import DirtyModel


@pytest.fixture
def data():
    X, y = load_boston(return_X_y=True)
    X = X[:20]
    y = y[:20]
    return X, y


def test_dirty(data):
    est = DirtyModel(alpha=0.1, beta=0.1)

    X, y = data
    X = [X, X]
    y = [y, y]
    est.fit(X, y)
    assert hasattr(est, 'is_fitted_')

    # assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))
