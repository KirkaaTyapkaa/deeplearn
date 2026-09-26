import numpy as np
from sklearn.datasets import load_iris


def load_data():
    """Iris: стратифікований поділ 35/15 для кожного класу, стандартизація за train."""
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    rng = np.random.default_rng(0)
    train_idx, test_idx = [], []
    for c in (0, 1, 2):
        idx = rng.permutation(np.flatnonzero(y == c))
        train_idx.append(idx[:35])
        test_idx.append(idx[35:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # середнє і std тільки по train
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test
