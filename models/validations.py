import numpy as np


def fit_checks(X, y):
    X = validate_features(X)
    y = validate_targets(y, n_samples=X.shape[0])
    return X, y

def predict_checks(self, X):
    if not hasattr(self, "n_features_in_"):
        raise RuntimeError(f"{type(self).__name__} must be fitted before calling predict.")

    X = validate_features(X)
    if X.shape[1] != self.n_features_in_:
        raise ValueError(
            f"X must have {self.n_features_in_} features, got {X.shape[1]}."
        )
    return X

def validate_targets(y, n_samples=None):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    if n_samples is not None and y.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    if y.size == 0:
        raise ValueError("X and y must contain at least one sample.")
    return y


def validate_binary_targets(y):
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only binary labels 0 and 1.")
    return y


def validate_features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must contain at least one sample and one feature.")
    return X
