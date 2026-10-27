import numpy as np

from .validations import validate_targets


class ClassifierMixin:
    def score(self, X, y):
        y = validate_targets(y)
        y_pred = self.predict(X)
        validate_targets(y, n_samples=y_pred.shape[0])
        return np.mean(y_pred == y)
