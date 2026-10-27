import numpy as np
from .classifier_mixin import ClassifierMixin
from .validations import fit_checks, predict_checks


class KNN(ClassifierMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X, y = fit_checks(X, y)
        self._validate_n_neighbors(y.size)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.classes_, self.y_encoded_ = np.unique(self.y_train_, return_inverse=True)
        self.n_features_in_ = self.X_train_.shape[1]
        return self

    def predict(self, X):
        X = predict_checks(self, X)
        n_neighbors = self._validate_n_neighbors(self.X_train_.shape[0])

        squared_distances = (
            np.sum(X**2, axis=1)[:, np.newaxis]
            + np.sum(self.X_train_**2, axis=1)[np.newaxis, :]
            - 2 * X @ self.X_train_.T
        )
        np.maximum(squared_distances, 0, out=squared_distances)
        nearest_indices = np.argsort(squared_distances, axis=1, kind="stable")[
            :, :n_neighbors
        ]
        nearest_classes = self.y_encoded_[nearest_indices]

        class_counts = np.zeros(
            (X.shape[0], self.classes_.size),
            dtype=np.intp,
        )
        sample_indices = np.repeat(np.arange(X.shape[0]), n_neighbors)
        np.add.at(class_counts, (sample_indices, nearest_classes.ravel()), 1)
        return self.classes_[np.argmax(class_counts, axis=1)]

    def _validate_n_neighbors(self, n_samples):
        n_neighbors = self.n_neighbors
        if (
            isinstance(n_neighbors, bool)
            or not isinstance(n_neighbors, (int, np.integer))
            or n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be a positive integer.")
        if n_neighbors > n_samples:
            raise ValueError(
                "n_neighbors cannot exceed the number of training samples."
            )
        return n_neighbors
