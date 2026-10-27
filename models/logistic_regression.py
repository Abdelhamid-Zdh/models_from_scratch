import numpy as np
from .classifier_mixin import ClassifierMixin
from .validations import fit_checks, predict_checks, validate_binary_targets


class LogisticRegression(ClassifierMixin):
    def __init__(self, lr=0.1, epochs=1000, eps=1e-8, patience=10):
        self.lr = lr
        self.epochs = epochs
        self.eps = eps
        self.patience = patience

    def fit(self, X, y):
        X, y = fit_checks(X, y)
        y = validate_binary_targets(y).astype(float)

        self.n_features_in_ = X.shape[1]
        self.weights_ = np.zeros(X.shape[1])
        self.bias_ = 0.0
        no_improve_count = 0
        best_loss = float('inf')
        best_weights = self.weights_.copy()
        best_bias = self.bias_

        for _ in range(self.epochs):
            old_w = self.weights_.copy()
            old_b = self.bias_
            prediction = self._sigmoid(X @ old_w + old_b)
            loss = self._cross_entropy_loss(y, prediction)
            grad_w, grad_b = self._gradients(X, y, prediction)

            if loss < best_loss - self.eps:
                best_loss = loss
                no_improve_count = 0
                best_weights = old_w.copy()
                best_bias = old_b
            else:
                no_improve_count += 1

            self.weights_ = old_w - self.lr * grad_w
            self.bias_ = old_b - self.lr * grad_b

            weight_diff = np.linalg.norm(self.weights_ - old_w)
            bias_diff = abs(self.bias_ - old_b)
            if weight_diff < self.eps and bias_diff < self.eps:
                break

            if no_improve_count >= self.patience:
                self.weights_ = best_weights
                self.bias_ = best_bias
                break
        return self

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X):
        X = predict_checks(self, X)
        return self._sigmoid(X @ self.weights_ + self.bias_)

    def _sigmoid(self, prediction):
        return 1.0 / (1.0 + np.exp(-prediction))

    def _cross_entropy_loss(self, y, prediction):
        eps = 1e-15  # Keep logarithms finite when predicted probabilities reach 0 or 1.
        prediction = np.clip(prediction, eps, 1 - eps)
        return -np.mean(
            y * np.log(prediction)
            + (1 - y) * np.log(1 - prediction)
        )

    def _gradients(self, X, y, predictions):
        errors = predictions - y
        grad_w = X.T @ errors / len(y)
        grad_b = np.mean(errors)
        return (grad_w, grad_b)
