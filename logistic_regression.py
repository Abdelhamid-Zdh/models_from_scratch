import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler


class Logistic_Regression:
    def __init__(self, X, y):
        """Initialize parameters and prepare standardized training and test data.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Binary labels with shape (n_samples,).
        """
        self._X, self._y, self.w, self.b = X, y, np.zeros(X.shape[1]), 0
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, test_size=0.2 ,random_state=93)
        self._standardization()

    def _inspect(self):
        """Print the dataset dimensions and a small sample of features and labels."""
        print(f"Shape: X({self._X.shape}), y({self._y.shape})")
        print(f"Data: {self._X[:5, :5]} \nTarget: {self._y[:5]}")

    def _standardization(self):
        """Standardize features using training-set means and standard deviations.

        The training statistics are also applied to the test features to avoid
        using test-set information during preprocessing.
        """
        mean = np.mean(self._X_train, axis=0)
        std = np.std(self._X_train, axis=0) + 1e-8
        self._X_train = (self._X_train - mean) / std
        # Reuse training statistics so test data does not leak into preprocessing.
        self._X_test = (self._X_test - mean) / std

    def _sigmoid(self, prediction):
        """Map a linear score or array of scores to probabilities in [0, 1].

        Args:
            prediction: Scalar or array of linear scores.

        Returns:
            The sigmoid-transformed score or array of probabilities.
        """
        return 1.0 / (1.0 + np.exp(-prediction))

    def _predict(self, y):
        """Calculate positive-class probabilities from one or more feature vectors.

        Args:
            y: Feature vector or matrix whose last dimension matches the weights.

        Returns:
            Predicted positive-class probability or array of probabilities.
        """
        return self._sigmoid(y@self.w + self.b)

    def _cross_entropy_loss(self, prediction):
        """Calculate mean binary cross-entropy for training predictions.

        Args:
            prediction: Predicted positive-class probabilities for training data.

        Returns:
            The mean binary cross-entropy loss.
        """
        y = self._y_train
        eps = 1e-15  # Keep logarithms finite when predicted probabilities reach 0 or 1.
        prediction = np.clip(prediction, eps, 1-eps) 
        return -np.mean(
            y * np.log(prediction)
            + (1 - y) * np.log(1 - prediction)
        )

    def _gradient_descent(self, predictions):
        """Compute gradients of the mean binary cross-entropy loss.

        Args:
            predictions: Predicted positive-class probabilities for training data.

        Returns:
            A tuple containing the weight gradient and bias gradient.
        """
        errors = predictions - self._y_train
        grad_w = self._X_train.T @ errors / len(self._y_train)
        grad_b = np.mean(errors)
        return (grad_w, grad_b)

    def fit(self, lr=0.1, epochs=5000, eps=1e-8, patience=10):
        """Train the model with gradient descent and early stopping.

        Args:
            lr: Learning rate used for each parameter update.
            epochs: Maximum number of full training-set updates.
            eps: Minimum loss improvement and parameter-change tolerance.
            patience: Number of unimproved epochs allowed before stopping.

        Returns:
            This fitted model.
        """
        no_improve_count = 0
        best_loss = float('inf')
        best_weights = None
        best_bias = self.b
        
        for epoch in range(epochs):
            prediction = self._predict(self._X_train)
            loss = self._cross_entropy_loss(prediction)
            grad_w, grad_b = self._gradient_descent(prediction)
            
            # Keep the previous parameters to detect when updates become negligible.
            old_w = self.w.copy()
            old_b = self.b
            
            self.w = self.w - lr * grad_w
            self.b = self.b - lr * grad_b
            
            weight_diff = np.linalg.norm(self.w - old_w)
            bias_diff = abs(self.b - old_b)
            if weight_diff < eps and bias_diff < eps:
                print(f"Converged at epoch {epoch}")
                break
            
            # Early stopping based on loss improvement
            if loss < best_loss - eps:
                best_loss = loss
                no_improve_count = 0
                best_weights = self.w.copy()
                best_bias = self.b
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"No improvement for {patience} epochs, stopping at {epoch}")
                    # Undo updates made after the best observed training loss.
                    self.w = best_weights
                    self.b = best_bias
                    break
        
        return self  

    def classify(self, sample):
        """Predict a binary label for one feature vector.

        Args:
            sample: Feature vector with one value per model weight.

        Returns:
            Integer class label, 1 when the predicted probability is at least 0.5,
            and 0 otherwise.
        """
        return int(self._predict(sample) >= .5)

    def test(self):
        """Evaluate accuracy on the held-out test set.

        Returns:
            Fraction of test labels predicted correctly, from 0 to 1.
        """
        y_pred = np.array([
            self.classify(sample)
            for sample in self._X_test
        ])
        return np.mean(y_pred == self._y_test) 

        
data = load_breast_cancer()
X, y = data.data, data.target

model = Logistic_Regression(X, y)
model.fit()
print(f"My Logistic Regression Acc= {model.test() * 100:.2f}%")

# Use the same split for a fair comparison with the NumPy implementation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=93
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
sklearn_model = SklearnLogisticRegression(max_iter=1000)
sklearn_model.fit(X_train_scaled, y_train)
y_pred = sklearn_model.predict(X_test_scaled)

print(f"Sklearn Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
