import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # For side to side comparaison
from sklearn.preprocessing import StandardScaler # same
from sklearn.metrics import accuracy_score # same

class KNN:
    def __init__(self, X, y):
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, test_size=0.2, random_state=93)
        self._standardization()  

    def _standardization(self):
        """Standardize the dataset (Mean:0, STD: 1)"""
        mean = np.mean(self._X_train, axis=0)
        std = np.std(self._X_train, axis=0) + 1e-8
        self._X_train = (self._X_train - mean) / std
        self._X_test = (self._X_test - mean) / std # using the X_train mean and std to avoid leakage

    def _euclidean_distance(self, X, Y):
        """
        Compute the distance between point Y and each row of matrix X.

        Args:
            X (np.ndarray): Matrix of shape (n, d).
            Y (np.ndarray): Point of shape (d,).

        Returns:
        np.ndarray: Array of shape (n,), where result[i] is the distance between X[i] and Y.
        """
        return np.sqrt(np.sum((X - Y) ** 2, axis=1))

    def _classify(self, X, k=5):
        """
        Predict the class label for a single input point.

        Args:
            X: A feature vector of shape (d,) representing one sample.
            k: Number of nearest neighbors to consider.

        Returns:
            The most common class among the k nearest training samples.
        """
        distances = self._euclidean_distance(self._X_train, X)
        nearest_k = self._y_train[np.argsort(distances)[:k]]
        values, counts = np.unique(nearest_k, return_counts=True)
        most_common = values[np.argmax(counts)]
        return most_common

    def fit(self):
        """
        Classify every sample in the test set and return the accuracy.

        Returns:
            The fraction of correctly classified test samples.
        """
        y_pred = np.array([
            self._classify(point)
            for point in self._X_test
        ])
        return np.mean(y_pred == self._y_test)


data = load_iris()
X, y = data.data, data.target

knn = KNN(X, y)
print(f"My KNN Acc= {knn.fit()*100:.2f}%")

# sklearn Prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=93)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

print(f"Sklearn Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
