from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import KNN


def main():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=93,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNN(n_neighbors=5).fit(X_train, y_train)
    sklearn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

    print(f"From-scratch KNN accuracy: {model.score(X_test, y_test):.2%}")
    print(f"scikit-learn KNN accuracy: {sklearn_model.score(X_test, y_test):.2%}")
    agreement = (model.predict(X_test) == sklearn_model.predict(X_test)).mean()
    print(f"Prediction agreement: {agreement:.2%}")


if __name__ == "__main__":
    main()
