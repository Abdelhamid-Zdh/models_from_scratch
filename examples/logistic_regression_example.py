from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import LogisticRegression


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=93,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        lr=0.1,
        epochs=20_000,
        eps=1e-10,
        patience=2_000,
    ).fit(X_train, y_train)
    sklearn_model = SklearnLogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5_000,
    ).fit(X_train, y_train)

    model_accuracy = model.score(X_test, y_test)
    sklearn_accuracy = sklearn_model.score(X_test, y_test)
    print(f"From-scratch logistic regression accuracy: {model_accuracy:.2%}")
    print(f"scikit-learn logistic regression accuracy: {sklearn_accuracy:.2%}")
    agreement = (model.predict(X_test) == sklearn_model.predict(X_test)).mean()
    print(f"Prediction agreement: {agreement:.2%}")


if __name__ == "__main__":
    main()