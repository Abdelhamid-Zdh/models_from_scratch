import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from models import KNN


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _expect_raises(exception_type, function, message):
    try:
        function()
    except exception_type:
        return
    raise AssertionError(message)


def _check_hand_examples():
    X = np.array([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]])
    y = np.array([0, 1, 2, 2, 1, 0])

    one_neighbor = KNN(n_neighbors=1).fit(X, y)
    np.testing.assert_array_equal(
        one_neighbor.predict([[-2.0], [0.0], [4.0]]),
        [0, 1, 2],
    )

    three_neighbors = KNN(n_neighbors=3).fit(X, y)
    # At x=0, the three nearest labels are 0, 1, and 2; ties choose the lowest label.
    np.testing.assert_array_equal(three_neighbors.predict([[0.0], [4.0]]), [0, 2])


def _check_sklearn_agreement(loader):
    X, y = loader(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=93,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    predictions = KNN(n_neighbors=5).fit(X_train, y_train).predict(X_test)
    reference = KNeighborsClassifier(n_neighbors=5).fit(
        X_train, y_train
    ).predict(X_test)
    _check(
        np.array_equal(predictions, reference),
        f"KNN predictions disagree with scikit-learn on {loader.__name__}.",
    )


def _check_properties_and_errors():
    X, y = load_iris(return_X_y=True)
    model = KNN(n_neighbors=1).fit(X, y)
    _check(
        np.array_equal(model.predict(X), y),
        "With k=1, predictions on training samples must equal their labels.",
    )

    list_targets_model = KNN(n_neighbors=1).fit(
        [[0.0], [1.0]],
        ["left", "right"],
    )
    _check(
        np.array_equal(list_targets_model.predict([[0.0], [1.0]]), ["left", "right"]),
        "KNN must accept list-valued targets.",
    )

    _expect_raises(
        RuntimeError,
        lambda: KNN().predict([[0.0]]),
        "predict before fit must raise RuntimeError.",
    )
    _expect_raises(
        ValueError,
        lambda: model.predict(np.zeros((1, X.shape[1] + 1))),
        "predict with the wrong feature count must raise ValueError.",
    )
    _expect_raises(
        ValueError,
        lambda: KNN(n_neighbors=3).fit([[0.0], [1.0]], [0, 1]),
        "k above the training sample count must raise ValueError.",
    )


def main():
    _check_hand_examples()
    _check_sklearn_agreement(load_iris)
    _check_sklearn_agreement(load_wine)
    _check_properties_and_errors()
    print("all checks passed")


if __name__ == "__main__":
    main()
