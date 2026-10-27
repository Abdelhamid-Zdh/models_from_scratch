import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import LogisticRegression


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _expect_raises(exception_type, function, message):
    try:
        function()
    except exception_type:
        return
    raise AssertionError(message)


def _check_gradient():
    X = np.array([[0.2, -0.4], [1.0, 0.3], [-0.5, 0.8]])
    y = np.array([0.0, 1.0, 0.0])
    weights = np.array([0.15, -0.2])
    bias = 0.07
    model = LogisticRegression()
    probabilities = model._sigmoid(X @ weights + bias)
    grad_weights, grad_bias = model._gradients(X, y, probabilities)

    def loss(parameters):
        prediction = model._sigmoid(X @ parameters[:-1] + parameters[-1])
        return model._cross_entropy_loss(y, prediction)

    parameters = np.append(weights, bias)
    step = 1e-5
    numerical_gradient = np.empty_like(parameters)
    for index in range(parameters.size):
        offset = np.zeros_like(parameters)
        offset[index] = step
        numerical_gradient[index] = (
            loss(parameters + offset) - loss(parameters - offset)
        ) / (2 * step)

    analytical_gradient = np.append(grad_weights, grad_bias)
    _check(
        np.allclose(analytical_gradient, numerical_gradient, atol=1e-6, rtol=0.0),
        "Analytical logistic-regression gradients disagree with central differences.",
    )


def _check_hand_example():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression(lr=0.1, epochs=20_000, eps=1e-10, patience=2_000)
    model.fit(X, y)

    _check(
        np.array_equal(model.predict(X), y),
        "The hand-checkable logistic-regression data must be classified perfectly.",
    )
    probabilities = model.predict_proba(X)
    _check(
        np.all(np.diff(probabilities) > 0),
        "Predicted positive-class probabilities must increase with x.",
    )


def _check_refitting_resets_state():
    X, y = make_classification(
        n_samples=250,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        flip_y=0.1,
        random_state=31,
    )
    model = LogisticRegression(lr=0.1, epochs=20_000, eps=1e-10, patience=2_000)
    model.fit(X, y)
    expected_weights = model.weights_.copy()
    expected_bias = model.bias_
    expected_probabilities = model.predict_proba(X)

    model.fit(X, y)
    _check(
        np.array_equal(model.weights_, expected_weights)
        and model.bias_ == expected_bias
        and np.array_equal(model.predict_proba(X), expected_probabilities),
        "Fitting twice on the same data must produce identical model state.",
    )


def _check_sklearn_agreement_and_weights():
    X, y = make_classification(
        n_samples=500,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        flip_y=0.1,
        class_sep=1.0,
        random_state=19,
    )
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=73,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        lr=0.1,
        epochs=50_000,
        eps=1e-12,
        patience=5_000,
    ).fit(X_train, y_train)
    reference = SklearnLogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5_000,
        tol=1e-12,
    ).fit(X_train, y_train)

    _check(
        np.array_equal(model.predict(X_test), reference.predict(X_test)),
        "Logistic-regression predictions disagree with scikit-learn.",
    )
    model_loss = model._cross_entropy_loss(y_train, model.predict_proba(X_train))
    reference_loss = model._cross_entropy_loss(
        y_train,
        reference.predict_proba(X_train)[:, 1],
    )
    _check(
        abs(model_loss - reference_loss) < 1e-6,
        "Training loss should be close to scikit-learn's unregularized loss.",
    )
    _check(
        np.allclose(model.weights_, reference.coef_[0], atol=1e-5, rtol=1e-5)
        and np.isclose(model.bias_, reference.intercept_[0], atol=1e-5, rtol=1e-5),
        "Logistic-regression weights should match on the fixed synthetic dataset.",
    )


def _check_errors_and_probabilities():
    _expect_raises(
        ValueError,
        lambda: LogisticRegression().fit([[0.0], [1.0]], [1, 2]),
        "Labels {1, 2} must raise ValueError.",
    )
    model = LogisticRegression()
    _expect_raises(
        RuntimeError,
        lambda: model.predict([[0.0]]),
        "predict before fit must raise RuntimeError.",
    )

    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    _check(
        np.all((probabilities >= 0.0) & (probabilities <= 1.0)),
        "Predicted probabilities must remain in [0, 1].",
    )


def main():
    _check_gradient()
    _check_hand_example()
    _check_refitting_resets_state()
    _check_sklearn_agreement_and_weights()
    _check_errors_and_probabilities()
    print("all checks passed")


if __name__ == "__main__":
    main()
