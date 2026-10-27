# ML Models from Scratch

Implementations of machine learning algorithms using **NumPy**, built to understand how they work internally rather than relying entirely on existing ML libraries.

## How to Run

Use Python 3.12 or later. Install the pinned dependencies and run an example from the repository root:

```bash
python -m pip install -r requirements.txt
python -m examples.knn_example
python -m examples.logistic_regression_example
```

Run the assertion-based checks (separate from the output-only demos) with:

```bash
python -m examples.check_knn
python -m examples.check_logistic_regression
```

## Algorithms

### 1. K-Nearest Neighbors (KNN)

**Type:** Classification

KNN is a *lazy learning* algorithm. Unlike models that learn parameters during training, KNN stores the training dataset and delays most computation until prediction time.

**How it works:**

1. Store the training features and their labels.
2. Calculate the distance between a new data point and the training points, commonly using Euclidean distance.
3. Select the $k$ nearest neighbors.
4. Predict the most frequent class among the neighbors.

**Key concepts:**
- Euclidean distance
- Nearest-neighbor search
- Majority voting
- External feature scaling
- Choosing the value of $k$

**Implementation details**

- **Implementation:** NumPy
- **Dataset:** Iris
- **Evaluation:** Achieved **100% accuracy** with $k = 5$, matching scikit-learn's KNN implementation on the same train/test split.

### 2. Logistic Regression

**Type:** Binary Classification

Logistic Regression is a supervised learning algorithm that predicts the probability of a sample belonging to a class. It learns weights and a bias during training to make its predictions as accurate as possible.

**How it works**

1. **Initialize parameters:** Start with weights set to zero and the bias set to zero.
2. **Calculate the linear score:** Multiply the input features by their corresponding weights and add the bias.
3. **Calculate the probability:** Apply the sigmoid function to convert the score into a probability between 0 and 1.
4. **Calculate the loss:** Use binary cross-entropy to measure how far the predicted probabilities are from the actual labels.
5. **Calculate gradients:** Use the chain rule to determine how each weight and the bias contribute to the loss.
6. **Update parameters:** Use gradient descent to adjust the weights and bias in the direction that reduces the loss.
7. **Repeat:** Continue making predictions, measuring the loss, and updating the parameters until convergence, early stopping, or the maximum number of epochs is reached.

**Understanding the training process**

The goal of training is to find weights and a bias that minimize the prediction loss.

The `fit()` method repeatedly performs the following cycle:

- **Forward pass:** Calculate predicted probabilities using the current parameters.
- **Loss calculation:** Measure the prediction error using binary cross-entropy.
- **Backpropagation:** Calculate gradients that indicate how the parameters affect the loss.
- **Gradient descent:** Update the parameters using the learning rate.
- **Stopping conditions:** Stop when parameter updates become sufficiently small, the loss stops improving for several epochs, or the epoch limit is reached.

The **learning rate** controls the size of each parameter update. A rate that is too large can make training unstable, while one that is too small can make training slow.

An **epoch** represents one complete pass through the training dataset. In this implementation, each epoch uses the entire training set to calculate the gradients.

**Key concepts**

- Linear scores and weighted features
- Sigmoid activation function
- Binary cross-entropy loss
- Chain rule and gradient calculation
- Gradient descent
- Learning rate and convergence
- Early stopping
- Probability threshold for classification

**Implementation details**

- **Implementation:** NumPy
- **Dataset:** Breast Cancer Wisconsin dataset from scikit-learn
- **Preprocessing:** Feature standardization is performed outside the model using training-set statistics, and the same transformation is applied to prediction data.
- **API:** Configure `lr`, `epochs`, `eps`, and `patience` when creating the model; `fit(X, y)` learns the parameters, and `predict_proba(X)` returns positive-class probabilities.
- **Evaluation:** The example compares accuracy with scikit-learn's unregularized Logistic Regression on the same standardized train/test split.

**Note**: For detailed math formulas, check `math.md`.

## Results

Each example uses an 80/20 train/test split with `random_state=93`. Features are standardized using training-set statistics before fitting and prediction.

| Model | Dataset | Metric | From scratch | scikit-learn | Prediction agreement |
|---|---|---|---:|---:|---:|
| KNN ($k=5$) | Iris | Accuracy | 100.00% | 100.00% | 100.00% |
| Logistic Regression | Breast Cancer Wisconsin | Accuracy | 96.49% | 96.49% | 98.25% |

## What I Learned

- Implementing algorithms directly makes the relationship between model assumptions, loss functions, gradients, and predictions explicit.
- Preprocessing must be learned from the training set and reused on test data to avoid data leakage.
- Comparing predictions with scikit-learn and checking gradients numerically are practical ways to validate an implementation.
- Shared classifier behavior, such as accuracy scoring, belongs in a reusable mixin rather than being duplicated.