# ML Models from Scratch

Implementations of machine learning algorithms using **NumPy**, built to understand how they work internally rather than relying entirely on existing ML libraries.

## Algorithms

### 1. K-Nearest Neighbors (KNN)

**Type:** Classification and Regression

KNN is a *lazy learning* algorithm. Unlike models that learn parameters during training, KNN stores the training dataset and delays most computation until prediction time.

**How it works:**

1. Store the training features and their labels.
2. Calculate the distance between a new data point and the training points, commonly using Euclidean distance.
3. Select the \(k\) nearest neighbors.
4. **Classification:** Predict the most frequent class among the neighbors.
5. **Regression:** Predict the average target value of the neighbors.

**Key concepts:**
- Euclidean distance
- Nearest-neighbor search
- Majority voting
- Feature standardization
- Choosing the value of \(k\)

**Implementation details**

- **Implementation:** NumPy
- **Dataset:** Iris
- **Evaluation:** Achieved **100% accuracy** with \(k = 5\), matching scikit-learn's KNN implementation on the same train/test split.

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
- **Preprocessing:** Feature standardization using training-set statistics
- **Evaluation:** Achieved **98.25% accuracy**, compared with **99.12%** for scikit-learn's Logistic Regression implementation using the same train/test split.

**Note**: For detailled math formulas check `math.md`