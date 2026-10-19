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

**Implementation:** NumPy

**Dataset:** Iris

**Evaluation:** Achieved **100% accuracy** with \(k = 5\), matching scikit-learn's KNN implementation on the same train/test split.
