#### Logistic Regression

**Mathematical Formulation**

**1. Linear score**

$$
z = Xw + b
$$

**2. Sigmoid function**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**3. Predicted probability**

$$
\hat{y} = \sigma(Xw+b)
$$

**4. Binary cross-entropy loss**

$$
L = -\frac{1}{n}\sum_{i=1}^{n}
\left[
y_i\ln(\hat{y}_i)
+(1-y_i)\ln(1-\hat{y}_i)
\right]
$$

**5. Gradient of the loss with respect to the weights**

$$
\frac{\partial L}{\partial w}
= \frac{1}{n}X^T(\hat{y}-y)
$$

**6. Gradient of the loss with respect to the bias**

$$
\frac{\partial L}{\partial b}
= \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)
$$

**7. Gradient descent updates**

$$
w \leftarrow w-\alpha\frac{\partial L}{\partial w}
$$

$$
b \leftarrow b-\alpha\frac{\partial L}{\partial b}
$$

Where:

- $X$: training feature matrix
- $w$: weight vector
- $b$: bias
- $y$: actual labels
- $\hat{y}$: predicted probabilities
- $n$: number of training samples
- $\alpha$: learning rate
- $L$: binary cross-entropy loss