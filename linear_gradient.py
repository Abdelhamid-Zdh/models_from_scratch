import numpy as np

def predict(X, w, b):
    """returns y_pred of shape (n,)"""
    return (X@w) + b 

def mse_loss(y_pred, y):
    """returns scalar mean squared error"""
    return np.mean((y_pred - y) ** 2)

def compute_gradients(X, y, y_pred):
    """returns (grad_w of shape (d,), grad_b scalar)"""
    grad_w = -(2/X.shape[0]) * X.T @ (y - y_pred)
    grad_b = -(2/X.shape[0]) * np.sum(y-y_pred)
    return(grad_w, grad_b)

def train(X, y, lr=0.01, epochs=1000):
    """initialize w=zeros, b=0, loop epochs, return (w,b)"""
    w = np.zeros(X.shape[1])
    b = 0
    for n in range(epochs):
        y_pred = predict(X, w, b)
        grad_w, grad_b = compute_gradients(X, y, y_pred)
        w -= lr * grad_w
        b -= lr * grad_b

    return(w, b)
