import numpy as np

X = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
], dtype=float)

y = np.array([
    [2],
    [4],
    [6]
], dtype=float)

w = np.array([
    [0.5],
    [1.0]
], dtype=float)


def compute_predictions(X, w):
    # Your implementation
    return X @ w

def compute_residuals(predictions, y):
    # Your implementation
    return predictions - y

def compute_loss(residuals):
    # Your implementation
    return np.sum(residuals**2)

def compute_gradient(X, residuals):
    # Your implementation
    return 2 * X.T @ residuals

print("Predictions:\n", compute_predictions(X, w))
print("Residuals:\n", compute_residuals(compute_predictions(X, w), y))
print("Loss:\n", compute_loss(compute_residuals(compute_predictions(X, w), y)))
print("Gradient:\n", compute_gradient(X, compute_residuals(compute_predictions(X, w), y)))