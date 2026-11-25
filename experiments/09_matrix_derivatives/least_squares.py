import numpy as np

X = np.array([
    [1,1],
    [1,2],
    [1,3],
    [1,4],
    [1,5]
])

y = np.array([
    [2],
    [4],
    [6],
    [8],
    [10]
], dtype=float)

w = np.zeros((2, 1))

def compute_predictions(X, w):
    # Your implementation
    return X @ w

def compute_residuals(predictions, y):
    # Your implementation
    return predictions - y

def compute_loss(residuals):
    # Your implementation
    return np.sum(residuals**2)


predictions = compute_predictions(X, w)


residuals = compute_residuals(predictions, y)

least_squares_loss = compute_loss(residuals)

print("Predictions:\n", predictions)
print("Residuals:\n", residuals)
print("Least Squares Loss:\n", least_squares_loss)
