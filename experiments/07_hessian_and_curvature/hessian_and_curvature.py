import numpy as np

def compute_loss(w1, w2):
    return w1**2 + 2*w2**2

def compute_gradient(w1, w2):
    return np.array([
        2*w1,
        4*w2
    ], dtype=float)

def compute_hessian():
    return np.array([
        [2, 0],
        [0, 4]
    ], dtype=float)

hessian = compute_hessian()

eigenvalues = np.linalg.eigvalsh(hessian)

# print("Hessian:\n", hessian)
# print("Eigenvalues:", eigenvalues)

tol = 1e-10
def classify_hessian(hessian):
    eigenvalues = np.linalg.eigvalsh(hessian)

    # Your classification logic here
    if np.all(eigenvalues > tol):
        return "Positive definite"
    elif np.all(eigenvalues < -tol):
        return "Negative definite"
    elif np.any(eigenvalues < -tol) and np.any(eigenvalues > tol):
        return "Saddle point"
    elif np.all(eigenvalues >= -tol):
        return "Positive semidefinite (Flat valley / Degenerate)"
    elif np.all(eigenvalues <= tol):
        return "Negative semidefinite (Flat ridge / Degenerate)"
    else:
        return "Zero matrix (Completely flat)"


test_hessians = {
    "Positive definite": np.array([[2, 0], [0, 4]]),
    "Negative definite": np.array([[-2, 0], [0, -4]]),
    "Indefinite": np.array([[2, 0], [0, -4]]),
    "Positive semidefinite": np.array([[0, 0], [0, 4]]),
    "Negative semidefinite": np.array([[0, 0], [0, -4]]),
    "Zero matrix": np.array([[0, 0], [0, 0]])
}

for name, matrix in test_hessians.items():
    print(name, ":", classify_hessian(matrix))