import numpy as np

def compute_f(x1, x2):
    y1 = x1**2 + x2
    y2 = x1 + 2*x2
    return np.array([y1, y2], dtype=float)

def compute_g(y1, y2):
    z1 = 3*y1 + y2
    z2 = y1 * y2
    return np.array([z1, z2], dtype=float)



def compute_jacobian_f(x1, x2):
    return np.array([
        [2*x1, 1],
        [1,    2]
    ], dtype=float)

def compute_jacobian_g(y1, y2):
    return np.array([
       [3, 1],
        [y2, y1]
    ], dtype=float)

def compute_composite_jacobian(jacobian_g, jacobian_f):
    return jacobian_g @ jacobian_f

x1, x2 = 2, 3

y = compute_f(x1, x2)

jacobian_f = compute_jacobian_f(x1, x2)
jacobian_g = compute_jacobian_g(y[0], y[1])

composite_jacobian = compute_composite_jacobian(
    jacobian_g, jacobian_f
)

# print("f(x) =", y)
# print("Jacobian of f:\n", jacobian_f)
# print("Jacobian of g:\n", jacobian_g)
# print("Composite Jacobian:\n", composite_jacobian)

def compute_direct_z1_grad(x1, x2):
    dz1_dx1 = 6*x1 + 1
    dz1_dx2 = 5
    return np.array([dz1_dx1, dz1_dx2], dtype=float)