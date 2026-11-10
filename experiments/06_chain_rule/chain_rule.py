import numpy as np

# Single variable chain rule
def compute_y(x):
    return x**2

def compute_z(y):
    return 3*y + 1

def compute_dy_dx(x):
    return 2*x

def compute_dz_dy():
    return 3

def compute_dz_dx(a):
    dy_dx = compute_dy_dx(a)
    dz_dy = compute_dz_dy()
    return  dy_dx * dz_dy

def compute_direct_derivative(x):
    return 6 * x

values = [1,2,3]

for i in values:
    chain_result = compute_dz_dx(i)
    direct_result = compute_direct_derivative(i)
    print(f"input: {i} | chain rule: {chain_result} | direct: {direct_result}")
