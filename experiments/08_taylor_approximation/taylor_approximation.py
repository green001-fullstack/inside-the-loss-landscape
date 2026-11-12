import numpy as np

def compute_function(x, y):
    return x**3 + x*y**2 

# original_value = compute_function(1,2)
# print(original_value)

def compute_gradient(x, y):
    dx = 3*x**2 + y**2
    dy = 2*x*y
    return np.array([dx,dy])

# gradient = compute_gradient(1,2)
# print(gradient)

def compute_hessian(x, y):
    dxx = 6*x
    dxy = 2*y
    dyx = 2*y
    dyy = 2*x

    return np.array([
        [dxx, dxy],
        [dyx, dyy]
    ])
# hessian = compute_hessian(1,2)
# print(hessian)

def first_order_taylor(x, y, a, b):
    original_value = compute_function(a,b)
    gradient = compute_gradient(a,b)
    value_change = np.array([x-a, y-b])

    return original_value + np.dot(gradient, value_change)

result = first_order_taylor(1.1,2.1, 1, 2)
# print(result)

def second_order_taylor(x, y, a, b):
  # Displacement vector Delta = [x - a, y - b]
  value_change = np.array([x - a, y - b])

  # 1. The original function value
  original_value = compute_function(a, b)

  # 2. The gradient contribution (first-order change)
  gradient = compute_gradient(a, b)
  first_order = np.dot(gradient, value_change)

  # 3. The Hessian contribution (second-order change)
  hessian = compute_hessian(a, b)
  second_order = 0.5 * (value_change.T @ hessian @ value_change)

  return original_value + first_order + second_order


second_result = second_order_taylor(1.1, 2.1, 1, 2)
print(second_result)

exact_result = compute_function(1.1, 2.1)
print(exact_result)