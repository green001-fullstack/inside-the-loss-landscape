import numpy as np

def compute_function(w1, w2):
    f1 = w1**2 + 2*w2
    f2 = 3*w1 + w2**2
    return np.array([f1, f2])

def compute_jacobian(w1, w2):
    row1 = [2*w1, 2]
    row2 = [3, 2*w2]

    return np.array([row1, row2])

jacobian = compute_jacobian(1, 1)
# print(jacobian)


def create_delta_w(x, y):
    return np.array([x,y])

w = np.array([1.0, 1.0])

def calc_change(a, b):
    w_new = w + create_delta_w(a,b)
    original = compute_function(w[0], w[1])
    new = compute_function(w_new[0], w_new[1])
    change = new - original
    approximation = np.dot(jacobian, create_delta_w(a, b))
    error = np.abs(change - approximation)
    return approximation, change, error

approx1, change1, error1 = calc_change(0.1, 0.2)
approx2, change2, error2 = calc_change(0.01, 0.02)
approx3, change3, error3 = calc_change(0.001, 0.002)


print("Experiment 1")
print("Approximation:", approx1)
print("Actual change:", change1)
print("Absolute error:", error1)

print("\nExperiment 2")
print("Approximation:", approx2)
print("Actual change:", change2)
print("Absolute error:", error2)

print("\nExperiment 3")
print("Approximation:", approx3)
print("Actual change:", change3)
print("Absolute error:", error3)


# Loop version
# perturbations = [
#     (0.1, 0.2),
#     (0.01, 0.02),
#     (0.001, 0.002)
# ]

# for dw1, dw2 in perturbations:
#     approximation, actual, error = calc_change(dw1, dw2)

#     print(f"\nPerturbation: ({dw1}, {dw2})")
#     print("Jacobian approximation:", approximation)
#     print("Actual change:", actual)
#     print("Absolute error:", error)

