import numpy as np
import matplotlib.pyplot as plt

def compute_loss(w1, w2):
    return w1**2 + 2 * w2**2

w1 = 1.0
w2 = 1.0

# 1. Define the function to compute the gradient
def compute_gradient(w1, w2):
    grad_w1 = 2 * w1
    grad_w2 = 4 * w2
    return np.array([grad_w1, grad_w2])


gradient = compute_gradient(w1, w2)
print("Gradient vector at (1, 1):", gradient)

# 2. Define our direction vectors

# Steepest increase: same direction as the gradient
direction_increase = np.array([2.0, 4.0])

# Steepest decrease: opposite direction to the gradient
direction_decrease = np.array([-2.0, -4.0])

# Zero instantaneous change: perpendicular to the gradient
direction_zero = np.array([4.0, -2.0])

# Normalize the directions
def normalize_direction(direction):
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction
    return direction / norm


increase_direction = normalize_direction(direction_increase)
decrease_direction = normalize_direction(direction_decrease)
zero_direction = normalize_direction(direction_zero)


# Calculate the directional derivative
def compute_directional_derivative(gradient, direction):
    return np.dot(gradient, direction)

increase = compute_directional_derivative(gradient, increase_direction)
decrease = compute_directional_derivative(gradient, decrease_direction)
zero = compute_directional_derivative(gradient, zero_direction)

print(increase)
print(decrease)
print(zero)

# Generate parameter values
w1_values = np.linspace(-3, 3, 100)
w2_values = np.linspace(-3, 3, 100)

# Create coordinate grid
W1, W2 = np.meshgrid(w1_values, w2_values)

# Calculate loss over the grid
L = compute_loss(W1, W2)

# Create contour plot
fig, ax = plt.subplots(figsize=(8, 6))

contour = ax.contour(W1, W2, L, levels=15)
ax.clabel(contour, inline=True, fontsize=8)

ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_title("Directional Derivatives on the Loss Landscape")

# Mark the current parameter point
ax.scatter(w1, w2, color="black", s=60, label="Current point (1, 1)")

# Steepest increase direction
ax.quiver(
    w1, w2,
    increase_direction[0], increase_direction[1],
    color="green",
    angles="xy",
    scale_units="xy",
    scale=1,
    label="Steepest increase"
)

# Steepest decrease direction
ax.quiver(
    w1, w2,
    decrease_direction[0], decrease_direction[1],
    color="red",
    angles="xy",
    scale_units="xy",
    scale=1,
    label="Steepest decrease"
)

# Zero change direction
ax.quiver(
    w1, w2,
    zero_direction[0], zero_direction[1],
    color="blue",
    angles="xy",
    scale_units="xy",
    scale=1,
    label="Zero change"
)

ax.legend()
plt.show()