import numpy as np
import matplotlib.pyplot as plt

# Generate parameter values
w1 = np.linspace(-3, 3, 100)
w2 = np.linspace(-3, 3, 100)

# Create coordinate grids
W1, W2 = np.meshgrid(w1, w2)

# Calculate the loss
L = W1**2 + 2 * W2**2

# print(W1.shape)
# print(W2.shape)
# print(L.shape)

fig, ax = plt.subplots(figsize=(8, 6))

contour = ax.contour(W1, W2, L, levels=10)

ax.clabel(contour, inline=True, fontsize=8)

ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_title("Level Curves of the Loss Function")

# Calculate gradient at (1, 1)
gradient_w1 = 2 * 1
gradient_w2 = 4 * 1

# Plot gradient vector
# ax.quiver(
#     1, 1,
#     gradient_w1, gradient_w2,
#     color="red",
#     angles="xy",
#     scale_units="xy",
#     scale=1
# )

# Define points
points_w1 = np.array([1, -1, 1, -1, 2, 0.1, 0.2])
points_w2 = np.array([1, 1, -1, -1, 0.5, 0.1, 0.2])

# Calculate gradients
gradient_w1 = 2 * points_w1
gradient_w2 = 4 * points_w2

# Plot gradient vectors
ax.quiver(
    points_w1,
    points_w2,
    gradient_w1,
    gradient_w2,
    color="red",
    angles="xy",
    scale_units="xy",
    scale = 1
)


plt.show()

# Mark the point
ax.scatter(1, 1, color="red")

plt.show()   