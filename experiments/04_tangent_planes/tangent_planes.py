import numpy as np
import matplotlib.pyplot as plt

p1 = 1.1
p2 = 1

def compute_loss(w1, w2):
    loss = w1**2 + 2*w2**2
    return loss


def compute_tangent_plane(w1, w2):
    plane_loss = 2*w1 + 4*w2 - 3
    return plane_loss

# print(compute_loss(p1, p2))
# print(compute_tangent_plane(p1, p2))


# Generate parameter values
w1_values = np.linspace(-3, 3, 100)
w2_values = np.linspace(-3, 3, 100)

# Create coordinate grid
W1, W2 = np.meshgrid(w1_values, w2_values)

# Calculate original loss and tangent plane
L = compute_loss(W1, W2)
T = compute_tangent_plane(W1, W2)

# print(W1.shape)
# print(W2.shape)
# print(L.shape)
# print(T.shape)

# Create 3D figure
fig, ax = plt.subplots(
    figsize=(10, 7),
    subplot_kw={"projection": "3d"}
)

# Plot original loss surface
ax.plot_surface(
    W1, W2, L,
    cmap="viridis",
    alpha=0.7
)

# Plot tangent plane
ax.plot_surface(
    W1, W2, T,
    cmap="plasma",
    alpha=0.5
)

# Mark the point of contact
ax.scatter(
    1, 1, 3,
    color="red",
    s=80,
    label="Point of contact (1, 1, 3)"
)

# Labels and title
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("Loss")
ax.set_title("Original Loss Surface and Tangent Plane")

ax.legend()
ax.view_init(elev=25, azim=-55)

plt.show()