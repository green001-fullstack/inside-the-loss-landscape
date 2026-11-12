import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 3, 100)
y = np.linspace(0, 4, 100)

X, Y = np.meshgrid(x, y)

Z_original = X**3 + X * Y**2
Z_first = 7 * X + 4 * Y - 10
Z_second = 3 * X**2 + 4 * X * Y + Y**2 - 7 * X - 4 * Y + 5

print(Z_original.shape)
print(Z_first.shape)
print(Z_second.shape)

fig = plt.figure(figsize=(18, 6))

# Subplot 1: Original Function
ax1 = fig.add_subplot(131, projection="3d")
ax1.plot_surface(X, Y, Z_original, cmap="viridis", alpha=0.9)
ax1.set_title("Original Function: f(x, y)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x, y)")

# Subplot 2: First-Order Taylor Approximation
ax2 = fig.add_subplot(132, projection="3d")
ax2.plot_surface(X, Y, Z_first, cmap="plasma", alpha=0.9)
ax2.set_title("First-Order Approximation: T1(x, y)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("T1(x, y)")

# Subplot 3: Second-Order Taylor Approximation
ax3 = fig.add_subplot(133, projection="3d")
ax3.plot_surface(X, Y, Z_second, cmap="coolwarm", alpha=0.9)
ax3.set_title("Second-Order Approximation: T2(x, y)")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("T2(x, y)")

plt.tight_layout()
plt.show()
