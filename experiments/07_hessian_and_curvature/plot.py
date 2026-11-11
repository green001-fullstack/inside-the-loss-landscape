import numpy as np
import matplotlib.pyplot as plt

def positive_definite_loss(x, y):
    return x**2 + y**2

def negative_definite_loss(x, y):
    return -x**2 - y**2

def indefinite_loss(x, y):
    return x**2 - y**2

def positive_semidefinite_loss(x, y):
    return x**2 + 0*y

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)

X, Y = np.meshgrid(x, y)

Z1 = positive_definite_loss(X, Y)
Z2 = negative_definite_loss(X, Y)
Z3 = indefinite_loss(X, Y)
Z4 = positive_semidefinite_loss(X, Y)

fig = plt.figure(figsize=(12, 10))

# 1. Positive Definite (Bowl)
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.plot_surface(X, Y, Z1, cmap="viridis", alpha=0.9)
ax1.set_title("Positive Definite (Bowl / Minimum)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("Loss")

# 2. Negative Definite (Hill)
ax2 = fig.add_subplot(2, 2, 2, projection="3d")
ax2.plot_surface(X, Y, Z2, cmap="plasma", alpha=0.9)
ax2.set_title("Negative Definite (Hill / Maximum)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("Loss")

# 3. Indefinite (Saddle)
ax3 = fig.add_subplot(2, 2, 3, projection="3d")
ax3.plot_surface(X, Y, Z3, cmap="coolwarm", alpha=0.9)
ax3.set_title("Indefinite (Saddle Point)")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("Loss")

# 4. Positive Semidefinite (Trough)
ax4 = fig.add_subplot(2, 2, 4, projection="3d")
ax4.plot_surface(X, Y, Z4, cmap="cividis", alpha=0.9)
ax4.set_title("Positive Semidefinite (Flat Valley)")
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_zlabel("Loss")

plt.tight_layout()
plt.show()
