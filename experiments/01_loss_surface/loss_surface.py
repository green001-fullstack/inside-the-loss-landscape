import numpy as np
import matplotlib.pyplot as plt


# Generate parameter values
w1 = np.linspace(-3, 3, 101)
w2 = np.linspace(-3, 3, 101)

# Create coordinate grids
W1, W2 = np.meshgrid(w1, w2)

# Calculate the loss
L = W1**2 + 2 * W2**2


p1 = 2
p2 = 1
learning_rate = 0.1

# Partial derivatives
Lp1 = 2*p1
Lp2 = 4*p2

# Updated parameters
p1new = p1 - learning_rate*Lp1
p2new = p2 - learning_rate*Lp2
new_loss = p1new**2 + 2*p2new**2
original_loss = p1**2 + 2*p2**2



p1_history = [p1]
p2_history = [p2]
loss_history = [p1**2 + 2*p2**2]

# Loop version
for i in range(100):
    # Calculate gradients
    Lp1 = 2 * p1
    Lp2 = 4 * p2

    # Update parameters
    p1 = p1 - learning_rate * Lp1
    p2 = p2 - learning_rate * Lp2

    # Calculate loss
    current_loss = p1**2 + 2 * p2**2
    # Store history
    p1_history.append(p1)
    p2_history.append(p2)
    loss_history.append(current_loss)


# print("Final w1:", p1)
# print("Final w2:", p2)
# print("Final loss:", current_loss)

print("Starting point:", (2, 1))
print("First iteration:", (p1_history[0], p2_history[0]))
print("Last iteration:", (p1_history[-1], p2_history[-1]))
print("Final loss:", loss_history[-1])

# Plotting the gradient descent trajectory

fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": "3d"})

ax.plot_surface(W1, W2, L, cmap="viridis", alpha=0.6)
ax.plot(p1_history, p2_history, loss_history, color="red", marker="o", markersize=2, linewidth=2, label="Gradient Descent Path")

# Mark the starting point
ax.scatter(
    p1_history[0],
    p2_history[0],
    loss_history[0],
    color="red",
    s=80,
    label="Starting Point"
)

# Mark the final point
ax.scatter(
    p1_history[-1],
    p2_history[-1],
    loss_history[-1],
    color="black",
    s=80,
    label="Final Point"
)

ax.view_init(elev=25, azim=-55)

ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("Loss")
ax.set_title("Loss Landscape")
ax.legend()

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 20)

# plt.show()


fig, ax = plt.subplots(figsize=(10, 6))

iterations = range(len(loss_history))

ax.plot(iterations, loss_history)

ax.set_xlabel("Iterations")
ax.set_ylabel("Loss")
ax.set_title("Gradient Descent Convergence")

plt.show()