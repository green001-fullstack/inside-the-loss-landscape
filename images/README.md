# Experiment 3: Directional Derivatives

## Objective

The objective of this experiment is to understand how the loss function changes when we move in different directions within the parameter space.

We explore how directional derivatives help us identify:

- The direction of steepest increase in loss.
- The direction of steepest decrease in loss.
- Directions in which the loss has zero instantaneous change.

We also visualize these directions on a contour plot to understand their geometric relationship with the loss landscape.

## Mathematical Foundation

We use the same loss function from the previous experiments:

\[
L(w_1,w_2)=w_1^2+2w_2^2
\]

The gradient is:

\[
\nabla L(w_1,w_2)=
\begin{bmatrix}
2w_1\\
4w_2
\end{bmatrix}
\]

At the parameter point \((1,1)\):

\[
\nabla L(1,1)=
\begin{bmatrix}
2\\
4
\end{bmatrix}
\]

The directional derivative measures the instantaneous rate of change of a function in a particular direction.

Its formula is:

\[
D_{\mathbf u}L=\nabla L\cdot\mathbf u
\]

Where:

- \(\nabla L\) is the gradient.
- \(\mathbf u\) is a unit direction vector.
- The dot product determines how rapidly the loss changes in that direction.

### Normalizing Direction Vectors

A direction vector must have a magnitude of 1 when calculating a directional derivative.

We normalize a vector using:

\[
\mathbf u=\frac{\mathbf v}{\|\mathbf v\|}
\]

Where \(\|\mathbf v\|\) represents the magnitude of the original vector.

### Three Important Directions

**1. Steepest Increase**

The gradient points in the direction of the greatest instantaneous increase in the loss.

\[
\mathbf u_{\text{increase}}=
\frac{\nabla L}{\|\nabla L\|}
\]

At \((1,1)\):

\[
D_{\mathbf u}L=2\sqrt5\approx4.472
\]

**2. Steepest Decrease**

The opposite direction of the gradient gives the greatest instantaneous decrease in the loss.

\[
\mathbf u_{\text{decrease}}=
-\frac{\nabla L}{\|\nabla L\|}
\]

Its directional derivative is:

\[
D_{\mathbf u}L=-2\sqrt5\approx-4.472
\]

**3. Zero Instantaneous Change**

A direction perpendicular to the gradient produces a directional derivative of zero.

\[
\nabla L\cdot\mathbf u=0
\]

This means there is no instantaneous change in the loss when moving in that direction.

For a smooth level curve, its tangent direction is perpendicular to the gradient. Therefore, moving along the tangent direction produces zero instantaneous change in the loss.

## Implementation

The experiment was implemented using Python, NumPy, and Matplotlib.

The implementation involved:

1. Defining the loss function.
2. Computing the gradient at the current parameter point.
3. Creating direction vectors for increase, decrease, and zero change.
4. Normalizing the direction vectors using `np.linalg.norm()`.
5. Computing directional derivatives using `np.dot()`.
6. Generating a contour plot of the loss function.
7. Visualizing the three directions using `ax.quiver()`.

## Results and Observations

At the parameter point \((1,1)\), the gradient was:

```text
[2. 4.]
```

The computed directional derivatives were approximately:

| Direction | Directional Derivative |
|---|---:|
| Steepest increase | 4.472 |
| Steepest decrease | -4.472 |
| Zero change | 0.000 |

The visualization demonstrated that:

- The green arrow points in the same direction as the gradient, representing the steepest increase in loss.
- The red arrow points in the opposite direction, representing the steepest decrease in loss.
- The blue arrow is perpendicular to the gradient and tangent to the level curve at the current point.

An important observation is that a zero directional derivative indicates zero instantaneous change, not necessarily that the loss will remain constant if we continue moving in that fixed direction.

## AI/ML Connection

Directional derivatives are important in machine learning because they help us understand how changes in model parameters affect the loss function.

During neural network training, the gradient tells us the direction in which the loss increases most rapidly.

Gradient descent uses the opposite direction to reduce the loss:

\[
\mathbf w_{\text{new}}=
\mathbf w_{\text{old}}-\alpha\nabla L
\]

Where:

- \(\mathbf w\) represents the model parameters.
- \(\alpha\) is the learning rate.
- \(\nabla L\) is the gradient of the loss function.

The learning rate controls the size of each parameter update, while the negative gradient determines the direction.

Understanding directional derivatives provides a foundation for understanding gradient descent, optimization, and backpropagation in neural networks.

## Conclusion

This experiment demonstrated how directional derivatives measure the rate of change of a loss function in different directions.

It also reinforced the relationship between gradients, level curves, and optimization.

Understanding these concepts provides an important mathematical foundation for studying how machine learning models learn by adjusting their parameters to minimize loss.