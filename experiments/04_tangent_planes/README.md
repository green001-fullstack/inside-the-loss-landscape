# Experiment 4: Tangent Planes

## Objective

The objective of this experiment is to understand how tangent planes provide local linear approximations of multivariable functions.

We explore how a flat plane can approximate a curved loss surface around a specific parameter point and investigate how the accuracy of this approximation changes as we move away from the point of contact.

## Mathematical Foundation

We use the loss function from our previous experiments:

L(w1, w2) = w1² + 2w2²

The function has two input variables and one output, so its graph forms a three-dimensional surface.

For a function of two variables, the tangent plane at a point (a, b) is given by:

T(w1, w2) = L(a, b) + [∂L/∂w1(a, b) * (w1 - a)] + [∂L/∂w2(a, b) * (w2 - b)]

### Deriving the Tangent Plane

We choose the point of contact:

(a, b) = (1, 1)

First, calculate the loss:

L(1, 1) = 1² + 2(1)² = 3

Next, calculate the partial derivatives:

∂L/∂w1 = 2w1

∂L/∂w2 = 4w2

At (1, 1):

∂L/∂w1 = 2

∂L/∂w2 = 4

Substituting into the tangent plane formula:

T(w1, w2) = 3 + 2(w1 - 1) + 4(w2 - 1)

Simplifying:

T(w1, w2) = 2w1 + 4w2 - 3

This plane touches the original loss surface at the point (1, 1, 3).

## Implementation

The experiment was implemented using Python, NumPy, and Matplotlib.

Two functions were defined:

**Original loss function:**

```python
def compute_loss(w1, w2):
    loss = w1**2 + 2*w2**2
    return loss
```

**Tangent plane function:**

```python
def compute_tangent_plane(w1, w2):
    plane_loss = 2*w1 + 4*w2 - 3
    return plane_loss
```

We generated parameter values using `np.linspace()` and created a coordinate grid using `np.meshgrid()`.

Both functions were evaluated over the grid and visualized together using Matplotlib's 3D `plot_surface()`.

The point of contact was marked at (1, 1, 3).

## Results and Observations

The visualization demonstrated the following:

- The original loss function forms a curved surface because it contains squared terms.
- The tangent plane is flat because it is a linear function.
- Both surfaces touch at (1, 1, 3).
- The distance between the two surfaces generally increases as we move away from the contact point.

### Comparing the Original Loss with the Approximation

We evaluated both functions at different parameter points:

| Parameter Point | Original Loss | Tangent Plane |
|---|---:|---:|
| (1.1, 1) | 3.21 | 3.20 |
| (1.01, 1) | 3.0201 | 3.02 |
| (1.001, 1) | 3.002001 | 3.002 |
| (2, 2) | 12 | 9 |

The results show that the tangent plane provides a more accurate approximation when we evaluate points closer to the contact point.

### Approximation Error

The approximation error is:

E = L(w1, w2) - T(w1, w2)

Substituting our functions:

E = (w1 - 1)² + 2(w2 - 1)²

At the contact point, the error is zero.

As we move away from the contact point, the error increases.

For example, at (1.1, 1):

E = 0.01

At (2, 2):

E = 3

This demonstrates how the curvature of the original loss surface affects the accuracy of the linear approximation.

## AI/ML Connection

Tangent planes are closely related to first-order Taylor approximations, which are fundamental to gradient-based optimization.

For a multivariable loss function:

L(w + Δw) ≈ L(w) + ∇L(w)^T * Δw

This approximation helps us estimate how the loss changes when model parameters are adjusted by small amounts.

During gradient descent, the parameters are updated using:

w_new = w_old - α * ∇L(w)

Where:

- w represents the model parameters.
- α represents the learning rate.
- ∇L is the gradient of the loss function.

A suitable learning rate helps control the size of parameter updates. Smaller updates generally keep us closer to the region where the local linear approximation is accurate.

However, an excessively small learning rate can make training slow.

Understanding tangent planes therefore provides a foundation for understanding gradient descent, learning-rate selection, and higher-order optimization methods.

## Conclusion

This experiment demonstrated how tangent planes approximate curved multivariable functions locally.

We learned that the approximation is exact at the point of contact and generally becomes less accurate as we move farther away.

We also connected tangent planes to first-order Taylor approximations and gradient-based optimization in machine learning.
