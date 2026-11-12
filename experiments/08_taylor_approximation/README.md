# Experiment 8: Multivariable Taylor Approximation

## Overview

This experiment explores how Taylor approximations can be used to estimate the value of a multivariable function around a chosen reference point.

We investigate two approximations:

- First-order Taylor approximation (linear approximation).
- Second-order Taylor approximation (quadratic approximation).

The goal is to understand how gradients and Hessians contribute to approximating nonlinear functions and why second-order approximations generally provide better local estimates.

This experiment builds on concepts from previous experiments, particularly gradients, tangent planes, Jacobians, and Hessians.

## Mathematical Foundation

We use the following function:

\[
f(x,y)=x^3+xy^2
\]

Our chosen reference point is:

\[
(a,b)=(1,2)
\]

The reference point is selected by us. Taylor approximations can be constructed around other points as well.

### 1. Gradient

The first partial derivatives are:

\[
f_x=3x^2+y^2
\]

\[
f_y=2xy
\]

Therefore, the gradient is:

\[
\nabla f(x,y)=
\begin{bmatrix}
3x^2+y^2\\
2xy
\end{bmatrix}
\]

At our reference point:

\[
\nabla f(1,2)=
\begin{bmatrix}
7\\
4
\end{bmatrix}
\]

The gradient describes the local direction and rate of steepest increase.

### 2. Hessian

The second partial derivatives are:

\[
f_{xx}=6x,\quad f_{xy}=2y
\]

\[
f_{yx}=2y,\quad f_{yy}=2x
\]

Therefore, the Hessian is:

\[
H(x,y)=
\begin{bmatrix}
6x&2y\\
2y&2x
\end{bmatrix}
\]

At our reference point:

\[
H(1,2)=
\begin{bmatrix}
6&4\\
4&2
\end{bmatrix}
\]

The Hessian describes how the gradient changes and provides information about the function's local curvature.

### 3. First-Order Taylor Approximation

The first-order Taylor approximation of a function of two variables is:

\[
T_1(x,y)=f(a,b)+f_x(a,b)(x-a)+f_y(a,b)(y-b)
\]

Using our reference point:

\[
T_1(x,y)=5+7(x-1)+4(y-2)
\]

Simplifying:

\[
\boxed{T_1(x,y)=7x+4y-10}
\]

This approximation represents a tangent plane to the original surface at the reference point.

It captures the function's value and local slope but does not account for curvature.

### 4. Second-Order Taylor Approximation

The second-order Taylor approximation is:

\[
T_2(\mathbf{x})=
f(\mathbf{a})+
\nabla f(\mathbf{a})^T(\mathbf{x}-\mathbf{a})
+\frac12(\mathbf{x}-\mathbf{a})^TH(\mathbf{a})(\mathbf{x}-\mathbf{a})
\]

Let:

\[
\Delta x=x-1,\quad \Delta y=y-2
\]

The quadratic contribution becomes:

\[
\frac12
\begin{bmatrix}\Delta x&\Delta y\end{bmatrix}
\begin{bmatrix}6&4\\4&2\end{bmatrix}
\begin{bmatrix}\Delta x\\\Delta y\end{bmatrix}
\]

Expanding:

\[
3\Delta x^2+4\Delta x\Delta y+\Delta y^2
\]

Therefore:

\[
T_2=5+7\Delta x+4\Delta y+
3\Delta x^2+4\Delta x\Delta y+\Delta y^2
\]

Substituting and simplifying:

\[
\boxed{
T_2(x,y)=3x^2+4xy+y^2-7x-4y+5
}
\]

Unlike the first-order approximation, this quadratic approximation also accounts for local curvature.

## Implementation

The experiment was implemented using Python, NumPy, and Matplotlib.

### Project Structure

```text
08_taylor_approximation/
│
├── taylor_approximation.py
├── plot.py
└── README.md
```

### 1. Computing the Original Function

```python
def compute_function(x, y):
    return x**3 + x*y**2
```

### 2. Computing the Gradient

```python
def compute_gradient(x, y):
    dx = 3*x**2 + y**2
    dy = 2*x*y

    return np.array([dx, dy])
```

### 3. Computing the Hessian

```python
def compute_hessian(x, y):
    dxx = 6*x
    dxy = 2*y
    dyx = 2*y
    dyy = 2*x

    return np.array([
        [dxx, dxy],
        [dyx, dyy]
    ])
```

### 4. First-Order Taylor Approximation

```python
def first_order_taylor(x, y, a, b):
    original_value = compute_function(a, b)
    gradient = compute_gradient(a, b)
    value_change = np.array([x-a, y-b])

    return original_value + np.dot(gradient, value_change)
```

### 5. Second-Order Taylor Approximation

```python
def second_order_taylor(x, y, a, b):
    value_change = np.array([x-a, y-b])

    original_value = compute_function(a, b)

    gradient = compute_gradient(a, b)
    first_order = np.dot(gradient, value_change)

    hessian = compute_hessian(a, b)
    second_order = 0.5 * (
        value_change.T @ hessian @ value_change
    )

    return original_value + first_order + second_order
```

The implementation uses NumPy's `dot()` operation for the gradient contribution and the `@` operator for matrix multiplication in the Hessian contribution.

## Numerical Comparison

We evaluated the original function and both approximations at:

\[
(x,y)=(1.1,2.1)
\]

The results were:

| Calculation | Result |
|---|---:|
| Original function | 6.182 |
| First-order approximation | 6.100 |
| Second-order approximation | 6.180 |

### Approximation Errors

First-order error:

\[
|6.182-6.100|=0.082
\]

Second-order error:

\[
|6.182-6.180|=0.002
\]

The second-order approximation was 41 times more accurate at this test point.

This demonstrates the benefit of including curvature information when estimating a nonlinear function locally.

## Visualization

We used Matplotlib to visualize three surfaces:

1. Original function.
2. First-order Taylor approximation.
3. Second-order Taylor approximation.

The plotting grid was generated using:

```python
x = np.linspace(-1, 3, 100)
y = np.linspace(0, 4, 100)

X, Y = np.meshgrid(x, y)
```

The three surfaces were calculated using:

```python
Z_original = X**3 + X * Y**2

Z_first = 7 * X + 4 * Y - 10

Z_second = (
    3 * X**2 + 4 * X * Y + Y**2
    - 7 * X - 4 * Y + 5
)
```

The visualization revealed:

- The original function is a nonlinear surface.
- The first-order approximation is a flat tangent plane.
- The second-order approximation is a curved quadratic surface.

All three surfaces share the same function value and first derivatives at the reference point.

The second-order approximation also matches the original function's second derivatives at that point.

**Important:** Taylor approximations are local. The second-order approximation is not guaranteed to be more accurate everywhere across the plotting domain.

## Key Findings

1. The gradient provides the first-order information needed to approximate a function locally.
2. The Hessian provides second-order information about curvature.
3. A first-order Taylor approximation produces a tangent plane for a function of two variables.
4. A second-order Taylor approximation produces a quadratic surface.
5. Including curvature information significantly improved our numerical estimate at the selected test point.
6. The accuracy of Taylor approximations generally improves as we evaluate points closer to the reference point.

## Connection to Artificial Intelligence and Machine Learning

Taylor approximations are important in machine learning because they help us understand how loss functions behave around current model parameters.

Consider a loss function:

\[
L(\mathbf{w})
\]

where \(\mathbf{w}\) represents the model parameters.

A first-order approximation tells us how the loss is expected to change when we slightly adjust the parameters.

A second-order approximation also considers how the gradient changes as the parameters move.

This is particularly important in optimization algorithms.

For example, Newton's method uses gradient and Hessian information to determine parameter updates.

The mathematical foundation developed in this experiment will help us understand:

- Gradient-based optimization.
- Second-order optimization methods.
- Newton's method.
- Neural network training.
- Loss landscape geometry.

## Conclusion

This experiment brought together several important concepts from multivariable calculus: partial derivatives, gradients, Hessians, tangent planes, matrix multiplication, and Taylor approximations.

We implemented both first- and second-order approximations, compared their numerical accuracy, and visualized their geometric differences.

Most importantly, we established how gradient and curvature information can be used to understand and approximate nonlinear functions.

This completes Experiment 8 of the **Inside the Loss Landscape** project.