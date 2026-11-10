# Experiment 6: Chain Rule

## Overview

This experiment explores the chain rule in single-variable and multivariable calculus, with a focus on how derivatives propagate through composed functions.

The goal is to understand how changes in an original input affect the final output when multiple functions are connected together. This is particularly important in machine learning because neural networks consist of multiple layers of transformations, and backpropagation relies on the chain rule to calculate gradients.

---

## Objectives

- Understand the chain rule for single-variable functions.
- Apply the chain rule to multivariable functions.
- Calculate Jacobians of vector-valued functions.
- Implement the multivariable chain rule using NumPy matrix multiplication.
- Verify the chain rule by comparing the composite Jacobian with direct differentiation.

---

## 1. Single-Variable Chain Rule

Consider two composed functions:

\[y = f(x) = x^2\]

\[z = g(y) = 3y + 1\]

Using the chain rule:

\[\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}\]

Since \(\frac{dy}{dx} = 2x\) and \(\frac{dz}{dy} = 3\), we obtain:

\[\frac{dz}{dx} = 6x\]

The implementation calculates the derivative through the chain rule and compares it with direct differentiation.

---

## 2. Multivariable Scalar Chain Rule

Consider:

\[y = f(x_1, x_2) = x_1^2 + 3x_2\]

\[z = g(y) = 2y\]

The chain rule gives:

\[\frac{\partial z}{\partial x_1} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x_1} = 2(2x_1) = 4x_1\]

\[\frac{\partial z}{\partial x_2} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x_2} = 2(3) = 6\]

This demonstrates how a scalar output can depend on multiple input variables through an intermediate function.

---

## 3. Vector-Valued Chain Rule

We now consider two composed vector-valued functions:

**First transformation:**

\[f(x_1, x_2) = \begin{bmatrix} x_1^2 + x_2 \\ x_1 + 2x_2 \end{bmatrix}\]

**Second transformation:**

\[g(y_1, y_2) = \begin{bmatrix} 3y_1 + y_2 \\ y_1 y_2 \end{bmatrix}\]

The Jacobian of the first function is:

\[J_f = \begin{bmatrix} 2x_1 & 1 \\ 1 & 2 \end{bmatrix}\]

The Jacobian of the second function is:

\[J_g = \begin{bmatrix} 3 & 1 \\ y_2 & y_1 \end{bmatrix}\]

The multivariable chain rule states:

\[J_{g \circ f} = J_g J_f\]

> **Note:** Matrix multiplication order matters. The Jacobian of the later transformation (\(J_g\)) sits on the left, and the Jacobian of the earlier transformation (\(J_f\)) sits on the right.

---

## 4. Numerical Implementation

We evaluate the functions at (x₁, x₂) = (2, 3).

The first transformation produces:

\[f(2, 3) = \begin{bmatrix} 7 \\ 8 \end{bmatrix}\]

Evaluating both Jacobians at these values gives:

\[J_f = \begin{bmatrix} 4 & 1 \\ 1 & 2 \end{bmatrix}, \qquad J_g = \begin{bmatrix} 3 & 1 \\ 8 & 7 \end{bmatrix}\]

Multiplying the Jacobians:

\[J_{g \circ f} = \begin{bmatrix} 3 & 1 \\ 8 & 7 \end{bmatrix} \begin{bmatrix} 4 & 1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 13 & 5 \\ 39 & 22 \end{bmatrix}\]

In Python, this is implemented using NumPy's matrix multiplication operator (`@`):

```python
def compute_composite_jacobian(jacobian_g, jacobian_f):
    return jacobian_g @ jacobian_f
```

---

## 5. Verification Through Direct Differentiation

To verify the result independently, we differentiate the first final output ($z_1$) directly in terms of $x_1$ and $x_2$:

$$
z_1 = 3y_1 + y_2
$$

Substituting the intermediate functions $y_1 = x_1^2 + x_2$ and $y_2 = x_1 + 2x_2$:

$$
z_1 = 3(x_1^2 + x_2) + (x_1 + 2x_2) = 3x_1^2 + x_1 + 5x_2
$$

Differentiating directly with respect to each input:

$$
\frac{\partial z_1}{\partial x_1} = 6x_1 + 1, \qquad \frac{\partial z_1}{\partial x_2} = 5
$$

At $(x_1, x_2) = (2, 3)$:

$$
\begin{bmatrix} \frac{\partial z_1}{\partial x_1} & \frac{\partial z_1}{\partial x_2} \end{bmatrix} = \begin{bmatrix} 13 & 5 \end{bmatrix}
$$

This matches the first row of the composite Jacobian ($J_{g \circ f}$).

---

## Key Findings

- The chain rule allows us to calculate how changes propagate through composed functions.
- For vector-valued functions, Jacobians combine through matrix multiplication.
- The order of Jacobian multiplication matters: the Jacobian of the later transformation comes first ($J_g J_f$).
- The composite Jacobian describes how every final output changes with respect to every original input.
- Direct differentiation provides an independent way to verify the result.

---

## Connection to Machine Learning

Neural networks are built from successive transformations. Each layer receives an input, performs a transformation, and passes its output to the next layer.

The chain rule allows us to calculate how the final loss changes with respect to earlier layers and their parameters. This is the mathematical foundation of **backpropagation**, which enables neural networks to update their parameters during training.

---

## Conclusion

This experiment connected single-variable differentiation, multivariable calculus, Jacobians, and matrix multiplication through the chain rule. It demonstrated how derivatives propagate through successive transformations and why this principle is essential to modern machine learning.
