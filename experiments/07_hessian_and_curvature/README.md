# Experiment 7: Hessian and Curvature

## Overview

In this experiment, we explore the **Hessian matrix**, its eigenvalues, and how they describe the curvature of a multivariable function.

While the gradient tells us the direction and rate of steepest increase of a function, the Hessian tells us how the gradient changes as we move through the function.

Understanding the Hessian is important in optimization because it helps us identify local minima, local maxima, saddle points, and flat regions in loss landscapes.

## 1. The Loss Function

We begin with our familiar quadratic loss function:

\[L(w_1,w_2)=w_1^2+2w_2^2\]

Its gradient is:

\[\nabla L= \begin{bmatrix} 2w_1\\ 4w_2 \end{bmatrix}\]

The Hessian is obtained by differentiating the gradient components again:

\[H= \begin{bmatrix} \frac{\partial^2L}{\partial w_1^2} & \frac{\partial^2L}{\partial w_1\partial w_2}\\ \frac{\partial^2L}{\partial w_2\partial w_1} & \frac{\partial^2L}{\partial w_2^2} \end{bmatrix}\]

Therefore:

\[H= \begin{bmatrix} 2&0\\ 0&4 \end{bmatrix}\]

The Hessian is constant because our loss function is quadratic.

## 2. Understanding Hessian Eigenvalues

For a real symmetric Hessian, its eigenvalues describe the curvature along its principal directions.

- Positive eigenvalue: curves upward.
- Negative eigenvalue: curves downward.
- Zero eigenvalue: zero second-order curvature in that direction.

We can use the eigenvalues to classify the Hessian.

| Hessian type | Eigenvalues | Geometric interpretation |
|---|---|---|
| Positive definite | All positive | Curves upward in every direction |
| Negative definite | All negative | Curves downward in every direction |
| Indefinite | Both positive and negative | Curves upward in some directions and downward in others |
| Positive semidefinite | All non-negative | Upward or flat curvature |
| Negative semidefinite | All non-positive | Downward or flat curvature |

At a stationary point, these classifications help us determine the local behaviour:

- Positive definite: strict local minimum.
- Negative definite: strict local maximum.
- Indefinite: saddle point.
- Semidefinite: the second derivative test is generally inconclusive.

## 3. Implementing the Hessian

We use NumPy to represent the gradient and Hessian.

```python
import numpy as np

def compute_loss(w1, w2):
    return w1**2 + 2*w2**2

def compute_gradient(w1, w2):
    return np.array([
        2*w1,
        4*w2
    ], dtype=float)

def compute_hessian():
    return np.array([,
        [0, 4]
    ], dtype=float)

hessian = compute_hessian()

eigenvalues = np.linalg.eigvalsh(hessian)

print("Hessian:\n", hessian)
print("Eigenvalues:", eigenvalues)
```

Output:

```text
Hessian:
 [[2. 0.]
  [0. 4.]]

Eigenvalues: [2. 4.]
```

We use `np.linalg.eigvalsh()` because our Hessian is real and symmetric.

## 4. Classifying Hessian Matrices

We implemented a classification function using the eigenvalues of a Hessian.

```python
tol = 1e-10

def classify_hessian(hessian):
    eigenvalues = np.linalg.eigvalsh(hessian)

    if np.all(eigenvalues > tol):
        return "Positive definite"

    elif np.all(eigenvalues < -tol):
        return "Negative definite"

    elif np.any(eigenvalues < -tol) and np.any(eigenvalues > tol):
        return "Indefinite"

    elif np.all(eigenvalues >= -tol):
        return "Positive semidefinite"

    elif np.all(eigenvalues <= tol):
        return "Negative semidefinite"
```

The tolerance helps us account for small floating-point errors when determining whether an eigenvalue is zero.

We tested the function using six different matrices:

1. Positive definite
2. Negative definite
3. Indefinite
4. Positive semidefinite
5. Negative semidefinite
6. Zero matrix

All six cases were classified as expected.

**Important:** An indefinite Hessian at a stationary point indicates a saddle point. The Hessian alone does not establish that an arbitrary point is stationary.

## 5. Visualizing Different Curvature Types

To understand the geometry behind the classifications, we implemented four different quadratic functions.

### A. Positive Definite

$$
L(x,y)=x^2+y^2
$$

Hessian:

$$
H=
\begin{bmatrix}
2&0\\
0&2
\end{bmatrix}
$$

Eigenvalues: $2,2$.

The surface curves upward in every direction, producing a bowl-shaped surface with a strict minimum at the origin.

### B. Negative Definite

$$
L(x,y)=-x^2-y^2
$$

Hessian:

$$
H=
\begin{bmatrix}
-2&0\\
0&-2
\end{bmatrix}
$$

Eigenvalues: $-2,-2$.

The surface curves downward in every direction, producing an inverted bowl with a strict maximum at the origin.

### C. Indefinite

$$
L(x,y)=x^2-y^2
$$

Hessian:

$$
H=
\begin{bmatrix}
2&0\\
0&-2
\end{bmatrix}
$$

Eigenvalues: $2,-2$.

The surface curves upward along the x-direction and downward along the y-direction, producing a saddle point at the origin.

### D. Positive Semidefinite

$$
L(x,y)=x^2
$$

Hessian:

$$
H=
\begin{bmatrix}
2&0\\
0&0
\end{bmatrix}
$$

Eigenvalues: $2,0$.

The surface curves upward along the x-direction but remains flat along the y-direction.

Consequently, the minimum is not isolated. Every point along the y-axis has a loss of zero.

## 6. Key Observations

Through the visualizations and numerical experiments, we observed that:

- Positive definite Hessians describe upward curvature in every direction.
- Negative definite Hessians describe downward curvature in every direction.
- Indefinite Hessians describe surfaces with both upward and downward curvature.
- Positive semidefinite Hessians can have flat directions.
- Zero eigenvalues indicate directions with zero second-order curvature.
- Eigenvalues reveal curvature strength, while eigenvectors identify the corresponding principal directions.

One important distinction is that the diagonal entries of a Hessian describe curvature along the coordinate axes, while its eigenvalues describe curvature along its principal directions.

## 7. Connection to AI and Machine Learning

The Hessian plays an important role in understanding neural network optimization.

Neural networks often have highly complex loss landscapes containing:

- Local minima
- Saddle points
- Flat regions
- Steep curvature directions

The Hessian helps us understand these geometric properties.

For example:

**1. Learning-rate behaviour**

Large curvature in a direction can make optimization sensitive to the learning rate. A learning rate that is too large may cause gradient descent to overshoot or become unstable.

**2. Saddle points**

A stationary point does not necessarily represent a minimum. An indefinite Hessian reveals directions in which the loss can still decrease.

**3. Flat regions**

Small or zero eigenvalues indicate directions with little or no second-order curvature. These can influence how quickly optimization progresses.

**4. Second-order optimization**

Methods such as Newton's method use Hessian information to adjust parameter updates according to local curvature.

For a parameter vector $\mathbf{w}$, the Newton update is:

$$
\mathbf{w}_{new}
=
\mathbf{w}
-
H^{-1}\nabla L
$$

This is one reason understanding Hessians is important before studying more advanced optimization algorithms.

## 8. Conclusion

This experiment helped us move beyond understanding only the direction of steepest increase or decrease.

We learned that the gradient describes first-order behaviour, while the Hessian describes second-order behaviour and curvature.

By studying Hessian eigenvalues and visualizing different quadratic surfaces, we developed a geometric understanding of minima, maxima, saddle points, and flat regions.

These concepts provide an important mathematical foundation for understanding how optimization algorithms navigate machine learning loss landscapes.

## Technologies Used

- Python
- NumPy
- Matplotlib

## Experiment Status

**Completed.**
