# Experiment 5: Jacobian

## Overview

This experiment explores the Jacobian matrix and how it describes the relationship between changes in multiple input variables and changes in multiple output functions.

Unlike the gradient, which describes the rate of change of a scalar-valued function, the Jacobian handles vector-valued functions with multiple outputs.

## Mathematical Foundations

We define a vector-valued function:
- **f1(w1, w2) = w1² + 2w2**
- **f2(w1, w2) = 3w1 + w2²**

Expressed as an output vector:
**f(w1, w2) = [ f1(w1, w2) , f2(w1, w2) ] = [ (w1² + 2w2) , (3w1 + w2²) ]**

The Jacobian is constructed by calculating the partial derivatives of every output with respect to every input, forming a 2x2 matrix:
- **Row 1:** [ ∂f1/∂w1 , ∂f1/∂w2 ] = **[ 2w1 , 2 ]**
- **Row 2:** [ ∂f2/∂w1 , ∂f2/∂w2 ] = **[ 3 , 2w2 ]**

Evaluating this Jacobian matrix at our starting point **(1, 1)** gives:
- **Row 1:** **[ 2 , 2 ]**
- **Row 2:** **[ 3 , 2 ]**

In this matrix, the rows represent the individual output functions, while the columns represent the input variables.

## Local Linear Approximation

For a small input change, the Jacobian matrix approximates the resulting output change using the linear transformation formula:

**Δf ≈ J * Δw**

Using our first input perturbation vector:
**Δw = [ 0.1 , 0.2 ]**

We execute matrix multiplication and obtain our predicted output change vector:
- **Δf1 ≈ (2 * 0.1) + (2 * 0.2) = 0.6**
- **Δf2 ≈ (3 * 0.1) + (2 * 0.2) = 0.7**
**Δf ≈ [ 0.6 , 0.7 ]**

However, evaluating the original functions at the new point yields an actual output change of:
**Δf_actual = [ 0.61 , 0.74 ]**

The difference occurs because the Jacobian captures first-order linear effects but does not account for higher-order nonlinear acceleration effects.

## Experiment: Approximation Accuracy

Three input perturbations were tested to track the error:

| Input Perturbation (dw1, dw2) | Approximation Error Vector |
| :--- | :--- |
| (0.1, 0.2) | [0.01, 0.04] |
| (0.01, 0.02) | [0.0001, 0.0004] |
| (0.001, 0.002) | [0.000001, 0.000004] |

The experiment demonstrated that reducing the input perturbations by a factor of 10 reduced the absolute approximation errors by a factor of 100.

This happened because the functions contain squared terms, making the leftover approximation errors second-order (quadratic) in relation to the input perturbations.

## Key Learnings

1. A Jacobian describes how multiple output functions respond to changes in multiple input variables.
2. Its rows correspond to output functions, while its columns correspond to input variables.
3. An m-output, n-input function has an m × n Jacobian matrix.
4. The Jacobian provides a local linear approximation of output changes.
5. The approximation becomes increasingly accurate as input perturbations become smaller.

## Connection to Artificial Intelligence

Jacobians are fundamental to understanding neural networks and backpropagation.

A neural network layer transforms one vector into another. Its Jacobian describes how changes in the input activations affect the output activations.

During backpropagation, the chain rule combines derivatives across successive layers to determine how changes propagate through the network.

Understanding the Jacobian therefore provides an important foundation for understanding how neural networks learn.

## Conclusion

This experiment demonstrated how to construct a Jacobian matrix, use matrix multiplication to approximate output changes, and compare those approximations against actual changes.

The central lesson is that the Jacobian describes the local flat transformation behavior of a system, serving as the foundational multi-variable engine behind deep learning gradient calculations.
