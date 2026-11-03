# Experiment 1: Exploring the Loss Surface

## 1. Objective
The objective of this experiment was to understand how a loss function behaves across different parameter values and how gradient descent uses gradients to minimise that loss. We used a simple two-parameter loss function to visualise the loss landscape and observe how gradient descent moves towards its minimum.

## 2. Mathematical Foundation
We defined our loss function as:

$$L(w_1,w_2)=w_1^2+2w_2^2$$

Where:
* **$w_1$ and $w_2$** represent model parameters (weights).
* **$L$** represents the loss associated with a particular combination of parameters.

The minimum of this function occurs at:

$$(w_1,w_2)=(0,0)$$

At this point, the loss is zero.

### Gradient calculation
To understand how the loss changes with respect to each parameter, we calculated the partial derivatives:

$$\frac{\partial L}{\partial w_1}=2w_1$$

$$\frac{\partial L}{\partial w_2}=4w_2$$

Therefore, the gradient is:

$$\nabla L= \begin{bmatrix} 2w_1\\ 4w_2 \end{bmatrix}$$

The gradient tells us the direction of steepest increase in the loss. Gradient descent moves in the opposite direction to reduce the loss.

## 3. Visualising the Loss Landscape
We used NumPy's `linspace()` to generate possible parameter values and `meshgrid()` to create combinations of those values. We then calculated the loss at every coordinate and used Matplotlib's `plot_surface()` to visualise the result. 

The resulting surface was a bowl-shaped quadratic surface, also known as an **elliptic paraboloid**. An important observation was that the surface rises more steeply along the $w_2$ direction because its squared term has a coefficient of 2.

## 4. Implementing Gradient Descent
We selected the following initial values:
* **Initial $w_1$:** 2
* **Initial $w_2$:** 1
* **Learning rate ($\eta$):** 0.1
* **Number of iterations:** 100

The gradient descent update rule is:

$$w_{\text{new}}=w_{\text{old}}-\eta\nabla L$$

For our two parameters, this becomes:

$$w_1^{\text{new}}=w_1-0.1(2w_1)$$

$$w_2^{\text{new}}=w_2-0.1(4w_2)$$

We implemented these updates inside a Python loop and stored the parameter values and corresponding loss after every iteration.

## 5. Results and Observations
After 100 iterations, we obtained approximately:

| Measurement | Result |
| :--- | ---: |
| Starting $w_1$ | 2 |
| Starting $w_2$ | 1 |
| First updated $w_1$ | 1.6 |
| First updated $w_2$ | 0.6 |
| Final $w_1$ | $4.0741 \times 10^{-10}$ |
| Final $w_2$ | $6.5332 \times 10^{-23}$ |
| Final loss | $1.6598 \times 10^{-19}$ |

The final parameters approached zero, and the final loss became extremely small.

### Visual observations
We produced two important visualisations:
1. **3D Loss Surface with Gradient Descent Trajectory:** This showed the path taken by gradient descent from the starting point towards the minimum of the loss surface.
2. **Loss Against Iterations:** This showed how the loss decreased throughout the optimisation process. 

We observed that the loss decreased quickly during the initial iterations and gradually flattened as it approached zero. This happens because the gradients become smaller as the parameters approach the minimum, resulting in smaller parameter updates.

## 6. Key Machine Learning Takeaways
This experiment helped establish the relationship between loss functions, gradients, and optimisation.
* **Loss function:** Measures how well or poorly a model performs.
* **Model parameters:** Determine the loss produced by the model.
* **Partial derivatives:** Tell us how the loss changes with respect to individual parameters.
* **Gradient:** Provides the direction of steepest increase in the loss.
* **Gradient descent:** Updates parameters in the opposite direction to the gradient.
* **Learning rate:** Controls the size of each update.
* **Repeated updates:** Progressively reduce the loss over time.
* **Loss landscape:** The unique shape of the surface directly influences the optimization path.

## 7. Conclusion
This experiment demonstrated how gradient descent can minimise a loss function by repeatedly updating model parameters using their gradients. Beyond implementing the algorithm, the most important understanding gained was how the mathematical concepts connect: the loss function defines the optimisation objective, partial derivatives measure parameter sensitivity, and gradient descent uses that information to improve the parameters. These concepts form an important mathematical foundation for understanding how machine learning models update and learn.
