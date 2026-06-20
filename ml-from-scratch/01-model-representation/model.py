"""
model.py
--------
A from-scratch implementation of the univariate linear regression model:

    f_w,b(x) = w*x + b

This module deliberately avoids scikit-learn / any ML library. The goal of
this lab is to build intuition for *what the model actually computes*,
before learning how to automatically find good values of w and b
(that comes in the Cost Function and Gradient Descent labs).

Notation (matches the course):
    x       : input feature  (years of experience)
    y       : target/label   (salary, in $1000s)
    w       : weight (slope) parameter
    b       : bias (intercept) parameter
    m       : number of training examples
    f_wb    : model prediction
"""

import numpy as np


def compute_model_output(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """
    Computes the prediction of a linear model for every example in x.

    Args:
        x (ndarray (m,)): input feature values, m examples
        w (scalar): weight parameter
        b (scalar): bias parameter

    Returns:
        f_wb (ndarray (m,)): model prediction for each x[i]
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb


def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """
    Computes the Mean Squared Error cost for given w, b.

    This isn't covered yet in the Model Representation lab, but it's
    included here so you can *see*, numerically, that some (w, b) choices
    fit the data better than others -- which is exactly the problem the
    next lab (Cost Function) formalizes, and the lab after that
    (Gradient Descent) solves automatically.

    Args:
        x (ndarray (m,)): input feature values
        y (ndarray (m,)): true target values
        w, b (scalar): model parameters

    Returns:
        total_cost (float): the cost of using w, b as parameters
    """
    m = x.shape[0]
    cost_sum = 0.0
    for i in range(m):
        f_wb_i = w * x[i] + b
        cost_sum += (f_wb_i - y[i]) ** 2
    total_cost = cost_sum / (2 * m)
    return total_cost


if __name__ == "__main__":
    # Quick manual sanity check
    x_train = np.array([1.0, 2.0, 3.0])
    y_train = np.array([38.0, 41.0, 44.0])

    w, b = 3.0, 35.0
    predictions = compute_model_output(x_train, w, b)
    cost = compute_cost(x_train, y_train, w, b)

    print(f"x_train = {x_train}")
    print(f"y_train = {y_train}")
    print(f"predictions (w={w}, b={b}) = {predictions}")
    print(f"cost = {cost:.4f}")
