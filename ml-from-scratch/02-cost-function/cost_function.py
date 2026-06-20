"""
cost_function.py
-----------------
Implements the cost function for univariate linear regression:

    J(w,b) = (1 / 2m) * sum_i ( f_w,b(x_i) - y_i )^2

This quantifies "how good" a given (w, b) pair is: the lower J(w,b), the
better the line fits the training data. In the previous topic
(Model Representation) we picked w, b by hand and eyeballed the fit. This
formalizes that comparison into a single number we can actually compare
and, eventually (next topic: Gradient Descent), minimize automatically.

This version is vectorized with NumPy (no explicit Python loop), as a
small step up from the loop-based version in 01-model-representation.
"""

import numpy as np


def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """
    Computes the Mean Squared Error cost J(w, b) for a single (w, b) pair.

    Args:
        x (ndarray (m,)): input feature values
        y (ndarray (m,)): true target values
        w, b (scalar): model parameters

    Returns:
        cost (float): J(w, b)
    """
    m = x.shape[0]
    f_wb = w * x + b
    cost = np.sum((f_wb - y) ** 2) / (2 * m)
    return cost


def compute_cost_grid(x: np.ndarray, y: np.ndarray,
                       w_range: np.ndarray, b_range: np.ndarray):
    """
    Computes J(w, b) over a grid of w and b values -- used to draw
    contour plots / 3D surfaces of the cost function.

    Args:
        x, y: training data
        w_range (ndarray (n_w,)): values of w to evaluate
        b_range (ndarray (n_b,)): values of b to evaluate

    Returns:
        W, B (ndarray (n_b, n_w)): meshgrid of w, b values
        J (ndarray (n_b, n_w)): cost at each (W[i,j], B[i,j])
    """
    W, B = np.meshgrid(w_range, b_range)
    J = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            J[i, j] = compute_cost(x, y, W[i, j], B[i, j])
    return W, B, J


def grid_search_minimum(x: np.ndarray, y: np.ndarray,
                         w_range: np.ndarray, b_range: np.ndarray):
    """
    Brute-force search over a grid for the (w, b) pair with lowest cost.
    Not how you'd do this in practice (that's what Gradient Descent is
    for) -- but useful here to numerically confirm where the minimum is
    before we learn an algorithm that finds it directly.

    Returns:
        best_w, best_b, best_cost
    """
    W, B, J = compute_cost_grid(x, y, w_range, b_range)
    idx = np.unravel_index(np.argmin(J), J.shape)
    return W[idx], B[idx], J[idx]


if __name__ == "__main__":
    x_train = np.array([1.0, 2.0, 3.0])
    y_train = np.array([38.0, 41.0, 44.0])

    for w, b in [(2.0, 30.0), (3.0, 35.0), (3.0, 36.0)]:
        print(f"J(w={w}, b={b}) = {compute_cost(x_train, y_train, w, b):.4f}")
