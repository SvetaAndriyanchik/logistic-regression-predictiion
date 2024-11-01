import numpy as np
from logistic_model import sigmoid, compute_cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        # Optionally, print the cost every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")

    return theta, cost_history
