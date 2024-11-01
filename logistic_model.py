import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def predict(X, theta):
    probabilities = sigmoid(np.dot(X, theta))
    return [1 if prob >= 0.5 else 0 for prob in probabilities]
