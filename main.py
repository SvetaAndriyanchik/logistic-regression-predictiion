import numpy as np
import pandas as pd

data = pd.read_csv('customer_data.csv')

X = data[['Age', 'Income', 'Family_members', 'House_size']].values
y = data['Purchased'].values

# Normalize
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)

X = (X - min_values) / (max_values - min_values)

# Add bias term
X = np.hstack((np.ones((X.shape[0], 1)), X))

n_features = X.shape[1]
theta = np.zeros(n_features)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history


learning_rate = 0.1
iterations = 1000

theta_optimal, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

print("\nOptimized weights (theta):")
print(theta_optimal)


def predict(X, theta):
    probabilities = sigmoid(np.dot(X, theta))
    return [1 if prob >= 0.5 else 0 for prob in probabilities]


def predict_new_customer():
    age = float(input("Enter the customer's age: "))
    income = float(input("Enter the customer's annual income (e.g., 50000): "))
    family_members = int(input("Enter the number of family members: "))
    house_size = float(input("Enter the house size in square feet: "))

    new_customer = np.array([age, income, family_members, house_size])
    new_customer_normalized = (new_customer - min_values) / (max_values - min_values)
    new_customer_with_bias = np.hstack(([1], new_customer_normalized))

    z_new = np.dot(new_customer_with_bias, theta_optimal)

    probability = sigmoid(z_new)

    print(f"\nPredicted probability that the new customer will buy the vacuum cleaner: {probability:.4f}")


predict_new_customer()
