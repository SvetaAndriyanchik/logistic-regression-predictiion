from data_processing import load_and_normalize_data
from gradient_descent import gradient_descent
from logistic_model import compute_cost, predict, sigmoid
from visualization import plot_cost_history, plot_predictions
import numpy as np

# Load and normalize data
features = ['Age', 'Income', 'Family_members', 'House_size']
X, y, min_values, max_values = load_and_normalize_data('customer_data.csv', features, 'Purchased')

# Initialize parameters
n_features = X.shape[1]
theta = np.zeros(n_features)
learning_rate = 0.1
iterations = 1000

# Run gradient descent
theta_optimal, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

# Display optimized weights
print("\nOptimized weights (theta):")
print(theta_optimal)

# Plot cost history
plot_cost_history(cost_history, iterations)

# Plot predictions vs actual values
plot_predictions(X, y, theta_optimal)

# Predict for a new customer
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

    print(f"\nPredicted probability that the new customer will buy the product: {probability:.4f}")

predict_new_customer()
