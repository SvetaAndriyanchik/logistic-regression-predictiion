import matplotlib.pyplot as plt
import numpy as np

from logistic_model import predict


def plot_cost_history(cost_history, iterations):
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), cost_history, color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function Convergence')
    plt.show()


def plot_predictions(X, y, theta):
    # Get predictions
    predictions = np.array(predict(X, theta))

    # Plot actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y)), y, label='Actual', color='blue', alpha=0.6, marker='o')

    # Offset predictions slightly to the right to distinguish them
    plt.scatter(range(len(predictions)), predictions, label='Predicted', color='red', alpha=0.6, marker='x')

    # Add labels and title
    plt.xlabel('Data Index')
    plt.ylabel('Purchased (0 or 1)')
    plt.title('Actual vs Predicted Purchases')
    plt.legend()
    plt.show()
