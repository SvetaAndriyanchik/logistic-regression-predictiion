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
    predictions = np.array(predict(X, theta))
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y)), y, label='Actual', color='blue', alpha=0.6)
    plt.scatter(range(len(predictions)), predictions, label='Predicted', color='red', alpha=0.6)
    plt.xlabel('Data Index')
    plt.ylabel('Purchased (0 or 1)')
    plt.title('Actual vs Predicted Purchases')
    plt.legend()
    plt.show()
