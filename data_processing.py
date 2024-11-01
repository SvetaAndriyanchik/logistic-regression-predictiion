import numpy as np
import pandas as pd

def load_and_normalize_data(filename, feature_columns, target_column):
    data = pd.read_csv(filename)
    X = data[feature_columns].values
    y = data[target_column].values

    # Min-max normalization
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    X_normalized = (X - min_values) / (max_values - min_values)

    # Add bias term
    X_normalized = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))

    return X_normalized, y, min_values, max_values
