import numpy as np
from Regression.LinearRegression import LinearRegression

class LinearLOESS:

    def __init__(self, X, y, x_window, weight_func = None):
        self.X = X
        self.y = y
        self.x_window = x_window
        self.weight_func = weight_func if weight_func is not None else self.weight_set_tricube

    def predict_set(self, X):
        predictions = np.zeros((X.shape[0]))
        for i in range(0, predictions.shape[0]):
            predictions[i] = self.predict(X[i])
        return predictions

    def predict(self, x):
        X_subset, y_subset, weight_subset = self.weight_func(x)
        try:
            lin_regression = LinearRegression(X_subset, y_subset, weight_subset)
            lin_regression.train()
            return lin_regression.predict(x)
        except:
            return 0

    def weight_set_tricube(self, x):
        x_distances = np.linalg.norm(self.X - x, axis = 1)
        x_distances /= self.x_window / 2
        indices_in_window = np.where(x_distances < 1)[0]

        point_weights = (1.0 - x_distances**3)**3

        X_subset_in_window = self.X[indices_in_window, :]
        y_subset_in_window = self.y[indices_in_window]
        weight_subset_in_window = point_weights[indices_in_window]
        return X_subset_in_window, y_subset_in_window, weight_subset_in_window
