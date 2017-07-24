import numpy as np

class LinearRegression:

    def __init__(self, X, y, data_weights = None):
        self.X = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
        self.y = y
        self.data_weights = data_weights if data_weights is not None else np.ones((self.X.shape[0]))

    def train(self):
        mat = np.zeros((self.X.shape[1], self.X.shape[1]))
        for i in range(0, mat.shape[0]):
            for j in range(0, mat.shape[1]):
                mat[i,j] = np.sum((self.X[:, i] * self.X[:,j]).dot(self.data_weights.T))

        mat_product = np.zeros((self.X.shape[1]))
        for i in range(0, mat_product.shape[0]):
            mat_product[i] = np.sum((self.y * self.X[:, i]).dot(self.data_weights))

        self.dot_weights = np.linalg.inv(mat).dot(mat_product.T)

    def predict_set(self, X):
        predictions = np.zeros((X.shape[0]))
        for i in range(0, predictions.shape[0]):
            predictions[i] = self.predict(X[i])
        return predictions

    def predict(self, x):
        x = np.append(x, np.array([1]))
        return np.dot(self.dot_weights, x)
