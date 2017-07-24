import numpy as np
from math import pi, cos, sin
from matplotlib import pyplot as plt

class AffineRegressTest:
    AFFINE_MAT_RANDOM_VALUE_RANGE = (-3.0, 3.0)
    def __init__(self, xy_range, noise_std_dev, num_points):
        self.num_points = num_points
        self.noise_std_dev = noise_std_dev
        self.xy_range = xy_range
        self.init_set()
        self.solve_fit_affine_mat()

    def init_set(self):
        '''inits points as three dimensional where the third index is always 1
        because it is an affine transform'''
        self.X = np.ones((self.num_points, 3))
        self.X[:, :2] = (np.random.rand(self.X.shape[0], 2)) * (self.xy_range[1] - self.xy_range[0]) + self.xy_range[0]

        self.ground_truth_affine_mat = np.identity(3, dtype = np.float32)

        random_affine_vals = np.random.rand(2, 3) * (AffineRegressTest.AFFINE_MAT_RANDOM_VALUE_RANGE[1] - AffineRegressTest.AFFINE_MAT_RANDOM_VALUE_RANGE[0]) + AffineRegressTest.AFFINE_MAT_RANDOM_VALUE_RANGE[0]
        self.ground_truth_affine_mat[:2, :3] = random_affine_vals

        self.Y = self.X.dot(self.ground_truth_affine_mat.T)
        self.Y[:, :2] += np.random.normal(scale = self.noise_std_dev, size = (self.Y.shape[0], 2))

    def solve_fit_affine_mat(self):
        Q = np.zeros((3,3))
        Q_product = np.zeros((3))
        P_product = np.zeros((3))
        for q in range(0, Q.shape[0]):
            Q_q_1 = np.dot(self.X[:, 0], self.X[:, q])
            Q_q_2 = np.dot(self.X[:, 1], self.X[:, q])
            Q_q_3 = np.sum(self.X[:, q])
            Q[q] = np.array([Q_q_1, Q_q_2, Q_q_3])


            Q_product[q] = np.dot(self.Y[:, 0], self.X[:, q])
            P_product[q] = np.dot(self.Y[:, 1], self.X[:, q])

        P = Q.copy()

        fit_affine_first_row = np.linalg.inv(Q).dot(Q_product.T)
        fit_affine_second_row = np.linalg.inv(P).dot(P_product.T)
        self.fit_affine_mat = np.array([fit_affine_first_row, fit_affine_second_row, [0,0,1]])

    def plot(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], marker = "v")
        plt.scatter(self.Y[:, 0], self.Y[:, 1], marker = "s")
        X_transformed = self.X.dot(self.fit_affine_mat.T)
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], marker = "+")
        plt.show()
