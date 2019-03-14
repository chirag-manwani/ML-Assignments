import numpy as np
from cvxopt import solvers
from cvxopt import matrix


class SVM():

    def __init__(
        self,
        X,
        Y,
        kernel='linear',
        c=0,
        threshold=1e-8
    ):
        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.c = c
        self.threshold = threshold
        self.alphas = []
        self.w = []
        self.b = []
        self.support_vectors = []
        self.support_vectors_labels = []

    def predict(
        self
    ):
        print('predict')

    def model(
    ):
        print('model')

    def fit(
        self
    ):
        num_samples = self.X.shape[0]
        G_0 = -np.identity(num_samples)
        G_1 = np.identity(num_samples)

        h_0 = np.zeros(num_samples)
        h_1 = np.ones(num_samples) * self.c

        P = matrix((np.outer(self.Y, self.Y) * (self.X @ self.X.T)))
        q = matrix(-np.ones((num_samples, 1)))
        G = matrix(np.vstack((G_0, G_1)))
        h = matrix(np.hstack((h_0, h_1)))
        A = matrix(self.Y.reshape(1, -1))
        b = matrix(np.zeros(1))

        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        support_vectors_idx = np.where(alphas > self.threshold)[0]
        alphas = alphas[support_vectors_idx]

        self.alphas = alphas.reshape(alphas.shape[0])
        self.support_vectors = self.X[support_vectors_idx]
        self.support_vectors_labels = self.Y[support_vectors_idx]

        self.w = np.zeros(self.support_vectors.shape[1])
        for i in range(self.support_vectors.shape[0]):
            self.w += (self.alphas[i] * self.support_vectors_labels[i])
            * self.support_vectors[i]

        max_prod = -1e6
        min_prod = 1e6
        for i in range(self.support_vectors.shape[0]):
            prod = self.w @ self.support_vectors[i]
            if self.support_vectors_labels[i] == 1:
                if prod < min_prod:
                    min_prod = prod
            else:
                if prod > max_prod:
                    max_prod = prod
        self.b = -0.5 * (min_prod + max_prod)

