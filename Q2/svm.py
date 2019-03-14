import numpy as np
from cvxopt import solvers
from cvxopt import matrix


class SVM():

    def __init__(
        self,
        kernel='linear',
        c=0,
        threshold=1e-8
    ):
        self.kernel = kernel
        self.c = c
        self.threshold = threshold
        self.alphas = []
        self.w = []
        self.b = []
        self.support_vectors = []
        self.support_vectors_labels = []

    def predict(
        self,
        X_test
    ):
        Y_pred = X_test @ self.w + self.b
        condition_list = [Y_pred > 0, Y_pred < 0]
        choice_list = [1, -1]
        Y_pred = np.select(condition_list, choice_list)

        return Y_pred

    def fit(
        self,
        X,
        Y
    ):
        num_samples = X.shape[0]
        G_0 = -np.identity(num_samples)
        G_1 = np.identity(num_samples)

        h_0 = np.zeros(num_samples)
        h_1 = np.ones(num_samples) * self.c

        P = matrix((np.outer(Y, Y) * (X @ X.T)))
        q = matrix(-np.ones((num_samples, 1)))
        G = matrix(np.vstack((G_0, G_1)))
        h = matrix(np.hstack((h_0, h_1)))
        A = matrix(Y.reshape(1, -1))
        b = matrix(np.zeros(1))

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        support_vectors_idx = np.where(alphas > self.threshold)[0]
        alphas = alphas[support_vectors_idx]

        self.alphas = alphas.reshape(alphas.shape[0])
        self.support_vectors = X[support_vectors_idx]
        self.support_vectors_labels = Y[support_vectors_idx]

        self.w = np.zeros(self.support_vectors.shape[1])
        for i in range(self.support_vectors.shape[0]):
            self.w += (self.alphas[i] * self.support_vectors_labels[i]) \
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
