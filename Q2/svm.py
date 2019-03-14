import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from cvxopt import solvers
from cvxopt import matrix


class SVM():

    def __init__(
        self,
        kernel='linear',
        c=0,
        threshold=1e-5,
        gamma=0.05
    ):
        if kernel not in ['linear', 'gaussian']:
            self.kernel = 'linear'
        else:
            self.kernel = kernel
        self.c = c
        self.threshold = threshold
        self.alphas = []
        self.w = []
        self.b = []
        self.support_vectors = []
        self.support_vectors_labels = []
        self.gamma = gamma

    def construct_K(
        self,
        X,
        Y
    ):
        if self.kernel == 'linear':
            return X @ X.T
        else:
            K = euclidean_distances(X, Y, squared=True)
            K = np.exp(-self.gamma * K)
            return K

    def predict(
        self,
        X_test
    ):
        Y_pred = []
        if self.kernel == 'linear':
            Y_pred = X_test @ self.w + self.b
        else:
            K = self.construct_K(X_test, self.support_vectors)
            print(X_test.shape)
            print(K.shape)
            Y_pred = (np.ones(X_test.shape[0]) * self.b +
                      np.sum(self.alphas * self.support_vectors_labels * K,
                             axis=1))
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

        K = self.construct_K(X, X)

        P = matrix((np.outer(Y, Y) * K))
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

        if self.kernel == 'linear':
            self.w = np.zeros(self.support_vectors.shape[1])
            for i in range(self.support_vectors.shape[0]):
                self.w += (self.alphas[i] * self.support_vectors_labels[i]) \
                    * self.support_vectors[i]

        b = np.zeros(self.support_vectors.shape[0])
        for sv_idx, idx in zip(support_vectors_idx, range(b.shape[0])):
            b[idx] = (self.support_vectors_labels[idx] -
                      np.sum(self.alphas *
                             self.support_vectors_labels *
                             K[sv_idx, support_vectors_idx]))

        self.b = np.sum(b)/b.shape[0]
        print(b)
        print(self.b)
