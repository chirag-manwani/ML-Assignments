import numpy as np
import utils
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class NN:

    def __init__(
        self,
        nI,
        nO,
        arch_list,
        activation_function='sigmoid'
    ):
        self.nI = nI
        self.nO = nO
        self.batch_size = None

        # A separate layer for inputs
        layer = Layer(0, nI)
        self.layers = [layer]

        arch_list.append(nO)
        i_size = nI
        for o_size in arch_list:
            n_layer = Layer(o_size, i_size, activation_function)
            self.layers.append(n_layer)
            i_size = o_size
        self.layers[-1].activation_function = 'sigmoid'

    def predict(
        self,
        X
    ):
        self.forward_pass(X)
        Y_pred = self.layers[-1].activations
        Y_pred = np.argmax(Y_pred, axis=0)
        return Y_pred

    def fit(
        self,
        X_train,
        Y_train,
        batch_size=100,
        lr=0.1,
        epochs=1000,
        adaptive='fixed',
        tol=1e-5,
        prints=False
    ):
        self.lr = lr
        is_adaptive = False
        if adaptive == 'variable':
            is_adaptive = True

        total_samples = X_train.shape[0]
        loss_hist = []

        for i in range(epochs):
            idx = 0
            while idx+batch_size < total_samples:
                batch_x = X_train[idx:idx+batch_size]
                batch_y = Y_train[idx:idx+batch_size]
                self.forward_pass(batch_x)
                self.backward_pass(batch_y)
                idx += batch_size

            accuracy = self.error(X_train, Y_train)

            Y_pred = self.layers[-1].activations.T
            loss = self.loss(Y_train, Y_pred)

            loss_hist.append(loss)
            if prints:
                print('Epoch:', i)
                print('Accuracy Score:', accuracy,
                      'Loss:', loss)
            if (is_adaptive and
                    i > 150 and
                    math.fabs(loss_hist[-2] - loss_hist[-1]) < tol and
                    math.fabs(loss_hist[-2] - loss_hist[-1]) < tol):
                print('Learning Rate adjusted\n\n\n')
                self.lr /= 5

    def forward_pass(
        self,
        X
    ):
        self.layers[0].activations = X.T
        for l_num in range(1, len(self.layers)):
            input = self.layers[l_num-1].activations

            bias = np.ones((1, input.shape[1]))
            input = np.vstack((bias, input))

            net = self.layers[l_num].weights @ input
            if self.layers[l_num].activation_function == 'sigmoid':
                self.layers[l_num].activations = utils.sigmoid(net)
            else:
                self.layers[l_num].activations = utils.relu(net)

    def backward_pass(
        self,
        Y
    ):
        batch_size = Y.shape[0]
        activations = self.layers[-1].activations

        # Last layer activation is always sigmoid
        gradient = activations * (1 - activations)
        diff = (Y.T - activations)

        delta_k = diff * gradient

        bias = np.ones((1, self.layers[-2].activations.shape[1]))
        X = np.vstack((bias, self.layers[-2].activations)).T

        new_weights = self.layers[-1].weights + self.lr * \
            (delta_k @ X) / batch_size
        for l_num in reversed(range(1, len(self.layers)-1)):
            activations = self.layers[l_num].activations

            gradient = activations * (1 - activations)
            self.layers[l_num+1].weights = new_weights
            delta_k = (gradient *
                       (self.layers[l_num+1].weights[:, 1:].T @ delta_k))

            bias = np.ones((1, self.layers[l_num-1].activations.shape[1]))
            X = np.vstack((bias, self.layers[l_num-1].activations)).T

            # self.layers[l_num+1].weights = new_weights
            new_weights = self.layers[l_num].weights + self.lr * \
                (delta_k @ X) / batch_size
        self.layers[1].weights = new_weights

    def error(
        self,
        X,
        Y_true
    ):
        Y_true_labels = np.argmax(Y_true, axis=1)
        Y_pred_labels = self.predict(X)

        # print(confusion_matrix(Y_true, Y_pred))
        accuracy = accuracy_score(Y_true_labels, Y_pred_labels)
        return accuracy

    def loss(
        self,
        Y_true,
        Y_pred
    ):
        norm = np.linalg.norm(Y_true - Y_pred, ord='fro')
        norm = norm ** 2
        return norm / (2 * Y_true.shape[0])


class Layer:

    def __init__(
        self,
        num_units,
        nI,
        activation_function='sigmoid'
    ):
        # self.weights = np.random.normal(size=(num_units, nI+1), scale=0.5)
        self.weights = np.random.uniform(low=-0.5, high=0.5,
                                         size=(num_units, nI+1))
        self.activations = np.zeros(num_units)
        self.activation_function = activation_function
