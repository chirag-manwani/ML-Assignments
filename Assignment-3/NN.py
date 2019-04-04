import numpy as np
import utils


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

    def fit(
        self,
        X_train,
        Y_train,
        batch_size=100,
        lr=0.1
    ):
        self.lr = lr
        for i in range(10):
            self.forward_pass(X_train)
            self.backward_pass(X_train, Y_train)
            self.error(X_train)

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
            input = self.layers[l_num].activations

    def backward_pass(
        self,
        X,
        Y
    ):
        activations = self.layers[-1].activations

        # Last layer activation is always sigmoid
        gradient = activations * (1 - activations)
        diff = (Y.T - activations)

        delta_k = diff * gradient

        bias = np.ones((1, self.layers[-2].activations.shape[1]))
        X = np.vstack((self.layers[-2].activations, bias)).T

        new_weights = self.layers[-1].weights - self.lr * (delta_k @ X)
        for l_num in reversed(range(1, len(self.layers)-1)):
            activations = self.layers[l_num].activations

            gradient = activations * (1 - activations)
            delta_k = (gradient *
                       (self.layers[l_num+1].weights[:, :-1].T @ delta_k))

            bias = np.ones((1, self.layers[l_num-1].activations.shape[1]))
            X = np.vstack((self.layers[l_num-1].activations, bias)).T

            self.layers[l_num+1].weights = new_weights
            new_weights = self.layers[l_num].weights - self.lr * (delta_k @ X)
        self.layers[1].weights = new_weights

    def error(
        self,
        Xtrain
    ):
        outputs = self.layers[-1].activations
        print(outputs.shape)        

class Layer:

    def __init__(
        self,
        num_units,
        nI,
        activation_function='sigmoid'
    ):
        self.weights = np.random.normal(size=(num_units, nI+1))
        self.activations = np.zeros(num_units)
        self.activation_function = activation_function
