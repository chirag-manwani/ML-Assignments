#!/usr/bin/env python
# coding: utf-8

# Imports and Initializations
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import fabs, exp

def standardize_features(X):
    num_cols = X.shape[1]
    means = np.zeros(num_cols)
    std_devs = np.ones(num_cols)
    for column in range(1, num_cols):
        means[column] = np.mean(X[:, column])
        std_devs[column] = np.std(X[:, column])
        X[:, column] -= means[column]
        X[:, column] /= std_devs[column]
    return X, means, std_devs

def sigmoid(x):
    m = x.shape[0]
    sigmoid_vector = np.zeros(m)
    for i in range(0, m):
        sigmoid_vector[i] = 1 / (1 + exp(-x[i]))
    return sigmoid_vector

def newton_root(X, y):
    theta_old = np.array([0, 0, 0])
    theta = np.array([100, 100, 100])
    converged = False
    iters = 0
    while(not converged):
        predictions = X @ theta_old
        predictions = sigmoid(predictions)

        diff = y - predictions
        first_derivative = diff @ X

        diagnol = np.diagflat(predictions * (1 - predictions))
        hessian = (-1) * (X.T @ diagnol @ X)

        theta = theta_old - np.linalg.inv(hessian) @ first_derivative

        if(np.linalg.norm(theta_old - theta) < epsilon):
            converged = True
        theta_old = theta
        iters += 1
    return theta

def load_data(input_file, output_file):
    # Loading features(X) and targets(y) from respective files
    X = np.loadtxt(input_file, delimiter=',')
    y = np.loadtxt(output_file)

    # Appending intercept term in features
    # X.shape = (100, 3)
    intercept = np.asarray([np.ones(X.shape[0])])
    X = np.hstack((intercept.T, X))
    X, means, std_devs = standardize_features(X)
    return X, y

if __name__ == '__main__':
    args = sys.argv
    input_file = ''
    output_file = ''
    if len(args) != 3:
        print("Invalid number of arguments, please pass 2 arguments")
        exit()
    else:
        input_file = str(args[1])
        output_file = str(args[2])
    X, y = load_data(input_file, output_file)

    ##################### Part (a) #########################
    epsilon = 1e-10
    theta = newton_root(X, y)

    ##################### Part (b) #########################
    predictions = X @ theta
    predictions = sigmoid(predictions)

    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0

    plt.figure(figsize=(6,6))
    plt.ion()
    class0 = patches.Patch(color='purple', label='Class 0')
    class1 = patches.Patch(color='yellow', label='Class 1')

    ax = plt.gca()

    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)

    ax.set_xlabel('X')
    ax.set_ylabel('y')

    x_line = np.linspace(-2, 2, 4)
    y_line = (-1) * (x_line * theta[1] + theta[0])/(theta[2])

    # Plotting data points and Decision Boundary
    ax.scatter(X[:,1], X[:,2], c=y, s=15)
    ax.plot(x_line, y_line, '-r')

    ax.legend(loc='upper left', handles=[class0, class1])
    plt.show(block=True)
