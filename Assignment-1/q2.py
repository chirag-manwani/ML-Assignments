#!/usr/bin/env python
# coding: utf-8

# Imports and Initializations
import numpy as np
import sys
import matplotlib.pyplot as plt
from math import fabs


# Function Definitions

# To solve the normal equation
# If W = I, then all weights=1, meaning normal Linear Regression
def solve_normal(X, y, W):
    inv_matrix = np.linalg.inv(X.T @ W @ X)
    theta = inv_matrix @ X.T @ W @ y
    return theta

# Calculates weights of all points wrt to given point and returns a vector
def calc_weights(X, point, tau):
    m = X.shape[0]
    diff_vector = np.zeros(m)
    X = X - point
    for row_num in range(0, m):
        diff = X[row_num]
        diff_vector[row_num] = diff @ diff
    diff_vector = np.exp((-1)*diff_vector / (2*tau*tau))
    return diff_vector

# Loading features(X) and targets(y) from respective files
def load_data(input_file, output_file):
    X = np.loadtxt(input_file)
    y = np.loadtxt(output_file)

    # Appending intercept term in features
    # X.shape = (100, 2)
    X = np.vstack((np.ones(X.shape[0]), X)).T
    return X, y

tau = 0.8
if __name__ == '__main__':
    args = sys.argv
    input_file = ''
    output_file = ''
    if len(args) != 4:
        print("Invalid number of arguments, please pass 3 arguments")
        exit()
    else:
        input_file = str(args[1])
        output_file = str(args[2])
        tau = float(args[3])
    X, y = load_data(input_file, output_file)

    ##################### Part (a) #########################
    theta = solve_normal(X, y, np.identity(y.shape[0]))

    # Plotting points on 2D
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.set_xlabel('X')
    ax.set_ylabel('y')

    # Dummy points for Linear Regression Boundary
    x_line = np.linspace(-5, 13, 4)
    y_line = x_line * theta[1] + theta[0]

    # Plotting the data-points and Decision Boundary
    ax.scatter(X[:, 1], y, label='data')
    ax.plot(x_line, y_line, '-r', label='Lin. Reg. Boundary')

    ax.legend(loc='upper left')
    plt.show()

    ##################### Part (a) #########################
    num_test_points = 20
    x_test = np.linspace(-5, 13, num_test_points)
    x_test = np.vstack((np.ones(num_test_points), x_test)).T
    y_test = np.zeros(num_test_points)

    iter = 0
    for x_t in x_test:
        weights = calc_weights(X, x_t, tau)
        w_mat = np.diagflat(weights)
        theta_t = solve_normal(X, y, w_mat)
        y_test[iter] = x_t @ theta_t
        iter += 1

    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_xlabel('X')
    ax.set_ylabel('y')

    # Plotting data points and Decision Boundary
    ax.scatter(X[:, 1], y, label='data')
    ax.plot(x_test[:, 1], y_test, '-r', label='Decision Boundary')

    ax.legend(loc='upper left')
    plt.show()
