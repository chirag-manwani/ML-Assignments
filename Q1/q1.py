#!/usr/bin/env python
# coding: utf-8

# Imports and Initializations
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation

from matplotlib import cm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from math import fabs

# Function declarations

# Standardizing to make the feature values have mean 0 and std 1
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

# Compute Cost calculates the MSE
# Returns J and difference(between predictions and targets
def compute_cost(X, y, theta):
    # Number of examples- m
    m = y.shape[0]

    # Calculating the predictions, h_theta(X) for each example
    predictions = X @ theta

    # diff stores difference in predictions and targets
    diff = predictions - y

    # Cost J using formula
    J = (1.0/(2*m))*(diff @ diff)
    return J, diff

# Gradient Descent Implementation
# Input- Feature Matrix (X), target data(y), Initial Theta(theta), Learning rate(eta)
# Output- Optimum Cost, Cost History, Theta0 history, Theta1 history
def gradient_descent(X, y, theta, eta):
    # Number of examples = m
    m = y.shape[0]

    converged = False
    iters = 0
    J_old = sys.maxsize

    # For storing data in each iteration
    J_hist = np.zeros(max_iters)
    theta_0_hist = np.zeros(max_iters)
    theta_1_hist = np.zeros(max_iters)

    while(not converged):
        iters += 1

        # Computing the cost and difference- Predicted - Actual
        J, diff = compute_cost(X, y, theta)
        J_hist[iters]= J
        theta_0_hist[iters] = theta[0]
        theta_1_hist[iters] = theta[1]

        # Gradient Computation
        gradient = (diff @ X)/m
        theta = theta - eta*gradient

        # Convergence Criteria, |J_old - J_new| < epsilon,
        # Default stop after max number of iterations
        # Convergence Check
        if((fabs(J_old-J) < epsilon)):
            converged = True

        J_old = J

    return theta, J_hist[1:iters+1], theta_0_hist[1:iters+1], theta_1_hist[1:iters+1]

# For computing the cost of the meshgrid
def J(theta_0, theta_1, X, y):
    rows, cols = theta_0.shape
    J_mat = np.zeros((rows, cols))
    for row in range(0, rows):
        for col in range(0, cols):
            theta = np.array([theta_0[row, col], theta_1[row, col]])
            J_mat[row, col], _ = compute_cost(X, y, theta)
    return J_mat

# FuncAnimation update function for Mesh
def update_mesh(num, data, lines):
    lines.set_data(data[:2, :num])
    lines.set_3d_properties(data[2, :num])
    lines.set_marker("o")
    return lines

# FuncAnimation update function for Contour
def update_contour(num, data, lines):
    lines.set_data(data[:2, :num])
    lines.set_marker("o")
    return lines

def load_data(input_file, output_file):
    # Loading features(X) and targets(y) from respective files
    X = np.loadtxt(input_file)
    y = np.loadtxt(output_file)

    # Appending intercept term in features
    # X.shape = (100, 2)
    X = np.vstack((np.ones(X.shape[0]), X)).T

    # Standardizing data, making mean = 0, and std = 1
    # Storing means and std_devs for later use
    X, means, std_devs = standardize_features(X)
    return X, y, means, std_devs

if __name__ == '__main__':
    args = sys.argv
    eta = 0.2
    epsilon = 1e-9
    max_iters = 1000
    input_file = ''
    output_file = ''
    time_gap = 2

    if len(args) != 5:
        print("Invalid number of arguments, please pass 4 arguments")
        exit()
    else:
        input_file = str(args[1])
        output_file = str(args[2])
        eta = float(args[3])
        time_gap = float(args[4])

    ##################### Part (a) #########################

    X, y, means, std_devs = load_data(input_file, output_file)

    # Initializing theta
    # theta.size = (X.shape[1], )
    theta = np.ones(X.shape[1]) * 3
    theta, J_hist, theta_0_hist, theta_1_hist = gradient_descent(X, y, theta, eta)

    ##################### Part (b) #########################
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()

    # Setting the x and y axes ranges
    plt.xlim(-3, 6)
    plt.ylim(0.9875, 1.008)

    ax.set_xlabel('X')
    ax.set_ylabel('y')

    x_line = np.linspace(-2.5, 5.5, 4)
    y_line = x_line * theta[1] + theta[0]

    # Plotting the data points and Decision Boundary
    ax.scatter(X[:, 1], y, label='data', s=10)
    ax.plot(x_line, y_line, '-r', label='Decision Boundary')

    ax.legend(loc='upper left')
    plt.show(block=True)

    ##################### Part (c) #########################
    # Creating data for meshgrid
    k = 4
    theta_0 = np.linspace(theta[0]-k, theta[0]+k, 301)
    theta_1 = np.linspace(theta[1]-k, theta[1]+k, 301)

    # Meshgrid data
    Theta_0, Theta_1 = np.meshgrid(theta_0, theta_1)

    # Cost data for each point on meshgrid
    J_mat = J(Theta_0, Theta_1, X, y)

    # Plotting
    fig = plt.figure(figsize=(7,7))

    # Setting Axes
    ax = Axes3D(fig)

    ax.set_xlabel('Theta_0')
    ax.set_ylabel('Theta_1')
    ax.set_zlabel('J(theta)')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-1, 15)

    ax.plot_wireframe(Theta_0, Theta_1, J_mat, alpha=0.2, label='Cost Function')

    data = np.vstack((theta_0_hist, theta_1_hist, J_hist))
    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], 'ro', linestyle='dashed', markersize=3, linewidth=1, label='Gradient Descent Movement')
    ani = animation.FuncAnimation(fig, update_mesh, frames=J_hist.shape[0], fargs=(data, line), interval=time_gap*1000)

    ax.legend(loc='upper left')
    plt.show(block=True)

    ##################### Part (d) #########################

    fig = plt.figure(figsize=(7,7))

    # Setting Axes
    ax = plt.gca()

    ax.set_xlabel('Theta_0')
    ax.set_ylabel('Theta_1')

    ax.set_xlim(-2, 4)
    ax.set_ylim(-3, 4)

    levels = np.geomspace(0.01, 10, num = 7)
    contour_levels = J_hist[J_hist > 0.01]
    if(contour_levels.shape[0] < 10):
        contour_levels = np.concatenate([J_hist, levels])
    cset = ax.contour(Theta_0, Theta_1, J_mat, levels=np.sort(contour_levels))

    line, = ax.plot(data[0, 0:1], data[1, 0:1], 'ro', linestyle='dashed', markersize=3, linewidth=1, label='Gradient Descent Movement')
    ani = animation.FuncAnimation(fig, update_contour, frames=J_hist.shape[0], fargs=(data, line), interval=time_gap*1000, repeat=True)

    ax.legend(loc='upper left')
    plt.show(block=True)
