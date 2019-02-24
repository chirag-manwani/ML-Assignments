#!/usr/bin/env python
# coding: utf-8

# Imports and Initializations
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import fabs, exp, log

# Loading features(X) and targets(y) from respective files
def load_data(input_file, output_file):
    X = np.loadtxt(input_file)
    y = np.loadtxt(output_file, dtype=object)

    condition_list = [y == 'Alaska', y == 'Canada']
    choice_list = [0, 1]
    y = np.select(condition_list, choice_list)

    return X, y

##################### Part (a) #########################

if __name__ == '__main__':
    bool_plot_linear = True
    bool_plot_quadratic = True
    args = sys.argv
    input_file = ''
    output_file = ''
    part = 0
    if len(args) != 4:
        print("Invalid number of arguments, please pass 3 arguments")
        exit()
    else:
        input_file = str(args[1])
        output_file = str(args[2])
        part = int(args[3])
        if(part == 0):
            bool_plot_quadratic = False
        elif(part==1):
            bool_plot_linear = False
        else:
            print("Invalid Input, last argument either 0 or 1")
            exit()
    X, y = load_data(input_file, output_file)

    ##################### Part (a) #########################
    m = y.shape[0]

    # phi_m is the number of examples with label=1
    phi_m = y @ y
    phi = phi_m / m

    y_ = y[:, np.newaxis]

    mu_0 = (1 - y) @ X
    mu_0 = mu_0 / (m - phi_m)

    mu_1 = y @ X
    mu_1 = mu_1 / phi_m

    x_0 = np.multiply((X-mu_0), (1 - y_))
    x_1 = np.multiply((X-mu_1), y_)

    sigma = (1.0/m)*(x_0.T @ x_0 + x_1.T @ x_1)

    ##################### Part (b) #########################
    if(bool_plot_linear):
        fig = plt.figure(figsize=(7, 7))

        class0 = patches.Patch(color='purple', label='Alaska')
        class1 = patches.Patch(color='yellow', label='Canada')

        ax = plt.gca()

        ax.set_xlim(40, 200)
        ax.set_ylim(250, 550)


        # Plotting data points
        ax.scatter(X[:, 0], X[:, 1], c=y)

        plt.legend(loc='upper left', handles=[class0, class1])
        plt.show(block=bool_plot_linear)

    ##################### Part (c) #########################
    sigma_inv = np.linalg.inv(sigma)

    # Calculating Theta values
    theta_x = sigma_inv @ (mu_1 - mu_0)
    theta_0 = (1/2)*(mu_0 @ sigma_inv @ mu_0 - mu_1 @ sigma_inv @ mu_1) - log((1-phi)/phi)

    # Dummy points on line, for plotting
    x_line = np.linspace(70, 165, 3)
    y_line = (-1) * (x_line * theta_x[0] + theta_0)/(theta_x[1])

    # Plotting Linear Decision Boundary
    if(bool_plot_linear):
        fig = plt.figure(figsize=(7, 7))
        class0 = patches.Patch(color='purple', label='Alaska')
        class1 = patches.Patch(color='yellow', label='Canada')

        first_legend = plt.legend(handles=[class0, class1], loc='upper left')
        ax = plt.gca().add_artist(first_legend)

        ax = plt.gca()
        ax.set_xlim(40, 200)
        ax.set_ylim(250, 550)

        # Plotting data points and Decision Boundary
        ax.scatter(X[:, 0], X[:, 1], c=y)
        ax.plot(x_line, y_line, '-r', label='Decision Boundary')

        ax.legend(loc='upper right')
        plt.show(block=bool_plot_linear)

    ##################### Part (d) #########################

    sigma_0 = (1.0/(m-phi_m))*(x_0.T @ x_0)
    sigma_1 = (1.0/phi_m)*(x_1.T @ x_1)
    
    ##################### Part (e) #########################

    sigma_0_inv = np.linalg.inv(sigma_0)
    sigma_1_inv = np.linalg.inv(sigma_1)

    sigma_0_det = np.linalg.det(sigma_0)
    sigma_1_det = np.linalg.det(sigma_1)

    # Calculating coefficients for square terms, linear terms and constants
    coeff_x_sq = (1/2) * (sigma_0_inv - sigma_1_inv)
    coeff_x_ = sigma_1_inv @ mu_1 - sigma_0_inv @ mu_0
    coeff_0_ = (1/2)*(mu_0 @ sigma_0_inv @ mu_0 - mu_1 @ sigma_1_inv @ mu_1) - log((1-phi)/phi) + (1/2) *(log(sigma_0_det) - log(sigma_1_det))

    if(bool_plot_quadratic):
        # Dummy points on line, for plotting
        x_1 = np.linspace(60, 160, 20)
        x_2 = np.linspace(250, 515, 20)
        x_1, x_2 = np.meshgrid(x_1, x_2)

        expression = coeff_x_sq[0,0]*x_1*x_1 + coeff_x_sq[1,1]*x_2*x_2 + (coeff_x_sq[0,1] + coeff_x_sq[1,0])*x_1*x_2 + coeff_x_[0]*x_1 + coeff_x_[1]*x_2 + coeff_0_

        fig = plt.figure(figsize=(7, 7))
        class0 = patches.Patch(color='purple', label='Alaska')
        class1 = patches.Patch(color='yellow', label='Canada')

        first_legend = plt.legend(handles=[class0, class1], loc='upper left')
        plt.gca().add_artist(first_legend)

        ax = plt.gca()
        ax.set_xlim(40, 200)
        ax.set_ylim(250, 600)

        ax.set_xlabel('Growth Ring diameter in Fresh Water')
        ax.set_ylabel('Growth Ring diameter in Marine Water')

        ax.scatter(X[:, 0], X[:, 1], c=y)
        ax.plot(x_line, y_line, '-b', label='Linear Decision Boundary')
        ax.contour(x_1, x_2, expression, [0], colors='r')
        ax.plot([0],[0],'r', label='Quadratic Decision Boundary')

        ax.legend(loc ='upper right', )
        plt.show(block=bool_plot_quadratic)
        coeff_x_

        # Plotting the Hyperbolic Boundary

        sigma_0_inv = np.linalg.inv(sigma_0)
        sigma_1_inv = np.linalg.inv(sigma_1)

        sigma_0_det = np.linalg.det(sigma_0)
        sigma_1_det = np.linalg.det(sigma_1)

        # Calculating coefficients for square terms, linear terms and constants
        coeff_x_sq = (1/2) * (sigma_0_inv - sigma_1_inv)
        coeff_x_ = sigma_1_inv @ mu_1 - sigma_0_inv @ mu_0
        coeff_0_ = (1/2)*(mu_0 @ sigma_0_inv @ mu_0 - mu_1 @ sigma_1_inv @ mu_1) - log((1-phi)/phi) + (1/2) *(log(sigma_0_det) - log(sigma_1_det))

        # Dummy points on line, for plotting
        x_1 = np.linspace(50, 200, 200)
        x_2 = np.linspace(-100, 700, 200)
        x_1, x_2 = np.meshgrid(x_1, x_2)

        expression = coeff_x_sq[0][0]*x_1*x_1 + coeff_x_sq[1][1]*x_2*x_2 + (coeff_x_sq[0][1] + coeff_x_sq[1][0])*x_1*x_2 + coeff_x_[0]*x_1 + coeff_x_[1]*x_2 + coeff_0_

        fig = plt.figure(figsize=(7, 7))
        class0 = patches.Patch(color='purple', label='Alaska')
        class1 = patches.Patch(color='yellow', label='Canada')

        first_legend = plt.legend(handles=[class0, class1], loc='upper left')
        plt.gca().add_artist(first_legend)

        ax = plt.gca()
        ax.set_xlim(50, 200)
        ax.set_ylim(-100, 700)

        ax.set_xlabel('Growth Ring diameter in Fresh Water')
        ax.set_ylabel('Growth Ring diameter in Marine Water')

        ax.scatter(X[:, 0], X[:, 1], c=y)
        # ax.plot(x_line, y_line, '-b', label='Linear Decision Boundary')
        ax.contour(x_1, x_2, expression, [0], colors='r')
        ax.plot([0],[0],'r', label='Hyperbolic Decision Boundary')

        ax.legend(loc ='upper right', )
        plt.show(block=bool_plot_quadratic)
