import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys, os, glob
from statistics import mean


def pre_processing(matrix):
    range_ = 10
    b = np.apply_along_axis(lambda x: (x-np.mean(x))/range_, 0, matrix)
    return b

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def cost_function(x, y, theta):
    h_theta = sigmoid( np.dot(x, theta) )
    if h_theta <= 0.0:
        print("ERROR: Cost function cannot be lessa than zero")
        sys.exit()
    log_l = (-y)*np.log(theta) + (1.0-y)*np.log(1.0-h_theta)
    return log_l.mean()

def calculate_gradient(x, y, theta, index, x_count):
    temp_theta = sigmoid( np.dot(x, theta) )
    cumsum = 0.0
    for i in range(temp_theta.shape[0]):
        cumsum += (temp_theta[i] - y[i]) * x[i][index]
    return cumsum

def gradient_descent(train_set, alpha, max_iter, plot_graph=False):
    iter_count = 0
    train_set = np.asarray(train_set)
    x = train_set.T[0:9].T
    y = train_set.T[9].T
    x_count = x.shape[1]

    theta = np.zeros(x_count); x_vals = list(); y_vals = list()
    regularization_parameter = 1

    while(iter_count < max_iter):
        iter_count += 1
        for i in range(x_count):
            prediction = calculate_gradient(x, y, theta, i, x_count)
            prev_theta = theta[i]
            if (i != 0):
                prediction += (regularization_parameter/x_count) * prev_theta
            theta[i] = prev_theta - (alpha*prediction)

            if (plot_graph == True):
                mean = cost_function(x, y, theta)
                x_vals.append(iter_count)
                y_vals.append(mean)

    if (plot_graph == True):
        plt.title('Gradient Desccent Plot')
        plt.gca().invert_yaxis()
        plt.plot(x_vals, y_vals)
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function J(theta)')
        plt.show()
        plt.savefig('gradient_descent.png', bbox_inches='tight')

    return theta

def compute_efficiency(test_set, theta):
    test_set = np.asarray(test_set)
    x = test_set.T[0:9].T
    y = test_set.T[9].T
    x_count = x.shape[0]
    correct = 0

    for i in range(x_count):
        prediction = 0
        value = np.dot(theta, x[i])
        if (value >= 0.5):
            prediction = 1
        else:
            prediction = 0

        if (prediction == y[i]):
            correct += 1

    return (correct/x_count)*100.0

def evaluate_algorithm(dataset, n_folds, alpha, max_ter, plot_graph):
    pass


if __name__ == '__main__':
    pass


