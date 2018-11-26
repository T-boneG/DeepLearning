#!/usr/bin/env python
"""
utils.py - Utility functions
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

__all__ = ['stable_log', 'sigmoid', 'softmax', 'one_hot', 'one_hot_inverse',
           'random_mini_batches', 'explore_data']

def stable_log(x):
    """
    compute log of a function adding epsilon for numeric stability when x -> 0
    """
    eps = 1e-10
    return np.log(x + eps)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(Z, axis=0):
    """

    :param Z: array of predictions of dimension
              (number of classes, number of samples)
    :param axis: optional, should be 0 if Z is of the aforementioned structure
    :return: array of same dimension as Z
    """

    # shift Z for numeric stability
    shiftZ = Z - np.max(Z, axis=axis, keepdims=True)

    exps = np.exp(shiftZ)

    return exps / np.sum(exps, axis=axis, keepdims=True)

def one_hot(Y, depth):
    """
    Convert a vector of integer class labels to a matrix of one-hot vectors
    :param Y: vector of length (number of samples)
    :param depth:
    :return:
    """
    num_samples = np.array(Y).size

    one_hot_matrix = np.zeros((depth, num_samples))
    one_hot_matrix[np.squeeze(Y), np.arange(num_samples)] = 1

    return one_hot_matrix

def one_hot_inverse(one_hot_matrix):
    """
    Inverse a one-hot matrix to a vector of integer class labels
    """
    return np.argmax(one_hot_matrix, axis=0)

def random_mini_batches(X, Y, batch_size):
    """
    Split the training data into random minibatches
    :param X: input data of shape (input dim, number of samples)
    :param Y: output data of shape (output dim, number of samples
    :param batch_size:
    :return:
        minibatches: a list of minibatch tuples where each tuple
                     is (minibatch_X, minibatch_Y)
        num_minibatches: the number of total minibatches
    """
    m = X.shape[1]

    num_minibatches = int(np.ceil(m / batch_size))

    # shuffle
    random_permutation = np.random.permutation(m)
    X_shuffled, Y_shuffled = (A[:, random_permutation] for A in (X, Y))

    minibatches = [
        (X_shuffled[:, i * batch_size:(i + 1) * batch_size],
         Y_shuffled[:, i * batch_size:(i + 1) * batch_size])
        for i in range(num_minibatches)]

    return minibatches, num_minibatches

def explore_data(*args):
    """
    print out information and display a plot of the data distributions
    shows the input dimension and the number of data samples per class label

    :param args: arbitrary number of (X, Y) tuples for train, validation, test, etc.
    """
    def explore_single_dataset(X, Y, num_datasets, set_number):
        # acquire information
        m = X.shape[0]

        class_labels = np.unique(Y)
        class_labels.sort()
        class_counts = np.zeros(class_labels.shape)

        for i, class_label in enumerate(class_labels):
            class_counts[i] = np.sum(np.equal(Y, class_label))

        # print information
        print('dataset size: %d' % m)
        for i, class_label in enumerate(class_labels):
            percent_of_set = 100 * class_counts[i] / m
            print('label: %3d - %5.2f%%  %5d/%d'
                  % (class_label, percent_of_set, class_counts[i], m))

        # plot information
        ax = plt.subplot(num_datasets, 1, set_number)
        ax.bar(class_labels, class_counts)
        ax.set_ylabel('count')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2 = ax.twinx()
        ax2.bar(class_labels, 100 * class_counts / m)
        ax2.set_ylabel('percent')
        # ax2.grid(axis='y')

    for i, (X, Y) in enumerate(args):
        explore_single_dataset(X, Y, len(args), i+1)

        if i == 0:
            input_shape = X.shape[1::]
            input_dimension = np.product(input_shape)
            s = 'input shape: %s = %d' % (str(input_shape), input_dimension)
            print(s)
            plt.title(s)

    plt.show()
