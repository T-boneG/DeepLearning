"""
utils.py - Utility functions
"""

from __future__ import absolute_import, division, print_function
import numpy as np

__all__ = ['stable_log', 'sigmoid', 'softmax', 'one_hot', 'one_hot_inverse']

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

    :param Z: array of predictions of dimension (number of classes, number of samples)
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
