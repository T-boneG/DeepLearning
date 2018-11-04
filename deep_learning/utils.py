"""

"""

from __future__ import division
import numpy as np

__all__ = ['stable_log', 'sigmoid', 'softmax', 'one_hot', 'one_hot_inverse']

def stable_log(x):
    # add epsilon for numeric stability
    eps = 1e-10
    return np.log(x + eps)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z, axis=0):
    # shift Z for numeric stability
    shiftZ = Z - np.max(Z, axis=axis, keepdims=True)

    exps = np.exp(shiftZ)

    return exps / np.sum(exps, axis=axis, keepdims=True)

def one_hot(Y, depth):
    one_hot_matrix = np.zeros((depth, len(Y)))
    one_hot_matrix[Y, np.arange(len(Y))] = 1

    return one_hot_matrix

def one_hot_inverse(one_hot_matrix):
    return np.argmax(one_hot_matrix, axis=0)