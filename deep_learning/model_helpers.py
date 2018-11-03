"""

"""

from __future__ import division
import numpy as np

__all__ = ['check_inputs_no_constraints', 'check_inputs_binary_classification',
           'check_inputs_multiclass_classification', 'binary_classification_prediction',
           'multiclass_one_hot_prediction', 'multiclass_prediction', 'linear_regression_prediction']

"""Check Inputs"""

def check_inputs_no_constraints(X, Y):
    """
    No specific constraints on the input and output of the system
        -Use for linear regression problems
    """
    assert X.shape[1] == Y.shape[1], 'inconsistent number of samples for X and Y, ' \
                                     '%d != %d' % (X.shape[1], Y.shape[1])

def check_inputs_binary_classification(X, Y):
    check_inputs_no_constraints(X, Y)

    assert Y.shape[0] == 1, 'invalid output vector dimension: %d. Expected: %d' % (Y.shape[0], 1)

def check_inputs_multiclass_classification(X, Y):
    check_inputs_no_constraints(X, Y)

    # verify Y is a set of one-hot vectors
    assert np.sum(Y, axis=0) == np.ones((1, Y.shape[1]))
    assert np.sum(Y != 0, axis=0) == np.ones((1, Y.shape[1]))

"""Prediction"""

def binary_classification_prediction(A):
    return np.array(A > 0.5, dtype=int)

def multiclass_one_hot_prediction(A):
    """
    Convert probabilities to one-hot prediction matrix
    :param A: matrix of softmax probabilities (number of classes, number of samples)
    :return: one-hot matrix of the same dimension
    """
    column_max = np.max(A, axis=0, keepdims=True)
    return np.array(A == column_max, dtype=int)

def multiclass_prediction(A):
    """
    Convert probabilities row vector of class predictions
    :param A: matrix of softmax probabilities (number of classes, number of samples)
    :return: row vector of integer class predictions
    """
    return np.argmax(A, axis=0)

def linear_regression_prediction(A):
    return A
