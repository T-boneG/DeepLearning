"""

"""
from __future__ import division
import numpy as np
from activation_functions import linear_af, sigmoid_af

#TODO implement regularization

#TODO implement this
def relaxation(X, Y, Z):
    raise NotImplementedError

#TODO implement this
def relaxation_with_margin(X, Y, Z):
    raise NotImplementedError

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

"""Activation Functions"""

#TODO move this to activation functions?
def softmax(Z):
    T = np.exp(Z)
    A = T / np.sum(T)
    return A

"""Compute Cost"""

def logistic_regression_cost(Y, A):
    m = Y.shape[1]

    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dA = np.divide(1 - Y, 1 - A) - np.divide(Y, A)

    return cost, dA

def perceptron_cost(Y, A):
    m = Y.shape[1]

    Y_prediction = binary_classification_prediction(A)
    misclassified_indicator = np.logical_xor(Y, Y_prediction)

    #TODO this computes the sigmoid inverse, is there a better way to get Z?
    Z = np.log(A) - np.log(1 - A)

    cost = -1 / m * np.dot(np.abs(Z), misclassified_indicator.T)

    return cost
def perceptron_dZ(Y, A):
    Y_prediction = binary_classification_prediction(A)
    misclassified_indicator = np.logical_xor(Y, Y_prediction)

    # misclassified Y's shifted so that 0 -> -1
    dZ = (2 * Y - 1) * misclassified_indicator

    return dZ

def softmax_regression_cost(Y, A):
    """
    Multi-class Classification
    Y.shape = (number of classes, number of samples)
    each column of Y is a one-hot vector
    """
    m = Y.shape[1]

    cost = -1 / m * np.sum(Y * np.log(A))

    return cost
def TODO_logistic_regression_dZ(Y, A):
    dZ = A - Y

    return dZ

def mean_square_error_cost(Y, A):
    """
    Linear Regression
    anything valid for output dimension
    anything valid for output value
    """
    m = Y.shape[1]

    cost = -1 / m * np.sum((Y - A) ** 2)

    return cost
def minimum_square_error_dZ(Y, A):
    dZ = 2 * (A - Y)

    return dZ


#TODO lamda function? how to generate a function that only takes (Y, A, params) as inputs
def compute_cost_with_regularization(Y, A, params, cost_function, lambd):
    cost = cost_function(A, Y)

    m = Y.shape[1]

    # select W's from params
    all_Ws = [val for (key, val) in params.items() if key.startswith('W')]

    L2_regularization_cost = (lambd / (2 * m)) * np.sum([np.sum(np.square(W)) for W in all_Ws])

    cost = cost + L2_regularization_cost

    return cost

#TODO replace with this??
#TODO YES!!! dA/dZ is the derivative of the cost, it is tied to the cost function and as such should need to be specified
#TODO is it dA or dZ?? What's going on here?
"""Backward Propagation"""

def logistic_regression_backward_propagation(X, Y, A):
    assert Y.shape[1] == X.shape[1]
    assert A.shape == Y.shape

    m = X.shape[1]

    dW = 1 / m * np.dot(A - Y, X.T)
    db = 1 / m * np.sum(A - Y, axis=1, keepdims=True)

    return dW, db

def perceptron_backward_propagation(X, Y, A):
    m = Y.shape[1]

    Y_prediction = binary_classification_prediction(A)
    misclassified_indicator = np.logical_xor(Y, Y_prediction)
    # misclassified Y's shifted so that 0 -> -1
    Y_shifted_indicator = (2 * Y - 1) * misclassified_indicator

    dW = 1 / m * np.dot(Y_shifted_indicator, X.T)
    db = 1 / m * np.sum(Y_shifted_indicator, axis=1, keepdims=True)

    return dW, db

def minimum_square_error_backward_propagation(X, Y, A):
    """
    Linear Regression
    """
    m = X.shape[1]

    dA = 2 * (A - Y)
    dW = 1 / m * np.dot(dA, X.T)
    db = 1 / m * np.sum(dA, axis=1, keepdims=True)

    return dW, db

"""Prediction"""

def binary_classification_prediction(A):
    return np.array(A > 0.5, dtype=int)

def softmax_prediction(A):
    #TODO could rewrite this as an argmax and not return in one-hot format
    column_max = np.max(A, axis=0, keepdims=True)
    return np.array(A == column_max, dtype=int)

def linear_regression_prediction(A):
    return A

"""Binary Classification Models"""

logistic_regression_model = {
    'final_activation': sigmoid_af,
    'compute_cost': logistic_regression_cost,
    'backward_propagate': logistic_regression_backward_propagation,
    'check_inputs': check_inputs_binary_classification,
    'prediction': binary_classification_prediction
}

perceptron_model = {
    'final_activation': sigmoid_af,
    'compute_cost': perceptron_cost,
    'backward_propagate': perceptron_backward_propagation,
    'check_inputs': check_inputs_binary_classification,
    'prediction': binary_classification_prediction
}

"""Multi-class Classification Models"""

softmax_regression_model = {
    'final_activation': softmax,
    'compute_cost': softmax_regression_cost,
    'backward_propagate': logistic_regression_backward_propagation,
    'check_inputs': check_inputs_multiclass_classification,
    'prediction': softmax_prediction
}

"""Linear Regression Models"""

mse_linear_regression_model = {
    'final_activation': linear_af,
    'compute_cost': mean_square_error_cost,
    'backward_propagate': minimum_square_error_backward_propagation,
    'check_inputs': check_inputs_no_constraints,
    'prediction': linear_regression_prediction
}

