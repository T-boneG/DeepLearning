"""
Logistic Regression and Linear Regression
"""

from __future__ import division
import numpy as np
from model_helpers import *
import cost_functions

"""Binary Classification Models"""

logistic_regression_model = {
    'check_inputs': check_inputs_binary_classification,
    'prediction': binary_classification_prediction,
    'final_activation_and_cost': cost_functions.SigmoidCrossEntropy()
}

perceptron_model = {
    'check_inputs': check_inputs_binary_classification,
    'prediction': binary_classification_prediction,
    'final_activation_and_cost': cost_functions.SigmoidPerceptron()
}

"""Multi-class Classification Models"""

softmax_regression_model = {
    'check_inputs': check_inputs_multiclass_classification,
    'prediction': multiclass_prediction,
    'final_activation_and_cost': cost_functions.SoftmaxCrossEntropy()
}

"""Linear Regression Models"""

mse_linear_regression_model = {
    'check_inputs': check_inputs_no_constraints,
    'prediction': linear_regression_prediction,
    'final_activation_and_cost': cost_functions.LinearMinimumSquareError()
}

class LinearDiscriminantFunction(object):
    """description..."""

    """Public Methods"""

    def __init__(self, n_x, n_y, model):
        self.n_x = n_x
        self.n_y = n_y

        self.check_inputs = model['check_inputs']
        self.prediction = model['prediction']
        self.faac = model['final_activation_and_cost']

        self._initialize_params()

    def get_params(self):
        return self.params

    # TODO implement batch processing
    def fit(self, X, Y, num_iterations, learning_rate, print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        X -- data of shape (n_x, number of examples)
        Y -- labels of shape (n_y, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule

        Returns:
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """
        assert X.shape[0] == self.n_x, 'invalid input vector dimension: %d. Expected: %d' % (X.shape[0], self.n_x)
        assert Y.shape[0] == self.n_y, 'invalid output vector dimension: %d. Expected: %d' % (Y.shape[0], self.n_y)
        self.check_inputs(X, Y)

        costs = []

        for i in range(num_iterations):
            # Cost and gradient calculation
            grads, cost = self._propagate(X, Y)

            # update rule
            self.params['W'] = self.params['W'] - learning_rate * grads['dW']
            self.params['b'] = self.params['b'] - learning_rate * grads['db']

            costs.append(cost)

            if print_cost and (i % 1000 == 0):
                print('%5d: %7.4f' % (i, cost))

        return costs

    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        X -- data of size (dim, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        A -- a numpy array (vector) containing probabilities corresponding to predictions
        '''
        assert X.shape[0] == self.n_x, 'invalid input vector dimension: %d. Expected: %d' % (X.shape[0], self.n_x)

        Z = self.forward_propagate(X, self.params['W'], self.params['b'])
        A = self.faac.final_activation(Z)

        Y_prediction = self.prediction(A)

        return Y_prediction, A

    #TODO address this in a more general way
    def score(self, X, Y):
        """
        prediction percent correct
        :param X:
        :param Y:
        :return:
        """
        assert X.shape[0] == self.n_x, 'invalid input vector dimension: %d. Expected: %d' % (X.shape[0], self.n_x)
        assert Y.shape[0] == self.n_y, 'invalid output vector dimension: %d. Expected: %d' % (Y.shape[0], self.n_y)
        self.check_inputs(X, Y)

        Y_prediction, _ = self.predict(X)

        correct = np.sum(Y_prediction == Y)

        return correct / Y.shape[1]

    """Private Methods"""

    def _initialize_params(self):
        W = np.zeros((self.n_y, self.n_x))
        b = np.zeros((self.n_y, 1))

        self.params = {
            'W': W,
            'b': b
        }

    def _propagate(self, X, Y):
        Z = self.forward_propagate(X, self.params['W'], self.params['b'])

        cost, A, dZ = self.faac.final_activation_and_cost(Y, Z)

        dW, db = self.linear_backward(X, dZ)

        grads = {'dW': dW,
                 'db': db}

        return grads, cost

    """Static Helper Methods"""

    @staticmethod
    def forward_propagate(X, W, b):
        """
        :param X: vector of shape (n_x, number of samples)
        :param W: vector of shape (n_y, n_x)
        :param b: vector of shape (n_y, 1)
        :return: linear function output
        """
        assert W.shape[1] == X.shape[0]
        assert W.shape[0] == b.shape[0]
        assert b.shape[1] == 1

        Z = np.dot(W, X) + b

        return Z

    @staticmethod
    def linear_backward(X, dZ):
        m = X.shape[1]

        dW = 1 / m * np.dot(dZ, X.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        return dW, db
