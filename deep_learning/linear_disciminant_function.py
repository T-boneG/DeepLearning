"""
linear_discriminant_function.py
Linear Discriminant Function machine learning algorithm

This class is actually superfluous because the same thing could be made using
the NeuralNetwork class with no hidden units, however it is still here for reference
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from . import model_helpers, cost_functions

__all__ = ['LinearDiscriminantFunction']

"""Example Models"""

# Binary Classification
logistic_regression_LDF_model = {
    'model_type': 'binary_classification',
    'final_activation_and_cost': cost_functions.SigmoidCrossEntropy()
}

# Multi-class Classification
softmax_regression_LDF_model = {
    'model_type': 'multiclass_classification',
    'final_activation_and_cost': cost_functions.SoftmaxCrossEntropy()
}

# Linear Regression
mse_linear_regression_LDF_model = {
    'model_type': 'linear_regression',
    'final_activation_and_cost': cost_functions.LinearMinimumSquareError()
}

"""Linear Discriminant Function Class"""

class LinearDiscriminantFunction(object):
    """
    LinearDiscriminantFunction

    This class is actually superfluous because the same thing could be made using
    the NeuralNetwork class with no hidden units, however it is still here for reference
    """

    """Public Methods"""

    def __init__(self, n_x, n_y, model):
        """

        :param n_x: dimension of input vector x
        :param n_y: dimension of output vector y
        :param model: dictionary with model details.
          required keys:
            'model_type' - the type of task ('binary_classification', 'multiclass_classification', 'linear_regression')
            'final_activation_and_cost' - the combination Final Activation and Cost function. refer to cost_functions.py
        """
        self.n_x = n_x
        self.n_y = n_y

        self._set_model_helper(model['model_type'])

        assert isinstance(model['final_activation_and_cost'], cost_functions._FinalActivationAndCost)
        self._final_activation = model['final_activation_and_cost'].final_activation
        self._final_activation_and_cost = model['final_activation_and_cost'].final_activation_and_cost

        self._initialize_parameters()

    def get_parameters(self):
        return self.parameters

    # TODO implement batch processing
    def fit(self, X, Y, num_iterations, learning_rate, print_cost=False):
        """
        This function optimizes W and b by running a gradient descent algorithm

        :param X: data of shape (n_x, number of examples)
        :param Y: labels of shape (n_y, number of examples)
        :param num_iterations: number of iterations of the optimization loop
        :param learning_rate: learning rate of the gradient descent update rule
        :param print_cost: boolean

        :return: the costs per iteration as a list (use to plot a learning curve)
        """
        assert num_iterations >= 0
        assert learning_rate > 0
        assert X.shape[0] == self.n_x, 'invalid input vector dimension: %d, expected: %d' % (X.shape[0], self.n_x)
        assert Y.shape[0] == self.n_y, 'invalid output vector dimension: %d, expected: %d' % (Y.shape[0], self.n_y)
        self._check_inputs(X, Y)

        costs = []

        for i in range(num_iterations):
            # cost and gradient calculation
            gradients, cost = self._propagate(X, Y)

            # update rule
            self.parameters['W'] = self.parameters['W'] - learning_rate * gradients['dW']
            self.parameters['b'] = self.parameters['b'] - learning_rate * gradients['db']

            costs.append(cost)

            if print_cost and (i % 1000 == 0):
                print('%5d: %7.4f' % (i, cost))

        return costs

    def predict(self, X):
        """
        Predict whether the output using the learned parameters (W, b)

        :param X: data of shape (n_x, number of examples)
        :return:
            Y_prediction: the predicted output vector os shape (n_y, number of examples)
            A: the final activation output
                effectively the probabilities for classification tasks
                the same as Y_prediction for regression tasks
        """
        assert X.shape[0] == self.n_x, 'invalid input vector dimension: %d, expected: %d' % (X.shape[0], self.n_x)

        Z = self.forward_propagate(X, self.parameters['W'], self.parameters['b'])
        A = self._final_activation(Z)

        Y_prediction = self._prediction(A)

        return Y_prediction, A

    def score(self, X, Y):
        """
        Prediction percent correct (for classification)
                   mean squared error (for regression)
        :param X: data of shape (n_x, number of examples)
        :param Y: labels of shape (n_y, number of examples)
        :return: percent correct
                   OR mean squared error
        """
        assert X.shape[0] == self.n_x, 'invalid input vector dimension: %d, expected: %d' % (X.shape[0], self.n_x)

        Y_prediction, _ = self.predict(X)

        return self._score(Y, Y_prediction)

    """Private Methods"""

    def _set_model_helper(self, model_type):
        """
        Acquire the _ModelHelper class corresponding to 'model_type' and instantiate as a class attribute

        :param model_type: string (refer to model_helper.py for options)
        """
        valid_model_types = []

        for model_helper_name in model_helpers.__all__:
            ModelHelper = getattr(model_helpers, model_helper_name)

            if ModelHelper.model_type == model_type:
                self.model_helper = ModelHelper()
            else:
                valid_model_types.append(ModelHelper.model_type)

        assert hasattr(self, 'model_helper'), 'invalid model_type: %s\n  valid model types: %s' \
                                              % (model_type, str(valid_model_types))

        # assign class functions to model_helper functions
        self._check_inputs = self.model_helper.check_inputs
        self._prediction = self.model_helper.prediction
        self._score = self.model_helper.score

    def _initialize_parameters(self):
        W = np.zeros((self.n_y, self.n_x))
        b = np.zeros((self.n_y, 1))

        self.parameters = {
            'W': W,
            'b': b
        }

    def _propagate(self, X, Y):
        """
        propagate one full pass of gradient descent:
            - forward propagation
            - compute cost
            - backward propagation
        :param X: input data
        :param Y: output labels
        :return: gradients and cost
        """
        Z = self.forward_propagate(X, self.parameters['W'], self.parameters['b'])

        cost, A, dZ = self._final_activation_and_cost(Y, Z)

        dW, db = self.linear_backward(X, dZ)

        gradients = {
            'dW': dW,
            'db': db
        }

        return gradients, cost

    """Static Helper Methods"""

    @staticmethod
    def forward_propagate(X, W, b):
        """
        linear step to forward propagation

        :param X: data of shape (n_x, number of samples)
        :param W: weight matrix of shape (n_y, n_x)
        :param b: bias vector of length n_y
        :return:
            Z: linear function output
        """
        assert W.shape[1] == X.shape[0]
        assert np.array(b).size == W.shape[0]

        Z = np.dot(W, X) + b

        return Z

    @staticmethod
    def linear_backward(X, dZ):
        """
        back propagation from linear activation (Z) to weight matrix (W) and bias vector (b)

        :param X: data of shape (n_x, number of samples)
        :param dZ: linear activation gradient matrix of shape (n_y, number of samples)
        :return: gradients (partial derivatives)
            dW: w.r.t. the weight matrix
            db: w.r.t. the bias vector
        """
        assert dZ.shape[1] == X.shape[1]

        m = X.shape[1]

        dW = 1 / m * np.dot(dZ, X.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        return dW, db
