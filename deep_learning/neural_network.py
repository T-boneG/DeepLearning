#!/usr/bin/env python
"""
neural_network.py
Neural Network machine learning algorithm
"""

from __future__ import absolute_import, division, print_function
import os
import numpy as np
import pandas as pd
import six

from .utils import random_mini_batches
from . import model_helpers, activation_functions, cost_functions

__all__ = ['NeuralNetwork']

"""Example Models"""

# Binary Classification
logistic_regression_NN_model = {
    'model_type': 'binary_classification',
    'hidden_activation': activation_functions.ReluActivation(),
    'final_activation_and_cost': cost_functions.SigmoidCrossEntropy()
}
leaky_logistic_regression_NN_model = {
    'model_type': 'binary_classification',
    'hidden_activation': activation_functions.LeakyReluActivation(leak=0.01),
    'final_activation_and_cost': cost_functions.SigmoidCrossEntropy()
}
LDF_logistic_regression_NN_model = {
    'model_type': 'binary_classification',
    # can assign 'hidden_activation' to None, but not necessary
    'final_activation_and_cost': cost_functions.SigmoidCrossEntropy()
}

# Multi-class Classification
softmax_regression_NN_model = {
    'model_type': 'multiclass_classification',
    'hidden_activation': activation_functions.ReluActivation(),
    'final_activation_and_cost': cost_functions.SoftmaxCrossEntropy()
}

# Linear Regression
mse_linear_regression_NN_model = {
    'model_type': 'linear_regression',
    'hidden_activation': activation_functions.ReluActivation(),
    'final_activation_and_cost': cost_functions.LinearMinimumSquareError()
}

"""Neural Network Class"""

class NeuralNetwork(object):
    """NeuralNetwork class"""

    """Public Methods"""

    def __init__(self, layer_dims, model):
        """
        :param layer_dims:
        :param hidden_af: activation function for all of the hidden layers/units
        :param final_af: activation function for the final layer/units

        :param layer_dims: dimensions of each layer.
                first layer dim - dimension of the input
                last layer dim - dimension of the output
                len(layer_dims) = network depth + 1
        :param model: dictionary specifying model parameters.
        Use example models above for reference.
          required keys:
            'model_type' - the type of task ('binary_classification',
                                             'multiclass_classification',
                                             'linear_regression')
            'final_activation_and_cost' - the combination Final Activation
                and Cost function. refer to cost_functions.py
        """
        self.layer_dims = np.array(layer_dims).copy()

        self._model = model

        self._set_model_helper(model['model_type'])

        if 'hidden_activation' in model.keys() \
                and model['hidden_activation'] is not None:
            self.hidden_activation = model['hidden_activation']
            assert isinstance(self.hidden_activation,
                              activation_functions._HiddenActivation)
        else:
            # must be a linear discriminant function
            # if there is no hidden activation
            assert self.get_num_layers() == 1

        assert isinstance(model['final_activation_and_cost'],
                          cost_functions._FinalActivationAndCost)
        self._final_activation = \
            model['final_activation_and_cost'].final_activation
        self._final_activation_and_cost = \
            model['final_activation_and_cost'].final_activation_and_cost

        self._initialize_parameters()
        self._clear_cache()

    def get_parameters(self):
        return self.parameters

    def get_layer_dims(self):
        return self.layer_dims

    def get_num_layers(self):
        """Number of layers"""
        return len(self.layer_dims) - 1

    def fit(self, X, Y, learning_rate, num_epochs, batch_size=32,
            print_cost=False):
        """
        This function optimizes the model parameters by running a gradient descent algorithm

        :param X: data of shape (n_x, number of examples)
        :param Y: labels of shape (n_y, number of examples)
        :param learning_rate: learning rate of the gradient descent update rule
        :param num_epochs: number of iterations of the optimization loop
        :param batch_size: size of each processing batch
        :param print_cost: boolean

        :return: the costs per iteration as a list (use to plot a learning curve)
        """
        assert learning_rate > 0
        assert num_epochs >= 0
        assert batch_size >= 0
        assert X.shape[0] == self.layer_dims[0], \
            'invalid input vector dimension: %d, expected: %d' \
            % (X.shape[0], self.layer_dims[0])
        assert Y.shape[0] == self.layer_dims[-1], \
            'invalid output vector dimension: %d, expected: %d' \
            % (Y.shape[0], self.layer_dims[-1])
        self._check_inputs(X, Y)

        epoch_costs = []

        minibatches, num_minibatches = \
            random_mini_batches(X, Y, batch_size=batch_size)

        for i in range(num_epochs):
            epoch_cost = 0

            for X_minibatch, Y_minibatch in minibatches:
                gradients, minibatch_cost = self._propagate(X_minibatch,
                                                            Y_minibatch)

                epoch_cost += minibatch_cost / num_minibatches

                # update rule for each parameter
                for l in range(self.get_num_layers()):
                    self.parameters['W' + str(l + 1)] = \
                        self.parameters['W' + str(l + 1)] \
                        - learning_rate * gradients['dW' + str(l + 1)]
                    self.parameters['b' + str(l + 1)] = \
                        self.parameters['b' + str(l + 1)] \
                        - learning_rate * gradients['db' + str(l + 1)]

            epoch_costs.append(epoch_cost)

            if print_cost and (i % 1 == 0):
                print('%5d: %7.4f' % (i, epoch_cost))

        return epoch_costs

    def predict(self, X):
        """
        Predict whether the output using the learned parameters (W, b)

        :param X: data of shape (n_x, number of examples)
        :return:
            Y_prediction: the predicted output vector os shape
                          (n_y, number of examples)
            AL: the final layer activation output
                effectively the probabilities for classification tasks
                the same as Y_prediction for regression tasks
        """
        assert X.shape[0] == self.layer_dims[0]

        ZL = self._forward_propagate(X)
        AL = self._final_activation(ZL)

        # Convert probabilities (AL) to actual predictions
        Y_prediction = self._prediction(AL)

        return Y_prediction, AL

    def score(self, X, Y):
        """
        Prediction percent correct (for classification)
                   mean squared error (for regression)
        :param X: data of shape (n_x, number of examples)
        :param Y: labels of shape (n_y, number of examples)
        :return: percent correct
                   OR mean squared error
        """
        assert X.shape[0] == self.layer_dims[0], \
            'invalid input vector dimension: %d, expected: %d' \
            % (X.shape[0], self.layer_dims[0])

        Y_prediction, _ = self.predict(X)

        return self._score(Y, Y_prediction)

    def save(self, filepath, overwrite=True):
        """
        save the NN's parameters to an HDF5 store

        :param filepath: path to the HDF5 store
        :param overwrite: Whether we should overwrite any existing
            model at the target location, or instead
            ask the user with a manual prompt.
        """
        assert filepath.endswith('.h5'), \
            'incorrect file suffix: .%s, should be .h5' \
            % filepath.split('.')[-1]

        # If file exists and should not be overwritten.
        if not overwrite and os.path.isfile(filepath):
            proceed = _ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return

        with pd.HDFStore(filepath, mode='w') as store:
            for key, value in self.get_parameters().items():
                store[key] = pd.DataFrame(value)

    def load(self, filepath):
        """
        load the NN's parameters from an HDF5 store created using NeuralNetwork.save(...)

        :param filepath: path to the HDF5 store
        """
        assert filepath.endswith('.h5'), \
            'incorrect file suffix: .%s, should be .h5' \
            % filepath.split('.')[-1]

        with pd.HDFStore(filepath, mode='r') as store:
            # print(store.info())

            for key in self.parameters.keys():
                assert key in store, 'parameter not found in the load file'

                parameter = store[key].get_values()
                assert parameter.shape == self.parameters[key].shape, \
                    'inconsistent shape between this NeuralNetwork\'s ' \
                    'parameters and the load file'

                self.parameters[key] = parameter

    def summary(self):
        """

        :return: a summary string of the network
        """
        s = ''

        for key, value in self._model.items():
            s = '\n'.join([s, '%s: %s' % (key, value)])

        s = '\n'.join([s, 'number of layers: %d' % self.get_num_layers()])

        layers = [{} for _ in range(self.get_num_layers())]

        for key, value in self.parameters.items():
            _, layer_number = _split_trailing_numbers(key)
            layers[layer_number-1][key] = value

        for i, cur_layer in enumerate(layers):
            cur_keys = cur_layer.keys()
            cur_keys.sort()

            s = '\n'.join([s, 'Layer %d:' % (i+1)])
            for key in cur_keys:
                s = '\n'.join([s, '\t%s - %s' % (key, cur_layer[key].shape)])

        return s

    """Private Methods"""

    def _set_model_helper(self, model_type):
        """
        Acquire the _ModelHelper class corresponding to 'model_type'
        and instantiate as a class attribute

        :param model_type: string (refer to model_helper.py for options)
        """
        valid_model_types = []

        for model_helper_name in model_helpers.__all__:
            ModelHelper = getattr(model_helpers, model_helper_name)

            if ModelHelper.model_type == model_type:
                self.model_helper = ModelHelper()
            else:
                valid_model_types.append(ModelHelper.model_type)

        assert hasattr(self, 'model_helper'), \
            'invalid model_type: %s\n  valid model types: %s' \
            % (model_type, str(valid_model_types))

        # assign class functions to model_helper functions
        self._check_inputs = self.model_helper.check_inputs
        self._prediction = self.model_helper.prediction
        self._score = self.model_helper.score

    def _initialize_parameters(self):
        """
        initialize parameters using Xavier initialization
        """
        self.parameters = {}
        L = self.get_num_layers()

        if isinstance(self.hidden_activation,
                      activation_functions.TanhActivation):
            scale_factor = 1
        elif isinstance(self.hidden_activation,
                        activation_functions.ReluActivation):
            scale_factor = 2
        else:
            raise NotImplementedError('unknown Xavier initialization scale '
                                      'factor for this activation function')

        for l in range(1, L+1):
            self.parameters['W' + str(l)] = \
                np.sqrt(scale_factor / self.layer_dims[l - 1]) \
                * np.random.randn(self.layer_dims[l], self.layer_dims[l - 1])
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

    def _clear_cache(self):
        """Cache used to store forward values for easier computation during
        back propagation"""
        self.cache = {}

    def _cache_insert(self, key, value):
        """Ensure this key is not already in the cache (for debugging)"""
        assert key not in self.cache
        self.cache[key] = value

    # Forward Propagation

    def _linear_forward(self, A_prev, W, b, layer):
        """
        Implement the linear part of a layer's forward propagation

        :param A_prev: activations from previous layer (or input data):
                    (size of previous layer, number of examples)
        :param W: current layer weight matrix: shape
                    (size of current layer, size of previous layer)
        :param b: current layer bias vector: shape
                    (size of current layer, 1)
        :param layer: current layer number
        :return: Z: linear activation of this layer
        """

        Z = np.dot(W, A_prev) + b

        self._cache_insert('A' + str(layer-1), A_prev)
        self._cache_insert('W' + str(layer), W)
        self._cache_insert('b' + str(layer), b)

        return Z

    def _linear_activation_forward(self, A_prev, W, b, layer):
        """
        Implement forward propagation for a single hidden layer

        :param A_prev: activations from previous layer (or input data):
                    (size of previous layer, number of examples)
        :param W: current layer weight matrix: shape
                    (size of current layer, size of previous layer)
        :param b: current layer bias vector: shape
                    (size of current layer, 1)
        :param layer: current layer number
        :return: Z: activation of this layer
        """
        Z = self._linear_forward(A_prev, W, b, layer)
        A = self.hidden_activation.forward(Z)

        self._cache_insert('Z' + str(layer), Z)

        assert A.shape == (W.shape[0], A_prev.shape[1])

        return A

    def _forward_propagate(self, X):
        """
        Implement forward propagation through all hidden units
        (except for the final activation function)

        :param X: data of shape (input size, number of examples)
        :return: ZL: the final layer linear activation
        """
        self._clear_cache() # clear for each full pass of forward propagation

        L = self.get_num_layers()
        A = X

        # propagate through the first L-1 layers
        for l in range(1, L):
            A_prev = A
            A = self._linear_activation_forward(A_prev,
                                                self.parameters['W' + str(l)],
                                                self.parameters['b' + str(l)],
                                                l)

            # if keep_prob != 1:
            #     # apply dropout
            #     D = np.random.rand(A.shape[0], A.shape[1])
            #     D = (D < keep_prob)
            #     A = A * D
            #     A = A / keep_prob
            #     cache['D' + str(l)] = D

        # propagate through the linear part of the final layer
        ZL = self._linear_forward(A,
                                  self.parameters['W' + str(L)],
                                  self.parameters['b' + str(L)], L)

        assert ZL.shape[0] == self.layer_dims[-1] and ZL.shape[1] == X.shape[1]

        return ZL

    # Backward Propagation

    def _linear_backward(self, dZ, layer):
        """
        Implement the linear part of a layer's backward propagation

        :param dZ: the linear activation gradients of this layer
        :param layer: current layer number
        :return:
            dA_prev: the gradients w.r.t. the previous layer's activations
            dW: the gradients w.r.t. this layer's weight matrix
            db: the gradients w.r.t. this layer's bias vector
        """
        A_prev = self.cache['A' + str(layer-1)]
        W = self.cache['W' + str(layer)]
        b = self.cache['b' + str(layer)]

        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert dW.shape == W.shape
        assert db.shape == b.shape
        assert dA_prev.shape == A_prev.shape

        return dA_prev, dW, db

    def _linear_activation_backward(self, dA, layer):
        """
        Implement backward propagation for a single hidden layer

        :param dA: the activation gradients of this layer
        :param layer: current layer number
        :return:
            dA_prev: the gradients w.r.t. the previous layer's activations
            dW: the gradients w.r.t. this layer's weight matrix
            db: the gradients w.r.t. this layer's bias vector
        """
        Z = self.cache['Z' + str(layer)]

        dZ = self.hidden_activation.backward(dA, Z)
        dA_prev, dW, db = self._linear_backward(dZ, layer)

        return dA_prev, dW, db

    def _backward_propagate(self, dZL):
        """
        Implements a full pass of backward propagation through the model

        :param dZL: the linear activation gradients of the final layer
        :return:
          gradients: a dictionary with the gradients with respect to
                     each parameter, activation and pre-activation variables
                 grads['dA' + str(l)] = ...
                 grads['dW' + str(l)] = ...
                 grads['db' + str(l)] = ...
        """
        L = self.get_num_layers()

        gradients = {}

        # Lth layer gradients.
        (gradients['dA' + str(L-1)],
         gradients['dW' + str(L)],
         gradients['db' + str(L)]) = self._linear_backward(dZL, L)

        # # apply dropout
        # if keep_prob < 1:
        #     D = cache['D' + str(L-1)]
        #     gradients['dA' + str(L-1)] = \
        #         gradients['dA' + str(L-1)] * D / keep_prob

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer gradients.
            (gradients['dA' + str(l)],
             gradients['dW' + str(l+1)],
             gradients['db' + str(l+1)]) = \
                self._linear_activation_backward(gradients['dA' + str(l + 1)],
                                                 l + 1)

            # # apply dropout
            # if keep_prob < 1 and l != 0:
            #     D = cache['D' + str(l)]
            #     gradients['dA' + str(l)] = \
            #           gradients['dA' + str(l)] * D / keep_prob

        return gradients

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
        ZL = self._forward_propagate(X)

        cost, _, dZL = self._final_activation_and_cost(Y, ZL)

        grads = self._backward_propagate(dZL)

        return grads, cost

def _split_trailing_numbers(s):
    """

    :param s: a string
    :return:
        head - the string with the trailing integer removed
        tail - the trailing numbers
    """
    head = s.rstrip('0123456789')
    tail = int(s[len(head):])

    return head, tail

def _ask_to_proceed_with_overwrite(filepath):
  """Produces a prompt asking about overwriting a file.

  Arguments:
      filepath: the path to the file to be overwritten.

  Returns:
      True if we can proceed with overwrite, False otherwise.
  """
  overwrite = six.moves.input('[WARNING] %s already exists - overwrite? '
                              '[y/n]' % (filepath)).strip().lower()
  while overwrite not in ('y', 'n'):
    overwrite = six.moves.input('Enter "y" (overwrite) or "n" '
                                '(cancel).').strip().lower()
  if overwrite == 'n':
    return False
  print('[TIP] Next time specify overwrite=True!')
  return True
