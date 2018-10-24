"""
Neural Network
"""

from __future__ import division

import numpy as np
import activation_functions

__all__ = ['NeuralNetwork']

class NeuralNetwork(object):
    _activation_functions = activation_functions.get_activation_function_names()

    ###################
    # Public Methods
    ###################

    def __init__(self, layer_dims, hidden_af='relu', final_af='sigmoid'):
        """

        :param layer_dims: dimensions of each layer.
                first layer dim -- dimension of the input
                last layer dim -- dimension of output
                len(layer_dims) = network depth + 1
        :param hidden_af: activation function for all of the hidden layers/units
        :param final_af: activation function for the final layer/units
        """
        self.layer_dims = np.array(layer_dims).copy()

        self.hidden_af = hidden_af.lower()
        assert self.hidden_af in NeuralNetwork._activation_functions
        self.final_af= final_af.lower()
        assert self.final_af in NeuralNetwork._activation_functions

        self._initialize_params()

    def get_params(self):
        return self.params

    def get_layer_dims(self):
        return self.layer_dims

    def get_num_layers(self):
        return len(self.layer_dims) - 1

    def fit(self, X, Y, num_iterations, learning_rate, lambd=0, keep_prob=1, print_cost=False):
        """
        This function optimizes W and b by running a gradient descent algorithm

        Arguments:
        X -- data of shape (layer_dims[0], number of examples)
        Y -- true "label" vector (containing 0's and 1's), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        lambd -- regularization term (must be >= 0 and lambd=0 >> no regularization)
        keep_prob -- activation unit keep probability used for dropout (0 < keep_prob <= 1
                     and keep_prob=1 >> no dropout)
        print_cost -- boolean value

        Returns:
        costs -- list of all the costs computed during the optimization
        """
        costs = []

        for i in range(num_iterations):
            # Cost and gradient calculation
            grads, cost = self._propagate(X, Y, lambd, keep_prob)

            L = len(self.params) // 2  # number of layers in the neural network

            # Update rule for each parameter
            for l in range(L):
                self.params['W' + str(l+1)] = self.params['W' + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
                self.params['b' + str(l+1)] = self.params['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]

            costs.append(cost)

            if print_cost and (i % 1000 == 0):
                print('%5d: %7.4f' % (i, cost))

        return costs

    def predict(self, X):
        '''
        predict the output Y for a given input X

        Arguments:
        X -- data of size (layer_dims[0], number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions
        A -- a numpy array (vector) containing probabilities corresponding to predictions
        '''
        assert X.shape[0] == self.layer_dims[0]

        AL, _ = NeuralNetwork.forward_propagate(X, self.get_num_layers(), self.params, keep_prob=1,
                                                hidden_activation=self.hidden_af, final_activation=self.final_af)

        # Convert probabilities (A) to actual predictions
        Y_prediction = np.array(AL > 0.5).astype(int)

        return Y_prediction, AL

    def score(self, X, Y):
        """
        compute prediction accuracy
        :param X: data of size (layer_dims[0], number of examples)
        :param Y: labels of size (layer_dims[-1], number of examples)
        :return: ratio of correct predictions / total predictions
        """
        assert Y.shape == (1, X.shape[1]), 'invalid input dimensions'

        Y_prediction, _ = self.predict(X)

        correct = np.sum(Y_prediction == Y)
        return correct / Y.shape[1]

    ###################
    # Private Methods
    ###################

    def _initialize_params(self):
        """
        initialize parameters using Xavier initialization
        """
        self.params = {}
        L = self.get_num_layers()

        if self.hidden_af == 'tanh':
            scale_factor = 1.
        elif self.hidden_af == 'relu':
            scale_factor = 2.
        else:
            raise NotImplementedError

        for l in range(1, L+1):
            self.params['W' + str(l)] = np.sqrt(scale_factor / self.layer_dims[l - 1]) \
                                        * np.random.randn(self.layer_dims[l], self.layer_dims[l - 1])
            self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

    def _propagate(self, X, Y, lambd=0, keep_prob=1):
        assert X.shape[0] == self.layer_dims[0]
        assert Y.shape == (1, X.shape[1]), 'invalid input dimensions'
        assert lambd >= 0
        assert keep_prob > 0 and keep_prob <= 1

        #TODO (eventually) allow both at the same time
        assert (lambd == 0 or keep_prob == 1)

        AL, cache = NeuralNetwork.forward_propagate(X, self.get_num_layers(), self.params, keep_prob,
                                                    hidden_activation=self.hidden_af, final_activation=self.final_af)

        cost = NeuralNetwork.compute_cost_with_regularization(AL, Y, self.params, lambd)
        grads = NeuralNetwork.backward_propagate_with_regularization(AL, Y, self.get_num_layers(), cache, lambd,
                                                                     self.params, keep_prob,
                                                                     hidden_activation=self.hidden_af,
                                                                     final_activation=self.final_af)

        return grads, cost

    #########################
    # Static Helper Methods
    #########################

    # Forward Propagation

    @staticmethod
    def linear_forward(A_prev, W, b, layer):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        layer -- current layer number

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python dictionary containing 'A', 'W' and 'b' ; stored for computing the backward pass efficiently
        """

        Z = np.dot(W, A_prev) + b

        cache = {
            'A' + str(layer-1): A_prev,
            'W' + str(layer): W,
            'b' + str(layer): b
        }

        return Z, cache

    @staticmethod
    def linear_activation_forward(A_prev, W, b, layer, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        layer -- current layer number
        activation -- the activation to be used in this layer, stored as a text string:
                        must be the prefix to a function in activation_functions that ends with: '_af'
                        examples:
                            for function linear_af, input: 'linear'
                            for function sigmoid_af, input: 'sigmoid'

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing 'A', 'W', 'b', 'Z'; stored for computing the backward pass efficiently
        """

        assert hasattr(activation_functions, activation + '_af'), 'invalid activation: %s' % activation
        activation_function = getattr(activation_functions, activation + '_af')

        Z, cache = NeuralNetwork.linear_forward(A_prev, W, b, layer)

        A = activation_function(Z)
        cache['Z' + str(layer)] = Z

        assert A.shape == (W.shape[0], A_prev.shape[1])

        return A, cache

    @staticmethod
    def forward_propagate(X, L, parameters, keep_prob, hidden_activation, final_activation):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        L -- number of layers in the neural network
        parameters -- output of initialize_parameters_deep()
        hidden_activation -- the activation function for the hidden layers, suggested: 'tanh' OR 'relu'
        final_activation -- activation function of the final Lth layer, suggested: 'linear' OR 'sigmoid'

        Returns:
        AL -- last post-activation value
        cache -- dictionary containing all cached elements to be used in back prop
        """

        cache = {}
        A = X

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A, current_cache = NeuralNetwork.linear_activation_forward(A_prev, parameters['W'+str(l)],
                                                                       parameters['b'+str(l)], l, hidden_activation)
            for key, value in current_cache.items():
                assert key not in cache
                cache[key] = value

            if keep_prob != 1:
                # apply dropout
                D = np.random.rand(A.shape[0], A.shape[1])
                D = (D < keep_prob)
                A = A * D
                A = A / keep_prob
                cache['D' + str(l)] = D

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, current_cache = NeuralNetwork.linear_activation_forward(A, parameters['W'+str(L)],
                                                                    parameters['b'+str(L)], L, final_activation)
        for key, value in current_cache.items():
            assert key not in cache
            cache[key] = value

        assert AL.shape[1] == X.shape[1]

        return AL, cache

    # Cost

    @staticmethod
    def compute_cost(AL, Y):
        """
        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]

        # Compute loss from aL and y.
        log_sum = Y * np.log(AL) + (1 - Y) * np.log(1 - AL)
        cost = (-1 / m) * np.sum(log_sum)

        return cost

    @staticmethod
    def compute_cost_with_regularization(AL, Y, params, lambd):
        cost = NeuralNetwork.compute_cost(AL, Y)

        # apply L2 regularization
        if lambd != 0:
            m = Y.shape[1]

            # select W's from params
            all_Ws = [val for (key, val) in params.items() if key.startswith('W')]

            L2_regularization_cost = (lambd / (2 * m)) * np.sum([np.sum(np.square(W)) for W in all_Ws])

            cost = cost + L2_regularization_cost

        return cost

    # Backward Propagation

    @staticmethod
    def linear_backward(dZ, cache, layer):
        """
        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
        layer -- current layer number

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev = cache['A' + str(layer-1)]
        W = cache['W' + str(layer)]
        b = cache['b' + str(layer)]
        m = A_prev.shape[1]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        #TODO debug this
        assert dW.shape == W.shape, 'L: ' + str(layer) + ' W:' + str(W.shape) + ' dZ:' + str(dZ.shape) + ' A_Prev:' + str(A_prev.shape)
        assert db.shape == b.shape
        assert dA_prev.shape == A_prev.shape

        return dA_prev, dW, db

    @staticmethod
    def linear_activation_backward(dA, cache, layer, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string:
                        must be the prefix to a function in activation_functions that ends with: '_backward'
                        examples:
                            for function linear_backward, input: 'linear'
                            for function sigmoid_backward, input: 'sigmoid'

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        assert hasattr(activation_functions, activation + '_backward'), 'invalid activation: %s' % activation
        af_backward = getattr(activation_functions, activation + '_backward')

        Z = cache['Z' + str(layer)]

        dZ = af_backward(dA, Z)
        dA_prev, dW, db = NeuralNetwork.linear_backward(dZ, cache, layer)

        return dA_prev, dW, db

    @staticmethod
    def backward_propagate(AL, Y, L, cache, keep_prob, hidden_activation, final_activation):
        """
        Implements the backward propagation of our baseline model to which we added an L2 regularization.

        Arguments:
        AL -- last layer activations
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation()
        keep_prob -- keep probability for dropout
        hidden_activation -- string name of the hidden activation function
        final_activation -- string name of the final activation function

        Returns:
        gradients -- a dictionary with the gradients with respect to each parameter,
                     activation and pre-activation variables
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        gradients = {}
        Y = Y.reshape(AL.shape)

        # Initializing the back propagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients.
        gradients['dA' + str(L - 1)], gradients['dW' + str(L)], gradients['db' + str(L)] = \
            NeuralNetwork.linear_activation_backward(dAL, cache, L, final_activation)

        # apply dropout
        if keep_prob < 1:
            D = cache['D' + str(L-1)]
            gradients['dA' + str(L-1)] = gradients['dA' + str(L-1)] * D / keep_prob

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            dA_prev_temp, dW_temp, db_temp = \
                NeuralNetwork.linear_activation_backward(gradients['dA' + str(l + 1)], cache, l+1, hidden_activation)

            # apply dropout
            if keep_prob < 1 and l != 0:
                D = cache['D' + str(l)]
                dA_prev_temp = dA_prev_temp * D / keep_prob

            gradients['dA' + str(l)] = dA_prev_temp
            gradients['dW' + str(l+1)] = dW_temp
            gradients['db' + str(l+1)] = db_temp

        return gradients

    @staticmethod
    def backward_propagate_with_regularization(AL, Y, L, cache, lambd, params, keep_prob,
                                               hidden_activation, final_activation):
        """
        calls backward_propogation without L2 regularization, then applies L2 regularization

        Arguments:
        AL -- last layer activations
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation()
        lambd -- regularization hyperparameter, scalar
        params -- dictionary of W and b parameters of the network
        keep_prob -- keep probability for dropout
        hidden_activation -- string name of the hidden activation function
        final_activation -- string name of the final activation function

        Returns:
        gradients -- a dictionary with the gradients with respect to each parameter,
                     activation and pre-activation variables
        """

        gradients = NeuralNetwork.backward_propagate(AL, Y, L, cache, keep_prob,
                                                     hidden_activation=hidden_activation,
                                                     final_activation=final_activation)

        # apply L2 regularization
        if lambd > 0:
            m = AL.shape[1]

            for l in range(L):
                assert gradients['dW' + str(l+1)].shape == params['W' + str(l+1)].shape
                gradients['dW' + str(l+1)] = gradients['dW' + str(l+1)] + (lambd / m) * params['W' + str(l+1)]

        return gradients
