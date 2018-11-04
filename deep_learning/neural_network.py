"""
Neural Network
"""

from __future__ import division
import numpy as np
from utils import *
import model_helpers
import activation_functions
import cost_functions

# __all__ = ['NeuralNetwork']

example_model = {
    'model_type': 'binary_classification',
    'hidden_activation': activation_functions.LeakyReluActivation(leak=0.01),
    'final_activation_and_cost': cost_functions.SigmoidCrossEntropy()
}

ldf_example_model = {
    'model_type': 'binary_classification',
    # can assign 'hidden_activation' to None, but not necessary
    'final_activation_and_cost': cost_functions.SigmoidCrossEntropy()
}

"""Binary Classification Models"""

logistic_regression_NN_model = {
    'model_type': 'binary_classification',
    'hidden_activation': activation_functions.ReluActivation(),
    'final_activation_and_cost': cost_functions.SigmoidCrossEntropy()
}

"""Multi-class Classification Models"""

softmax_regression_NN_model = {
    'model_type': 'multiclass_classification',
    'hidden_activation': activation_functions.ReluActivation(),
    'final_activation_and_cost': cost_functions.SoftmaxCrossEntropy()
}

"""Linear Regression Models"""

mse_linear_regression_NN_model = {
    'model_type': 'linear_regression',
    'hidden_activation': activation_functions.ReluActivation(),
    'final_activation_and_cost': cost_functions.LinearMinimumSquareError()
}

class NeuralNetwork(object):
    """description..."""

    """Public Methods"""

    def __init__(self, layer_dims, model):
        """

        :param layer_dims: dimensions of each layer.
                first layer dim -- dimension of the input
                last layer dim -- dimension of output
                len(layer_dims) = network depth + 1
        :param hidden_af: activation function for all of the hidden layers/units
        :param final_af: activation function for the final layer/units
        """
        self.layer_dims = np.array(layer_dims).copy()

        self._set_model_helper(model['model_type'])

        self.faac = model['final_activation_and_cost']

        if 'hidden_activation' in model.keys() and model['hidden_activation'] is not None:
            self.hidden_activation = model['hidden_activation']
            assert isinstance(self.hidden_activation, activation_functions._HiddenActivation)
        else:
            # must be a linear discriminant function if there is no hidden activation
            assert self.get_num_layers() == 1

        self._initialize_parameters()
        self._clear_cache()

    def get_parameters(self):
        return self.parameters

    def get_layer_dims(self):
        return self.layer_dims

    def get_num_layers(self):
        return len(self.layer_dims) - 1

    def fit(self, X, Y, num_iterations, learning_rate, print_cost=False):
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
        #TODO check inputs?
        costs = []

        for i in range(num_iterations):
            # Cost and gradient calculation
            grads, cost = self._propagate(X, Y)

            L = len(self.parameters) // 2  # number of layers in the neural network

            # Update rule for each parameter
            for l in range(L):
                self.parameters['W' + str(l + 1)] = self.parameters['W' + str(l + 1)] \
                                                    - learning_rate * grads['dW' + str(l + 1)]
                self.parameters['b' + str(l + 1)] = self.parameters['b' + str(l + 1)] \
                                                    - learning_rate * grads['db' + str(l + 1)]

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

        ZL = self._forward_propagate(X)
        AL = self.faac.final_activation(ZL)

        # Convert probabilities (A) to actual predictions
        Y_prediction = self._prediction(AL)

        return Y_prediction, AL

    def score(self, X, Y):
        """
        prediction percent correct
        :param X:
        :param Y:
        :return:
        """
        assert X.shape[0] == self.layer_dims[0], 'invalid input vector dimension: %d, Expected: %d' \
                                                 % (X.shape[0], self.layer_dims[0])

        Y_prediction, _ = self.predict(X)

        return self._score(Y, Y_prediction)

    """Private Methods"""

    def _set_model_helper(self, model_type):
        # set the model helper
        valid_model_types = []
        for model_helper_name in model_helpers.__all__:
            ModelHelper = getattr(model_helpers, model_helper_name)
            if ModelHelper.model_type == model_type:
                self.model_helper = ModelHelper()
            else:
                valid_model_types.append(ModelHelper.model_type)
        assert hasattr(self, 'model_helper'), 'invalid model_type: %s\n  valid model types: %s' \
                                              % (model_type, str(valid_model_types))

        self._check_inputs = self.model_helper.check_inputs
        self._prediction = self.model_helper.prediction
        self._score = self.model_helper.score

    def _initialize_parameters(self):
        """
        initialize parameters using Xavier initialization
        """
        self.parameters = {}
        L = self.get_num_layers()

        if isinstance(self.hidden_activation, activation_functions.TanhActivation):
            scale_factor = 1.
        elif isinstance(self.hidden_activation, activation_functions.ReluActivation):
            scale_factor = 2.
        else:
            raise NotImplementedError, 'unknown Xavier initialization scale factor for this activation function'

        for l in range(1, L+1):
            self.parameters['W' + str(l)] = np.sqrt(scale_factor / self.layer_dims[l - 1]) \
                                            * np.random.randn(self.layer_dims[l], self.layer_dims[l - 1])
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

    def _clear_cache(self):
        self.cache = {}

    def _cache_insert(self, key, value):
        assert key not in self.cache
        self.cache[key] = value

    # Forward Propagation

    def _linear_forward(self, A_prev, W, b, layer):
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

        self._cache_insert('A' + str(layer-1), A_prev)
        self._cache_insert('W' + str(layer), W)
        self._cache_insert('b' + str(layer), b)

        return Z

    def _linear_activation_forward(self, A_prev, W, b, layer):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        layer -- current layer number
        activation -- the activation function to be used in this layer

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing 'A', 'W', 'b', 'Z'; stored for computing the backward pass efficiently
        """
        Z = self._linear_forward(A_prev, W, b, layer)
        A = self.hidden_activation.forward(Z)

        self._cache_insert('Z' + str(layer), Z)

        assert A.shape == (W.shape[0], A_prev.shape[1])

        return A

    def _forward_propagate(self, X):
        # def forward_propagate(X, L, parameters, keep_prob, hidden_activation, final_activation):
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

        self._clear_cache()

        L = self.get_num_layers()
        A = X

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A = self._linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], l)

            # if keep_prob != 1:
            #     # apply dropout
            #     D = np.random.rand(A.shape[0], A.shape[1])
            #     D = (D < keep_prob)
            #     A = A * D
            #     A = A / keep_prob
            #     cache['D' + str(l)] = D

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ZL = self._linear_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], L)

        assert ZL.shape[1] == X.shape[1]

        return ZL

    # Backward Propagation

    def _linear_backward(self, dZ, layer):
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
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        af_backward -- the activation backward function

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        Z = self.cache['Z' + str(layer)]

        dZ = self.hidden_activation.backward(dA, Z)
        dA_prev, dW, db = self._linear_backward(dZ, layer)

        return dA_prev, dW, db

    def _backward_propagate(self, dZL):
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
        L = self.get_num_layers()

        gradients = {}

        # Lth layer gradients.
        (gradients['dA' + str(L-1)],
         gradients['dW' + str(L)],
         gradients['db' + str(L)]) = self._linear_backward(dZL, L)

        # # apply dropout
        # if keep_prob < 1:
        #     D = cache['D' + str(L-1)]
        #     gradients['dA' + str(L-1)] = gradients['dA' + str(L-1)] * D / keep_prob

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer gradients.
            (gradients['dA' + str(l)],
             gradients['dW' + str(l+1)],
             gradients['db' + str(l+1)]) = self._linear_activation_backward(gradients['dA' + str(l + 1)], l + 1)

            # # apply dropout
            # if keep_prob < 1 and l != 0:
            #     D = cache['D' + str(l)]
            #     gradients['dA' + str(l)] = gradients['dA' + str(l)] * D / keep_prob

        return gradients

    #TODO regularization
    @staticmethod
    def _backward_propagate_with_regularization(AL, Y, L, cache, lambd, params, keep_prob,
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

        gradients = self._backward_propagate(AL, Y, L, cache, keep_prob,
                                                      hidden_activation=hidden_activation,
                                                      final_activation=final_activation)

        # apply L2 regularization
        if lambd > 0:
            m = AL.shape[1]

            for l in range(L):
                assert gradients['dW' + str(l+1)].shape == params['W' + str(l+1)].shape
                gradients['dW' + str(l+1)] = gradients['dW' + str(l+1)] + (lambd / m) * params['W' + str(l+1)]

        return gradients


    def _propagate(self, X, Y):
        ZL = self._forward_propagate(X)

        cost, _, dZL = self.faac.final_activation_and_cost(Y, ZL)

        grads = self._backward_propagate(dZL)

        return grads, cost
