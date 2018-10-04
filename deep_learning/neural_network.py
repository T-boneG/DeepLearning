"""
Neural Network
"""

import numpy as np
import activation_functions
#TODO account for the changed format of the activation functions passes the cache back as well
#TODO and the renaming from "derivative" to "backward"

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string:
                    must be the prefix to a function in activation_functions that ends with: '_af'
                    examples:
                        for function linear_af, input: 'linear'
                        for function sigmoid_af, input: 'sigmoid'

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    assert hasattr(activation_functions, activation + '_af'), 'invalid activation: %s' % activation
    activation_function = getattr(activation_functions, activation + '_af')

    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = activation_function(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def forward_propagate(X, parameters, hidden_activation='relu', final_activation='sigmoid'):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    hidden_activation -- the activation function for the hidden layers, suggested: 'tanh' OR 'relu'
    final_activation -- activation function of the final Lth layer, suggested: 'linear' OR 'sigmoid'

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             hidden_activation)
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = A, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)],
                                                     final_activation)
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches

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
    logsum = Y * np.log(AL) + (1 - Y) * np.log(1 - AL)
    cost = (-1 / m) * np.sum(logsum)

    return cost

def linear_backward(dZ, cache):
    """
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string:
                    must be the prefix to a function in activation_functions that ends with: '_derivative'
                    examples:
                        for function linear_derivative, input: 'linear'
                        for function sigmoid_derivative, input: 'sigmoid'

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    assert hasattr(activation_functions, activation + '_derivative'), 'invalid activation: %s' % activation
    af_derivative = getattr(activation_functions, activation + '_derivative')

    dZ = af_derivative(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def backward_propagate(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the back propagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients.
    current_cache = caches[-1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, 'sigmoid')

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


class NeuralNetwork(object):
    #TODO maybe make this a function? that gets the list from activation_functions.py
    _activation_functions = ['linear', 'sigmoid', 'tanh', 'relu']

    ###################
    # Public Functions
    ###################

    def __init__(self, layers_dims, hidden_af='relu', final_af='sigmoid'):
        """

        :param layers_dims: dimensions of each layer.
                first layer dim -- dimension of the input
                last layer dim -- dimension of output
                len(layers_dims) = network depth + 1
        :param hidden_af: activation function for all of the hidden layers/units
        :param final_af: activation function for the final layer/units
        """
        self.layers_dims = np.array(layers_dims).copy()

        self.hidden_af = hidden_af.lower()
        assert self.hidden_af in NeuralNetwork._activation_functions
        self.final_af= final_af.lower()
        assert self.final_af in NeuralNetwork._activation_functions

        self._initialize_params()

    def get_params(self):
        return self.params

    def fit(self, X, Y, num_iterations, learning_rate):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        X -- data of shape (dim, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule

        Returns:
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """
        costs = []

        for i in range(num_iterations):
            # Cost and gradient calculation
            grads, cost = self._propagate(X, Y)

            L = len(self.params) // 2  # number of layers in the neural network

            # Update rule for each parameter
            for l in range(L):
                self.params['W' + str(l+1)] = self.params['W' + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
                self.params['b' + str(l+1)] = self.params['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]

            costs.append(cost)

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
        assert X.shape[0] == self.layers_dims[0]

        AL, _ = forward_propagate(X, self.params, hidden_activation=self.hidden_af, final_activation=self.final_af)

        # Convert probabilities (A) to actual predictions
        Y_prediction = np.array(AL > 0.5).astype(int)

        return Y_prediction, AL

    def score(self, X, Y):
        """
        prediction percent correct
        :param X:
        :param Y:
        :return:
        """
        assert Y.shape == (1, X.shape[1]), 'invalid input dimensions'

        Y_prediction, _ = self.predict(X)

        correct = np.sum(Y_prediction == Y)
        return correct / Y.shape[1]

    ###################
    # Private Functions
    ###################

    def _initialize_params(self):
        """
        Arguments:
        layer_dims -- python array (list) containing the size of each layer.

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
        """

        self.params = {}
        L = len(self.layers_dims) - 1  # integer representing the number of layers

        if self.hidden_af == 'tanh':
            scale_factor = 1.
        elif self.hidden_af == 'relu':
            scale_factor = 2.
        else:
            raise NotImplementedError

        for l in range(1, L + 1):
            self.params['W' + str(l)] = np.sqrt(scale_factor / self.layers_dims[l - 1]) \
                                        * np.random.randn(self.layers_dims[l], self.layers_dims[l - 1])
            self.params['b' + str(l)] = np.zeros((self.layers_dims[l], 1))

    def _propagate(self, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        X -- data of size (dim, number of examples)
        Y -- true "label" vector  of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """
        assert X.shape[0] == self.layers_dims[0]
        assert Y.shape == (1, X.shape[1]), 'invalid input dimensions'

        AL, caches = forward_propagate(X, self.params, hidden_activation=self.hidden_af, final_activation=self.final_af)

        cost = compute_cost(AL, Y)

        grads = backward_propagate(AL, Y, caches)

        return grads, cost


# TODO
# NN cat example:
"""
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
learning_rate = 0.0075
num_iterations = 3000
"""