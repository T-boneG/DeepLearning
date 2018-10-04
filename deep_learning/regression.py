"""
Logistic Regression and Linear Regression
"""

from __future__ import division
import numpy as np

#TODO change this to "import activation_functions" and make them parameters to pass
#TODO account for the changed format of the activation functions passes the cache back as well
from activation_functions import sigmoid_af

#TODO force choice of dims on declaration (remove update dims, etc.)

#TODO split to linear and logistic

#TODO (maybe...have both inherit from Regression class:
"""
class C(abc.ABC):
    @abstractmethod
    def my_abstract_method(self, ...):
"""

#TODO move the helpers into the class (as class methods that don't take self as a parameter: static?)
def logistic_forward_propagate(X, w, b):
    """
    :param X: vector of shape (dim, number of samples)
    :param w: vector of shape (dim, 1)
    :param b: scalar
    :return: activation function output
    """
    assert (w.shape == (X.shape[0], 1))
    assert (isinstance(b, float) or isinstance(b, int))

    Z = np.dot(w.T, X) + b
    A = sigmoid_af(Z)

    return A

def logistic_backward_propagate(X, Y, A):
    """
    :param X: vector of shape (dim, number of samples)
    :param Y: vector of shape (1, number of samples)
    :param A: vector of shape (1, number of samples)
    :return: gradients for w and b
    """
    assert Y.shape == (1, X.shape[1])
    assert A.shape == Y.shape

    m = X.shape[1]

    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    return dw, db

def logistic_compute_cost(Y, A):
    """
    :param Y: vector of shape (1, number of samples)
    :param A: vector of shape (1, number of samples)
    :return: cost J(w, b)
    """
    assert Y.shape[0] == 1
    assert A.shape == Y.shape

    m = Y.shape[1]

    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    return cost

class LogisticRegression(object):
    ###################
    # Public Functions
    ###################

    def __init__(self):
        self.dim = 1
        self._initialize_params()

    def get_params(self):
        return self.params.copy()

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
        self._update_dim(X.shape[0])

        costs = []

        for i in range(num_iterations):
            # Cost and gradient calculation
            grads, cost = self._propagate(X, Y)

            # update rule
            self.params['w'] = self.params['w'] - learning_rate * grads['dw']
            self.params['b'] = self.params['b'] - learning_rate * grads['db']

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
        assert X.shape[0] == self.dim, 'invalid input vector dimension: %d. Expected: %d' % (X.shape[0], self.dim)

        A = logistic_forward_propagate(X, self.params['w'], self.params['b'])

        # Convert probabilities (A) to actual predictions
        Y_prediction = np.array(A > 0.5).astype(int)

        return Y_prediction, A

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
        This method creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)

        sets dictionary:
        params
            w -- initialized vector of shape (dim, 1)
            b -- initialized scalar (corresponds to the bias)
        """

        w = np.zeros((self.dim, 1))
        b = 0

        assert (w.shape == (self.dim, 1))
        assert (isinstance(b, float) or isinstance(b, int))

        self.params = {
            'w': w,
            'b': b
        }

    def _update_dim(self, new_dim):
        if self.dim != new_dim:
            self.dim = new_dim
            self._initialize_params()

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
        assert Y.shape == (1, X.shape[1]), 'invalid input dimensions'

        # FORWARD PROPAGATION (FROM X TO COST)
        A = logistic_forward_propagate(X, self.params['w'], self.params['b'])
        cost = logistic_compute_cost(Y, A)

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw, db = logistic_backward_propagate(X, Y, A)

        grads = {'dw': dw,
                 'db': db}

        return grads, cost
