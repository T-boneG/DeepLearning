"""

"""

from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
from utils import *
import warnings

__all__ = ['SigmoidCrossEntropy', 'SigmoidPerceptron', 'SoftmaxCrossEntropy', 'LinearMinimumSquareError']


class _FinalActivationAndCost:
    __metaclass__ = ABCMeta

    @abstractmethod
    def final_activation(self, ZL):
        pass

    @abstractmethod
    def final_activation_and_cost(self, Y, ZL):
        pass

class SigmoidCrossEntropy(_FinalActivationAndCost):
    """
    Sigmoid final activation with Cross Entropy cost function
      Use for binary classification
      Y is of dim (1, number of samples) and each element is a label: 0 or 1
    :param Y: labels
    :param ZL: final layer linear output
    :return:
        cost: ...
        AL: final activation function output
        dZL: partial derivative of cost w.r.t. the final linear output
    """
    def final_activation(self, ZL):
        return sigmoid(ZL)

    def final_activation_and_cost(self, Y, ZL):
        assert Y.shape == ZL.shape, 'inconsistent shapes %s != %s' % (str(Y.shape), str(ZL.shape))

        m = Y.shape[1]

        # compute final activation
        AL = self.final_activation(ZL)

        # compute cost
        cost = -1 / m * np.sum(np.multiply(Y, stable_log(AL)) + np.multiply(1 - Y, stable_log(1 - AL)))

        # # compute partial derivatives (don't delete for reference)
        # dAL = np.divide(1 - Y, 1 - AL) - np.divide(Y, AL)
        # dZL = dAL * AL * (1 - AL)

        # compute dZ directly from the output
        dZL = AL - Y

        return cost, AL, dZL

class SigmoidPerceptron(_FinalActivationAndCost):
    """
    Sigmoid final activation with Perceptron cost function
      Use for binary classification
      Y is of dim (1, number of samples) and each element is a label: 0 or 1

    NOTE: the perceptron algorithm is effectively the limit of Cross Entropy as learning_rate -> infinity
        it is invariant to constant scaling of the learning_rate
        and Cross Entropy is a strictly better algorithm (I think?)

    NOTE: furthermore, (this isn't too important but...) the cost is linearly scaled by the learning_rate

    :param Y: labels
    :param ZL: final layer linear output
    :return:
        cost: ...
        AL: final activation function output
        dZL: partial derivative of cost w.r.t. the final linear output

    """
    def __init__(self):
        warnings.warn('the perceptron algorithm is effectively the limit of Cross Entropy as learning_rate -> infinity'
                      '\n it is invariant to constant scaling of the learning_rate'
                      '\n and Cross Entropy is a strictly better algorithm (I think?)')

    def final_activation(self, ZL):
        return sigmoid(ZL)

    def final_activation_and_cost(self, Y, ZL):
        assert Y.shape == ZL.shape, 'inconsistent shapes %s != %s' % (str(Y.shape), str(ZL.shape))

        m = Y.shape[1]

        # compute final activation
        AL = self.final_activation(ZL)

        Y_prediction = ZL > 0
        misclassified_indicator = np.logical_xor(Y, Y_prediction)

        # compute cost
        cost = 1 / m * np.sum(np.abs(ZL) * misclassified_indicator)

        # compute partial derivative
        dZL = -(2 * Y - 1) * misclassified_indicator

        return cost, AL, dZL

class SoftmaxCrossEntropy(_FinalActivationAndCost):
    """
    Softmax final activation with Cross Entropy cost function
      Use for multi-class classification
      Y is of dim (number of classes, number of samples) and each column is a one-hot vector
    :param Y: labels
    :param ZL: final layer linear output
    :return:
        cost: ...
        AL: final activation function output
        dZL: partial derivative of cost w.r.t. the final linear output
    """
    def final_activation(self, ZL):
        return softmax(ZL)

    def final_activation_and_cost(self, Y, ZL):
        assert Y.shape == ZL.shape, 'inconsistent shapes %s != %s' % (str(Y.shape), str(ZL.shape))

        m = Y.shape[1]

        # compute final activation
        AL = self.final_activation(ZL)

        # compute cost
        cost = -1 / m * np.sum(Y * stable_log(AL))

        # compute partial derivative
        dZL = AL - Y

        return cost, AL, dZL

class LinearMinimumSquareError(_FinalActivationAndCost):
    """
    Linear final activation with Mean Squared Error cost function
      Use for linear regression
      Y is of dim (output dim, number of samples)
    :param Y: labels
    :param ZL: final layer linear output
    :return:
        cost: ...
        AL: final activation function output
        dZL: partial derivative of cost w.r.t. the final linear output
    """
    def final_activation(self, ZL):
        return ZL

    def final_activation_and_cost(self, Y, ZL):
        assert Y.shape == ZL.shape, 'inconsistent shapes %s != %s' % (str(Y.shape), str(ZL.shape))

        m = Y.shape[1]

        # compute final activation (linear)
        AL = ZL

        # compute cost
        cost = 1 / m * np.sum((AL - Y) ** 2)

        # compute partial derivative
        dZL = 2 * (AL - Y)

        return cost, AL, dZL

#TODO lamda function? how to generate a function that only takes (Y, A, params) as inputs
def compute_cost_with_regularization(Y, A, params, cost_function, lambd):
    cost = cost_function(A, Y)

    m = Y.shape[1]

    # select W's from params
    all_Ws = [val for (key, val) in params.items() if key.startswith('W')]

    L2_regularization_cost = (lambd / (2 * m)) * np.sum([np.sum(np.square(W)) for W in all_Ws])

    cost = cost + L2_regularization_cost

    return cost
