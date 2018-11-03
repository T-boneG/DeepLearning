"""
Activation functions and their derivatives.
"""

from __future__ import division
import sys
from abc import ABCMeta, abstractmethod
import numpy as np

__all__ = ['sigmoid', 'softmax',
           'LinearActivation', 'SigmoidActivation', 'TanhActivation', 'ReluActivation', 'LeakyReluActivation']

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z):
    shiftZ = Z - np.max(Z)
    exps = np.exp(shiftZ)
    return exps / np.sum(exps)

class _HiddenActivation:
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, Z):
        pass

    @abstractmethod
    def backward(self, dA, Z):
        pass

class LinearActivation(_HiddenActivation):
    """
    Linear Activation Function
    (not really an activation function) This has the effect of reducing
    the neural network to a logistic or linear regression problem
    """
    def forward(self, Z):
        A = np.array(Z, copy=True)

        return A

    def backward(self, dA, Z):
        dZ = np.array(dA, copy=True)

        assert (dZ.shape == Z.shape)

        return dZ

class SigmoidActivation(_HiddenActivation):
    """
    Sigmoid Activation Function
    """
    def forward(self, Z):
        A = sigmoid(Z)

        return A

    def backward(self, dA, Z):
        A = sigmoid(Z)
        dZ = dA * A * (1 - A)

        assert (dZ.shape == Z.shape)

        return dZ

class TanhActivation(_HiddenActivation):
    """
    tanh Activation Function
    """
    def forward(self, Z):
        A = np.tanh(Z)

        return A

    def backward(self, dA, Z):
        A = np.tanh(Z)
        dZ = dA * (1 - A**2)

        assert dZ.shape == Z.shape

        return dZ

class ReluActivation(_HiddenActivation):
    """
    Rectified Linear Unit Activation Function
    """
    def forward(self, Z):
        A = np.maximum(0, Z)
        assert (A.shape == Z.shape)

        return A

    def backward(self, dA, Z):
        dZ = np.array(dA, copy=True)

        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ

class LeakyReluActivation(_HiddenActivation):
    """
    Leaky Rectified Linear Unit Activation Function
    """
    def __init__(self, leak=0.01):
        assert (leak >= 0) and (leak <= 1)

        self.leak = leak

    def forward(self, Z):
        A = np.maximum(self.leak * Z, Z)

        assert A.shape == Z.shape

        return A

    def backward(self, dA, Z):
        dZ = np.array(dA, copy=True)

        dZ[Z <= 0] = self.leak * dZ[Z <= 0]

        assert dZ.shape == Z.shape

        return dZ

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    this_module = sys.modules[__name__]

    activation_function_names = [x for x in __all__ if x[0].isupper()]

    Z = np.linspace(-5, 5, 1000)
    dA = np.ones(Z.shape)

    for af_name in activation_function_names:
        Activation = getattr(this_module, af_name)

        # plot activation function
        plt.subplot(2, 1, 1)
        A = Activation().forward(Z)
        plt.plot(Z, A, label=af_name)

        # plot backward function
        plt.subplot(2, 1, 2)
        dZ = Activation().backward(dA, Z)
        plt.plot(Z, dZ, label=af_name)

    plt.subplot(2, 1, 1)
    plt.xlim((Z[0], Z[-1]))
    plt.ylim((-1.1, 1.1))
    plt.title(('Activation Functions'))
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.xlim((Z[0], Z[-1]))
    plt.ylim((-0.1, 1.1))
    plt.title(('Backward Functions'))
    plt.legend(loc='best')

    plt.show()
