#!/usr/bin/env python
"""
activation_functions.py
Activation Functions that are typically used as hidden activations

This file contains a collection of classes that inherit from an abstract _HiddenActivation class.
 Each class implements an activation function: forward(Z) and its respective backward function: backward(dA, Z) where:
  Z - final linear activation
  dA - the derivative of the cost w.r.t. A (the final activation)

Running this file as __main__ plots the activation functions (forward) and their derivatives (backward)
"""

from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod
import numpy as np

from .utils import sigmoid

__all__ = ['LinearActivation', 'SigmoidActivation', 'TanhActivation', 'ReluActivation', 'LeakyReluActivation']

class _HiddenActivation:
    """Abstract class providing the structure for the hidden activation classes"""

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

        assert A.shape == Z.shape

        return A

    def backward(self, dA, Z):
        assert dA.shape == Z.shape

        dZ = np.array(dA, copy=True)

        assert dZ.shape == Z.shape

        return dZ

class SigmoidActivation(_HiddenActivation):
    """
    Sigmoid Activation Function
    """
    def forward(self, Z):
        A = sigmoid(Z)

        assert A.shape == Z.shape

        return A

    def backward(self, dA, Z):
        assert dA.shape == Z.shape

        A = sigmoid(Z)
        dZ = dA * A * (1 - A)

        assert dZ.shape == Z.shape

        return dZ

class TanhActivation(_HiddenActivation):
    """
    Hyperbolic Tangent Activation Function
    """
    def forward(self, Z):
        A = np.tanh(Z)

        assert A.shape == Z.shape

        return A

    def backward(self, dA, Z):
        assert dA.shape == Z.shape

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

        assert A.shape == Z.shape

        return A

    def backward(self, dA, Z):
        assert dA.shape == Z.shape

        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        assert dZ.shape == Z.shape

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
        assert dA.shape == Z.shape

        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = self.leak * dZ[Z <= 0]

        assert dZ.shape == Z.shape

        return dZ

if __name__ == '__main__':
    """Plot Activation Functions and their derivatives"""

    import matplotlib.pyplot as plt
    import sys

    this_module = sys.modules[__name__]
    activation_function_names = [x for x in __all__ if x[0].isupper()]

    Z = np.linspace(-5, 5, 1000)
    dA = np.ones(Z.shape)

    for af_name in activation_function_names:
        Activation = getattr(this_module, af_name)

        # plot activation (forward) function
        plt.subplot(2, 1, 1)
        A = Activation().forward(Z)
        plt.plot(Z, A, label=af_name)

        # plot derivative (backward) function
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
