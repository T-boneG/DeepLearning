"""
Activation functions and their derivatives.

"""

from __future__ import division
import numpy as np

#TODO
def linear_af(Z):
    return Z

def sigmoid_af(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

#TODO
def tanh_af(Z):
    raise NotImplementedError

def relu_af(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z

    return A, cache

#TODO (refer to relu_backward)
def linear_backward(X):
    raise NotImplementedError

def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ

#TODO
def tanh_backward(x):
    raise NotImplementedError

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ