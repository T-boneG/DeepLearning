"""
Activation functions and their derivatives.

"""

from __future__ import division
import numpy as np

#TODO verify this is correct
def linear_af(Z):
    A = Z
    cache = Z

    return A, cache

def sigmoid_af(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

#TODO implement this
def tanh_af(Z):
    raise NotImplementedError

    A = None
    cache = Z

    return A, cache

def relu_af(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z

    return A, cache

#TODO verify this is correct
def linear_backward(dA, cache):
    Z = cache

    dZ = np.array(dA, copy=True)

    assert (dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ

#TODO implement this
def tanh_backward(dA, cache):
    raise NotImplementedError

    assert (dZ.shape == Z.shape)

    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

#TODO assert that there is a matching '_backward' for each '_af'
# _activation_functions = [x.rstrip('_af') for x in dir(activation_functions) if x.endswith('_af')]