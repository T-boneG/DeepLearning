"""
Activation functions and their derivatives.

"""

from __future__ import division
import numpy as np

#TODO write these functions

def linear_af():
    pass

def sigmoid_af(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size

    Return:
    a -- sigmoid(z)
    """

    a = 1 / (1 + np.exp(-z))

    return a

def tanh_af():
    pass

def relu_af():
    pass

def linear_derivative():
    pass

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """

    s = sigmoid_af(x)
    ds = s * (1 - s)

    return ds

def tanh_derivative():
    pass

def relu_derivative():
    pass