"""
Activation functions and their derivatives.
"""

from __future__ import division
import sys
import numpy as np

__all__ = ['linear_af', 'sigmoid_af', 'tanh_af', 'relu_af', 'leaky_relu_af',
           'linear_backward', 'sigmoid_backward', 'tanh_backward', 'relu_backward', 'leaky_relu_backward']

def linear_af(Z):
    """
    Linear Activation Function
    (not really an activation function) This has the effect of reducing
    the neural network to a logistic or linear regression problem
    """
    A = np.array(Z, copy=True)
    cache = {'Z': Z}

    return A, cache

def sigmoid_af(Z):
    """
    Sigmoid Activation Function
    """
    A = 1 / (1 + np.exp(-Z))
    cache = {'Z': Z}

    return A, cache

def tanh_af(Z):
    """
    tanh Activation Function
    """
    A = np.tanh(Z)
    cache = {'Z': Z}

    return A, cache

def relu_af(Z):
    """
    Rectified Linear Unit Activation Function
    """
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = {'Z': Z}

    return A, cache

def leaky_relu_af(Z, leak=0.01):
    """
    Rectified Linear Unit Activation Function
    """
    assert (leak > 0) and (leak < 1)

    A = np.maximum(leak*Z, Z)
    assert (A.shape == Z.shape)
    cache = {'Z': Z}

    return A, cache

def linear_backward(dA, cache):
    Z = cache['Z']

    dZ = np.array(dA, copy=True)

    assert (dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA, cache):
    Z = cache['Z']

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ

def tanh_backward(dA, cache):
    Z = cache['Z']

    s = np.tanh(Z)
    dZ = dA * (1 - s**2)

    assert (dZ.shape == Z.shape)

    return dZ

def relu_backward(dA, cache):
    Z = cache['Z']
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

def leaky_relu_backward(dA, cache, leak=0.01):
    Z = cache['Z']
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = leak * dZ[Z <= 0]

    assert (dZ.shape == Z.shape)

    return dZ

this_module = sys.modules[__name__]
# assert that there is a matching '_backward' for each '_af'
activation_function_names = [x.rsplit('_', 1)[0] for x in dir(this_module) if x.endswith('_af')]
for af_name in activation_function_names:
    assert (af_name + '_backward') in dir(this_module), \
        'missing backward function for activation function %s' % af_name

# and vice versa
backward_function_names = [x.rsplit('_', 1)[0] for x in dir(this_module) if x.endswith('_backward')]
for bf_name in backward_function_names:
    assert (bf_name + '_af') in dir(this_module), \
        'missing activation function for backward function %s' % bf_name

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('testing activation functions...')

    Z = np.linspace(-5, 5, 1000)
    dA = np.ones(Z.shape)

    for af_name in activation_function_names:
        # plot activation function
        plt.subplot(2, 1, 1)
        af = getattr(this_module, af_name + '_af')
        A, cache = af(Z)
        plt.plot(Z, A, label=af_name)

        # plot backward function
        plt.subplot(2, 1, 2)
        backward = getattr(this_module, af_name + '_backward')
        dZ = backward(dA, cache)
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
