"""

"""

from __future__ import division
import numpy as np

__all__ = ['stable_log', 'sigmoid', 'softmax']

def stable_log(x):
    # add epsilon for numeric stability
    eps = 1e-10
    return np.log(x + eps)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z):
    # shift Z for numeric stability
    shiftZ = Z - np.max(Z)

    exps = np.exp(shiftZ)

    return exps / np.sum(exps)


