"""

"""

from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
from utils import one_hot_inverse

__all__ = ['BinaryClassificationHelper', 'MulticlassClassificationHelper', 'LinearRegressionHelper']

def _check_inputs_no_constraints(X, Y):
    """
    No specific constraints on the input and output of the system
        -Use for linear regression problems
    """
    assert X.shape[1] == Y.shape[1], 'inconsistent number of samples for X and Y, ' \
                                     '%d != %d' % (X.shape[1], Y.shape[1])

class _ModelHelper:
    __metaclass__ = ABCMeta

    @abstractmethod
    def check_inputs(self, X, Y):
        pass

    @abstractmethod
    def prediction(self, A):
        pass

    @abstractmethod
    def score(self, Y, Y_prediction):
        pass

class BinaryClassificationHelper(_ModelHelper):
    model_type = 'binary_classification'

    def check_inputs(self, X, Y):
        _check_inputs_no_constraints(X, Y)

        assert Y.shape[0] == 1, 'invalid output vector dimension: %d. Expected: %d' % (Y.shape[0], 1)

    def prediction(self, A):
        return np.greater(A, 0.5).astype(int)

    def score(self, Y, Y_prediction):
        Y = np.squeeze(Y)
        Y_prediction = np.squeeze(Y_prediction)
        assert len(Y_prediction) == len(Y), 'inconsistent number of samples: %d != %d' % (len(Y_prediction), len(Y))

        correct = np.sum(np.equal(Y_prediction, Y))
        return correct / len(Y)

class MulticlassClassificationHelper(_ModelHelper):
    model_type = 'multiclass_classification'

    def check_inputs(self, X, Y):
        _check_inputs_no_constraints(X, Y)

        # verify Y is a set of one-hot vectors
        assert np.array_equal(np.sum(Y, axis=0), np.ones((Y.shape[1])))
        assert np.array_equal(np.sum(np.not_equal(Y, 0), axis=0), np.ones((Y.shape[1])))

    def prediction(self, A):
        """
        Convert probabilities row vector of class predictions
        :param A: matrix of softmax probabilities (number of classes, number of samples)
        :return: row vector of integer class predictions
        """
        return np.argmax(A, axis=0)

    def score(self, Y, Y_prediction):
        Y = np.squeeze(Y)
        Y_prediction = np.squeeze(Y_prediction)

        if len(Y.shape) > 1:
            Y = one_hot_inverse(Y)
        if len(Y_prediction.shape) > 1:
            Y_prediction = one_hot_inverse(Y_prediction)

        assert len(Y_prediction) == len(Y), 'inconsistent number of samples: %d != %d' % (len(Y_prediction), len(Y))

        correct = np.sum(np.equal(Y_prediction, Y))
        return correct / len(Y)

class LinearRegressionHelper(_ModelHelper):
    model_type = 'linear_regression'

    def check_inputs(self, X, Y):
        _check_inputs_no_constraints(X, Y)

    def prediction(self, A):
        return A

    def score(self, Y, Y_prediction):
        """
        Not technically a score...returns the average Mean Squared Error (MSE) over the provided samples
        """
        return np.mean(np.linalg.norm(Y - Y_prediction, axis=0)**2)
