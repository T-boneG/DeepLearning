"""
Example of Binary Classification
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage

"""Data Preparation"""

# local imports
import os, sys
# print(os.path.abspath(os.path.dirname(sys.argv[0])))
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(sys.argv[0])))[0])
from deep_learning.linear_disciminant_function import *

def load_dataset():
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

def show_example_image(index):
    assert(index >= 0 and index <= len(train_set_x_orig))
    plt.imshow(train_set_x_orig[index])
    print ("y = " + str(train_set_y[:, index]) + ", it's a '"
           + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
    plt.show()

# Example of a picture
# show_example_image(1)

def preprocess(train_set_x_orig, test_set_x_orig):
    # Reshape the training and test examples
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # Standardize the data
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.

    return train_set_x, test_set_x

train_set_x, test_set_x = preprocess(train_set_x_orig, test_set_x_orig)

"""Classifier Training and Analysis"""

n_x = train_set_x.shape[0]
n_y = 1

def train_and_plot(clf, label, train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate):
    print('training...')
    costs = clf.fit(train_set_x, train_set_y, num_iterations=num_iterations, learning_rate=learning_rate)

    train_accuracy = clf.score(train_set_x, train_set_y)
    test_accuracy = clf.score(test_set_x, test_set_y)

    print('Training Accuracy: %6.2f%%' % (100 * train_accuracy))
    print('Test Accuracy:     %6.2f%%' % (100 * test_accuracy))

    plt.plot(costs, label=label)
    plt.xlabel('number of iterations')
    plt.ylabel('cost')
    plt.title('Learning rate = ' + str(learning_rate))

plt.figure()

clf = LinearDiscriminantFunction(n_x, n_y, logistic_regression_model)
learning_rate = 0.005
num_iterations = 1000
label = 'LogRegLDF'
train_and_plot(clf, label, train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate)

clf = LinearDiscriminantFunction(n_x, n_y, perceptron_model)
learning_rate = 0.005
num_iterations = 1000
label = 'PerceptronLDF'
# train_and_plot(clf, label, train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate)

plt.title(('Comparison of Classifiers'))
plt.legend(loc='best')
plt.show()
