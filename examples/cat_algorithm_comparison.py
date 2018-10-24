"""
NN cat example

Compares logistic regression with multiple neural network architectures with multiple learning rates
"""

from __future__ import division

import h5py
import matplotlib.pyplot as plt
import numpy as np

#################################################
# Data Preparation
#################################################

# local imports
import os, sys
# print(os.path.abspath(os.path.dirname(sys.argv[0])))
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(sys.argv[0])))[0])
from deep_learning.regression import LogisticRegression
from deep_learning.neural_network import NeuralNetwork

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

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standardize the data
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

input_dim = train_set_x.shape[0]

#################################################
# Learning
#################################################

num_iterations = 2000

learning_rate = 0.005

print('training logistic regression...')
clf = LogisticRegression(input_dim)
costs = clf.fit(train_set_x, train_set_y, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)
print('Logistic Regression Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Logistic Regression Test Accuracy:     %6.2f%%' % (100 * test_accuracy))
plt.subplot(2,1,1)
plt.plot(costs, label='LR')

print('training neural network...')
layers_dims = [input_dim, 12, 1]
clf = NeuralNetwork(layers_dims)
costs = clf.fit(train_set_x, train_set_y, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)
print('Neural Network Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Neural Network Test Accuracy:     %6.2f%%' % (100 * test_accuracy))
plt.subplot(2,1,1)
plt.plot(costs, label='NN: ' + str(layers_dims[1::]))

print('training neural network...')
layers_dims = [input_dim, 12, 5, 1]
clf = NeuralNetwork(layers_dims)
costs = clf.fit(train_set_x, train_set_y, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)
print('Neural Network Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Neural Network Test Accuracy:     %6.2f%%' % (100 * test_accuracy))
plt.subplot(2,1,1)
plt.plot(costs, label='NN: ' + str(layers_dims[1::]))

print('training neural network...')
layers_dims = [input_dim, 12, 5, 1]
clf = NeuralNetwork(layers_dims)
costs = clf.fit(train_set_x, train_set_y, lambd=0.1,
                num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)
print('Neural Network Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Neural Network Test Accuracy:     %6.2f%%' % (100 * test_accuracy))
plt.subplot(2,1,1)
plt.plot(costs, label='NN w L2reg: ' + str(layers_dims[1::]))

print('training neural network...')
layers_dims = [input_dim, 12, 5, 1]
clf = NeuralNetwork(layers_dims)
costs = clf.fit(train_set_x, train_set_y, keep_prob=0.7,
                num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)
print('Neural Network Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Neural Network Test Accuracy:     %6.2f%%' % (100 * test_accuracy))
plt.subplot(2,1,1)
plt.plot(costs, label='NN w dropout: ' + str(layers_dims[1::]))

learning_rate = 0.01

print('training logistic regression...')
clf = LogisticRegression(input_dim)
costs = clf.fit(train_set_x, train_set_y, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)
print('Logistic Regression Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Logistic Regression Test Accuracy:     %6.2f%%' % (100 * test_accuracy))
plt.subplot(2,1,2)
plt.plot(costs, label='LR')

print('training neural network...')
layers_dims = [input_dim, 12, 1]
clf = NeuralNetwork(layers_dims)
costs = clf.fit(train_set_x, train_set_y, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)
print('Neural Network Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Neural Network Test Accuracy:     %6.2f%%' % (100 * test_accuracy))
plt.subplot(2,1,2)
plt.plot(costs, label='NN: ' + str(layers_dims[1::]))

print('training neural network...')
layers_dims = [input_dim, 12, 5, 1]
clf = NeuralNetwork(layers_dims)
costs = clf.fit(train_set_x, train_set_y, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)
print('Neural Network Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Neural Network Test Accuracy:     %6.2f%%' % (100 * test_accuracy))
plt.subplot(2,1,2)
plt.plot(costs, label='NN: ' + str(layers_dims[1::]))

print('training neural network...')
layers_dims = [input_dim, 12, 5, 1]
clf = NeuralNetwork(layers_dims)
costs = clf.fit(train_set_x, train_set_y, lambd=0.1,
                num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)
print('Neural Network Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Neural Network Test Accuracy:     %6.2f%%' % (100 * test_accuracy))
plt.subplot(2,1,2)
plt.plot(costs, label='NN w L2reg: ' + str(layers_dims[1::]))

print('training neural network...')
layers_dims = [input_dim, 12, 5, 1]
clf = NeuralNetwork(layers_dims)
costs = clf.fit(train_set_x, train_set_y, keep_prob=0.7,
                num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)
print('Neural Network Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Neural Network Test Accuracy:     %6.2f%%' % (100 * test_accuracy))
plt.subplot(2,1,2)
plt.plot(costs, label='NN w dropout: ' + str(layers_dims[1::]))



for i in range(1, 3):
    plt.subplot(2,1,i)
    plt.ylabel('cost')
    plt.legend(loc='best')
    plt.xlim([0, num_iterations])
plt.xlabel('number of iterations')
plt.show()