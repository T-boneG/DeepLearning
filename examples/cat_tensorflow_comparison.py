"""
Tensorflow Example

Uses the same cat image dataset to compare this libraries logistic regression and neural network implementations with a
tensorflow neural network
"""

from __future__ import division

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

num_iterations = 3000

learning_rate = 0.002

# layer_dims = [input_dim, 5, 1]
# layer_dims = [input_dim, 12, 5, 1]
layer_dims = [input_dim, 128, 12, 5, 1]

print('training logistic regression...')
clf = LogisticRegression(input_dim)
costs = clf.fit(train_set_x, train_set_y, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)
print('Logistic Regression Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Logistic Regression Test Accuracy:     %6.2f%%' % (100 * test_accuracy))
plt.plot(costs, label='LR')

print('training neural network...')
clf = NeuralNetwork(layer_dims)
costs = clf.fit(train_set_x, train_set_y, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True)
train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)
print('Neural Network Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Neural Network Test Accuracy:     %6.2f%%' % (100 * test_accuracy))
plt.plot(costs, label='NN: ' + str(layer_dims[1::]))


#################################################
print('training tensorflow neural network...')

(n_x, m) = train_set_x.shape
n_y = train_set_y.shape[0]
L = len(layer_dims) - 1

# Create Placeholders
X = tf.placeholder(tf.float32, shape=[n_x, None])
Y = tf.placeholder(tf.float32, shape=[n_y, None])

# Initialize parameters
parameters = {}
for i in range(L):
    parameters['W' + str(i+1)] = tf.get_variable('W' + str(i+1), [layer_dims[i+1], layer_dims[i]],
                                                 initializer=tf.contrib.layers.xavier_initializer())
    parameters['b' + str(i+1)] = tf.get_variable('b' + str(i + 1), [layer_dims[i+1], 1],
                                                 initializer=tf.zeros_initializer())

# Forward propagation
A_prev = X
for i in range(L):
    W = parameters['W' + str(i+1)]
    b = parameters['b' + str(i+1)]
    Z = tf.add(tf.matmul(W, A_prev), b)
    if i != L-1:
        A_prev = tf.nn.relu(Z)
ZL = Z

# Cost function: Add cost function to tensorflow graph
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(ZL), labels=tf.transpose(Y)))

# Back propagation: Define the tensorflow optimizer. Use an AdamOptimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize all the variables
init = tf.global_variables_initializer()
costs = []

# Start the session to compute the tensorflow graph
with tf.Session() as sess:
    # Run the initialization
    sess.run(init)

    # Do the training loop
    for epoch in range(num_iterations):
        _, epoch_cost = sess.run([optimizer, cost], feed_dict={X: train_set_x, Y: train_set_y})

        # Print the cost every epoch
        if epoch % 1000 == 0:
            print ('%5d: %7.4f' % (epoch, epoch_cost))

        costs.append(epoch_cost)

    correct_prediction = tf.equal(tf.to_float(tf.greater(ZL, 0)), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print('Tensorflow NN Training Accuracy: %6.2f%%' % (100 * accuracy.eval({X: train_set_x, Y: train_set_y})))
    print('Tensorflow NN Test Accuracy:     %6.2f%%' % (100 * accuracy.eval({X: test_set_x, Y: test_set_y})))

plt.plot(costs, label='TF-NN: ' + str(layer_dims[1::]))
#################################################


plt.ylabel('cost')
plt.legend(loc='best')
plt.xlim([0, num_iterations])
plt.xlabel('number of iterations')
plt.show()