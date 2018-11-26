#!/usr/bin/env python
"""
Example of Multi-class classification
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

# local imports
import os, sys
# print(os.path.abspath(os.path.dirname(sys.argv[0])))
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(sys.argv[0])))[0])
from deep_learning.utils import one_hot, one_hot_inverse, explore_data
from deep_learning.linear_disciminant_function import LinearDiscriminantFunction
from deep_learning.neural_network import NeuralNetwork
from deep_learning import cost_functions, activation_functions

"""load and view data"""

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train) / 255.0, np.array(x_test) / 255.0

def view_data(x_train, y_train):
    n_x = np.product(x_train.shape[1:])
    n_y = 1
    m_train = x_train.shape[0]
    m_test = x_test.shape[0]
    print('n_x: %d' % n_x)
    print('n_y: %d' % n_y)
    print('m_train: %d' % m_train)
    print('m_test:  %d' % m_test)

    for i in range(4):
        plt.subplot(2,2,i+1)
        index = i
        plt.imshow(x_train[index,:], cmap='binary')
        plt.title('Y label: %s' % str(y_train[index]))
    plt.show()
view_data(x_train, y_train)

explore_data((x_train, y_train), (x_test, y_test))

"""preprocess data"""

x_train = x_train.reshape(x_train.shape[0], -1).T
x_test = x_test.reshape(x_test.shape[0], -1).T
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

n_x = x_train.shape[0]
n_y = y_train.shape[0]

# use a smaller training set
x_train = x_train[:, :500]
y_train = y_train[:, :500]

def train_and_plot(clf, label, train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate):
    print('training...')
    costs = clf.fit(train_set_x, train_set_y, learning_rate=learning_rate, num_epochs=num_iterations, print_cost=False)

    train_accuracy = clf.score(train_set_x, one_hot_inverse(train_set_y))
    test_accuracy = clf.score(test_set_x, one_hot_inverse(test_set_y))

    print('Training Accuracy: %6.2f%%' % (100 * train_accuracy))
    print('Test Accuracy:     %6.2f%%' % (100 * test_accuracy))

    plt.plot(costs, label=label)
    plt.xlabel('number of iterations')
    plt.ylabel('cost')
    plt.title('Learning rate = ' + str(learning_rate))

plt.figure()

learning_rate = 0.01
num_iterations = 2000

softmax_regression_LDF_model = {
    'model_type': 'multiclass_classification',
    'final_activation_and_cost': cost_functions.SoftmaxCrossEntropy()
}
clf = LinearDiscriminantFunction(n_x, n_y, softmax_regression_LDF_model)
label = 'SoftmaxLDF'
train_and_plot(clf, label, x_train, y_train, x_test, y_test, num_iterations, learning_rate)

layer_dims = [n_x, 12, n_y]
softmax_regression_NN_model = {
    'model_type': 'multiclass_classification',
    'hidden_activation': activation_functions.ReluActivation(),
    'final_activation_and_cost': cost_functions.SoftmaxCrossEntropy()
}
clf = NeuralNetwork(layer_dims, softmax_regression_NN_model)
label = 'SoftmaxNN'
train_and_plot(clf, label, x_train, y_train, x_test, y_test, num_iterations, learning_rate)

plt.legend(loc='best')
plt.show()
