"""
This library's NN implementation

"""
from __future__ import absolute_import, division, print_function

import os, sys
sys.path.append(os.path.split(os.path.split(os.path.abspath(os.path.dirname(sys.argv[0])))[0])[0])

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from deep_learning import activation_functions, cost_functions
from deep_learning.neural_network import NeuralNetwork
from deep_learning.utils import one_hot, one_hot_inverse

"""quick settings toggle"""

load_saved_model = False
# load_saved_model = True

m_train = 1000
# m_train = -1

learning_rate = 0.001

BATCH_SIZE = 32
NUM_EPOCHS = 1000

"""preparation stuff"""

# the name of this file
base_filename = os.path.basename(sys.argv[0]).split('.')[0]

saved_model_path = 'models/%s_%02d.h5' % (base_filename, 1)

"""load the data"""

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:m_train, :]
y_train = y_train[:m_train]

"""preprocess"""

x_train, x_test = (x.reshape(x.shape[0], -1).T / 255.0 for x in (x_train, x_test))
y_train, y_test = (y.reshape(y.shape[0], -1).T for y in (y_train, y_test))
y_train, y_test = one_hot(y_train, 10), one_hot(y_test, 10)

n_x, n_y = x_train.shape[0], y_train.shape[0]

"""create/load model"""

model_specs = softmax_regression_NN_model = {
    'model_type': 'multiclass_classification',
    'hidden_activation': activation_functions.ReluActivation(),
    'final_activation_and_cost': cost_functions.SoftmaxCrossEntropy()
}

layer_dims = [n_x, 512, 10, n_y]

model = NeuralNetwork(layer_dims, model_specs)

if load_saved_model:
    model.load(saved_model_path)

    accuracy = model.score(x_train, y_train)
    print('Train Accuracy: %4.2f%%' % (100 * accuracy))
    accuracy = model.score(x_test, y_test)
    print('Test Accuracy:  %4.2f%%' % (100 * accuracy))

    print(model.summary())

else:
    costs = model.fit(x_train, y_train,
                      num_iterations=NUM_EPOCHS,
                      learning_rate=learning_rate,
                      print_cost=True)

    model.save(saved_model_path, overwrite=False)

    accuracy = model.score(x_train, y_train)
    print('Train Accuracy: %4.2f%%' % (100 * accuracy))
    accuracy = model.score(x_test, y_test)
    print('Test Accuracy:  %4.2f%%' % (100 * accuracy))

    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), costs)
    plt.ylabel('cost (loss)')
    plt.xlabel('epoch')
    plt.show()
