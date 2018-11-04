"""
Idea to try learning an FFT using a linear regression and/or NN

To compute correct energy:    X = 1/N      * fft(x)
To preserve energy (unitary): X = 1/N**0.5 * fft(x)
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# local imports
import os, sys
# print(os.path.abspath(os.path.dirname(sys.argv[0])))
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(sys.argv[0])))[0])
from deep_learning import *
import time

"""Create the function (DTFT) to be learned"""
N = 2 ** 5

def make_DTFT(N):
    DTFT = np.zeros((N, N), dtype=complex)
    base_row_twiddle = np.exp(-1j * 2 * np.pi / N * np.arange(N))
    for k in range(N):
        DTFT[k,:] = base_row_twiddle ** k
    DTFT = (1 / N**0.5) * np.concatenate((np.real(DTFT), np.imag(DTFT)))
    return DTFT

DTFT = make_DTFT(N)

# x = (1 / N**0.5) * np.random.randn(N)
# y = np.dot(DTFT, x)
# temp = DTFT + np.expand_dims(np.arange(DTFT.shape[0]), 1) * (2/N**0.5)
# plt.plot(temp[:N,:].T)
# plt.show()

"""Create the training set"""

training_size = 1000

train_set_x = (1 / N**0.5) * np.random.randn(N, training_size)
train_set_y = np.dot(DTFT, train_set_x)
test_set_x = None
test_set_y = None

"""Train a linear classifier"""

learning_rate = 0.1
num_iterations = 1000
# num_iterations = 1

n_x = train_set_x.shape[0]
n_y = train_set_y.shape[0]

print('training LDF...')
clf = LinearDiscriminantFunction(n_x, n_y, mse_linear_regression_LDF_model)
tic = time.time()
costs = clf.fit(train_set_x, train_set_y, num_iterations, learning_rate)
print('time: %fs' % (time.time() - tic))
plt.plot(np.squeeze(costs), label='LDF')
W = clf.get_parameters()['W']
b = clf.get_parameters()['b']

print('training NN-LDF...')
layer_dims = [n_x, n_y]
clf = NeuralNetwork(layer_dims, mse_linear_regression_NN_model)
tic = time.time()
costs = clf.fit(train_set_x, train_set_y, num_iterations, learning_rate)
print('time: %fs' % (time.time() - tic))
plt.plot(np.squeeze(costs), label='linearNN')
W = clf.get_parameters()['W1']
b = clf.get_parameters()['b1']

print('training wide NN...')
layer_dims = [n_x, 512, n_y]
clf = NeuralNetwork(layer_dims, mse_linear_regression_NN_model)
tic = time.time()
costs = clf.fit(train_set_x, train_set_y, num_iterations, learning_rate)
print('time: %fs' % (time.time() - tic))
plt.plot(np.squeeze(costs), label='wideNN')

print('training deep NN...')
layer_dims = [n_x, 128, 128, 128, 128, n_y]
clf = NeuralNetwork(layer_dims, mse_linear_regression_NN_model)
tic = time.time()
costs = clf.fit(train_set_x, train_set_y, num_iterations, learning_rate)
print('time: %fs' % (time.time() - tic))
plt.plot(np.squeeze(costs), label='deepNN')

print('training best? NN...')
layer_dims = [n_x, 128, n_y]
clf = NeuralNetwork(layer_dims, mse_linear_regression_NN_model)
tic = time.time()
costs = clf.fit(train_set_x, train_set_y, num_iterations, learning_rate)
print('time: %fs' % (time.time() - tic))
plt.plot(np.squeeze(costs), label='NN')

plt.ylabel('cost')
plt.xlabel('iterations')
plt.legend(loc='best')

plt.figure()
plt.subplot(2, 1, 1)
temp = DTFT + np.expand_dims(np.arange(DTFT.shape[0]), 1) * (2/N**0.5)
plt.plot(temp[:N,:].T)
plt.ylabel('DTFT')

plt.subplot(2, 1, 2)
temp = W / np.linalg.norm(W, axis=1, keepdims=True) + np.expand_dims(np.arange(W.shape[0]), 1) * (2 / N ** 0.5)
plt.plot(temp[:N, :].T)
plt.ylabel('W')
plt.show()

