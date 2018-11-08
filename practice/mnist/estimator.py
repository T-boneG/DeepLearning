"""
TensorFlow Estimator

"""
from __future__ import absolute_import, division, print_function

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""quick settings toggle"""

load_saved_model = False
# load_saved_model = True

m_train = 1000
# m_train = -1

"""preparation stuff"""

# the name of this file
base_filename = os.path.basename(sys.argv[0]).split('.')[0]

saved_model_path = 'models/%s_%02d.h5' % (base_filename, 1)

"""load the data"""

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = (x / 255.0 for x in (x_train, x_test))
y_train, y_test = (y.astype(int) for y in (y_train, y_test))   # necessary to cast for eager execution?

# let's use a subset
x_train = x_train[:m_train, :]
y_train = y_train[:m_train]

"""create/load model"""

if load_saved_model:
    # TODO load a model
    model = None

    _, accuracy = model.evaluate(x_train, y_train)
    print('Train Accuracy: %4.2f%%' % (100 * accuracy))
    _, accuracy = model.evaluate(x_test, y_test)
    print('Test Accuracy:  %4.2f%%' % (100 * accuracy))

else:
    # TODO continue here
    model = tf.estimator.DNNClassifier()

    # TODO save the model

    _, accuracy = model.evaluate(x_train, y_train)
    print('Train Accuracy: %4.2f%%' % (100 * accuracy))
    _, accuracy = model.evaluate(x_test, y_test)
    print('Test Accuracy:  %4.2f%%' % (100 * accuracy))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), history.history['loss'], label='loss')
    plt.plot(range(1, NUM_EPOCHS + 1), history.history['val_loss'], label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.subplot(2, 1, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), history.history['acc'], label='acc')
    plt.plot(range(1, NUM_EPOCHS + 1), history.history['val_acc'], label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
