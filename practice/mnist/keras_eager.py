"""
TensorFlow & Keras with Eager Execution

 Write eager-compatible code:
 tf.keras.layers
 tf.keras.Model
 tf.contrib.summary
 tfe.metrics
 (use object-based saving)
 ...then your model will worth with both eager and graph

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, sys

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model

"""quick settings toggle"""

# tf.enable_eager_execution()

# load_saved_model = False
load_saved_model = True

m_train = 1000
# m_train = -1

BATCH_SIZE = 32
NUM_EPOCHS = 5

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
    model = load_model(saved_model_path)

    if not model._is_compiled:
        print('compiling saved model...')
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])

    # print(model.summary())

    _, accuracy = model.evaluate(x_train, y_train)
    print('Train Accuracy: %4.2f%%' % (100 * accuracy))
    _, accuracy = model.evaluate(x_test, y_test)
    print('Test Accuracy:  %4.2f%%' % (100 * accuracy))

else:
    model = Sequential([layers.Flatten(input_shape=(28, 28,)),
                        layers.Dense(512, activation=tf.nn.relu),
                        layers.Dropout(0.2),
                        # layers.Dense(512, bias_regularizer=tf.keras.regularizers.l2(0.01)),
                        layers.Dense(10, activation=tf.nn.softmax)])

    model.compile(
        # optimizer=tf.train.GradientDescentOptimizer(),
        # optimizer=tf.train.RMSPropOptimizer(),
        # optimizer=tf.train.AdamOptimizer(),
        # optimizer=tf.keras.optimizers.SGD(),
        # optimizer=tf.keras.optimizers.RMSprop(),
        optimizer=tf.keras.optimizers.Adam(),
        # loss=tf.keras.losses.categorical_crossentropy,
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_split=0.1,
                        verbose=1)

    model.save(saved_model_path,
               overwrite=False,
               include_optimizer=True)

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
