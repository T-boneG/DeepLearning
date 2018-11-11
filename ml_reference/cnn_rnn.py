#!/usr/bin/env python
"""
CNN-LSTM code example for processing very large/long 1D time-series

source:
https://medium.com/@alexrachnog/deep-learning-the-final-frontier-for-signal-processing-and-time-series-analysis-734307167ad6
"""
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Embedding(20000, 32, input_length=100))
model.add(layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=3))
model.add(layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=3))
model.add(layers.LSTM(50, return_sequences=True))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.45))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
