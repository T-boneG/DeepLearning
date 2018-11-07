"""
TensorFlow Estimator

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.estimator.canned.baseline import BaselineClassifier
from tensorflow.python.estimator.canned.dnn import DNNClassifier

clf = BaselineClassifier()
