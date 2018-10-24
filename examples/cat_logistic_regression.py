"""
Example of Logistic Regression
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage

#################################################
# Data Preparation
#################################################

# local imports
import os, sys
# print(os.path.abspath(os.path.dirname(sys.argv[0])))
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(sys.argv[0])))[0])
from deep_learning.regression import LogisticRegression

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
show_example_image(1)

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standardize the data
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

input_dim = train_set_x.shape[0]

#################################################
# Classifier Training and Analysis
#################################################

# Create classifier
clf = LogisticRegression(input_dim)

learning_rate = 0.005
num_iterations = 8000

print('training...')
costs = clf.fit(train_set_x, train_set_y, num_iterations=num_iterations, learning_rate=learning_rate)

train_accuracy = clf.score(train_set_x, train_set_y)
test_accuracy= clf.score(test_set_x, test_set_y)

print('Training Accuracy: %6.2f%%' % (100 * train_accuracy))
print('Test Accuracy:     %6.2f%%' % (100 * test_accuracy))

plt.plot(costs)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.title('Learning rate = ' + str(learning_rate))
plt.show()

#################################################
# Effects of Learning Rate
#################################################

learning_rates = [0.01, 0.001, 0.0001]
num_iterations = 1500

print('training multiple learning rates...')
plt.figure()
for cur_learning_rate in learning_rates:
    new_clf = LogisticRegression(input_dim)

    cur_costs = new_clf.fit(train_set_x, train_set_y, num_iterations=num_iterations, learning_rate=cur_learning_rate)

    plt.plot(cur_costs, label=str(cur_learning_rate))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

#################################################
# Process an Arbitrary Image (of Greg the Cat)
#################################################
num_px = train_set_x_orig.shape[1]
my_image = "tall_cat.jpg"   # change this to the name of your image file
fname = "data/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
print(clf.predict(my_image))
my_predicted_image, _ = clf.predict(my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \""
      + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
plt.show()