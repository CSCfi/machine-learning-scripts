
# coding: utf-8

# # Script for testing the TensorFlow 2.0 setup
# 
# This script is for testing the TensorFlow
# (https://www.tensorflow.org/) setup using the Keras API
# (https://keras.io/).  Below is a set of required imports.
# 
# No error messages should appear.  In particular, **TensorFlow 2 is
# required**.
# 
# Some warnings may appear, this should be fine.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU 
from tensorflow.keras.utils import to_categorical

from distutils.version import LooseVersion as LV

from tensorflow.keras.datasets import mnist, fashion_mnist, imdb

from sklearn.model_selection import train_test_split

import numpy as np

print('Using Tensorflow version: {}, '
      'and Keras version: {}.'.format(tf.__version__,
                                      tf.keras.__version__))
assert(LV(tf.__version__) >= LV("2.0.0"))

# Let's check if we have GPU available.

if tf.test.is_gpu_available():
    from tensorflow.python.client import device_lib
    for d in device_lib.list_local_devices():
        if d.device_type == 'GPU':
            print('GPU', d.physical_device_desc)
else:
    print('No GPU, using CPU instead.')

# ## Getting started: 30 seconds to Keras
# 
# (This section is adapted from https://keras.io/)
# 
# The core data structure of Keras is a **model**, a way to organize
# layers. The main type of model is the Sequential model, a linear
# stack of layers.
# 
# A model is initialized by calling Sequential():

model = Sequential()

# Stacking layers is as easy as .add():

model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))

# A summary of the model:

print(model.summary())

# Once your model looks good, configure its learning process with
# .compile():

model.compile(loss='categorical_crossentropy', 
              optimizer='sgd', 
              metrics=['accuracy'])

# You can now begin training your model with .fit().  Let's generate
# some random data and use it to train the model:

X_train = np.random.rand(128, 100)
Y_train = to_categorical(np.random.randint(10, size=128))

model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=2);

# Evaluate your performance on test data with .evaluate():

X_test = np.random.rand(64, 100)
Y_test = to_categorical(np.random.randint(10, size=64))

loss, acc = model.evaluate(X_test, Y_test, batch_size=32)
print()
print('loss:', loss, 'acc:', acc)
