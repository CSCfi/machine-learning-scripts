
# coding: utf-8

# # Dogs-vs-cats classification with CNNs
# 
# In this script, we'll train a convolutional neural network (CNN,
# ConvNet) to classify images of dogs from images of cats using Keras
# (version $\ge$ 2 is required). This script is largely based on the
# blog post [Building powerful image classification models using very
# little data]
# (https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
# by François Chollet.
# 
# **Note that using a GPU with this script is highly recommended.**
# 
# First, the needed imports. Keras tells us which backend (Theano,
# Tensorflow, CNTK) it will be using.

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D 
from keras.preprocessing.image import (ImageDataGenerator, array_to_img, 
                                      img_to_array, load_img)
from keras import applications, optimizers

from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

import numpy as np
import sys

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

# ## Data
# 
# The training dataset consists of 2000 images of dogs and cats, split
# in half.  In addition, the validation set consists of 1000 images,
# and the test set of 22000 images.

datapath = "/wrk/makoskel/dogs-vs-cats/train-2000"
(nimages_train, nimages_validation, nimages_test) = (2000, 1000, 22000)

input_image_size = (150, 150)

noopgen = ImageDataGenerator(rescale=1./255)

batch_size = 25

print('Test: ', end="")
test_generator = noopgen.flow_from_directory(
        datapath+'/test',  
        target_size=input_image_size,
        batch_size=batch_size,
        class_mode='binary')

# ### Initialization

if len(sys.argv)<2:
    print('ERROR: model file missing')
    sys.exit()
    
model = load_model(sys.argv[1])

print(model.summary())

# ### Inference

workers = 14
use_multiprocessing = True

print('Evaluating model', sys.argv[1], 'with', workers,
      'workers, use_multiprocessing is', use_multiprocessing)

scores = model.evaluate_generator(test_generator,
                                  steps=nimages_test // batch_size,
                                  use_multiprocessing=use_multiprocessing,
                                  workers=workers)

print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
