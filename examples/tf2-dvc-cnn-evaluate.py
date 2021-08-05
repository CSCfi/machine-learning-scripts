
# coding: utf-8

# # Dogs-vs-cats classification with CNNs
# 
# In this notebook, we'll train a convolutional neural network (CNN,
# ConvNet) to classify images of dogs from images of cats using
# TensorFlow 2.0 / Keras. This notebook is largely based on the blog
# post [Building powerful image classification models using very
# little data]
# (https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
# by François Chollet.
# 
# **Note that using a GPU with this notebook is highly recommended.**
# 
# First, the needed imports.

import os, datetime, sys
import random
import pathlib

import tensorflow as tf
from tensorflow import keras
import numpy as np

print('Using Tensorflow version:', tf.__version__,
      'Keras version:', keras.__version__,
      'backend:', keras.backend.backend())

# ## Data
# 
# The test set consists of 22000 images.

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2003747/data/"

datapath = os.path.join(DATADIR, "dogs-vs-cats/train-2000/")
assert os.path.exists(datapath), "Data not found at "+datapath

nimages = dict()
nimages['test'] = 22000

# ### Image paths and labels

def get_paths(dataset):
    data_root = pathlib.Path(datapath+dataset)
    image_paths = list(data_root.glob('*/*'))
    image_paths = [str(path) for path in image_paths]
    image_count = len(image_paths)
    assert image_count == nimages[dataset], "Found {} images, expected {}".format(image_count, nimages[dataset])
    return image_paths

image_paths = dict()
image_paths['test'] = get_paths('test')

label_names = sorted(item.name for item in pathlib.Path(datapath+'train').glob('*/')
                     if item.is_dir())
label_to_index = dict((name, index) for index,name in enumerate(label_names))

def get_labels(dataset):
    return [label_to_index[pathlib.Path(path).parent.name]
            for path in image_paths[dataset]]
    
image_labels = dict()
image_labels['test'] = get_labels('test')

# ### Data augmentation
# 
# We need to resize all test images to a fixed size. Here we'll use
# 160x160 pixels.
# 
# Unlike the training images, we do not apply any random
# transformations to the test images.

INPUT_IMAGE_SIZE = [160, 160, 3]

def preprocess_image(image, augment):
    image = tf.image.decode_jpeg(image, channels=3)
    if augment:
        image = tf.image.resize(image, [256, 256])
        image = tf.image.random_crop(image, INPUT_IMAGE_SIZE)
        if random.random() < 0.5:
            image = tf.image.flip_left_right(image)
    else:
        image = tf.image.resize(image, INPUT_IMAGE_SIZE[:2])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_augment_image(path, label):
    image = tf.io.read_file(path)
    return preprocess_image(image, True), label

def load_and_not_augment_image(path, label):
    image = tf.io.read_file(path)
    return preprocess_image(image, False), label

# ### TF Datasets
# 
# Let's now define our TF Dataset
# (https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/Dataset#class_dataset)
# for the test data. First the Datasets contain the filenames of the
# images and the corresponding labels.

test_dataset = tf.data.Dataset.from_tensor_slices((image_paths['test'],
                                                   image_labels['test']))

# We then map() the filenames to the actual image data and decode the images.

BATCH_SIZE = 32

test_dataset = test_dataset.map(load_and_not_augment_image, num_parallel_calls=10)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# ### Initialization

if len(sys.argv)<2:
    print('ERROR: model file missing')
    sys.exit()
    
model = keras.models.load_model(sys.argv[1])

print(model.summary())

# ### Inference

print('Evaluating model', sys.argv[1])
scores = model.evaluate(test_dataset, verbose=2)
print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
