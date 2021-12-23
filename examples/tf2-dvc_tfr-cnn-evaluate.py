# coding: utf-8

# # Dogs-vs-cats classification with CNNs
# 
# This script is used to evaluate neural networks trained using
# TensorFlow 2 / Keras to classify images of dogs from images of cats.
# 
# **Note that using a GPU with this notebook is highly recommended.**
# 
# First, the needed imports.

import os, sys
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
    DATADIR = "/scratch/project_2005299/data/"

print('Using DATADIR', DATADIR)
datapath = os.path.join(DATADIR, "dogs-vs-cats/train-2000/tfrecord/")
assert os.path.exists(datapath), "Data not found at "+datapath

# ### Data loading
#
# We now define a function to load the images from TFRecord
# entries. Also we need to resize the images to a fixed size
# (INPUT_IMAGE_SIZE).

INPUT_IMAGE_SIZE = [256, 256]

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    return tf.image.resize(image, INPUT_IMAGE_SIZE)

feature_description = {
    "image/encoded": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/height": tf.io.FixedLenFeature((), tf.int64, default_value=0),
    "image/width": tf.io.FixedLenFeature((), tf.int64, default_value=0),
    "image/colorspace": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/channels": tf.io.FixedLenFeature((), tf.int64, default_value=0),
    "image/format": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/filename": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/class/label": tf.io.FixedLenFeature((), tf.int64, default_value=0),
    "image/class/text": tf.io.FixedLenFeature((), tf.string, default_value="")}

def load_image(example_proto):
    ex = tf.io.parse_single_example(example_proto, feature_description)
    return (preprocess_image(ex["image/encoded"]), ex["image/class/label"]-1)

# ### TF Datasets
#
# Let's now define our TF Datasets for training and validation data.
# We use the TFRecordDataset class, which reads the data records from
# multiple TFRecord files.

test_filenames = [datapath+"test-{0:05d}-of-00022".format(i)
                   for i in range(22)]
test_dataset = tf.data.TFRecordDataset(test_filenames)

# We then map() the filenames to the actual image data and decode the
# images.

BATCH_SIZE = 32

test_dataset = test_dataset.map(load_image,
                                  num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# ### Initialization

if len(sys.argv)<2:
    print('ERROR: model file missing')
    sys.exit()

print('Loading model', sys.argv[1])
model = keras.models.load_model(sys.argv[1])
print(model.summary())

# ### Inference

print('Evaluating model')
scores = model.evaluate(test_dataset, verbose=2)
print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
