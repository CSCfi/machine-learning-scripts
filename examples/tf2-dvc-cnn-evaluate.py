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

try:
    from tensorflow_hub import KerasLayer
except:
    KerasLayer = None
    print('WARNING: Package tensorflow_hub not found, models based on '
          'TF Hub components will not work.')

# ## Data
#
# The test set consists of 22000 images.

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2005299/data/"

print('Using DATADIR', DATADIR)
datapath = os.path.join(DATADIR, "dogs-vs-cats/train-2000/")
assert os.path.exists(datapath), "Data not found at "+datapath

nimages = {'test':22000}

# ### Image paths and labels

def get_paths(dataset):
    data_root = pathlib.Path(datapath+dataset)
    image_paths = list(data_root.glob('*/*'))
    image_paths = [str(path) for path in image_paths]
    image_count = len(image_paths)
    assert image_count == nimages[dataset], \
        "Found {} images, expected {}".format(image_count, nimages[dataset])
    return image_paths

image_paths = dict()
image_paths['test'] = get_paths('test')

label_names = sorted(item.name for item in
                     pathlib.Path(datapath+'train').glob('*/')
                     if item.is_dir())
label_to_index = dict((name, index) for index,name in enumerate(label_names))

def get_labels(dataset):
    return [label_to_index[pathlib.Path(path).parent.name]
            for path in image_paths[dataset]]

image_labels = dict()
image_labels['test'] = get_labels('test')

# ### Data loading
#
# We now define a function to load the images. Also we need to resize
# the images to a fixed size (INPUT_IMAGE_SIZE).

INPUT_IMAGE_SIZE = [256, 256]

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return tf.image.resize(image, INPUT_IMAGE_SIZE), label

# ### TF Datasets
#
# Let's now define our TF Datasets for the test data.

BATCH_SIZE = 32

test_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths['test'], image_labels['test']))
test_dataset = test_dataset.map(load_image,
                                num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# ### Initialization

if len(sys.argv)<2:
    print('ERROR: model file missing')
    sys.exit()

print('Loading model', sys.argv[1])

custom_objects = {}
if KerasLayer is not None:
    custom_objects["KerasLayer"] = KerasLayer
with keras.utils.custom_object_scope(custom_objects):
    model = keras.models.load_model(sys.argv[1])
print(model.summary())

# ### Inference

print('Evaluating model')
scores = model.evaluate(test_dataset, verbose=2)
print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
