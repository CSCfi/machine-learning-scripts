
# coding: utf-8

# # The Oxford-IIIT Pet Dataset classification with CNNs
#
# In this script, we'll train a convolutional neural network (CNN,
# ConvNet) to classify images of breeds of dogs and cats using
# TensorFlow 2 / Keras. This script is largely based on the blog
# post [Building powerful image classification models using very
# little data]
# (https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
# by François Chollet.
# 
# **Note that using a GPU with this notebook is highly recommended.**
# 
# First, the needed imports.

import os, datetime
import random
import pathlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import applications, optimizers

from tensorflow.keras.callbacks import TensorBoard

import numpy as np

print('Using Tensorflow version:', tf.__version__,
      'Keras version:', keras.__version__,
      'backend:', keras.backend.backend())


# ## Data
# 
# The training dataset consists of 2000 images of dogs and cats, split
# in half.  In addition, the validation set consists of 1000 images.

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_xxx/data/"

print('Using DATADIR', DATADIR)
datapath = os.path.join(DATADIR, "pets/")
assert os.path.exists(datapath), "Data not found at "+datapath

if 'DECODED_IMAGES_TFRECORD' in os.environ:
    DECODED_IMAGES_TFRECORD = True
else:
    DECODED_IMAGES_TFRECORD = False
print('DECODED_IMAGES_TFRECORD is', DECODED_IMAGES_TFRECORD)

# ### Data loading
#
# We now define a function to load the images. Also we need to resize
# the images to a fixed size.

N_SHARDS = 10

def preprocess_image(image):
    if DECODED_IMAGES_TFRECORD:
        return tf.io.parse_tensor(image, tf.uint8)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    return tf.cast(image, tf.uint8)

feature_description = {
    "image": tf.io.FixedLenFeature((), tf.string),
    "filename": tf.io.FixedLenFeature((), tf.string),
    "classname": tf.io.FixedLenFeature((), tf.string),
    "classidx": tf.io.FixedLenFeature((), tf.int64)}

def load_image(example_proto):
    ex = tf.io.parse_single_example(example_proto, feature_description)
    return (preprocess_image(ex["image"]), ex["classidx"])

# ### TF Datasets
#
# Let's now define our TF Datasets for training and validation
# data. First the Datasets contain the filenames of the images and the
# corresponding labels.

# We then map() the filenames to the actual image data and decode the images.
# Note that we shuffle the training data.

BATCH_SIZE = 64

suffix = "_decoded" if DECODED_IMAGES_TFRECORD else ""

train_filenames = [datapath+"/images{}_{:03d}.tfrec".format(suffix, i)
                   for i in range(N_SHARDS)]
train_dataset = tf.data.TFRecordDataset(train_filenames)
train_dataset = train_dataset.map(load_image,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(2000).batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# ## Reuse a pre-trained CNN
# 
# We now reuse a pretrained network.  Here we'll use one of the
# pre-trained networks available from Keras
# (https://keras.io/applications/).

# ### Initialization

# We first choose either VGG16 or MobileNet as our pretrained network:

pretrained = 'VGG16'
#pretrained = 'MobileNet'

# Due to the small number of training images, a large network will
# easily overfit. Therefore, to make the most of our limited number of
# training examples, we'll apply random augmentation transformations
# (crop and horizontal flip) to them each time we are looping over
# them. This way, we "augment" our training dataset to contain more
# data.
#
# The augmentation transformations are implemented as preprocessing
# layers in Keras. There are various such layers readily available,
# see https://keras.io/guides/preprocessing_layers/ for more
# information.

inputs = keras.Input(shape=[256, 256, 3])
x = layers.Rescaling(scale=1./255)(inputs)

x = layers.RandomCrop(160, 160)(x)
x = layers.RandomFlip(mode="horizontal")(x)

# We load the pretrained network, remove the top layers, and
# freeze the pre-trained weights.

if pretrained == 'VGG16':
    pt_model = applications.VGG16(weights='imagenet', include_top=False,      
                                  input_tensor=x)
elif pretrained == 'MobileNet':
    pt_model = applications.MobileNet(weights='imagenet', include_top=False,
                                      input_tensor=x)
else:
    assert 0, "Unknown model: "+pretrained
    
pt_name = pt_model.name
print('Using "{}" pre-trained model with {} layers'
      .format(pt_name, len(pt_model.layers)))

for layer in pt_model.layers:
    layer.trainable = False

# We then stack our own, randomly initialized layers on top of the
# pre-trained network.

x = layers.Flatten()(pt_model.output)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(37, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs,
                    name="pets-"+pt_name+"-pretrained")
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# ### Learning

logdir = os.path.join(os.getcwd(), "logs", "pets_tfr"+suffix+"-"+pt_name+"-reuse-"+
                      datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir)]

epochs = 10

history = model.fit(train_dataset, epochs=epochs,
                    callbacks=callbacks, verbose=2)

fname = "pets-" + pt_name + "-reuse.h5"
print('Saving model to', fname)
model.save(fname)
print('All done')
