
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

train_filenames = [datapath+"train-{0:05d}-of-00004".format(i)
                   for i in range(4)]
train_dataset = tf.data.TFRecordDataset(train_filenames)

validation_filenames = [datapath+"validation-{0:05d}-of-00002".format(i)
                        for i in range(2)]
validation_dataset = tf.data.TFRecordDataset(validation_filenames)

# We then map() the filenames to the actual image data and decode the images.
# Note that we shuffle the training data.

BATCH_SIZE = 32

train_dataset = train_dataset.map(load_image,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(2000).batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

validation_dataset = validation_dataset.map(load_image,
                                            num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

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

inputs = keras.Input(shape=INPUT_IMAGE_SIZE+[3])
x = layers.Rescaling(scale=1./255)(inputs)

x = layers.RandomCrop(160, 160)(x)
x = layers.RandomFlip(mode="horizontal")(x)

# We load the pretrained network, remove the top layers, and
# freeze the pre-trained weights.

if pretrained == 'VGG16':
    pt_model = applications.VGG16(weights='imagenet', include_top=False,      
                                  input_tensor=x)
    finetuning_first_trainable_layer = "block5_conv1" 
elif pretrained == 'MobileNet':
    pt_model = applications.MobileNet(weights='imagenet', include_top=False,
                                      input_tensor=x)
    finetuning_first_trainable_layer = "conv_dw_12"
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
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs,
                    name="dvc-"+pt_name+"-pretrained")
print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# ### Learning 1: New layers

logdir = os.path.join(os.getcwd(), "logs", "dvc_tfr-"+pt_name+"-reuse-"+
                      datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir)]

epochs = 20

history = model.fit(train_dataset, epochs=epochs,
                    validation_data=validation_dataset,
                    callbacks=callbacks, verbose=2)

fname = "dvc_tfr-" + pt_name + "-reuse.h5"
print('Saving model to', fname)
model.save(fname)

# ### Learning 2: Fine-tuning
# 
# Once the top layers have learned some reasonable weights, we can
# continue training by unfreezing the last blocks of the pre-trained
# network so that it may adapt to our data. The learning rate should
# be smaller than usual.

print('Setting last pre-trained layers to be trainable')
train_layer = False
for layer in model.layers:
    if layer.name == finetuning_first_trainable_layer:
        train_layer = True
    layer.trainable = train_layer
    
for i, layer in enumerate(model.layers):
    print(i, layer.name, "trainable:", layer.trainable)    
print(model.summary())    

model.compile(loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-5),
    metrics=['accuracy'])

logdir = os.path.join(os.getcwd(), "logs", "dvc_tfr-"+pt_name+"-finetune-"+
                      datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir)]

epochs = 20

history = model.fit(train_dataset, epochs=epochs,
                    validation_data=validation_dataset,
                    callbacks=callbacks, verbose=2)

fname = "dvc_tfr-" + pt_name + "-finetune.h5"
print('Saving model to', fname)
model.save(fname)
