
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
datapath = os.path.join(DATADIR, "dogs-vs-cats/train-2000/")
assert os.path.exists(datapath), "Data not found at "+datapath

nimages = {'train':2000, 'validation':1000}

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
image_paths['train'] = get_paths('train')
image_paths['validation'] = get_paths('validation')

label_names = sorted(item.name for item in
                     pathlib.Path(datapath+'train').glob('*/')
                     if item.is_dir())
label_to_index = dict((name, index) for index,name in enumerate(label_names))

def get_labels(dataset):
    return [label_to_index[pathlib.Path(path).parent.name]
            for path in image_paths[dataset]]
    
image_labels = dict()
image_labels['train'] = get_labels('train')
image_labels['validation'] = get_labels('validation')

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
# Let's now define our TF Datasets for training and validation
# data. First the Datasets contain the filenames of the images and the
# corresponding labels.

train_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths['train'], image_labels['train']))
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths['validation'], image_labels['validation']))

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

logdir = os.path.join(os.getcwd(), "logs", "dvc-"+pt_name+"-reuse-"+
                      datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir)]

epochs = 20

history = model.fit(train_dataset, epochs=epochs,
                    validation_data=validation_dataset,
                    callbacks=callbacks, verbose=2)

fname = "dvc-" + pt_name + "-reuse.h5"
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
    optimizer=optimizers.RMSprop(learning_rate=1e-5),
    metrics=['accuracy'])

logdir = os.path.join(os.getcwd(), "logs", "dvc-"+pt_name+"-finetune-"+
                      datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir)]

epochs = 20

history = model.fit(train_dataset, epochs=epochs,
                    validation_data=validation_dataset,
                    callbacks=callbacks, verbose=2)

fname = "dvc-" + pt_name + "-finetune.h5"
print('Saving model to', fname)
model.save(fname)
