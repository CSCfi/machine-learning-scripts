

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Activation, Dropout, Conv2D,
                                    Flatten, MaxPooling2D, InputLayer)
from tensorflow.keras.preprocessing.image import (ImageDataGenerator, 
                                                  array_to_img, 
                                                  img_to_array, load_img)
from tensorflow.keras import applications, optimizers

from tensorflow.keras.callbacks import TensorBoard

import numpy as np

# Horovod: import
import horovod.tensorflow.keras as hvd

# Horovod: initialize Horovod
hvd.init()

if hvd.rank() == 0:
    print('Using Tensorflow version:', tf.__version__,
          'Keras version:', tf.keras.__version__,
          'backend:', tf.keras.backend.backend())
    print('Using Horovod with', hvd.size(), 'workers')

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# ## Data
# 
# The training dataset consists of 2000 images of dogs and cats, split
# in half.  In addition, the validation set consists of 1000 images,

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2003747/data/"

datapath = os.path.join(DATADIR, "dogs-vs-cats/train-2000/")

nimages = dict()
nimages['train'] = 2000
nimages['validation'] = 1000

# ### Image paths and labels

def get_paths(dataset):
    data_root = pathlib.Path(datapath+dataset)
    image_paths = list(data_root.glob('*/*'))
    image_paths = [str(path) for path in image_paths]
    image_count = len(image_paths)
    assert image_count == nimages[dataset], "Found {} images, expected {}".format(image_count, nimages[dataset])
    return image_paths

image_paths = dict()
image_paths['train'] = get_paths('train')
image_paths['validation'] = get_paths('validation')

label_names = sorted(item.name for item in pathlib.Path(datapath+'train').glob('*/')
                     if item.is_dir())
label_to_index = dict((name, index) for index,name in enumerate(label_names))

def get_labels(dataset):
    return [label_to_index[pathlib.Path(path).parent.name]
            for path in image_paths[dataset]]
    
image_labels = dict()
image_labels['train'] = get_labels('train')
image_labels['validation'] = get_labels('validation')

# ### Data augmentation
# 
# We need to resize all training and validation images to a fixed
# size. Here we'll use 160x160 pixels.
# 
# Then, to make the most of our limited number of training examples,
# we'll apply random transformations (crop and horizontal flip) to
# them each time we are looping over them. This way, we "augment" our
# training dataset to contain more data. There are various
# transformations readily available in TensorFlow, see tf.image
# (https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/image)
# for more information.

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
# Let's now define our TF Datasets
# (https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/Dataset#class_dataset)
# for training and validation data. First the Datasets contain the
# filenames of the images and the corresponding labels.

train_dataset = tf.data.Dataset.from_tensor_slices((image_paths['train'],
                                                    image_labels['train']))
validation_dataset = tf.data.Dataset.from_tensor_slices((image_paths['validation'],
                                                         image_labels['validation']))

# We then map() the filenames to the actual image data and decode the images.
# Note that we shuffle and augment only the training data.

BATCH_SIZE = 32

train_dataset = train_dataset.map(load_and_augment_image, num_parallel_calls=10)
train_dataset = train_dataset.shuffle(2000).batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

validation_dataset = validation_dataset.map(load_and_not_augment_image,
                                            num_parallel_calls=10)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# ## Train a small CNN from scratch
# 
# Similarly as with MNIST digits, we can start from scratch and train
# a CNN for the classification task. However, due to the small number
# of training images, a large network will easily overfit, regardless
# of the data augmentation.
# 
# ### Initialization

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=INPUT_IMAGE_SIZE, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Horovod: adjust learning rate based on number of GPUs.
opt = tf.keras.optimizers.RMSprop(0.001 * hvd.size())

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],
              experimental_run_tf_function=False)
#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])

if hvd.rank() == 0:
    print(model.summary())

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
]

# ### Learning

# We'll use TensorBoard to visualize our progress during training.

# Horovod: 
logfile = "dvc-cnn-simple-{}-".format(hvd.rank())
logfile = logfile+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logdir = os.path.join(os.getcwd(), "logs", logfile)
print('Rank:', hvd.rank(), 'TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks.append(TensorBoard(log_dir=logdir))

# Horovod: reduce epochs
epochs = 20 // hvd.size()

# Horovod: write logs on worker 0.
verbose = 2 if hvd.rank() == 0 else 0

history = model.fit(train_dataset, epochs=epochs,
                    validation_data=validation_dataset,
                    callbacks=callbacks, verbose=verbose)

# Horovod: 
if hvd.rank() == 0:
    fname = "dvc-cnn-simple-hvd.h5"
    print('Saving model to', fname)
    model.save(fname)
