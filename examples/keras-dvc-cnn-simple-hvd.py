
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

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D 
from keras.preprocessing.image import (ImageDataGenerator, array_to_img, 
                                      img_to_array, load_img)
from keras import applications, optimizers

from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

import tensorflow as tf
import horovod.keras as hvd

import numpy as np
import math

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

# If we are using TensorFlow as the backend, we can use TensorBoard to
# visualize our progress during training.
# Horovod: create TensorBoard log directory only on rank 0.
callbacks =  []
if hvd.rank() == 0 and K.backend() == "tensorflow":
    from keras.callbacks import TensorBoard
    import os, datetime
    logdir = os.path.join(os.getcwd(), "logs",
                     "dvc-simple-hvd-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory:', logdir)
    try:
        os.makedirs(logdir)
        callbacks.append(TensorBoard(log_dir=logdir))
    except FileExistsError:
        pass

# ## Data
# 
# The training dataset consists of 2000 images of dogs and cats, split
# in half.  In addition, the validation set consists of 1000 images,
# and the test set of 22000 images.

datapath = "/wrk/makoskel/dogs-vs-cats/train-2000"
(nimages_train, nimages_validation, nimages_test) = (2000, 1000, 22000)

# ### Data augmentation
# 
# First, we'll resize all training and validation images to a fized size. 
# 
# Then, to make the most of our limited number of training examples,
# we'll apply random transformations to them each time we are looping
# over them. This way, we "augment" our training dataset to contain
# more data. There are various transformations readily available in
# Keras, see [ImageDataGenerator]
# (https://keras.io/preprocessing/image/) for more information.

input_image_size = (150, 150)

datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        #rotation_range=40,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        horizontal_flip=True)

noopgen = ImageDataGenerator(rescale=1./255)

# Let's put a couple of training images with the augmentation to a
# TensorBoard event file.

augm_generator = datagen.flow_from_directory(
        datapath+'/train',  
        target_size=input_image_size,  
        batch_size=10)

for batch, _ in augm_generator:
    break

# ### Data loaders
# 
# Let's now define our real data loaders for training and validation data.

batch_size = 25

print('Train: ', end="")
train_generator = datagen.flow_from_directory(
        datapath+'/train',  
        target_size=input_image_size,
        batch_size=batch_size, 
        class_mode='binary')

print('Validation: ', end="")
validation_generator = noopgen.flow_from_directory(
        datapath+'/validation',  
        target_size=input_image_size,
        batch_size=batch_size,
        class_mode='binary')

# Similarly as with MNIST digits, we can start from scratch and train
# a CNN for the classification task. However, due to the small number
# of training images, a large network will easily overfit, regardless
# of the data augmentation.
# 
# ### Initialization

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_image_size+(3,), activation='relu'))
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
opt = optimizers.Adadelta(1.0 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Horovod: broadcast initial variable states from rank 0 to all other processes.
# This is necessary to ensure consistent initialization of all processes when
# training is started with random weights or restored from a checkpoint.

callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

# Horovod: print model summary only on rank 0.
if hvd.rank() == 0:
    print(model.summary())

# ### Learning

epochs = 20
workers = 4
use_multiprocessing = False

# Horovod: adjust number of epochs based on number of GPUs.
epochs_hvd = int(math.ceil(epochs / hvd.size()))

print('Rank', hvd.rank(), ': Training for', epochs_hvd, 'epochs with',
      workers, 'workers, use_multiprocessing is', use_multiprocessing)

history = model.fit_generator(train_generator,
                              steps_per_epoch=nimages_train // batch_size,
                              epochs=epochs_hvd,
                              validation_data=validation_generator,
                              validation_steps=nimages_validation // batch_size,
                              verbose=2, callbacks=callbacks,
                              use_multiprocessing=use_multiprocessing,
                              workers=workers)

# Horovod: save model only on rank 0.
if hvd.rank() == 0:
    fname = "dvc-small-cnn-hvd.h5"
    print('Saving model to', fname)
    model.save(fname)
