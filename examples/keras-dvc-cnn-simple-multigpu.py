
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

from keras.utils import multi_gpu_model
NCPUS, NGPUS = 8, 2

from distutils.version import LooseVersion as LV
from keras import __version__

import numpy as np

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

# If we are using TensorFlow as the backend, we can use TensorBoard to
# visualize our progress during training.

if K.backend() == "tensorflow":
    import tensorflow as tf
    from keras.callbacks import TensorBoard
    import os, datetime
    logdir = os.path.join(os.getcwd(), "logs",
                     "dvc-simple-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory:', logdir)
    os.makedirs(logdir)
    callbacks = [TensorBoard(log_dir=logdir)]
else:
    callbacks =  None

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

if K.backend() == "tensorflow":
    imgs = tf.convert_to_tensor(batch)
    summary_op = tf.summary.image("augmented", imgs, max_outputs=10)
    with tf.Session() as sess:
        summary = sess.run(summary_op)
        writer = tf.summary.FileWriter(logdir)
        writer.add_summary(summary)
        writer.close()

# ### Data loaders
# 
# Let's now define our real data loaders for training and validation data.

batch_size = NGPUS*25

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

print('Test: ', end="")
test_generator = noopgen.flow_from_directory(
        datapath+'/test',  
        target_size=input_image_size,
        batch_size=batch_size,
        class_mode='binary')

# Similarly as with MNIST digits, we can start from scratch and train
# a CNN for the classification task. However, due to the small number
# of training images, a large network will easily overfit, regardless
# of the data augmentation.
# 
# ### Initialization

with tf.device('/cpu:0'):

    _model = Sequential()

    _model.add(Conv2D(32, (3, 3), input_shape=input_image_size+(3,), activation='relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))

    _model.add(Conv2D(32, (3, 3), activation='relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))

    _model.add(Conv2D(64, (3, 3), activation='relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))

    _model.add(Flatten())
    _model.add(Dense(64, activation='relu'))
    _model.add(Dropout(0.5))
    _model.add(Dense(1, activation='sigmoid'))

model = multi_gpu_model(_model, gpus=NGPUS)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

# ### Learning

epochs = 20

history = model.fit_generator(train_generator,
                              steps_per_epoch=nimages_train // batch_size,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=nimages_validation // batch_size,
                              verbose=2, callbacks=callbacks,
                              use_multiprocessing=True, workers=NCPUS)

#model.save("dvc-small-cnn.h5")

# ### Inference

scores = model.evaluate_generator(test_generator,
                                  steps=nimages_test // batch_size)
print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
