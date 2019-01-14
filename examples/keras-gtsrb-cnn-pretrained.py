
# coding: utf-8

# # Traffic sign classification with CNNs
# 
# In this script, we'll train a convolutional neural network (CNN,
# ConvNet) to classify images of traffic signs from [The German
# Traffic Sign Recognition Benchmark]
# (http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) using
# Keras (version $\ge$ 2 is required). This script is largely based on
# the blog post [Building powerful image classification models using
# very little data]
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
                     "gtsrb-pretrained-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory:', logdir)
    os.makedirs(logdir)
    callbacks = [TensorBoard(log_dir=logdir)]
else:
    callbacks =  None

# ## Data
# 
# The training dataset consists of 5535 images of traffic signs of
# varying size. There are 43 different types of traffic signs.
# 
# The validation and test sets consist of 999 and 12630 images,
# respectively.

datapath = "/wrk/makoskel/gtsrb/train-5535"
(nimages_train, nimages_validation, nimages_test) = (5535, 999, 12630)

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

input_image_size = (75, 75)

datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        #rotation_range=40,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        horizontal_flip=False)

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

batch_size = 50

print('Train: ', end="")
train_generator = datagen.flow_from_directory(
        datapath+'/train',  
        target_size=input_image_size,
        batch_size=batch_size)

print('Validation: ', end="")
validation_generator = noopgen.flow_from_directory(
        datapath+'/validation',  
        target_size=input_image_size,
        batch_size=batch_size)

print('Test: ', end="")
test_generator = noopgen.flow_from_directory(
        datapath+'/test',  
        target_size=input_image_size,
        batch_size=batch_size)

# We now reuse a pretrained network.  Here we'll use the
# [VGG16](https://keras.io/applications/#vgg16) network architecture
# with weights learned using Imagenet.  We remove the top layers and
# freeze the pre-trained weights.
# 
# ### Initialization

model = Sequential()

vgg_model = applications.VGG16(weights='imagenet', 
                               include_top=False, 
                               input_shape=input_image_size+(3,))
for layer in vgg_model.layers:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

print(model.summary())

# We then stack our own, randomly initialized layers on top of the
# VGG16 network.

model.add(Flatten())
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

# ### Learning 1: New layers

epochs = 20

history = model.fit_generator(train_generator,
                              steps_per_epoch=nimages_train // batch_size,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=nimages_validation // batch_size,
                              verbose=2, callbacks=callbacks)

model.save("gtsrb-vgg16-reuse.h5")

# ### Learning 2: Fine-tuning
# 
# Once the top layers have learned some reasonable weights, we can
# continue training by unfreezing the last convolution block of VGG16
# (`block5`) so that it may adapt to our data. The learning rate
# should be smaller than usual.

for layer in model.layers[15:]:
    layer.trainable = True
    print(layer.name, "now trainable")
    
model.compile(loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-5),
    metrics=['accuracy'])

print(model.summary())

# Note that before continuing the training, we create a separate
# TensorBoard log directory:

epochs = 20

if K.backend() == "tensorflow":
    logdir_ft = logdir + "-ft"
    os.makedirs(logdir_ft)
    callbacks_ft = [TensorBoard(log_dir=logdir_ft)]
else:
    callbacks_ft = None

history = model.fit_generator(train_generator,
                              steps_per_epoch=nimages_train // batch_size,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=nimages_validation // batch_size,
                              verbose=2, callbacks=callbacks_ft)

model.save("gtsrb-vgg16-finetune.h5")

# ### Inference

scores = model.evaluate_generator(test_generator,
                                  steps=nimages_test // batch_size)
print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
