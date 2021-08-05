# # Traffic sign classification with CNNs
#
# In this notebook, we'll train a convolutional neural network (CNN,
# ConvNet) to classify images of traffic signs from [The German
# Traffic Sign Recognition
# Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)
# using TensorFlow 2.0 / Keras. This notebook is largely based on the
# blog post [Building powerful image classification models using very
# little
# data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) by François Chollet.
#
# **Note that using a GPU with this notebook is highly recommended.**
#
# First, the needed imports.

import os
import datetime
import pathlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
from PIL import Image

print('Using Tensorflow version: {}, and Keras version: {}.'.format(
    tf.__version__, keras.__version__))

# # Data
#
# The training dataset consists of 5535 images of traffic signs of
# varying size. There are 43 different types of traffic signs. In
# addition, the validation consists of 999.

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2003747/data/"

datapath = os.path.join(DATADIR, "gtsrb/train-5535/")
assert os.path.exists(datapath), "Data not found at "+datapath

nimages = dict()
(nimages['train'], nimages['validation']) = (5535, 999)

# ### Parameters

INPUT_IMAGE_SIZE = [75, 75, 3]
BATCH_SIZE = 50
NUM_CLASSES = 43


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
                     pathlib.Path(datapath+'train').glob('*/') if
                     item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))


def get_labels(dataset):
    return [label_to_index[pathlib.Path(path).parent.name]
            for path in image_paths[dataset]]


image_labels = dict()
image_labels['train'] = get_labels('train')
image_labels['validation'] = get_labels('validation')


# ### Data augmentation
#
# We need to resize all training and validation images to a fixed
# size.
#
# Then, to make the most of our limited number of training examples,
# we'll apply random transformations (crop and horizontal flip) to
# them each time we are looping over them. This way, we "augment" our
# training dataset to contain more data. There are various
# transformations readily available in TensorFlow, see
# [tf.image](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/image)
# for more information.

def _load_image(path, label):
    image = Image.open(path.numpy())
    return np.array(image), label


def load_image(path, label):
    return tf.py_function(_load_image, (path, label), (tf.float32, tf.int32))


def preprocess_image(image, augment):
    image.set_shape([None, None, None])
    if augment:
        image = tf.image.resize(image, [80, 80])
        image = tf.image.random_crop(image, INPUT_IMAGE_SIZE)
        #image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.clip_by_value(image, 0.0, 255.0)
    else:
        image = tf.image.resize(image, INPUT_IMAGE_SIZE[:2])
    image /= 255.0  # normalize to [0,1] range
    image.set_shape(INPUT_IMAGE_SIZE)
    return image


def process_and_augment_image(image, label):
    label.set_shape([])
    return preprocess_image(image, True), label


def process_and_not_augment_image(image, label):
    label.set_shape([])
    return preprocess_image(image, False), label


# ### TF Datasets

train_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths['train'], image_labels['train']))
train_dataset = train_dataset.map(load_image,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(process_and_augment_image,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(500).batch(BATCH_SIZE,
                                                          drop_remainder=True)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths['validation'], image_labels['validation']))
validation_dataset = validation_dataset.map(load_image,
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.map(process_and_not_augment_image,
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)


# ## Train a small CNN from scratch
#
# Similarly as with MNIST digits, we can start from scratch and train
# a CNN for the classification task.
#
# ### Initialization

inputs = keras.Input(shape=INPUT_IMAGE_SIZE)

x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(43, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs,
                    name="gtsrb-cnn-simple")

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

# ### Learning

# We'll use TensorBoard to visualize our progress during training.

logdir = os.path.join(os.getcwd(), "logs", "gtsrb-cnn-simple-" +
                      datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir)]

epochs = 20
history = model.fit(train_dataset, epochs=epochs,
                    validation_data=validation_dataset,
                    callbacks=callbacks, verbose=2)

fname = "gtsrb-cnn-simple.h5"
print('Saving model to', fname)
model.save(fname)
