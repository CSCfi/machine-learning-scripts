# # Traffic sign classification with CNNs
#
# In this notebook, we'll train a convolutional neural network (CNN,
# ConvNet) to classify images of traffic signs from The German Traffic
# Sign Recognition Benchmark using TensorFlow 2 / Keras. This notebook
# is largely based on the blog post [Building powerful image
# classification models using very little data]
# (https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
# by François Chollet.
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
# addition, the validation set consists of 999 images.

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2005299/data/"

print('Using DATADIR', DATADIR)
datapath = os.path.join(DATADIR, "gtsrb/train-5535/")
assert os.path.exists(datapath), "Data not found at "+datapath

nimages = {'train':5535, 'validation':999}

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

# ###Data loading
#
# We now define a function to load the images. The images are in PPM
# format, so we use the PIL library. Also we need to resize the images
# to a fixed size (INPUT_IMAGE_SIZE).

INPUT_IMAGE_SIZE = [80, 80]

def _load_image(path, label):
    image = Image.open(path.numpy())
    return np.array(image), label

def load_image(path, label):
    image, label = tf.py_function(_load_image, (path, label),
                                  (tf.float32, tf.int32))
    image.set_shape([None, None, None])
    label.set_shape([])
    return tf.image.resize(image, INPUT_IMAGE_SIZE), label

# ### TF Datasets
#
# Let's now define our TF Datasets for training and validation data.

BATCH_SIZE = 50

train_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths['train'], image_labels['train']))
train_dataset = train_dataset.map(load_image,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(2000).batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths['validation'], image_labels['validation']))
validation_dataset = validation_dataset.map(load_image,
                                            num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# ## Train a small CNN from scratch
#
# Similarly as with MNIST digits, we can start from scratch and train
# a CNN for the classification task.
#
# However, due to the small number of training images, a large network
# will easily overfit. Therefore, to make the most of our limited
# number of training examples, we'll apply random augmentation
# transformations (small random crop and contrast adjustment) to them
# each time we are looping over them. This way, we "augment" our
# training dataset to contain more data.
#
# The augmentation transformations are implemented as preprocessing
# layers in Keras. There are various such layers readily available,
# see https://keras.io/guides/preprocessing_layers/ for more
# information.
#
# ### Initialization

inputs = keras.Input(shape=INPUT_IMAGE_SIZE+[3])
x = layers.Rescaling(scale=1./255)(inputs)

x = layers.RandomCrop(75, 75)(x)
x = layers.RandomContrast(0.1)(x)

x = layers.Conv2D(32, (3, 3), activation='relu')(x)
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
