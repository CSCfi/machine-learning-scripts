# coding: utf-8

# # Dogs-vs-cats classification with BiT
#
# In this script, we'll finetune a [BigTransfer]
# (https://arxiv.org/abs/1912.11370) (BiT) model from [TensorFlow
# Hub](https://tfhub.dev/) to classify images of dogs from images of
# cats using TensorFlow 2 / Keras. This script is somewhat based on
# the Keras code example [Image Classification using BigTransfer
# (BiT)](https://keras.io/examples/vision/bit/) by Sayan Nath.
#
# **Note that using a GPU with this script is highly recommended.**
#
# First, the needed imports.

import os, datetime
import pathlib

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

import numpy as np

print('Using Tensorflow version:', tf.__version__,
      'Keras version:', keras.__version__,
      'backend:', keras.backend.backend())


# ## Data
#
# The training dataset consists of 2000 images of dogs and cats, split
# in half.  In addition, the validation set consists of 1000 images,

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
label_to_index = dict((name, index) for index, name in enumerate(label_names))


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

# ## BiT
#
# ### Initialization
#
# Now we specify the pre-trained BiT model we are going to use. The
# model ["BiT-M R50x1"] (https://tfhub.dev/google/bit/m-r50x1/1) is
# pre-trained on ImageNet-21k (14 million images, 21,843 classes). It
# outputs 2048-dimensional feature vectors.

bit_model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
bit_model = hub.KerasLayer(bit_model_url)

# First we'll apply random augmentation transformations (crop and
# horizontal flip) to them each time we are looping over them. This
# way, we "augment" our training dataset to contain more data. The
# augmentation transformations are implemented as preprocessing layers
# in Keras. There are various such layers readily available, see
# https://keras.io/guides/preprocessing_layers/ for more information.
#
# Then we add the BiT model as a layer and finally add the output
# layer with a single unit and sigmoid activation. Note that we
# initialize the output layer to all zeroes as instructed in
# https://keras.io/examples/vision/bit/.

inputs = keras.Input(shape=INPUT_IMAGE_SIZE+[3])
x = layers.Rescaling(scale=1./255)(inputs)

x = layers.RandomCrop(160, 160)(x)
x = layers.RandomFlip(mode="horizontal")(x)

x = bit_model(x)

outputs = layers.Dense(1, kernel_initializer="zeros",
                       activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs,
                    name="dvc-bit")

learning_rate, momentum = 0.003, 0.9

optimizer = keras.optimizers.SGD(learning_rate=learning_rate,
                                 momentum=momentum)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)

model.compile(loss=loss_fn, optimizer=optimizer,
              metrics=['accuracy'])

print(model.summary())

# ### Learning
#
# We'll set up two callbacks. *EarlyStopping* is used to stop training
# when the monitored metric has stopped improving. *TensorBoard* is
# used to visualize our progress during training.

logdir = os.path.join(
    os.getcwd(), "logs",
    "dvc-bit-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True),
    TensorBoard(log_dir=logdir)]

EPOCHS = 20

history = model.fit(train_dataset, batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=validation_dataset,
                    callbacks=callbacks)

fname = "dvc-bit.h5"
print('Saving model to', fname)
model.save(fname)
print('All done')
