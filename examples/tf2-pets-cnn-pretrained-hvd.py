
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

# Horovod: import and initialize
import horovod.tensorflow.keras as hvd
import horovod
hvd.init()

if hvd.rank() == 0:
    print('Using Horovod', horovod.__version__,
          'with hvd.size():', hvd.size())

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# ## Data
# 
# The training dataset consists of 2000 images of dogs and cats, split
# in half.  In addition, the validation set consists of 1000 images.

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_xxx/data/"

if hvd.rank() == 0:
    print('Using DATADIR', DATADIR)
datapath = os.path.join(DATADIR, "pets/")
assert os.path.exists(datapath), "Data not found at "+datapath

nimages = {'train':7390}

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

label_names = sorted(item.name for item in
                     pathlib.Path(datapath+'train').glob('*/')
                     if item.is_dir())
label_to_index = dict((name, index) for index,name in enumerate(label_names))

def get_labels(dataset):
    return [label_to_index[pathlib.Path(path).parent.name]
            for path in image_paths[dataset]]
    
image_labels = dict()
image_labels['train'] = get_labels('train')

# ### Data loading
#
# We now define a function to load the images. Also we need to resize
# the images to a fixed size (INPUT_IMAGE_SIZE).

AUGMENT = False
if AUGMENT:
    INPUT_IMAGE_SIZE = [256, 256]
else:
    INPUT_IMAGE_SIZE = [160, 160]

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

# We then map() the filenames to the actual image data and decode the images.
# Note that we shuffle the training data.

BATCH_SIZE = 64

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

inputs = keras.Input(shape=INPUT_IMAGE_SIZE+[3])
x = layers.Rescaling(scale=1./255)(inputs)
if AUGMENT:
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
if hvd.rank() == 0:
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
if hvd.rank() == 0:
    print(model.summary())

# Horovod: adjust learning rate based on number of GPUs.
initial_lr = 0.001 * hvd.size()
opt = tf.keras.optimizers.RMSprop(initial_lr)

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)

# Horovod: Specify `experimental_run_tf_function=False` to ensure
# TensorFlow uses hvd.DistributedOptimizer() to compute gradients.
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],
              experimental_run_tf_function=False)

# ### Learning

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all
    # other processes.  This is necessary to ensure consistent
    # initialization of all workers when training is started with
    # random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every
    # epoch.
    #
    # Note: This callback must be in the list before the
    # ReduceLROnPlateau, TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning
    # leads to worse final accuracy. Scale the learning rate `lr =
    # 1.0` ---> `lr = 1.0 * hvd.size()` during the first three
    # epochs. See https://arxiv.org/abs/1706.02677 for details.
    #hvd.callbacks.LearningRateWarmupCallback(initial_lr, warmup_epochs=3,
    #                                         verbose=1),
]

if hvd.rank() == 0:
    logdir = os.path.join(os.getcwd(), "logs", "pets-"+pt_name+"-reuse-"+
                          datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory:', logdir)
    os.makedirs(logdir)
    callbacks.append(TensorBoard(log_dir=logdir))

epochs = 10

# Horovod: write logs on worker 0.
verbose = 2 if hvd.rank() == 0 else 0

history = model.fit(train_dataset, epochs=epochs,
                    callbacks=callbacks, verbose=verbose)

if hvd.rank() == 0:
    fname = "pets-" + pt_name + "-reuse.h5"
    print('Saving model to', fname)
    model.save(fname)

print('All done for rank', hvd.rank())
