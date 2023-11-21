
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

print('Using Tensorflow version:', tf.__version__)

slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15345)
options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
)
#communication = tf.distribute.experimental.CommunicationImplementation.NCCL
strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver,
                                                     communication_options=options)

gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs found:', len(gpus))
print('Number of replicas:', strategy.num_replicas_in_sync)
print('Cluster specification:', slurm_resolver.cluster_spec())
print('Task info:', slurm_resolver.get_task_info())
n_gpus = slurm_resolver.num_accelerators()['GPU']
print('Number of GPUs:', n_gpus)

# ## Data
# 
# The training dataset consists of 2000 images of dogs and cats, split
# in half.  In addition, the validation set consists of 1000 images.

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_xxx/data/"

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
if len(n_gpus)>1:
      BATCH_SIZE *= len(n_gpus)

train_dataset = train_dataset.map(load_image,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(2000).batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)


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

with strategy.scope():

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
    print('Using "{}" pre-trained model with {} layers'
          .format(pt_name, len(pt_model.layers)))

    for layer in pt_model.layers:
       layer.trainable = False

    # WE then stack our own, randomly initialized layers on top of the
    # pre-trained network.

    x = layers.Flatten()(pt_model.output)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(37, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name="pets-"+pt_name+"-pretrained")
    print(model.summary())

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

# ### Learning

#logdir = os.path.join(os.getcwd(), "logs", "pets-"+pt_name+"-reuse-"+
#                      datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
#print('TensorBoard log directory:', logdir)
#os.makedirs(logdir)
#callbacks = [TensorBoard(log_dir=logdir)]

epochs = 10*2

history = model.fit(train_dataset, epochs=epochs, verbose=2)
#history = model.fit(dist_train_dataset, epochs=epochs, steps_per_epoch=20, verbose=2)
                    #callbacks=callbacks, verbose=2)

#fname = "pets-" + pt_name + "-reuse.h5"
#print('Saving model to', fname)
#model.save(fname)
#print('All done')
