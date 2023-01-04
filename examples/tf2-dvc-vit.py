# coding: utf-8

# # Dogs-vs-cats classification with ViT
#
# In this notebook, we'll finetune a [Vision Transformer]
# (https://arxiv.org/abs/2010.11929) (ViT) to classify images of dogs
# from images of cats using TensorFlow 2 / Keras and HuggingFace's
# [Transformers](https://github.com/huggingface/transformers).
#
# **Note that using a GPU with this notebook is highly recommended.**
#
# First, the needed imports.

from transformers import __version__ as transformers_version
from transformers.utils import check_min_version
check_min_version("4.13.0.dev0")
from transformers import ViTFeatureExtractor, TFViTForImageClassification

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from PIL import Image

import os, sys, datetime
import pathlib

import numpy as np

print('Using TensorFlow version:', tf.__version__,
      'Keras version:', tf.keras.__version__,
      'Transformers version:', transformers_version)

# ## Data

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2005299/data/"

print('Using DATADIR', DATADIR)
datapath = os.path.join(DATADIR, "dogs-vs-cats/train-2000/")
assert os.path.exists(datapath), "Data not found at "+datapath

# The training dataset consists of 2000 images of dogs and cats, split
# in half.  In addition, the validation set and test set consists of
# 1000 and 22000 images, respectively.

nimages = {'train':2000, 'validation':1000, 'test':22000}

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
image_paths['test'] = get_paths('test')

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
image_labels['test'] = get_labels('test')

# ### Data loading
#
# First we specify the pre-trained ViT model we are going to use. The
# model ["google/vit-base-patch16-224"]
# (https://huggingface.co/google/vit-base-patch16-224) is pre-trained
# on ImageNet-21k (14 million images, 21,843 classes) at resolution
# 224x224, and fine-tuned on ImageNet 2012 (1 million images, 1,000
# classes) at resolution 224x224.
#
# We'll use a pre-trained ViT feature extractor that matches the ViT
# model to preprocess the input images.

VITMODEL = 'google/vit-base-patch16-224'

feature_extractor = ViTFeatureExtractor.from_pretrained(VITMODEL)

# Next we define functions to load and preprocess the images:

def _load_and_process_image(path, label):
    img = Image.open(path.numpy()).convert("RGB")
    proc_img = feature_extractor(images=img,
                                 return_tensors="np")['pixel_values']
    return np.squeeze(proc_img), label

def load_and_process_image(path, label):
    image, label = tf.py_function(_load_and_process_image,
                                  (path, label), (tf.float32, tf.int32))
    image.set_shape([None, None, None])
    label.set_shape([])
    return image, label

# ### TF Datasets
#
# Let's now define our TF Datasets for training and validation data.

BATCH_SIZE = 32

dataset_train = tf.data.Dataset.from_tensor_slices((image_paths['train'],
                                                    image_labels['train']))
dataset_train = dataset_train.map(load_and_process_image,
                                  num_parallel_calls=tf.data.AUTOTUNE)
dataset_train = dataset_train.shuffle(len(dataset_train)).batch(
    BATCH_SIZE, drop_remainder=True)

dataset_validation = tf.data.Dataset.from_tensor_slices(
    (image_paths['validation'], image_labels['validation']))
dataset_validation = dataset_validation.map(load_and_process_image,
                                  num_parallel_calls=tf.data.AUTOTUNE)
dataset_validation = dataset_validation.batch(BATCH_SIZE, drop_remainder=True)

# ## Model
#
# ### Initialization

model = TFViTForImageClassification.from_pretrained(
    VITMODEL, num_labels=1, ignore_mismatched_sizes=True)

LR = 1e-5

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metric = 'accuracy'

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

print(model.summary())

# ### Learning

logdir = os.path.join(
    os.getcwd(), "logs",
    "dvc-vit-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir)]

EPOCHS = 4

history = model.fit(dataset_train, validation_data=dataset_validation,
                    epochs=EPOCHS, verbose=2, callbacks=callbacks)

# ### Inference
#
# We now evaluate the model using the test set. First we'll define the
# TF Dataset for the test images.

dataset_test = tf.data.Dataset.from_tensor_slices((image_paths['test'],
                                                   image_labels['test']))
dataset_test = dataset_test.map(load_and_process_image,
                                num_parallel_calls=tf.data.AUTOTUNE)
dataset_test = dataset_test.batch(BATCH_SIZE, drop_remainder=False)

scores = model.evaluate(dataset_test, verbose=2)
print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
