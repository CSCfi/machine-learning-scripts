import os
import sys
import pathlib

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

from PIL import Image

print('Using Tensorflow version: {}, and Keras version: {}.'.format(
    tf.__version__, tf.keras.__version__))

# # Data
#
# The test set consists of 12630 images.

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2003747/data/"

datapath = os.path.join(DATADIR, "gtsrb/train-5535/")

nimages = dict()
nimages['test'] = 12630

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
image_paths['test'] = get_paths('test')

label_names = sorted(item.name for item in
                     pathlib.Path(datapath+'train').glob('*/') if
                     item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))


def get_labels(dataset):
    return [label_to_index[pathlib.Path(path).parent.name]
            for path in image_paths[dataset]]


image_labels = dict()
image_labels['test'] = get_labels('test')


# ### Data augmentation

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

test_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths['test'], image_labels['test']))
test_dataset = test_dataset.map(load_image)
test_dataset = test_dataset.map(process_and_not_augment_image,
                                num_parallel_calls=10)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)

# ### Initialization

if len(sys.argv) < 2:
    print('ERROR: model file missing')
    sys.exit()

model = load_model(sys.argv[1])

print(model.summary())

print('Evaluating model', sys.argv[1])
scores = model.evaluate(test_dataset, verbose=2)
print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
