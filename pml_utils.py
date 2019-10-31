#!/usr/bin/env python3

# Adapted from https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py

import gzip
import os
import shutil
import tempfile
import struct

import numpy as np
import urllib.request


def download_mnist(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""

    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        print('Not downloading, file already exists:', filepath)
        return filepath
    if not os.path.isdir(directory):
        os.mkdir(directory)
    # original: http://yann.lecun.com/exdb/mnist/
    # CVDF mirror: https://storage.googleapis.com/cvdf-datasets/mnist/
    # CSC mirror
    url_base = 'https://object.pouta.csc.fi/swift/v1/AUTH_dac/mldata/'
    url = url_base + filename + '.gz'
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        os.remove(zipped_filepath)
    return filepath


def read_mnist_idx(filename):
    """Read MNIST file."""

    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def get_mnist_dataset(directory, images_file, labels_file):
    """Download and parse MNIST dataset."""

    images_file = download_mnist(directory, images_file)
    labels_file = download_mnist(directory, labels_file)

    images = read_mnist_idx(images_file)
    labels = read_mnist_idx(labels_file)

    return (images, labels)


def get_mnist(directory, labels_as_strings=True, flatten=True):
    X_train, y_train = get_mnist_dataset(directory, 'train-images-idx3-ubyte',
                                         'train-labels-idx1-ubyte')
    X_test, y_test = get_mnist_dataset(directory, 't10k-images-idx3-ubyte',
                                       't10k-labels-idx1-ubyte')
    if labels_as_strings:
        y_train = y_train.astype(np.str)
        y_test = y_test.astype(np.str)

    if flatten:
        X_train = X_train.astype(np.float64).reshape(-1, 28*28)
        X_test = X_test.astype(np.float64).reshape(-1, 28*28)

    return (X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_mnist('MNIST')
    print()
    print('MNIST data loaded:')
    print('X_train:', X_train.shape, X_train.dtype)
    print('y_train:', y_train.shape, y_train.dtype)
    print('X_test:', X_test.shape, X_test.dtype)
    print('y_test:', y_test.shape, y_test.dtype)

    print()
    print(X_train[:3, :])
    print(y_train[:3])
