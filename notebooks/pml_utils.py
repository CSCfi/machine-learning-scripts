#!/usr/bin/env python3

# Adapted from https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py

import gzip
import os
import shutil
import tempfile
import struct

import numpy as np
import urllib.request

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def show_failures(predictions, y_test, X_test, trueclass=None,
                  predictedclass=None, maxtoshow=10):
    import matplotlib.pyplot as plt

    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)
    errors = predictions != y_test
    print('Showing max', maxtoshow, 'first failures. The predicted class is '
          'shown first and the correct class in parenthesis.')
    ii = 0
    plt.figure(figsize=(maxtoshow, 1))
    for i in range(X_test.shape[0]):
        if ii >= maxtoshow:
            break
        if errors[i]:
            if trueclass is not None and y_test[i] != trueclass:
                continue
            if predictedclass is not None and predictions[i] != predictedclass:
                continue
            plt.subplot(1, maxtoshow, ii+1)
            plt.axis('off')
            plt.imshow(X_test[i, :].reshape(28, 28), cmap="gray")
            plt.title("%s (%s)" % (predictions[i], y_test[i]))
            ii = ii + 1

def show_clusters(labels, n_clust, X, n_img_per_row = 32):
    img = np.zeros((28 * n_clust, 28 * n_img_per_row))

    for i in range(n_clust):
        ix = 28 * i
        X_cluster = X[labels==i,:]
        try:
            for j in range(n_img_per_row):
                iy = 28 * j
                img[ix:ix + 28, iy:iy + 28] = X_cluster[j,:].reshape(28,28)
        except IndexError:
            pass

    plt.figure(figsize=(12, 12))
    plt.imshow(img, cmap='gray')
    plt.title('Some MNIST digits from each cluster')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('clusters');

def show_anomalies(predictions, X, n_img_per_row = 32):
    img = np.zeros((28 * 2, 28 * n_img_per_row))
    anolabels = [-1, 1]

    for i in range(2):
        ix = 28 * i
        X_ano = X[predictions==anolabels[i], :]
        try:
            for j in range(n_img_per_row):
                iy = 28 * j
                img[ix:ix + 28, iy:iy + 28] = X_ano[j,:].reshape(28,28)
        except IndexError:
            pass

    plt.figure(figsize=(12, 12))
    plt.imshow(img, cmap='gray')
    plt.title('Examples of anomalies (upper row) and normal data (lower row)')
    plt.xticks([])
    plt.yticks([]);

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
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)

    if flatten:
        X_train = X_train.astype(np.float64).reshape(-1, 28*28)
        X_test = X_test.astype(np.float64).reshape(-1, 28*28)

    return (X_train, y_train, X_test, y_test)


def load_not_mnist(directory, filename):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        print('Not downloading, file already exists:', filepath)
    else:
        if not os.path.isdir(directory):
            os.mkdir(directory)
        url_base = 'https://a3s.fi/mldata/'
        url = url_base + filename
        print('Downloading {} to {}'.format(url, filepath))
        urllib.request.urlretrieve(url, filepath)
    return np.load(filepath)


def get_notmnist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    X_train = load_not_mnist(directory, 'notMNIST_large_images.npy').reshape(-1, 28*28).astype(np.float32)
    y_train = load_not_mnist(directory, 'notMNIST_large_labels.npy')
    X_test = load_not_mnist(directory, 'notMNIST_small_images.npy').reshape(-1, 28*28).astype(np.float32)
    y_test = load_not_mnist(directory, 'notMNIST_small_labels.npy')
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
