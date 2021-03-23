#!/usr/bin/env python
# coding: utf-8

# MNIST handwritten digits classification with MLPs
# =================================================
#
# Copyright 2018 Google LLC.<br/>
# Copyright 2021 CSC
#
# Licensed under the Apache License, Version 2.0 (the "License");you
# may not use this file except in compliance with the License.  You
# may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.
#
# ----------------------------------------------------------------------
#
# In this notebook, we'll train a multi-layer perceptron model to
# classify MNIST digits using pure **Jax**.
#
# This script is partly copied and adapted from
# https://github.com/google/jax/blob/master/docs/notebooks/Neural_Network_and_Data_Loading.ipynb
#
# First, the needed imports.

import time

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random, __version__
from jax.scipy.special import logsumexp

import torch
from torchvision import datasets, transforms

import numpy as np

from jax.lib import xla_bridge
device = xla_bridge.get_backend().platform

print('Using Jax version:', __version__, ' Device:', device)

# Data
# ----
#
# Next we'll load the MNIST data.  We are using Pytorch's
# `torchvision` for this purpose. First time we may have to download
# the data, which can take a while.
#
# Note that we are here using the MNIST test data for *validation*,
# instead of for testing the final model.

batch_size = 32
n_targets = 10

train_dataset = datasets.MNIST('../MNIST',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

validation_dataset = datasets.MNIST('../MNIST',
                                    train=False,
                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)

# The train and test data are provided via data loaders that provide
# iterators over the datasets. The first element of training data
# (`X_train`) is a 4th-order tensor of size (`batch_size`, 1, 28, 28),
# i.e. it consists of a batch of images of size 1x28x28
# pixels. `y_train` is a vector containing the correct classes ("0",
# "1", ..., "9") for each training digit.

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

# Here are the first 10 training digits:

pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
    plt.title('Class: '+str(y_train[i].item()))

# MLP network definition
# ----------------------
#
# ### Hyperparameters

layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 10

# ### Initialization

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return (scale * random.normal(w_key, (n, m)),
          scale * random.normal(b_key, (n,)))

# Initialize all layers for a fully-connected neural network with
# sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in
          zip(sizes[:-1], sizes[1:], keys)]

params = init_network_params(layer_sizes, random.PRNGKey(0))

# ### Forward pass

def relu(x):
    return jnp.maximum(0, x)

def predict(params, image):
    # per-example predictions
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)

# Forward pass for a single image:

random_flattened_image = random.normal(random.PRNGKey(1), (28 * 28,))
preds = predict(params, random_flattened_image)
print(preds.shape)

# Forward pass for a batch of images, using `vmap()`:

random_flattened_images = random.normal(random.PRNGKey(1), (10, 28 * 28))

# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))

# `batched_predict` has the same call signature as `predict`
batched_preds = batched_predict(params, random_flattened_images)
print(batched_preds.shape)

# ## Learning

def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

train_images = jnp.array(train_dataset.data.numpy()
                        .reshape(len(train_dataset.data),-1),
                        dtype=jnp.float32)
train_labels = one_hot(jnp.array(train_dataset.targets), n_targets)

validation_images = jnp.array(validation_dataset.data.numpy()
                              .reshape(len(validation_dataset.data),-1),
                              dtype=jnp.float32)
validation_labels = one_hot(jnp.array(validation_dataset.targets), n_targets)

# Now we are ready to train our model.  An *epoch* means one pass
# through the whole training data. After each epoch, we evaluate the
# model by computing accuracy and loss values for training and
# validation data.

for epoch in range(num_epochs):
    start_time = time.time()
    for x, y in train_loader:
        y = one_hot(jnp.array(y), n_targets)
        x = jnp.array(x).reshape(batch_size, -1)
        params = update(params, x, y)
    epoch_time = time.time() - start_time

    train_acc = accuracy(params, train_images, train_labels)
    train_loss = loss(params, train_images, train_labels)
    validation_acc = accuracy(params, validation_images, validation_labels)
    validation_loss = loss(params, validation_images, validation_labels)

    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy:   "
          "{:0.4f}, loss: {:0.4f}".format(train_acc, train_loss))
    print("Validation set accuracy: "
          "{:0.4f}, loss: {:0.4f}".format(validation_acc, validation_loss))
