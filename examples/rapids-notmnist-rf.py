#!/usr/bin/env python
# coding: utf-8

# notMNIST letters classification with ensembles of decision trees
# 
# In this notebook, we'll use random forest
# (https://docs.rapids.ai/api/cuml/stable/api.html#random-forest)
# [gradient boosted trees](https://xgboost.readthedocs.io/en/latest/)
# to classify notMNIST letters using a GPU, the
# RAPIDS (https://rapids.ai/) libraries (cudf, cuml) and
# XGBoost (https://xgboost.readthedocs.io/en/latest/).
# 
# **Note that a GPU is required with this notebook.**
# 
# This version of the notebook has been tested with RAPIDS version
# 0.15.

# First, the needed imports.

import cudf
import numpy as np
import pandas as pd

import os
import urllib.request
from time import time

from cuml.ensemble import RandomForestClassifier
import xgboost as xgb
from cuml import __version__ as cuml_version

from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
from sklearn import __version__ as sklearn_version

print('Using cudf version:', cudf.__version__)
print('Using cuml version:', cuml_version)
print('Using sklearn version:', sklearn_version)

# Then we load the notMNIST data. First time we need to download the
# data, which can take a while. The data is stored as Numpy arrays in
# host (CPU) memory.

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

DATA_DIR = os.path.expanduser('~/data/notMNIST/')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

print()
print('Loading data begins')
t0 = time()

X_train = load_not_mnist(DATA_DIR,
                         'notMNIST_large_images.npy').reshape(-1, 28*28)
y_train = load_not_mnist(DATA_DIR, 'notMNIST_large_labels.npy')
X_test = load_not_mnist(DATA_DIR,
                        'notMNIST_small_images.npy').reshape(-1, 28*28)
y_test = load_not_mnist(DATA_DIR, 'notMNIST_small_labels.npy')

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

print()
print('notMNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', type(X_train), 'shape:', X_train.shape)
print('y_train:', type(y_train), 'shape:', y_train.shape)
print('X_test:', type(X_test), 'shape:', X_test.shape)
print('y_test:', type(y_test), 'shape:', y_test.shape)

print('Loading data done in {:.2f} seconds'.format(time()-t0))

# Let's convert our training data to cuDF DataFrames in device (GPU)
# memory. We will also convert the classes in `y_train` to integers in
# [0..9].
#
# We do not explicitly need to convert the test data as the GPU-based
# inference functionality will take care of it.

print()
print('Copying data to GPU begins')
t0 = time()

cu_X_train = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
cu_y_train = cudf.Series(y_train.view(np.int32)-ord('A'))

print('cu_X_train:', type(cu_X_train), 'shape:', cu_X_train.shape)
print('cu_y_train:', type(cu_y_train), 'shape:', cu_y_train.shape)

print('Copying data to GPU done in {:.2f} seconds'.format(time()-t0))

# ### Learning
#
# Random forest classifiers are quick to train, quite robust to
# hyperparameter values, and often work relatively well.

print()
print('Learning begins')
t0 = time()

n_estimators = 100
max_depth = 16
clf_rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth)
print(clf_rf)
clf_rf.fit(cu_X_train, cu_y_train)

print('Learning done in {:.2f} seconds'.format(time()-t0))

# ### Inference
#
# We will use GPU-based inference to predict the classes for the test
# data.

print()
print('Inference begins')
t0 = time()

pred_rf = clf_rf.predict(X_test, predict_model='GPU')
pred_rf = [chr(x) for x in pred_rf+ord('A')]
pred_rf = np.array(pred_rf)

print('Inference done in {:.2f} seconds'.format(time()-t0))
print()

print('Predicted {} digits with accuracy: {:.4f}'
      .format(len(pred_rf), accuracy_score(y_test, pred_rf)))
print()

# #### Confusion matrix, accuracy, precision, and recall
# 
# We can also compute the confusion matrix to see which digits get
# mixed the most, and look at classification accuracies separately for
# each class:

labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
print('Confusion matrix (rows: true classes; columns: predicted classes):')
print()
cm=confusion_matrix(y_test, pred_rf, labels=labels)
df_cm = pd.DataFrame(cm, columns=labels, index=labels)
print(df_cm.to_string())
print()

print('Classification accuracy for each class:');
print()
for i,j in enumerate(cm.diagonal()/cm.sum(axis=1)):
    print("%s: %.4f" % (labels[i], j))

# Precision and recall for each class:

print()
print('Precision and recall for each class:');
print()
print(classification_report(y_test, pred_rf, labels=labels))

import sys
sys.exit()
# ## Gradient boosted trees (XGBoost)
#
# Gradient boosted trees (or extreme gradient boosted trees) is another way of constructing ensembles of decision trees, using the *boosting* framework.  Here we use the GPU accelerated [XGBoost](http://xgboost.readthedocs.io/en/latest/) library to train gradient boosted trees to classify MNIST digits. 
#
# ### Data
#
# We begin by converting our training and test data to XGBoost's
# internal DMatrix data structures.
#
# We will also convert the classes in `y_train` and `y_test` to
# integers in [0..9]

dtrain = xgb.DMatrix(X_train, label=y_train.view(np.int32)-ord('A'))
dtest = xgb.DMatrix(X_test, label=y_test.view(np.int32)-ord('A'))

# ### Learning
#
# XGBoost has been used to obtain record-breaking results on many
# machine learning competitions, but have quite a lot of
# hyperparameters that need to be carefully tuned to get the best
# performance.
#
# For more information, see the documentation for XGBoost Parameters
# (https://xgboost.readthedocs.io/en/latest/parameter.html)

# instantiate params
params = {}

# general params
general_params = {'verbosity': 2}
params.update(general_params)

# booster params
booster_params = {'tree_method': 'gpu_hist'}
params.update(booster_params)

# learning task params
learning_task_params = {'objective': 'multi:softmax', 'num_class': 10}
params.update(learning_task_params)

print(params)

# We specify the number of boosted trees and are then ready to train
# our gradient boosted trees model on the GPU.

num_round = 100
clf_xgb = xgb.train(params, dtrain, num_round)

# ### Inference
#
# Inference is also run on the GPU.
#
# To match `y_test`, we also convert the predicted integer classes
# back to letters.

pred_xgb = clf_xgb.predict(dtest)
pred_xgb = [chr(x) for x in pred_xgb.astype(np.int32)+ord('A')]
pred_xgb = np.array(pred_xgb)
print('Predicted', len(pred_xgb),'letters with accuracy:',
      accuracy_score(y_test, pred_xgb))

# You can also use `show_failures()` to inspect the failures, and
# calculate the confusion matrix and other metrics as was done with
# the random forest above.
