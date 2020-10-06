#!/usr/bin/env python
# coding: utf-8

# # notMNIST letters classification with ensembles of decision trees 
# 
# In this notebook, we'll use two different ensembles of decision trees: [random forest](https://docs.rapids.ai/api/cuml/stable/api.html#random-forest) and [gradient boosted trees](https://xgboost.readthedocs.io/en/latest/) to classify notMNIST letters using a GPU, the [RAPIDS](https://rapids.ai/) libraries (cudf, cuml) and [XGBoost](https://xgboost.readthedocs.io/en/latest/).
# 
# **Note that a GPU is required with this notebook.**
# 
# This version of the notebook has been tested with RAPIDS version 0.15.
# 
# First, the needed imports. 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pml_utils import show_failures

import cudf
import numpy as np
import pandas as pd

import os
import urllib.request

from cuml.ensemble import RandomForestClassifier
import xgboost as xgb
from cuml import __version__ as cuml_version

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import __version__ as sklearn_version

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

print('Using cudf version:', cudf.__version__)
print('Using cuml version:', cuml_version)
print('Using sklearn version:', sklearn_version)


# Then we load the notMNIST data. First time we need to download the data, which can take a while. The data is stored as Numpy arrays in host (CPU) memory.

# In[ ]:


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


# In[ ]:


# Load notMNIST
DATA_DIR = os.path.expanduser('~/data/notMNIST/')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
X_train = load_not_mnist(DATA_DIR, 'notMNIST_large_images.npy').reshape(-1, 28*28)
y_train = load_not_mnist(DATA_DIR, 'notMNIST_large_labels.npy')
X_test = load_not_mnist(DATA_DIR, 'notMNIST_small_images.npy').reshape(-1, 28*28)
y_test = load_not_mnist(DATA_DIR, 'notMNIST_small_labels.npy')


# In[ ]:


X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

print()
print('notMNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', type(X_train), 'shape:', X_train.shape, X_train.dtype)
print('y_train:', type(y_train), 'shape:', y_train.shape, y_train.dtype)
print('X_test:', type(X_test), 'shape:', X_test.shape)
print('y_test:', type(y_test), 'shape:', y_test.shape)


# ## Random forest
# 
# Random forest is an ensemble (or a group; hence the name *forest*) of decision trees, obtained by introducing randomness into the tree generation. The prediction of the random forest is obtained by *averaging* the predictions of the individual trees.
# 
# ### Data
# 
# Let's convert our training data to cuDF DataFrames in device (GPU) memory. We will also convert the classes in `y_train` to integers in 
# $[0 \mathrel{{.}\,{.}} 9]$. 
# 
# We do not explicitly need to convert the test data as the GPU-based inference functionality will take care of it.

# In[ ]:


cu_X_train = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
cu_y_train = cudf.Series(y_train.view(np.int32)-ord('A'))

print('cu_X_train:', type(cu_X_train), 'shape:', cu_X_train.shape)
print('cu_y_train:', type(cu_y_train), 'shape:', cu_y_train.shape)


# ### Learning
# 
# Random forest classifiers are quick to train, quite robust to hyperparameter values, and often work relatively well.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nn_estimators = 100\nmax_depth = 16\nclf_rf = RandomForestClassifier(n_estimators=n_estimators,\n                                max_depth=max_depth)\nclf_rf.fit(cu_X_train, cu_y_train)')


# ### Inference
# 
# We will use GPU-based inference to predict the classes for the test data.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\npred_rf = clf_rf.predict(X_test, predict_model='GPU')\npred_rf = [chr(x) for x in pred_rf+ord('A')]\npred_rf = np.array(pred_rf)\n\nprint('Predicted', len(pred_rf), 'digits with accuracy:',\n      accuracy_score(y_test, pred_rf))")


# #### Failure analysis
# 
# The random forest classifier worked quite well, so let's take a closer look.

# Here are the first 10 test digits the random forest model classified to a wrong class:

# In[ ]:


show_failures(pred_rf, y_test, X_test)


# We can use `show_failures()` to inspect failures in more detail. For example:
# 
# * show failures in which the true class was "5":

# In[ ]:


show_failures(pred_rf, y_test, X_test, trueclass="F")


# * show failures in which the prediction was "A":

# In[ ]:


show_failures(pred_rf, y_test, X_test, predictedclass="A")


# * show failures in which the true class was "A" and the prediction was "C":

# In[ ]:


show_failures(pred_rf, y_test, X_test, trueclass="A", predictedclass="C")


# #### Confusion matrix, accuracy, precision, and recall
# 
# We can also compute the confusion matrix to see which digits get mixed the most, and look at classification accuracies separately for each class:

# In[ ]:


labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
print('Confusion matrix (rows: true classes; columns: predicted classes):'); print()
cm=confusion_matrix(y_test, pred_rf, labels=labels)
print(cm); print()

print('Classification accuracy for each class:'); print()
for i,j in enumerate(cm.diagonal()/cm.sum(axis=1)): print("%d: %.4f" % (i,j))


# Precision and recall for each class:

# In[ ]:


print(classification_report(y_test, pred_rf, labels=labels))


# ## Gradient boosted trees (XGBoost)
# 
# Gradient boosted trees (or extreme gradient boosted trees) is another way of constructing ensembles of decision trees, using the *boosting* framework.  Here we use the GPU accelerated [XGBoost](http://xgboost.readthedocs.io/en/latest/) library to train gradient boosted trees to classify MNIST digits. 
# 
# ### Data
# 
# We begin by converting our training and test data to XGBoost's internal `DMatrix` data structures. 
# 
# We will also convert the classes in `y_train` and `y_test` to integers in $[0 \mathrel{{.}\,{.}} 9]$.

# In[ ]:


dtrain = xgb.DMatrix(X_train, label=y_train.view(np.int32)-ord('A'))
dtest = xgb.DMatrix(X_test, label=y_test.view(np.int32)-ord('A'))


# ### Learning
# 
# XGBoost has been used to obtain record-breaking results on many machine learning competitions, but have quite a lot of hyperparameters that need to be carefully tuned to get the best performance.
# 
# For more information, see the documentation for [XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html).

# In[ ]:


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


# We specify the number of boosted trees and are then ready to train our gradient boosted trees model on the GPU.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nnum_round = 100\nclf_xgb = xgb.train(params, dtrain, num_round)')


# ### Inference
# 
# Inference is also run on the GPU. 
# 
# To match `y_test`, we also convert the predicted integer classes back to letters.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\npred_xgb = clf_xgb.predict(dtest)\npred_xgb = [chr(x) for x in pred_xgb.astype(np.int32)+ord('A')]\npred_xgb = np.array(pred_xgb)\nprint('Predicted', len(pred_xgb), 'letters with accuracy:', accuracy_score(y_test, pred_xgb))")


# You can also use `show_failures()` to inspect the failures, and calculate the confusion matrix and other metrics as was done with the random forest above.
# 
# ## Model tuning

# Study the documentation of the different decision tree models used in this notebook ([cuml random forest](https://docs.rapids.ai/api/cuml/stable/api.html#random-forest) and [XGBoost gradient boosted trees](https://xgboost.readthedocs.io/en/latest/)), and experiment with different hyperparameter values.  
# 
# Report the highest classification accuracy you manage to obtain for each model type.  Also mark down the parameters you used, so others can try to reproduce your results. 

# In[ ]:




