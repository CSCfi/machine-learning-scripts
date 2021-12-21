#!/usr/bin/env python
# coding: utf-8

# # Dogs-vs-cats classification with ViT
# 
# In this notebook, we'll finetune a [Vision Transformer](https://arxiv.org/abs/2010.11929) (ViT) to classify images of dogs from images of cats using TensorFlow 2 / Keras and HuggingFace's [Transformers](https://github.com/huggingface/transformers). 
# 
# **Note that using a GPU with this notebook is highly recommended.**
# 
# First, the needed imports.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from transformers import ViTFeatureExtractor, TFViTForImageClassification
from transformers.utils import check_min_version
from transformers import __version__ as transformers_version

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from PIL import Image

import os, sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

check_min_version("4.13.0.dev0")

print('Using TensorFlow version:', tf.__version__,
      'Keras version:', tf.keras.__version__,
      'Transformers version:', transformers_version)


# ## Data
# 
# The training dataset consists of 2000 images of dogs and cats, split in half.  In addition, the validation set consists of 1000 images, and the test set of 22000 images.  Here are some random training images:
# 
# ![title](imgs/dvc.png)

# In[ ]:


datapath = "/media/data/dogs-vs-cats/train-2000/"
nimages = {'train':2000, 'validation':1000, 'test':22000}


# ### Image paths and labels

# In[ ]:


def get_paths(dataset):
    data_root = pathlib.Path(datapath+dataset)
    image_paths = list(data_root.glob('*/*'))
    image_paths = [str(path) for path in image_paths]
    image_count = len(image_paths)
    assert image_count == nimages[dataset], "Found {} images, expected {}".format(image_count, nimages[dataset])
    return image_paths

image_paths = dict()
image_paths['train'] = get_paths('train')
image_paths['validation'] = get_paths('validation')
image_paths['test'] = get_paths('test')


# In[ ]:


label_names = sorted(item.name for item in pathlib.Path(datapath+'train').glob('*/') if item.is_dir())
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
# We now define a function to load the images. 

# In[ ]:


def pil_loadimg(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")
    
def pil_loader(imglist: list):
    res = []
    for i in imglist:
        res.append(pil_loadimg(i))
    return res


# Next we specify the pre-trained ViT model we are going to use. The model [`"google/vit-base-patch16-224"`](https://huggingface.co/google/vit-base-patch16-224) is pre-trained on ImageNet-21k (14 million images, 21,843 classes) at resolution 224x224, and fine-tuned on ImageNet 2012 (1 million images, 1,000 classes) at resolution 224x224. 
# 
# We'll use a pre-trained ViT feature extractor that matches the ViT model to preprocess the input images. 

# In[ ]:


VITMODEL = 'google/vit-base-patch16-224'

feature_extractor = ViTFeatureExtractor.from_pretrained(VITMODEL)


# We load and preprocess the training and validation images:

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ninputs_train = feature_extractor(images=pil_loader(image_paths[\'train\']),\n                                 return_tensors="tf")\ninputs_validation = feature_extractor(images=pil_loader(image_paths[\'validation\']),\n                                      return_tensors="tf")')


# ### TF Datasets
# 
# Let's now define our TF `Dataset`s for training and validation data. 

# In[ ]:


BATCH_SIZE = 32

dataset_train = tf.data.Dataset.from_tensor_slices((inputs_train.data, image_labels['train']))
dataset_train = dataset_train.shuffle(len(dataset_train)).batch(BATCH_SIZE,
                                                                drop_remainder=True)
dataset_validation = tf.data.Dataset.from_tensor_slices((inputs_validation.data,
                                                         image_labels['validation']))
dataset_validation = dataset_validation.batch(BATCH_SIZE, drop_remainder=True)


# ## Model
# 
# ### Initialization

# In[ ]:


model = TFViTForImageClassification.from_pretrained(VITMODEL, num_labels=1,
                                                    ignore_mismatched_sizes=True)


# In[ ]:


LR = 1e-5

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metric = 'accuracy'

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

print(model.summary())


# ### Learning

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nEPOCHS = 4\n\nhistory = model.fit(dataset_train,\n                    validation_data=dataset_validation,\n                    epochs=EPOCHS, verbose=2) #callbacks=callbacks)')


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))

ax1.plot(history.epoch,history.history['loss'], label='training')
ax1.plot(history.epoch,history.history['val_loss'], label='validation')
ax1.set_title('loss')
ax1.set_xlabel('epoch')
ax1.legend(loc='best')

ax2.plot(history.epoch,history.history['accuracy'], label='training')
ax2.plot(history.epoch,history.history['val_accuracy'], label='validation')
ax2.set_title('accuracy')
ax2.set_xlabel('epoch')
ax2.legend(loc='best');


# ### Inference
# 
# We now evaluate the model using the test set. First we'll load and preprocess the test images and define the TF `Dataset`. Due to memory issues, we limit the evaluation to 5000 images.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ninputs_test = feature_extractor(images=pil_loader(image_paths[\'test\'][:5000]),\n                                return_tensors="tf")\ndataset_test = tf.data.Dataset.from_tensor_slices((inputs_test.data,\n                                                   image_labels[\'test\'][:5000]))\ndataset_test = dataset_test.batch(BATCH_SIZE, drop_remainder=True)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nscores = model.evaluate(dataset_test, verbose=2)\nprint("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))')


# Alternatively, we could evaluate the model using the validation set
%%time

scores = model.evaluate(dataset_validation, verbose=2)
print("Validation set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))