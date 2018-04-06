
# coding: utf-8

# # Dogs-vs-cats classification with CNNs
# 
# In this notebook, we'll train a convolutional neural network (CNN, ConvNet) to classify images of dogs from images of cats using Keras (version $\ge$ 2 is required). This notebook is largely based on the blog post [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) by François Chollet.
# 
# **Note that using a GPU with this notebook is highly recommended.**
# 
# First, the needed imports. Keras tells us which backend (Theano, Tensorflow, CNTK) it will be using.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D 
from keras.preprocessing.image import (ImageDataGenerator, array_to_img, 
                                      img_to_array, load_img)
from keras import applications, optimizers

from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))


# ## Data
# 
# The training dataset consists of 2000 (or optionally 6000, 10000 or 24000) images of dogs and cats, split in half.  In addition, the validation set consists of 500 images of dogs and 500 images of cats.
# 
# ### Downloading the data

# In[ ]:


datapath = "/wrk/makoskel/dogs-vs-cats/train-2000"
(nimages_train, nimages_validation) = (2000, 1000)


# ### Data augmentation
# 
# First, we'll resize all training and validation images to a fized size. 
# 
# Then, to make the most of our limited number of training examples, we'll apply random transformations to them each time we are looping over them. This way, we "augment" our training dataset to contain more data. There are various transformations readily available in Keras, see [ImageDataGenerator](https://keras.io/preprocessing/image/) for more information.

# In[ ]:


input_image_size = (150, 150)

datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        #rotation_range=40,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        horizontal_flip=True)

noopgen = ImageDataGenerator(rescale=1./255)


# Let's see a couple of training images with and without the augmentation.

# In[ ]:


orig_generator = noopgen.flow_from_directory(
        datapath+'/train',  
        target_size=input_image_size,  
        batch_size=9)

augm_generator = datagen.flow_from_directory(
        datapath+'/train',  
        target_size=input_image_size,  
        batch_size=9)

for batch, _ in orig_generator:
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(batch[i,:,:,:])
        plt.suptitle('only resized training images', fontsize=16, y=0.93)
    plt.savefig("dvc-input-resized.png")
    break

for batch, _ in augm_generator:
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(batch[i,:,:,:])
        plt.suptitle('augmented training images', fontsize=16, y=0.93)
    plt.savefig("dvc-input-augmented.png")
    break

# ### Data loaders
# 
# Let's now define our real data loaders for training and validation data.

# In[ ]:


batch_size = 16

train_generator = datagen.flow_from_directory(
        datapath+'/train',  
        target_size=input_image_size,
        batch_size=batch_size, 
        class_mode='binary')

validation_generator = noopgen.flow_from_directory(
        datapath+'/validation',  
        target_size=input_image_size,
        batch_size=batch_size,
        class_mode='binary')


# ## Option 1: Train a small CNN from scratch
# 
# Similarly as with MNIST digits, we can start from scratch and train a CNN for the classification task. However, due to the small number of training images, a large network will easily overfit, regardless of the data augmentation.
# 
# ### Initialization

# In[ ]:


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_image_size+(3,), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())


# ### Learning

# In[ ]:


epochs = 20

history = model.fit_generator(train_generator,
                              steps_per_epoch=nimages_train // batch_size,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=nimages_validation // batch_size,
                              verbose=2)

model.save_weights("dvc-small-cnn.h5")

# In[ ]:


plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'], label='training')
plt.plot(history.epoch,history.history['val_loss'], label='validation')
plt.title('loss')
plt.legend(loc='best')
plt.savefig("dvc-small-cnn-loss.png")

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['acc'], label='training')
plt.plot(history.epoch,history.history['val_acc'], label='validation')
plt.title('accuracy')
plt.legend(loc='best');
plt.savefig("dvc-small-cnn-accuracy.png")
