
# coding: utf-8

# # 20 Newsgroups text classification with pre-trained word embeddings
# 
# In this notebook, we'll use pre-trained [GloVe word embeddings](http://nlp.stanford.edu/projects/glove/) for text classification using Keras (version $\ge$ 2 is required). This notebook is largely based on the blog post [Using pre-trained word embeddings in a Keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) by François Chollet.
# 
# **Note that using a GPU with this notebook is highly recommended.**
# 
# First, the needed imports. Keras tells us which backend (Theano, Tensorflow, CNTK) it will be using.

# In[ ]:


from keras.preprocessing import sequence, text
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import LSTM
from keras.utils import to_categorical

from distutils.version import LooseVersion as LV
from keras import __version__
from keras import backend as K

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))


# ## GloVe word embeddings

# In[ ]:


GLOVE_DIR = "/wrk/makoskel/glove.6B"

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

print('Examples of embeddings:')
for w in ['some', 'random', 'words']:
    print(w, embeddings_index[w])


# ## 20 Newsgroups data set
# 
# Next we'll load the [20 Newsgroups](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html) data set. 
# 
# The dataset contains 20000 messages collected from 20 different Usenet newsgroups (1000 messages from each group):
# 
# * alt.atheism
# * talk.politics.guns
# * talk.politics.mideast
# * talk.politics.misc
# * talk.religion.misc
# * soc.religion.christian
# * comp.sys.ibm.pc.hardware
# * comp.graphics
# * comp.os.ms-windows.misc
# * comp.sys.mac.hardware
# * comp.windows.x
# * rec.autos
# * rec.motorcycles
# * rec.sport.baseball
# * rec.sport.hockey
# * sci.crypt
# * sci.electronics
# * sci.space
# * sci.med
# * misc.forsale

# In[ ]:


TEXT_DATA_DIR = "/wrk/makoskel/20_newsgroup"

print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)

print('Found %s texts.' % len(texts))


# First message and its label:

# In[ ]:


print(texts[0])
print('label:', labels[0], labels_index)


# Vectorize the text samples into a 2D integer tensor.

# In[ ]:


MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 1000 

tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# Split the data into a training set and a validation set

# In[ ]:


VALIDATION_SPLIT = 0.2

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
print('Shape of training data tensor:', x_train.shape)
print('Shape of training label tensor:', y_train.shape)
print('Shape of validation data tensor:', x_val.shape)
print('Shape of validation label tensor:', y_val.shape)


# Prepare embedding matrix

# In[ ]:


print('Preparing embedding matrix.')

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_dim = 100

embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
print('Shape of embedding matrix:', embedding_matrix.shape)


# ## 1-D CNN
# 
# ### Initialization
# 

# In[ ]:


print('Build model...')
model = Sequential()

model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))
#model.add(Dropout(0.2))

model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())

model.add(Dense(128, activation='relu'))
model.add(Dense(20, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())




# ### Learning

# In[ ]:


epochs = 10

history = model.fit(x_train, y_train, batch_size=128,
                    epochs=epochs,
                    validation_data=(x_val, y_val))

model.save_weights("20ng-1dcnn.h5")

# In[ ]:


plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'], label='training')
plt.plot(history.epoch,history.history['val_loss'], label='validation')
plt.title('loss')
plt.legend(loc='best')
plt.savefig("20ng-1dcnn-loss.png")

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['acc'], label='training')
plt.plot(history.epoch,history.history['val_acc'], label='validation')
plt.title('accuracy')
plt.legend(loc='best')
plt.savefig("20ng-1dcnn-accuracy.png")


# ## LSTM
# 
# ### Initialization

# In[ ]:


print('Build model...')
model = Sequential()

model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))
#model.add(Dropout(0.2))

model.add(LSTM(128))

model.add(Dense(128, activation='relu'))
model.add(Dense(20, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())



# ### Learning

# In[ ]:


epochs = 10

history = model.fit(x_train, y_train, batch_size=128,
                    epochs=epochs,
                    validation_data=(x_val, y_val))

model.save_weights("20ng-lstm.h5")


# In[ ]:


plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'], label='training')
plt.plot(history.epoch,history.history['val_loss'], label='validation')
plt.title('loss')
plt.legend(loc='best')
plt.savefig("20ng-lstm-loss.png")

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['acc'], label='training')
plt.plot(history.epoch,history.history['val_acc'], label='validation')
plt.title('accuracy')
plt.legend(loc='best')
plt.savefig("20ng-lstm-accuracy.png")

