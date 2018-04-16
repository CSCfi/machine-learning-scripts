
# coding: utf-8

# # Ted talks keyword labeling with pre-trained word embeddings
# 
# In this notebook, we'll use pre-trained [GloVe word embeddings](http://nlp.stanford.edu/projects/glove/) for keyword labeling using Keras (version $\ge$ 2 is required). This notebook is largely based on the blog post [Using pre-trained word embeddings in a Keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) by François Chollet.
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

import xml.etree.ElementTree as ET
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
# 
# Let's begin by loading a datafile containing pre-trained word embeddings.  The datafile contains 100-dimensional embeddings for 400,000 English words.  

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


# ## Ted talks data set
# 
# Next we'll load the Ted talks data set. 
# 
# The dataset contains transcripts and metadata of 2085 Ted talks. Each talk is annotated with a set of keywords. In this notebook, we'll use the 10 most common keywords.

# In[ ]:


TEXT_DATA_DIR = "/wrk/makoskel/ted"

keywords = {"technology": 0, "culture": 1, "science": 2, "global issues": 3, "design": 4, 
            "business": 5, "entertainment": 6, "arts": 7, "education": 8, "politics": 9}

print('Processing xml')

tree = ET.parse(TEXT_DATA_DIR+"/ted_en-20160408.xml")
root = tree.getroot() 

texts = []  # list of text samples
labels = []  # list of label ids

for i in root:
    l = np.zeros(10)
    for j in i.findall("./head/keywords"):
        kws = j.text.split(",")
        kws = [x.strip() for x in kws]
        for k in kws:
            if k in keywords:
                l[keywords[k]] = 1.
        labels.append(l)
    for c in i.findall("./content"):
        texts.append(c.text)

print('Found %s texts, %s labels.' % (len(texts), len(labels)))


# First talk and its labels:

# In[ ]:


print(texts[0])
print('labels:', labels[0])


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
labels = np.asarray(labels)

print('Shape of data tensor:', data.shape)
print('Shape of labels tensor:', labels.shape)


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
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop')

print(model.summary())



# ### Learning

# In[ ]:


epochs = 20

history = model.fit(x_train, y_train, batch_size=16,
                    epochs=epochs,
                    validation_data=(x_val, y_val))

model.save_weights("ted-1dcnn.h5")


# In[ ]:


plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'], label='training')
plt.plot(history.epoch,history.history['val_loss'], label='validation')
plt.title('loss')
plt.legend(loc='best');
plt.savefig("ted-1dcnn-loss.png")


# To further analyze the results, we can produce the actual predictions for the validation data.

# In[ ]:


predictions = model.predict(x_val)


# Let's look at the correct and predicted labels for some talks in the validation set.

# In[ ]:


threshold = 0.5
nb_talks = 10

inv_keywords = {v: k for k, v in keywords.items()}
for t in range(nb_talks):
    print(t,':')
    print('    correct: ', end='')
    for idx in np.where(y_val[t]>0.5)[0].tolist():
        sys.stdout.write('['+inv_keywords[idx]+'] ')
    print()
    print('  predicted: ', end='')
    for idx in np.where(predictions[t]>threshold)[0].tolist():
        sys.stdout.write('['+inv_keywords[idx]+'] ')
    print()


# Scikit-learn has some applicable [multilabel ranking metrics](http://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics) we can try: 

# In[ ]:


from sklearn.metrics import coverage_error, label_ranking_average_precision_score
print('Coverage:', coverage_error(y_val, predictions))
print('LRAP:', label_ranking_average_precision_score(y_val, predictions))


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
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop')

print(model.summary())



# ### Learning

# In[ ]:


epochs = 5

history = model.fit(x_train, y_train, batch_size=16,
                    epochs=epochs,
                    validation_data=(x_val, y_val))

model.save_weights("ted-lstm.h5")


# In[ ]:


plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'], label='training')
plt.plot(history.epoch,history.history['val_loss'], label='validation')
plt.title('loss')
plt.legend(loc='best');
plt.savefig("ted-lstm-loss.png")


# In[ ]:


predictions = model.predict(x_val)


# In[ ]:


threshold = 0.5
nb_talks = 10

inv_keywords = {v: k for k, v in keywords.items()}
for t in range(nb_talks):
    print(t,':')
    print('    correct: ', end='')
    for idx in np.where(y_val[t]>0.5)[0].tolist():
        sys.stdout.write('['+inv_keywords[idx]+'] ')
    print()
    print('  predicted: ', end='')
    for idx in np.where(predictions[t]>threshold)[0].tolist():
        sys.stdout.write('['+inv_keywords[idx]+'] ')
    print()


# In[ ]:


from sklearn.metrics import coverage_error, label_ranking_average_precision_score
print('Coverage:', coverage_error(y_val, predictions))
print('LRAP:', label_ranking_average_precision_score(y_val, predictions))

