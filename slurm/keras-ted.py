
# coding: utf-8

# # Ted talks keyword labeling with pre-trained word embeddings
# 
# In this script, we'll use pre-trained [GloVe word embeddings]
# (http://nlp.stanford.edu/projects/glove/) for keyword labeling using
# Keras (version $\ge$ 2 is required). This script is largely based on
# the blog post [Using pre-trained word embeddings in a Keras model]
# (https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
# by François Chollet.
# 
# **Note that using a GPU with this script is highly recommended.**
# 
# First, the needed imports. Keras tells us which backend (Theano,
# Tensorflow, CNTK) it will be using.

from keras.preprocessing import sequence, text
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import LSTM, CuDNNLSTM
from keras.utils import to_categorical

from distutils.version import LooseVersion as LV
from keras import __version__
from keras import backend as K

from sklearn import metrics

import xml.etree.ElementTree as ET
import os
import sys

import numpy as np

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

# If we are using TensorFlow as the backend, we can use TensorBoard to
# visualize our progress during training.

if K.backend() == "tensorflow":
    import tensorflow as tf
    from keras.callbacks import TensorBoard
    import os, datetime
    logdir = os.path.join(os.getcwd(), "logs",
                     "ted-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory:', logdir)
    os.makedirs(logdir)
    callbacks = [TensorBoard(log_dir=logdir)]
else:
    callbacks =  None

# ## GloVe word embeddings
# 
# Let's begin by loading a datafile containing pre-trained word
# embeddings.  The datafile contains 100-dimensional embeddings for
# 400,000 English words.

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

# ## Ted talks data set
# 
# Next we'll load the Ted talks data set. 
# 
# The dataset contains transcripts and metadata of 2085 Ted
# talks. Each talk is annotated with a set of keywords. In this
# notebook, we'll use the 10 most common keywords.

TEXT_DATA_DIR = "/wrk/makoskel/ted"

keywords = {"technology": 0, "culture": 1, "science": 2, "global issues": 3, "design": 4, 
            "business": 5, "entertainment": 6, "arts": 7, "education": 8, "politics": 9}

print('Processing xml')

tree = ET.parse(TEXT_DATA_DIR+"/ted_en-20160408.xml")
root = tree.getroot() 

talks = []

for i in root:
    labels = np.zeros(10)
    kws = i.find("./head/keywords").text.split(",")
    kws = [x.strip() for x in kws]
    for k in kws:
        if k in keywords:
            labels[keywords[k]] = 1.
    title = i.find("./head/title").text
    date = i.find("./head/date").text
    description = i.find("./head/description").text
    content = i.find("./content").text
    
    talks.append({"title": title, "date": date, "description": description,
                  "content": content, "labels": labels})

print('Found %s talks.' % len(talks))

l = np.empty(len(talks))
for i in range(len(talks)):
    l[i]=np.sum(talks[i]["labels"])
nlabels_mean = np.mean(l)

# Let's take a look at *i*th talk:

i = 3
print('* Title:', talks[i]["title"])
print('* Date:', talks[i]["date"])
print('* Description:', talks[i]["description"])
print('* Content:', talks[i]["content"][:1000], '...',
      '(%d chars in total)' % len(talks[i]["content"])) 
print('* Labels:', talks[i]["labels"])
print('* Keywords:')
kwlist = sorted(keywords, key=lambda key: keywords[key])
for l, lval in enumerate(talks[i]["labels"]):
    if lval>0.5:
        print('  ['+kwlist[l]+']')

# Now we decide whether to use the `description` or `content` fields.

texttype = "content"
#texttype = "description"

# Vectorize the text samples into a 2D integer tensor.

MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 1000 

tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts([x[texttype] for x in talks])
sequences = tokenizer.texts_to_sequences([x[texttype] for x in talks])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.asarray([x['labels'] for x in talks])

print('Shape of data tensor:', data.shape)
print('Shape of labels tensor:', labels.shape)

# Split the data into a training set and a validation set.  Note that
# we do not use a separate test set in this notebook, due to the small
# size of the dataset.

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

# Prepare the embedding matrix:

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

model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop')

print(model.summary())

# ### Learning

epochs = 20
batch_size=16

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    verbose=2, callbacks=callbacks)

# ### Inference
# 
# To further analyze the results, we can produce the actual
# predictions for the validation data.

predictions = model.predict(x_val)

# Let's look at the correct and predicted labels for some talks in the
# validation set.

threshold = 0.5
nb_talks_to_show = 20

inv_keywords = {v: k for k, v in keywords.items()}
for t in range(nb_talks_to_show):
    print(t,':')
    print('    correct: ', end='')
    for idx in np.where(y_val[t]>0.5)[0].tolist():
        sys.stdout.write('['+inv_keywords[idx]+'] ')
    print()
    print('  predicted: ', end='')
    for idx in np.where(predictions[t]>threshold)[0].tolist():
        sys.stdout.write('['+inv_keywords[idx]+'] ')
    print()

# Scikit-learn has some applicable performance [metrics]
# (http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
# we can try:

print('Precision: {0:.3f} (threshold: {1:.2f})'
      .format(metrics.precision_score(y_val.flatten(), predictions.flatten()>threshold), threshold))
print('Recall: {0:.3f} (threshold: {1:.2f})'
      .format(metrics.recall_score(y_val.flatten(), predictions.flatten()>threshold), threshold))
print('F1 score: {0:.3f} (threshold: {1:.2f})'
      .format(metrics.f1_score(y_val.flatten(), predictions.flatten()>threshold), threshold))

average_precision = metrics.average_precision_score(y_val.flatten(), predictions.flatten())
print('Average precision: {0:.3f}'.format(average_precision))
print('Coverage: {0:.3f}, optimal: {1:.3f}'
      .format(metrics.coverage_error(y_val, predictions), nlabels_mean))
print('LRAP: {0:.3f}'
      .format(metrics.label_ranking_average_precision_score(y_val, predictions)))

# ## LSTM
# 
# ### Initialization

print('Build model...')
model = Sequential()

model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))
#model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))

model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop')

print(model.summary())

# ### Learning

epochs = 20
batch_size=16

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    verbose=2, callbacks=callbacks)

# ### Inference

# To further analyze the results, we can produce the actual
# predictions for the validation data.

predictions = model.predict(x_val)

# Let's look at the correct and predicted labels for some talks in the
# validation set.

threshold = 0.5
nb_talks_to_show = 20

inv_keywords = {v: k for k, v in keywords.items()}
for t in range(nb_talks_to_show):
    print(t,':')
    print('    correct: ', end='')
    for idx in np.where(y_val[t]>0.5)[0].tolist():
        sys.stdout.write('['+inv_keywords[idx]+'] ')
    print()
    print('  predicted: ', end='')
    for idx in np.where(predictions[t]>threshold)[0].tolist():
        sys.stdout.write('['+inv_keywords[idx]+'] ')
    print()


# Scikit-learn has some applicable performance [metrics]
# (http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
# we can try:

print('Precision: {0:.3f} (threshold: {1:.2f})'
      .format(metrics.precision_score(y_val.flatten(), predictions.flatten()>threshold), threshold))
print('Recall: {0:.3f} (threshold: {1:.2f})'
      .format(metrics.recall_score(y_val.flatten(), predictions.flatten()>threshold), threshold))
print('F1 score: {0:.3f} (threshold: {1:.2f})'
      .format(metrics.f1_score(y_val.flatten(), predictions.flatten()>threshold), threshold))

average_precision = metrics.average_precision_score(y_val.flatten(), predictions.flatten())
print('Average precision: {0:.3f}'.format(average_precision))
print('Coverage: {0:.3f}, optimal: {1:.3f}'
      .format(metrics.coverage_error(y_val, predictions), nlabels_mean))
print('LRAP: {0:.3f}'
      .format(metrics.label_ranking_average_precision_score(y_val, predictions)))
