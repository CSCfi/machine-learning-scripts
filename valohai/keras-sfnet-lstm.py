#!/usr/bin/env python3

from keras.preprocessing import sequence, text
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
# from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import CuDNNLSTM
from keras.utils import to_categorical

from distutils.version import LooseVersion as LV
from keras import __version__
from keras import backend as K

# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

import os
import gzip
import re
import pickle

import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

FASTTEXT_FILE = "/valohai/inputs/embedding/cc.fi.300.vec.gz"
TEXT_DATA_DIR = "/valohai/inputs/dataset/sfnet2007-2008/raw_texts/"


# if K.backend() == "tensorflow":
#     # import tensorflow as tf
#     from keras.callbacks import TensorBoard
#     import datetime
#     logdir = os.path.join(os.getcwd(), "logs",
#                           "sfnet-lstm-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
#     print('TensorBoard log directory:', logdir)
#     os.makedirs(logdir)
#     callbacks = [TensorBoard(log_dir=logdir)]
# else:
callbacks = None


# Finnish word embeddings
#
# TODO: try also
# http://bionlp.utu.fi/finnish-internet-parsebank.html ?
#

pickle_name = 'fasttext.cc.fi.300.pickle'

if os.path.isfile(pickle_name):
    with open(pickle_name, 'rb') as f:
        embeddings_index = pickle.load(f)
    print('Loaded word vectors from {}.'.format(pickle_name))
else:
    print('Indexing word vectors.')

    embeddings_index = {}

    with gzip.open(FASTTEXT_FILE, 'rt', encoding='utf-8') as f:
        num_lines, dim = (int(x) for x in f.readline().rstrip().split())
        print('{} has {} words with {}-dimensional embeddings.'.format(
            os.path.basename(FASTTEXT_FILE), num_lines, dim))

        for line in tqdm(f, total=num_lines):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            assert coefs.shape[0] == dim
            embeddings_index[word] = coefs

        assert len(embeddings_index) == num_lines

    # with open(pickle_name, 'wb') as f:
    #     # Pickle the 'data' dictionary using the highest protocol available.
    #     pickle.dump(embeddings_index, f, pickle.HIGHEST_PROTOCOL)

# FASTTEXT_FILE = "/media/data/yle-embeddings/fasttext_fin.csv.gz"

# with gzip.open(FASTTEXT_FILE, 'rt', encoding='utf-8') as f:
#     f.readline()
#     # num_lines, dim = (int(x) for x in f.readline().rstrip().split())
#     # print('{} has {} words with {}-dimensional embeddings.'.format(
#     #     os.path.basename(FASTTEXT_FILE), num_lines, dim))

#     for line in tqdm(f, total=880327):
#         values = line.split(',')
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         assert coefs.shape[0] == 100
#         embeddings_index[word] = coefs

#     # assert len(embeddings_index) == num_lines

# print('Loaded {} embeddings'.format(len(embeddings_index)))
# print('Examples of embeddings:')
# for w in ['jotain', 'satunnaisia', 'sanoja']:
#     print(w, embeddings_index[w])

print('Examples of embeddings:')
for w in ['jotain', 'satunnaisia', 'sanoja']:
    print(w, embeddings_index[w])

# SFNet data set

print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        print(name, label_id)
        for fname in sorted(os.listdir(path)):
            print('*', fname)
            if fname.endswith('.gz'):
                fpath = os.path.join(path, fname)
                with gzip.open(fpath, 'rt', encoding='latin-1') as f:
                    header = True  # keep track if we are in header area, or in message
                    t = ''  # accumulate current message into t
                    prev_line = None
                    for line in f:
                        m = re.match(r'^([a-zA-Z]+): (.*)$', line)
                        if m and m.group(1) in ['Path', 'Subject', 'From', 'Newsgroups']:
                            # yes, we are definitely inside a header now...
                            header = True
                            if t != '':  # if we have accumulated text, we save it
                                texts.append(t)
                                labels.append(label_id)
                                t = ''
                                continue
                        # empty line indicates end of headers
                        if line == '\n' and header:
                            header = False
                            continue

                        # if not a header, accumulate line to text in t
                        if not header:
                            t += line

                        prev_line = line

                    if t != '':  # store also the last message
                        texts.append(t)
                        labels.append(label_id)

print('Found %s texts.' % len(texts))

# First message and its label:

print(texts[0])
print('label:', labels[0], labels_index)

# Vectorize the text samples into a 2D integer tensor.

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

VALIDATION_SET, TEST_SET = 1000, 4000

x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=TEST_SET,
                                                    shuffle=True, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=VALIDATION_SET,
                                                  shuffle=False)

print('Shape of training data tensor:', x_train.shape)
print('Shape of training label tensor:', y_train.shape)
print('Shape of validation data tensor:', x_val.shape)
print('Shape of validation label tensor:', y_val.shape)
print('Shape of test data tensor:', x_test.shape)
print('Shape of test label tensor:', y_test.shape)


# Prepare the embedding matrix:

print('Preparing embedding matrix.')

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_dim = 300
not_found = 0

embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        not_found += 1

print('Shape of embedding matrix:', embedding_matrix.shape)
print('Number of words not found in embedding index:', not_found)

print('Build model...')
model = Sequential()

model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))
model.add(Dropout(0.5))

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(CuDNNLSTM(128))

model.add(Dense(128, activation='relu'))
model.add(Dense(9, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())


# Learning

epochs = 20
batch_size = 128

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    verbose=2, callbacks=callbacks)
model.save('/valohai/outputs/sfnet-lstm-fasttext_cc300-epochs{}.h5'.format(epochs))

# Inference
scores = model.evaluate(x_test, y_test, verbose=2)
print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


predictions = model.predict(x_test)

cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1),
                      labels=list(range(9)))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print('Classification accuracy for each newsgroup:')
print()
labels = [l[0] for l in sorted(labels_index.items(), key=lambda x: x[1])]
for i, j in enumerate(cm.diagonal()/cm.sum(axis=1)):
    print("%s: %.4f" % (labels[i].ljust(26), j))
print()

print('Confusion matrix (rows: true newsgroup; columns: predicted newsgroup):')
print()
np.set_printoptions(linewidth=9999)
print(cm)
print()
