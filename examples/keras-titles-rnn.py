
# coding: utf-8

# # Book and movie title generation with RNNs
# 
# In this notebook, we'll train a character-level recurrent neural network (RNN) to generate book and movie titles using Keras (with either Theano or Tensorflow as the compute backend).  Keras version $\ge$ 2 is required. This notebook is based on the Keras text generation example found [here](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py).
# 
# First, the needed imports. Keras tells us which backend (Theano or Tensorflow) it will be using.

import sys
import os
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.4/site-packages/'))

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU 
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot

import numpy as np
import random
import sys
#import matplotlib.pyplot as plt
#import seaborn as sns

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))


# Next, let's load our training data.  The data consists of movie and book titles originally downloaded from [here](https://github.com/markriedl/WikiPlots).  For our purposes, the data has been slightly modified to reduce the number of rare characters. 

path = get_file('titles-translated', origin='https://kannu.csc.fi/s/Md68oCy6l62CuKC/download')
text = open(path).read().lower()

# This can be used to reduce the size of training data
text = text[:500000]

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

textlines = text.splitlines()

print()
print('Corpus length:', len(text), 'lines:', len(textlines))
print('First 10 lines:', textlines[:10])
print()
print('Number of unique chars:', len(chars))
print(chars)

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 10
step = 3

sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))
print('First 10 sequences and next chars:')
for i in range(10):
    print('[{}]:[{}]'.format(sentences[i], next_chars[i]))




print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
print('Size of X: {:.2f} MB'.format(X.nbytes/1024/1024))
print('Size of y: {:.2f} MB'.format(y.nbytes/1024/1024))


# ### Initialization
# 
# Now we are ready to create a recurrent model.  Keras contains three types of recurrent layers:
# 
#  * `SimpleRNN`, a fully-connected RNN where the output is fed back to input.
#  * `LSTM`, the Long-Short Term Memory unit layer.
#  * `GRU`, the Gated Recurrent Unit layer.
# 
# See https://keras.io/layers/recurrent/ for more information.

# Number of hidden units to use:
nb_units = 128

model = Sequential()

# Recurrent layers supported: SimpleRNN, LSTM, GRU:
model.add(LSTM(nb_units,
                    input_shape=(maxlen, len(chars))))

# To stack multiple RNN layers, all RNN layers except the last one need
# to have "return_sequences=True".  An example of using two RNN layers:
#model.add(SimpleRNN(16,
#                    input_shape=(maxlen, len(chars)),
#                    return_sequences=True))
#model.add(SimpleRNN(32))

model.add(Dense(units=len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer)

print(model.summary())


#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


# ### Learning
# 
# Let's first define a helper function to sample the next character. 

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# Now let's train the RNN model.
# 
# This is a relatively complex model, so training (especially with LSTM and GRU layers) can be quite slow without GPUs. 

lossv = []


epochs = 10

for iteration in range(0, epochs):
    print();print()
    print('######', iteration)
    history = model.fit(X, y, 
                        epochs=1, 
                        batch_size=512,
                        verbose=2)
    lossv.append(history.history['loss'])
    
    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print()
        print('----- Generating with diversity', diversity, 'seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(100):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()


#plt.figure(figsize=(5,3))
#plt.plot(lossv)
#plt.title('loss');


# ### Inference 

print('###### INFERENCE ######')

diversity = 0.8
sentence = " " * 10

for i in range(1000):
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()





