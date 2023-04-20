# coding: utf-8

# # IMDB movie review (aclImdb) generation with scaled-down GPT 
# 
# In this notebook, we'll train a scaled-down GPT model from scratch
# for generating text using **Tensorflow** with the **Keras API**.
# This notebook is based on Keras code examples "Text generation with
# a miniature GPT"
# (https://keras.io/examples/generative/text_generation_with_miniature_gpt/)
# and "GPT text generation from scratch with
# KerasNLP"(https://keras.io/examples/generative/text_generation_gpt/).
# 
# First, the needed imports.

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import string
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

import keras_nlp

import numpy as np

print('Using Tensorflow version: {}, and Keras version: {}.'
      .format(tf.__version__, tf.keras.__version__))

# Let's check if we have GPU available.

USE_FP16 = False

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    from tensorflow.python.client import device_lib
    for d in device_lib.list_local_devices():
        if d.device_type == 'GPU':
            print('GPU', d.physical_device_desc)
    if USE_FP16:
        keras.mixed_precision.set_global_policy("mixed_float16")
else:
    print('No GPU, using CPU instead.')


# ## IMDB data set
# 
# Next we'll load the entire aclImdb data set. The dataset contains
# 100,000 movies reviews from the Internet Movie Database.
# 
# The reviews have been collected into a single text file, and we use
# the `TextLineDataset()` function to create a `tf.data.Dataset()`
# from the reviews.

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2007759/data/"

DATAFILE = "aclImdb-nobr-100k.txt"
BATCH_SIZE = 128
VOCAB_SIZE = 5000
SEQ_LEN = 80  

text_ds = tf.data.TextLineDataset(DATADIR+"/"+DATAFILE)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(BATCH_SIZE)

print('An example review:')
print(text_ds.unbatch().take(1).get_single_element())

# ## Load or create the vocabulary and create the tokenizer

VOCABFILE = "aclImdb-nobr-100k-vocab.txt"

vocabfullpath = DATADIR+"/"+VOCABFILE
vocab = []
if os.path.exists(vocabfullpath):
    print('Loading vocabulary from', VOCABFILE)
    with open(vocabfullpath, 'r') as fp:
        for line in fp:
            vocab.append(line[:-1])

else:
    print('Creating vocabulary')
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        text_ds,
        vocabulary_size=VOCAB_SIZE,
        lowercase=True,
        reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
    )
    print('Saving vocabulary to', VOCABFILE)
    with open(DATADIR+"/aclImdb-nobr-100k-vocab.txt", 'w') as fp:
        for v in vocab:
            fp.write("%s\n" % v)

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
    lowercase=True,
)

# ## Tokenize data

# packer adds a start token
start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)

def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels

text_ds = text_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)

# ## GPT model
 
# ### Initialization

MODELFILE = "aclImdb-gpt.h5"

if os.path.exists(MODELFILE):
    model = keras.models.load_model(MODELFILE)

else:
    EMBED_DIM = 256
    FEED_FORWARD_DIM = 256
    NUM_HEADS = 3
    NUM_LAYERS = 2

    inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)

    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=VOCAB_SIZE,
        sequence_length=SEQ_LEN,
        embedding_dim=EMBED_DIM,
        mask_zero=True,
    )
    x = embedding_layer(inputs)

    for _ in range(NUM_LAYERS):
        decoder_layer = keras_nlp.layers.TransformerDecoder(
            num_heads=NUM_HEADS,
            intermediate_dim=FEED_FORWARD_DIM,
        )
        x = decoder_layer(x)

    outputs = keras.layers.Dense(VOCAB_SIZE)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
    model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])

    print(model.summary())

    # ### Learning

    logdir = os.path.join(os.getcwd(), "logs",
                          "aclImdb-gpt-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory:', logdir)
    os.makedirs(logdir)
    callbacks = [TensorBoard(log_dir=logdir)]

    EPOCHS = 6

    history = model.fit(text_ds, verbose=2, epochs=EPOCHS,
                        callbacks=callbacks)

    print('Saving model to', MODELFILE)
    model.save(MODELFILE)

# ### Inference

if len(sys.argv)<2:
    review = ""
else:
    review = " ".join(sys.argv[1:])
    print('Starting review with "{}"'.format(review))

#review = "This was a great scary movie which"
#review = "This was the best movie of"
#review = "A funny movie"

tokenized_review = np.trim_zeros(tokenizer.tokenize(review).numpy(), 'b')
tokenized_review = np.insert(tokenized_review, 0, tokenizer.token_to_id("[BOS]"))
prompt_tokens = tf.convert_to_tensor(tokenized_review)

NUM_TOKENS_TO_GENERATE = 80

def token_logits_fn(inputs):
    cur_len = inputs.shape[1]
    output = model(inputs)
    return output[:, cur_len - 1, :]

# Greedy search:

output_tokens = keras_nlp.utils.greedy_search(
    token_logits_fn,
    prompt_tokens,
    max_length=NUM_TOKENS_TO_GENERATE,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Greedy search generated text: \n{txt}\n")

# Random search:

print("Random search generated text:")
for i in range(5):
    output_tokens = keras_nlp.utils.random_search(
        token_logits_fn,
        prompt_tokens,
        max_length=NUM_TOKENS_TO_GENERATE,
        from_logits=True,
    )
    print("{}: {}".format(i, tokenizer.detokenize(output_tokens)))
print()

# Top-P search:
    
print("Top-P search generated text:")
for i in range(5):
    output_tokens = keras_nlp.utils.top_p_search(
        token_logits_fn,
        prompt_tokens,
        max_length=NUM_TOKENS_TO_GENERATE,
        p=0.5,
        from_logits=True,
    )
    print("{}: {}".format(i, tokenizer.detokenize(output_tokens)))
print()

print("All done.")

