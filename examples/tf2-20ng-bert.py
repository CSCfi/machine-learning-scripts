
# coding: utf-8

# # 20 newsgroup text classification with BERT finetuning
# 
# In this script, we'll use a pre-trained BERT
# (https://arxiv.org/abs/1810.04805) model for text classification
# using TensorFlow 2 / Keras and HuggingFace's Transformers
# (https://github.com/huggingface/transformers). This notebook is
# based on "Predicting Movie Review Sentiment with BERT on TF Hub"
# (https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb)
# by Google and "BERT Fine-Tuning Tutorial with PyTorch"
# (https://mccormickml.com/2019/07/22/BERT-fine-tuning/) by Chris
# McCormick.

# **Note that using a GPU with this script is highly recommended.**
# 
# First, the needed imports.

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from transformers import BertTokenizer, BertConfig
from transformers import TFBertForSequenceClassification
from transformers import __version__ as transformers_version

from distutils.version import LooseVersion as LV

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import io, sys, os, datetime

from zipfile import ZipFile
import numpy as np

print('Using TensorFlow version:', tf.__version__,
      'Keras version:', tf.keras.__version__,
      'Transformers version:', transformers_version, flush=True)
assert(LV(tf.__version__) >= LV("2.3.0"))

if len(tf.config.list_physical_devices('GPU')):
    from tensorflow.python.client import device_lib
    for d in device_lib.list_local_devices():
        if d.device_type == 'GPU':
            print('GPU', d.physical_device_desc)
else:
    print('No GPU, using CPU instead.')

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2000745/data/"

# ## 20 Newsgroups data set
# 
# Next we'll load the 20 Newsgroups
# (http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html)
# data set.
# 
# The dataset contains 20000 messages collected from 20 different
# Usenet newsgroups (1000 messages from each group):
#
# alt.atheism           | soc.religion.christian   | comp.windows.x     | sci.crypt
# talk.politics.guns    | comp.sys.ibm.pc.hardware | rec.autos          | sci.electronics
# talk.politics.mideast | comp.graphics            | rec.motorcycles    | sci.space
# talk.politics.misc    | comp.os.ms-windows.misc  | rec.sport.baseball | sci.med
# talk.religion.misc    | comp.sys.mac.hardware    | rec.sport.hockey   | misc.forsale

TEXT_DATA_ZIP = os.path.join(DATADIR, "20_newsgroup.zip")
zf = ZipFile(TEXT_DATA_ZIP, 'r')

print('Processing text dataset from', TEXT_DATA_ZIP, flush=True)

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for fullname in sorted(zf.namelist()):
    parts = fullname.split('/')
    dirname = parts[1]
    fname = parts[2] if len(parts) > 2 else None
    zinfo = zf.getinfo(fullname)
    if zinfo.is_dir() and len(dirname) > 0:
        label_id = len(labels_index)
        labels_index[dirname] = label_id
        print(dirname, label_id)
    elif fname is not None and fname.isdigit():
        with zf.open(fullname) as f:
            t = f.read().decode('latin-1')
            i = t.find('\n\n')  # skip header
            if 0 < i:
                t = t[i:]
            texts.append(t)
        labels.append(label_id)

labels = np.array(labels)
print('Found %s texts.' % len(texts))

# Split the data into a training set and a test set using
# scikit-learn's train_test_split().

TEST_SET = 4000

(sentences_train, sentences_test,
 labels_train, labels_test) = train_test_split(texts, labels, 
                                               test_size=TEST_SET,
                                               shuffle=True, random_state=42)

print('Length of training texts:', len(sentences_train))
print('Length of training labels:', len(labels_train))
print('Length of test texts:', len(sentences_test))
print('Length of test labels:', len(labels_test))

# The token "[CLS]" is a special token required by BERT at the beginning
# of the sentence.

sentences_train = ["[CLS] " + s for s in sentences_train]
sentences_test = ["[CLS] " + s for s in sentences_test]

print ("The first training sentence:")
print(sentences_train[0], 'LABEL:', labels_train[0])

# Next we specify the pre-trained BERT model we are going to use. The
# model "bert-base-uncased" is the lowercased "base" model (12-layer,
# 768-hidden, 12-heads, 110M parameters).
# 
# We load the used vocabulary from the BERT model, and use the BERT
# tokenizer to convert the sentences into tokens that match the data
# the BERT model was trained on.

print('Initializing BertTokenizer')

BERTMODEL='bert-base-uncased'
CACHE_DIR=os.path.join(DATADIR, 'transformers-cache')

tokenizer = BertTokenizer.from_pretrained(BERTMODEL, cache_dir=CACHE_DIR,
                                          do_lower_case=True)

tokenized_train = [tokenizer.tokenize(s) for s in sentences_train]
tokenized_test  = [tokenizer.tokenize(s) for s in sentences_test]

print ("The full tokenized first training sentence:")
print (tokenized_train[0])

# Now we set the maximum sequence lengths for our training and test
# sentences as MAX_LEN_TRAIN and MAX_LEN_TEST. The maximum length
# supported by the used BERT model is 512.
# 
# The token "[SEP]" is another special token required by BERT at the
# end of the sentence.

MAX_LEN_TRAIN, MAX_LEN_TEST = 128, 512

tokenized_train = [t[:(MAX_LEN_TRAIN-1)]+['SEP'] for t in tokenized_train]
tokenized_test  = [t[:(MAX_LEN_TEST-1)]+['SEP'] for t in tokenized_test]

print ("The truncated tokenized first training sentence:")
print (tokenized_train[0])

# Next we use the BERT tokenizer to convert each token into an integer
# index in the BERT vocabulary. We also pad any shorter sequences to
# MAX_LEN_TRAIN or MAX_LEN_TEST indices with trailing zeros.

ids_train = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_train]
ids_train = np.array([np.pad(i, (0, MAX_LEN_TRAIN-len(i)), 
                             mode='constant') for i in ids_train])

ids_test = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_test]
ids_test = np.array([np.pad(i, (0, MAX_LEN_TEST-len(i)), 
                            mode='constant') for i in ids_test])

print ("The indices of the first training sentence:")
print (ids_train[0])

# BERT requires *attention masks*, with 1 for each real token in the
# sequences and 0 for the padding:

trainval_masks, test_masks= [], []

for seq in ids_train:
  seq_mask = [float(i>0) for i in seq]
  trainval_masks.append(seq_mask)
    
for seq in ids_test:
  seq_mask = [float(i>0) for i in seq]
  test_masks.append(seq_mask)
    
trainval_masks = np.array(trainval_masks)
test_masks = np.array(test_masks)


# We use again scikit-learn's train_test_split() to use 10% of our
# training data as a validation set.

(train_inputs, validation_inputs, 
 train_labels, validation_labels) = train_test_split(ids_train, labels_train, 
                                                     random_state=42,
                                                     test_size=0.1)
(train_masks, validation_masks, 
 _, _) = train_test_split(trainval_masks, ids_train,
                          random_state=42, test_size=0.1)

# BERT also requires *type ids*, which are zero-valued vectors in our
# case:

train_type_ids = np.zeros(train_masks.shape)
validation_type_ids = np.zeros(validation_masks.shape)
test_type_ids = np.zeros(test_masks.shape)

# ## BERT model initialization
# 
# We now load a pretrained BERT model with a single linear
# classification layer added on top.

model = TFBertForSequenceClassification.from_pretrained(BERTMODEL,
                                                        cache_dir=CACHE_DIR,
                                                        num_labels=20)

# We use Adam as the optimizer, categorical crossentropy as loss, and
# then compile the model.
# 
# LR is the learning rate for the Adam optimizer (2e-5 to 5e-5
# recommended for BERT finetuning).

LR = 2e-5

optimizer = tf.keras.optimizers.Adam(learning_rate=LR, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# ## Learning
# 
# For fine-tuning BERT on a specific task, the authors recommend a
# batch size of 16 or 32, and 2-4 epochs.

logdir = os.path.join(os.getcwd(), "logs",
                      "20ng-bert-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir)]

EPOCHS = 4
BATCH_SIZE = 32

history = model.fit([train_inputs, train_masks, train_type_ids], train_labels,
                    validation_data=([validation_inputs, validation_masks,
                                      validation_type_ids],
                                     validation_labels),
                    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2,
                    callbacks=callbacks)

# ## Inference
# 
# For a better measure of the quality of the model, let's see the
# model accuracy for the test messages.

test_scores = model.evaluate([ids_test, test_masks, test_type_ids],
                             labels_test, batch_size=BATCH_SIZE, verbose=2)
print("Test set %s: %.2f%%" % (model.metrics_names[1], test_scores[1]*100))
