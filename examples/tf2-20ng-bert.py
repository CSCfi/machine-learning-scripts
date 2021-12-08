
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
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard

from transformers import BertTokenizer, TFBertForSequenceClassification
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
            print('GPU:', d.physical_device_desc)
else:
    print('No GPU, using CPU instead.')

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2005299/data/"
print('Using DATADIR', DATADIR)

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
        print(' ', dirname, label_id)
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

(texts_train, texts_test,
 labels_train, labels_test) = train_test_split(texts, labels,
                                               test_size=TEST_SET,
                                               shuffle=True, random_state=42)

(texts_train, texts_valid,
 labels_train, labels_valid) = train_test_split(texts_train, labels_train,
                                                shuffle=False,
                                                test_size=0.1)

print('Length of training texts:', len(texts_train),
      'labels:', len(labels_train))
print('Length of validation texts:', len(texts_valid),
      'labels:', len(labels_valid))
print('Length of test texts:', len(texts_test), 'labels:', len(labels_test))

# ## BERT
#
# Next we specify the pre-trained BERT model we are going to use. The
# model "bert-base-uncased" is the lowercased "base" model (12-layer,
# 768-hidden, 12-heads, 110M parameters).
#
# ### Tokenization
#
# We load the used vocabulary from the BERT model, and use the BERT
# tokenizer to convert the sentences into tokens that match the data
# the BERT model was trained on.

BERTMODEL='bert-base-uncased'
CACHE_DIR=os.path.join(DATADIR, 'transformers-cache')

tokenizer = BertTokenizer.from_pretrained(BERTMODEL,
                                          do_lower_case=True,
                                          cache_dir=CACHE_DIR)

# Next we tokenize all datasets. We set the maximum sequence lengths
# for our training and test sentences as MAX_LEN_TRAIN and
# MAX_LEN_TEST. The maximum length supported by the used BERT model is
# 512 tokens.

MAX_LEN_TRAIN, MAX_LEN_TEST = 128, 512

data_train = tokenizer(texts_train, padding=True, truncation=True,
                       return_tensors="tf", max_length=MAX_LEN_TRAIN)
data_valid = tokenizer(texts_valid, padding=True, truncation=True,
                       return_tensors="tf", max_length=MAX_LEN_TRAIN)
data_test = tokenizer(texts_test, padding=True, truncation=True,
                      return_tensors="tf", max_length=MAX_LEN_TEST)

print("Truncated tokenized first training message: As token ids:")
print(data_train["input_ids"][0])
print("Converted back to tokens:")
print(tokenizer.decode(data_train["input_ids"][0]))

# ### TF Datasets
#
# Let's now define our TF Datasets for training, validation, and
# test data. A batch size of 16 or 32 is often recommended for
# fine-tuning BERT on a specific task.

BATCH_SIZE = 32

dataset_train = tf.data.Dataset.from_tensor_slices((data_train.data,
                                                    labels_train))
dataset_train = dataset_train.shuffle(len(dataset_train)).batch(BATCH_SIZE)
dataset_valid = tf.data.Dataset.from_tensor_slices((data_valid.data,
                                                    labels_valid))
dataset_valid = dataset_valid.batch(BATCH_SIZE)
dataset_test = tf.data.Dataset.from_tensor_slices((data_test.data,
                                                   labels_test))
dataset_test = dataset_test.batch(BATCH_SIZE)

# ### Model initialization
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

optimizer = tf.keras.optimizers.Adam(learning_rate=LR, epsilon=1e-08,
                                     clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

print(model.summary())

# ## Learning

logdir = os.path.join(os.getcwd(), "logs",
                      "20ng-bert-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir)]

# For fine-tuning BERT on a specific task, 2-4 epochs is often
# recommended.

EPOCHS = 4

history = model.fit(dataset_train, validation_data=dataset_valid,
                    epochs=EPOCHS, verbose=2, callbacks=callbacks)

# ## Inference
#
# For a better measure of the quality of the model, let's see the
# model accuracy for the test messages.

test_scores = model.evaluate(dataset_test, verbose=2)
print("Test set %s: %.2f%%" % (model.metrics_names[1], test_scores[1]*100))
