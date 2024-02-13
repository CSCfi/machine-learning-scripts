#!/usr/bin/env python
# coding: utf-8

# 20 Newsgroups text classification with pre-trained word embeddings
#
# In this script, we'll use pre-trained [GloVe word
# embeddings](http://nlp.stanford.edu/projects/glove/) for text
# classification using PyTorch.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from packaging.version import Version as LV

from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from datetime import datetime

import os
import sys

import numpy as np

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)
assert(LV(torch.__version__) >= LV("1.0.0"))


class Net(nn.Module):
    def __init__(self, embedding_matrix):
        super(Net, self).__init__()
        self.emb = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.layers = nn.Sequential(
            nn.Conv1d(100, 128, 5),  # output: batch_size x 128 x seq_len-4
            nn.ReLU(),
            nn.MaxPool1d(5),         # output: bs x 128 x 199
            nn.Conv1d(128, 128, 5),  # output: bs x 128 x 199
            nn.ReLU(),
            nn.MaxPool1d(5),         # output: bs x 128 x 39
            nn.Conv1d(128, 128, 5),  # output: bs x 128 x 35
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # output: bs x 128 x 1
            )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )

    def forward(self, x):
        x = self.emb(x)      # output from embedding: batch_size x seq_len x embedding dim.
        x = x.transpose(1,2) # change to: batch_size x embedding_dim x seq_len
        x = self.layers(x)
        x = self.linear_layers(x)
        return x


def correct(output, target):
    predicted = output.argmax(1) # pick class with largest network output
    correct_ones = (predicted == target).type(torch.float)
    return correct_ones.sum().item() # count number of correct ones


def train(data_loader, model, criterion, optimizer):
    model.train()

    num_batches = 0
    num_items = 0

    total_loss = 0
    total_correct = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)

        # Do a forward pass
        output = model(data)

        # Calculate the loss
        loss = criterion(output, target)
        total_loss += loss
        num_batches += 1

        # Count number of correct
        total_correct += correct(output, target)
        num_items += len(target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return {
        'loss': total_loss/num_batches,
        'accuracy': total_correct/num_items
        }


def test(test_loader, model, criterion):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)

            # Do a forward pass
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()

            # Count number of correct digits
            total_correct += correct(output, target)

    return {
        'loss': test_loss/num_batches,
        'accuracy': total_correct/num_items
    }


def log_measures(ret, log, prefix, epoch):
    if log is not None:
        for key, value in ret.items():
            log.add_scalar(prefix + "_" + key, value, epoch)


def main():
    try:
        import tensorboardX
        time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logdir = os.path.join(os.getcwd(), "logs", "20ng-cnn-" + time_str)
        print('TensorBoard log directory:', logdir)
        os.makedirs(logdir)
        log = tensorboardX.SummaryWriter(logdir)
    except (ImportError, FileExistsError):
        log = None

    # ## GloVe word embeddings
    #
    # Let's begin by loading a datafile containing pre-trained word
    # embeddings.  The datafile contains 100-dimensional embeddings for
    # 400,000 English words.

    datapath = os.getenv('DATADIR')
    if datapath is None:
        print("Please set DATADIR environment variable!")
        sys.exit(1)

    glove_dir = os.path.join(datapath, "glove.6B")

    print('Indexing word vectors.')

    embeddings_index = {}
    with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    
    # ## 20 Newsgroups data set
    #
    # Next we'll load the [20 Newsgroups]
    # (http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html)
    # data set.
    #
    # The dataset contains 20000 messages collected from 20 different
    # Usenet newsgroups (1000 messages from each group):
    #
    # | alt.atheism           | soc.religion.christian   | comp.windows.x     | sci.crypt
    # | talk.politics.guns    | comp.sys.ibm.pc.hardware | rec.autos          | sci.electronics
    # | talk.politics.mideast | comp.graphics            | rec.motorcycles    | sci.space
    # | talk.politics.misc    | comp.os.ms-windows.misc  | rec.sport.baseball | sci.med
    # | talk.religion.misc    | comp.sys.mac.hardware    | rec.sport.hockey   | misc.forsale

    text_data_dir = os.path.join(datapath, "20_newsgroup")

    print('Processing text dataset')

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            print('-', name, label_id)
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

    # Tokenize the texts using gensim.

    tokens = list()
    for text in texts:
        tokens.append(simple_preprocess(text))

    # Vectorize the text samples into a 2D integer tensor.

    MAX_NUM_WORDS = 10000 # 2 words reserved: 0=pad, 1=oov
    MAX_SEQUENCE_LENGTH = 1000

    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=0, no_above=1.0,
                               keep_n=MAX_NUM_WORDS-2)

    word_index = dictionary.token2id
    print('Found %s unique tokens.' % len(word_index))

    data = [dictionary.doc2idx(t) for t in tokens]

    # Truncate and pad sequences.

    data = [i[:MAX_SEQUENCE_LENGTH] for i in data]
    data = np.array([np.pad(i, (MAX_SEQUENCE_LENGTH-len(i), 0),
                            mode='constant', constant_values=-2)
                     for i in data], dtype=int)
    data = data + 2

    print('Shape of data tensor:', data.shape)
    print('Length of label vector:', len(labels))

    # Split the data into a training set and a validation set

    VALIDATION_SET, TEST_SET = 1000, 4000

    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=TEST_SET,
                                                        shuffle=True,
                                                        random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=VALIDATION_SET,
                                                      shuffle=False)

    print('Shape of training data tensor:', x_train.shape)
    print('Length of training label vector:', len(y_train))
    print('Shape of validation data tensor:', x_val.shape)
    print('Length of validation label vector:', len(y_val))
    print('Shape of test data tensor:', x_test.shape)
    print('Length of test label vector:', len(y_test))

    # Create PyTorch DataLoaders for all data sets:

    BATCH_SIZE = 128

    print('Train: ', end="")
    train_dataset = TensorDataset(torch.LongTensor(x_train),
                                  torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    print(len(train_dataset), 'messages')

    print('Validation: ', end="")
    validation_dataset = TensorDataset(torch.LongTensor(x_val),
                                       torch.LongTensor(y_val))
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=4)
    print(len(validation_dataset), 'messages')

    print('Test: ', end="")
    test_dataset = TensorDataset(torch.LongTensor(x_test),
                                 torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4)
    print(len(test_dataset), 'messages')

    # Prepare the embedding matrix:

    print('Preparing embedding matrix.')

    EMBEDDING_DIM = 100

    embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))
    n_not_found = 0
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS-2:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i+2] = embedding_vector
        else:
            n_not_found += 1

    embedding_matrix = torch.FloatTensor(embedding_matrix)
    print('Shape of embedding matrix:', embedding_matrix.shape)
    print('Words not found in pre-trained embeddings:', n_not_found)

    model = Net(embedding_matrix)
    model = model.to(device)
    
    # optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    criterion = nn.CrossEntropyLoss()

    print(model)

    num_epochs = 40

    # Training loop
    start_time = datetime.now()
    for epoch in range(num_epochs):
        train_ret = train(train_loader, model, criterion, optimizer)
        log_measures(train_ret, log, "train", epoch)

        val_ret = test(validation_loader, model, criterion)
        log_measures(val_ret, log, "val", epoch)
        print(f"Epoch {epoch+1}: "
              f"train loss: {train_ret['loss']:.6f} "
              f"train accuracy: {train_ret['accuracy']:.2%}, "
              f"val accuracy: {val_ret['accuracy']:.2%}")

    end_time = datetime.now()
    print('Total training time: {}.'.format(end_time - start_time))

    # Inference
    ret = test(test_loader, model, criterion)
    print(f"\nTesting: accuracy: {ret['accuracy']:.2%}")


if __name__ == "__main__":
    main()
