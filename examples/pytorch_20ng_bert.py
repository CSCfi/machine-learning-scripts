#!/usr/bin/env python
# coding: utf-8

# # 20 newsgroup text classification with BERT finetuning
#
# In this script, we'll use a pre-trained BERT
# (https://arxiv.org/abs/1810.04805) model for text classification
# using PyTorch and HuggingFace's Transformers
# (https://github.com/huggingface/transformers).

import torch
from torch.utils.data import (TensorDataset, DataLoader,
                              RandomSampler, SequentialSampler)
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from packaging.version import Version as LV

from sklearn.model_selection import train_test_split

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
assert LV(torch.__version__) >= LV("1.0.0")


def correct(output, target):
    predicted = output.argmax(1)  # pick class with largest network output
    correct_ones = (predicted == target).type(torch.float)
    return correct_ones.sum().item()  # count number of correct ones


def train(data_loader, model, scheduler, optimizer):
    model.train()

    num_batches = 0
    num_items = 0

    total_loss = 0
    total_correct = 0
    for input_ids, input_mask, labels in data_loader:
        # Copy data and targets to GPU
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        labels = labels.to(device)

        # Do a forward pass
        output = model(input_ids, token_type_ids=None,
                       attention_mask=input_mask, labels=labels)

        loss = output[0]
        logits = output[1]

        total_loss += loss
        num_batches += 1

        # Count number of correct
        total_correct += correct(logits, labels)
        num_items += len(labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return {
        'loss': total_loss/num_batches,
        'accuracy': total_correct/num_items
        }


def test(test_loader, model):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for input_ids, input_mask, labels in test_loader:
            # Copy data and targets to GPU
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)

            # Do a forward pass
            output = model(input_ids, token_type_ids=None,
                           attention_mask=input_mask)

            logits = output[0]

            # Count number of correct digits
            total_correct += correct(logits, labels)

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
        logdir = os.path.join(os.getcwd(), "logs", "20ng-bert-" + time_str)
        print('TensorBoard log directory:', logdir)
        os.makedirs(logdir)
        log = tensorboardX.SummaryWriter(logdir)
    except (ImportError, FileExistsError):
        log = None

    datapath = os.getenv('DATADIR')
    if datapath is None:
        print("Please set DATADIR environment variable!")
        sys.exit(1)

    # 20 Newsgroups data set
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

    # Split the data into a training set and a test set using
    # scikit-learn's train_test_split().

    TEST_SET = 4000

    (sentences_train, sentences_test,
     labels_train, labels_test) = train_test_split(texts, labels,
                                                   test_size=TEST_SET,
                                                   shuffle=True,
                                                   random_state=42)

    print('Length of training texts:', len(sentences_train))
    print('Length of training labels:', len(labels_train))
    print('Length of test texts:', len(sentences_test))
    print('Length of test labels:', len(labels_test))

    # The token [CLS] is a special token required by BERT at the beginning
    # of the sentence.

    sentences_train = ["[CLS] " + s for s in sentences_train]
    sentences_test = ["[CLS] " + s for s in sentences_test]

    print("The first training sentence:")
    print(sentences_train[0], 'LABEL:', labels_train[0])

    # Next we specify the pre-trained BERT model we are going to use. The
    # model `"bert-base-uncased"` is the lowercased "base" model
    # (12-layer, 768-hidden, 12-heads, 110M parameters).
    #
    # We load the used vocabulary from the BERT model, and use the BERT
    # tokenizer to convert the sentences into tokens that match the data
    # the BERT model was trained on.

    print('Initializing BertTokenizer')

    BERTMODEL = 'bert-base-uncased'
    CACHE_DIR = os.path.join(datapath, 'transformers-cache')

    tokenizer = BertTokenizer.from_pretrained(BERTMODEL, cache_dir=CACHE_DIR,
                                              do_lower_case=True)

    tokenized_train = [tokenizer.tokenize(s) for s in sentences_train]
    tokenized_test = [tokenizer.tokenize(s) for s in sentences_test]

    print("The full tokenized first training sentence:")
    print(tokenized_train[0])

    # Now we set the maximum sequence lengths for our training and test
    # sentences as `MAX_LEN_TRAIN` and `MAX_LEN_TEST`. The maximum length
    # supported by the used BERT model is 512.
    #
    # The token `[SEP]` is another special token required by BERT at the
    # end of the sentence.

    MAX_LEN_TRAIN, MAX_LEN_TEST = 128, 512

    tokenized_train = [t[:(MAX_LEN_TRAIN-1)]+['SEP'] for t in tokenized_train]
    tokenized_test = [t[:(MAX_LEN_TEST-1)]+['SEP'] for t in tokenized_test]

    print("The truncated tokenized first training sentence:")
    print(tokenized_train[0])

    # Next we use the BERT tokenizer to convert each token into an integer
    # index in the BERT vocabulary. We also pad any shorter sequences to
    # `MAX_LEN_TRAIN` or `MAX_LEN_TEST` indices with trailing zeros.

    ids_train = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_train]
    ids_train = np.array([np.pad(i, (0, MAX_LEN_TRAIN-len(i)),
                                 mode='constant') for i in ids_train])

    ids_test = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_test]
    ids_test = np.array([np.pad(i, (0, MAX_LEN_TEST-len(i)),
                                mode='constant') for i in ids_test])

    print("The indices of the first training sentence:")
    print(ids_train[0])

    # BERT also requires *attention masks*, with 1 for each real token in
    # the sequences and 0 for the padding:

    amasks_train, amasks_test = [], []

    for seq in ids_train:
        seq_mask = [float(i > 0) for i in seq]
        amasks_train.append(seq_mask)

    for seq in ids_test:
        seq_mask = [float(i > 0) for i in seq]
        amasks_test.append(seq_mask)

    # We use again scikit-learn's train_test_split to use 10% of our
    # training data as a validation set, and then convert all data into
    # torch.tensors.

    (train_inputs, validation_inputs,
     train_labels, validation_labels) = train_test_split(ids_train,
                                                         labels_train,
                                                         random_state=42,
                                                         test_size=0.1)
    (train_masks, validation_masks,
     _, _) = train_test_split(amasks_train, ids_train,
                              random_state=42, test_size=0.1)

    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    test_inputs = torch.tensor(ids_test)
    test_labels = torch.tensor(labels_test)
    test_masks = torch.tensor(amasks_test)

    # Next we create PyTorch DataLoaders for all data sets.
    #
    # For fine-tuning BERT on a specific task, the authors recommend a
    # batch size of 16 or 32.

    BATCH_SIZE = 32

    print('Train: ', end="")
    train_dataset = TensorDataset(train_inputs, train_masks,
                                  train_labels)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                              batch_size=BATCH_SIZE)
    print(len(train_dataset), 'messages')

    print('Validation: ', end="")
    validation_dataset = TensorDataset(validation_inputs, validation_masks,
                                       validation_labels)
    validation_sampler = SequentialSampler(validation_dataset)
    validation_loader = DataLoader(validation_dataset,
                                   sampler=validation_sampler,
                                   batch_size=BATCH_SIZE)
    print(len(validation_dataset), 'messages')

    print('Test: ', end="")
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler,
                             batch_size=BATCH_SIZE)
    print(len(test_dataset), 'messages')

    # ## BERT model initialization
    #
    # We now load a pretrained BERT model with a single linear
    # classification layer added on top.

    print('Initializing BertForSequenceClassification')

    model = BertForSequenceClassification.from_pretrained(BERTMODEL,
                                                          cache_dir=CACHE_DIR,
                                                          num_labels=20)
    model = model.to(device)

    # We set the remaining hyperparameters needed for fine-tuning the
    # pretrained model:
    #   * num_epochs: the number of training epochs in fine-tuning
    #     (recommended values between 2 and 4)
    #   * weight_decay: weight decay for the Adam optimizer
    #   * lr: learning rate for the Adam optimizer (2e-5 to 5e-5 recommended)
    #   * warmup_steps: number of warmup steps to (linearly) reach the set
    #     learning rate
    #
    # We also need to grab the training parameters from the pretrained model.

    num_epochs = 4
    weight_decay = 0.01
    lr = 2e-5
    warmup_steps = int(0.2*len(train_loader))

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=len(train_loader)*num_epochs)

    # Training loop
    start_time = datetime.now()
    for epoch in range(num_epochs):
        train_ret = train(train_loader, model, scheduler, optimizer)
        log_measures(train_ret, log, "train", epoch)

        val_ret = test(validation_loader, model)
        log_measures(val_ret, log, "val", epoch)
        print(f"Epoch {epoch+1}: "
              f"train loss: {train_ret['loss']:.6f} "
              f"train accuracy: {train_ret['accuracy']:.2%}, "
              f"val accuracy: {val_ret['accuracy']:.2%}")

    end_time = datetime.now()
    print('Total training time: {}.'.format(end_time - start_time))

    # Inference
    ret = test(test_loader, model)
    print(f"\nTesting: accuracy: {ret['accuracy']:.2%}")


if __name__ == "__main__":
    main()
