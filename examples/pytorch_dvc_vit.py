#!/usr/bin/env python
# coding: utf-8

# # Dogs-vs-cats classification with CNNs
#
#
# **Note that using a GPU with this notebook is highly recommended.**
#
# First, the needed imports.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from distutils.version import LooseVersion as LV

from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import __version__ as transformers_version

torch.manual_seed(42)
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__,
      'Transformers version:', transformers_version,
      'Device:', device)
assert(LV(torch.__version__) >= LV("1.0.0"))


# TensorBoard is a tool for visualizing progress during training. Although
# TensorBoard was created for TensorFlow, it can also be used with PyTorch. It
# is easiest to use it with the tensorboardX module.

try:
    import tensorboardX
    import os
    import datetime
    logdir = os.path.join(os.getcwd(), "logs",
                          "dvc-"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory:', logdir)
    os.makedirs(logdir)
    log = tensorboardX.SummaryWriter(logdir)
except ImportError:
    log = None


# ## Data
#
# The training dataset consists of 2000 images of dogs and cats, split in half.
# In addition, the validation set consists of 1000 images, and the test set of
# 22000 images.
#
# ### Downloading the data

local_scratch = os.getenv('LOCAL_SCRATCH')

datapath = local_scratch if local_scratch is not None else '/scratch/project_2005299/data'
datapath = os.path.join(datapath, 'dogs-vs-cats/train-2000')

if local_scratch is not None and not os.path.exists(datapath):
    os.system('tar xf /scratch/project_2005299/data/dogs-vs-cats.tar -C ' + local_scratch)

print("Loading data from", datapath)

(nimages_train, nimages_validation, nimages_test) = (2000, 1000, 22000)


# ### Data loaders
#
# Let's now define our real data loaders for training, validation, and test data.

class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        x = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        x['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.float32)
        return x


VITMODEL = 'google/vit-base-patch16-224'
feature_extractor = ViTFeatureExtractor.from_pretrained(VITMODEL)
collator = ImageClassificationCollator(feature_extractor)

batch_size = 25

print('Train: ', end="")
train_dataset = datasets.ImageFolder(root=datapath+'/train',
                                     transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=4,
                          collate_fn=collator)
print('Found', len(train_dataset), 'images belonging to',
      len(train_dataset.classes), 'classes')

print('Validation: ', end="")
validation_dataset = datasets.ImageFolder(root=datapath+'/validation',
                                          transform=transforms.ToTensor())
validation_loader = DataLoader(validation_dataset,
                               batch_size=batch_size, shuffle=False,
                               num_workers=4, collate_fn=collator)
print('Found', len(validation_dataset), 'images belonging to',
      len(validation_dataset.classes), 'classes')

print('Test: ', end="")
test_dataset = datasets.ImageFolder(root=datapath+'/test',
                                    transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=4,
                         collate_fn=collator)
print('Found', len(test_dataset), 'images belonging to',
      len(test_dataset.classes), 'classes')


# ### Learning

model = ViTForImageClassification.from_pretrained(VITMODEL,
                                                  num_labels=1,
                                                  ignore_mismatched_sizes=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCELoss()


def train(epoch, scores=None):
    # Set model to training mode
    model.train()
    epoch_loss = 0

    # Loop over each batch from the training set
    for batch_idx, data in enumerate(train_loader):
        # Copy data to GPU if needed
        data["pixel_values"] = data["pixel_values"].to(device)
        data["labels"] = data["labels"].to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(**data)

        #output = torch.squeeze(output.logits.argmax(1))

        # Calculate loss
        epoch_loss += output.loss.data.item()

        # Backpropagate
        output.loss.backward()

        # Update weights
        optimizer.step()

    epoch_loss /= len(train_loader.dataset)
    print('Train Epoch: {}, Loss: {:.6f}'.format(epoch, epoch_loss))

    if scores is not None:
        if 'loss' not in scores:
            scores['loss'] = []
        scores['loss'].append(epoch_loss)

    if log is not None:
        log.add_scalar('loss', epoch_loss, epoch-1)


def evaluate(loader, scores=None, iteration=-1):
    model.eval()
    loss, correct = 0, 0
    for data in loader:
        data["pixel_values"] = data["pixel_values"].to(device)
        data["labels"] = data["labels"].to(device)

        output = model(**data)

        loss += output.loss.data.item()

        pred = output.logits>0.5
        lab = data["labels"].unsqueeze(1)
        correct += pred.eq(lab).cpu().sum()

    loss /= len(loader.dataset)

    accuracy = 100. * correct.to(torch.float32) / len(loader.dataset)

    print('Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss, correct, len(loader.dataset), accuracy))

    if scores is not None:
        if 'loss' not in scores:
            scores['loss'] = []
        if 'accuracy' not in scores:
            scores['accuracy'] = []
        scores['loss'].append(loss)
        scores['accuracy'].append(accuracy)

    if log is not None and iteration >= 0:
        log.add_scalar('val_loss', loss, iteration)
        log.add_scalar('val_acc', accuracy, iteration)


epochs = 10

train_scores = {}
valid_scores = {}

from datetime import datetime
start_time = datetime.now()

for epoch in range(1, epochs + 1):
    train(epoch, train_scores)

    with torch.no_grad():
       print('\nValidation:')
       evaluate(validation_loader, valid_scores, epoch-1)

end_time = datetime.now()
print('Total training time: {}.'.format(end_time - start_time))

### Inference

print('\nTesting:')
with torch.no_grad():
    evaluate(test_loader)
