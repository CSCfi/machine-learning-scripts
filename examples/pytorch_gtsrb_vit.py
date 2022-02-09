#!/usr/bin/env python
# coding: utf-8

# # Traffic sign classification with CNNs
# 
# In this notebook, we'll finetune a [Vision Transformer](https://arxiv.org/abs/2010.11929) (ViT) to classify images of traffic signs from [The German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) using PyTorch and HuggingFace's [Transformers](https://github.com/huggingface/transformers). 
# 
# **Note that using a GPU with this notebook is highly recommended.**
# 
# First, the needed imports.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from distutils.version import LooseVersion as LV

from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import __version__ as transformers_version

from datetime import datetime

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)
assert(LV(torch.__version__) >= LV("1.0.0"))


# TensorBoard is a tool for visualizing progress during training.  Although TensorBoard was created for TensorFlow, it can also be used with PyTorch.  It is easiest to use it with the tensorboardX module.

try:
    import tensorboardX
    import os
    logdir = os.path.join(os.getcwd(), "logs",
                          "gtsrb-"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory:', logdir)
    os.makedirs(logdir)
    log = tensorboardX.SummaryWriter(logdir)
except ImportError as e:
    log = None


# ## Data
# 
# The training dataset consists of 5535 images of traffic signs of varying size. There are 43 different types of traffic signs.
# 
# The validation and test sets consist of 999 and 12630 images, respectively.
# 
# ### Downloading the data

local_scratch = os.getenv('LOCAL_SCRATCH')

datapath = local_scratch if local_scratch is not None else '/scratch/project_2005299/data'
datapath = os.path.join(datapath, 'gtsrb/train-5535')

if local_scratch is not None and not os.path.exists(datapath):
    os.system('tar xf /scratch/project_2005299/data/gtsrb.tar -C ' + local_scratch)

(nimages_train, nimages_validation, nimages_test) = (5535, 999, 12630)


# ### Data augmentation
# 
# First, we'll resize all training and validation images to a fized size.
# 
# Then, to make the most of our limited number of training examples, we'll apply random transformations to them each time we are looping over them. This way, we "augment" our training dataset to contain more data. There are various transformations available in `torchvision`, see [torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html) for more information.

input_image_size = (75, 75)

data_transform = transforms.Compose([
        transforms.Resize(input_image_size),
        transforms.RandomAffine(degrees=0, translate=None,
                                scale=(0.8, 1.2), shear=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

noop_transform = transforms.Compose([
        transforms.Resize(input_image_size),
        transforms.ToTensor()
    ])


# ### Data loaders
# 
# First we specify the pre-trained ViT model we are going to use. The model [`"google/vit-base-patch16-224"`](https://huggingface.co/google/vit-base-patch16-224) is pre-trained on ImageNet-21k (14 million images, 21,843 classes) at resolution 224x224, and fine-tuned on ImageNet 2012 (1 million images, 1,000 classes) at resolution 224x224.
# 
# We'll use a pre-trained ViT feature extractor that matches the ViT model to preprocess the input images. 

VITMODEL = 'google/vit-base-patch16-224'
model = ViTForImageClassification.from_pretrained(VITMODEL,
                                                  num_labels=43,
                                                  ignore_mismatched_sizes=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

feature_extractor = ViTFeatureExtractor.from_pretrained(VITMODEL)


# The we define a "collator" function. This is just a function passed to the `DataLoader` which will pre-process each batch of data. In our case we will pass the images through the `ViTFeatureExtractor` which will process the images into the correct format for ViT.

class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        
    def __call__(self, batch):
        x = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        x['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.int64)
        return x

collator = ImageClassificationCollator(feature_extractor)


# Let's now define our data loaders for training, validation, and test data.

class ImageFolderRemoveDirs(datasets.ImageFolder):
    def __init__(self, root, transform, remove_dirs):
        self.remove_dirs = remove_dirs
        super(ImageFolderRemoveDirs, self).__init__(root=root, transform=transform)

    def find_classes(self, directory):
        classes, class_to_idx = super(ImageFolderRemoveDirs, self).find_classes(directory)
        for d in self.remove_dirs:
            print('Removing directory', d)
            classes.remove(d)
            del class_to_idx[d]
        return classes, class_to_idx


batch_size = 50

print('Train: ', end="")
train_dataset = datasets.ImageFolder(root=datapath+'/train',
                                     transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                         shuffle=True, num_workers=4,
                         collate_fn=collator)
print('Found', len(train_dataset), 'images belonging to',
     len(train_dataset.classes), 'classes')

print('Validation: ', end="")
validation_dataset = ImageFolderRemoveDirs(root=datapath+'/validation',
                                           transform=transforms.ToTensor(),
                                           remove_dirs=['00027', '00039'])
validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4,
                              collate_fn=collator)
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

        # output is actually a SequenceClassifierOutput object that contains
        # the loss and classification scores (logits)

        # add up the loss
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

        _, pred = output.logits.max(1)
        lab = data["labels"]
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


epochs = 5

train_scores = {}
valid_scores = {}

start_time = datetime.now()
for epoch in range(1, epochs + 1):
    train(epoch, train_scores)

    with torch.no_grad():
        print('Validation:')
        evaluate(validation_loader, valid_scores, epoch-1)

end_time = datetime.now()
print('Total training time: {}.'.format(end_time - start_time))

# torch.save("gtsrb-small-cnn.pt")')

# ### Inference

print('\nTesting:')
with torch.no_grad():
    evaluate(test_loader)

