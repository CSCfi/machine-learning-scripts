#!/usr/bin/env python
# coding: utf-8

# # Traffic sign classification with CNNs
# 
# In this notebook, we'll train a convolutional neural network (CNN, ConvNet) to classify images of traffic signs from [The German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) using PyTorch. 
# 
# **Note that using a GPU with this notebook is highly recommended.**
# 
# First, the needed imports.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from distutils.version import LooseVersion as LV
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



# ### Data loaders
# 
# Let's now define our real data loaders for training and validation data.

batch_size = 50

print('Train: ', end="")
train_dataset = datasets.ImageFolder(root=datapath+'/train',
                                     transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                         shuffle=True, num_workers=4)
print('Found', len(train_dataset), 'images belonging to',
     len(train_dataset.classes), 'classes')

print('Validation: ', end="")
validation_dataset = ImageFolderRemoveDirs(root=datapath+'/validation',
                                           transform=noop_transform,
                                           remove_dirs=['00027', '00039'])
validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=4)
print('Found', len(validation_dataset), 'images belonging to',
     len(validation_dataset.classes), 'classes')

print('Test: ', end="")
test_dataset = datasets.ImageFolder(root=datapath+'/test',
                                     transform=noop_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=4)
print('Found', len(test_dataset), 'images belonging to',
     len(test_dataset.classes), 'classes')


# ## Option 1: Train a small CNN from scratch
# 
# Similarly as with MNIST digits, we can start from scratch and train a CNN for the classification task. However, due to the small number of training images, a large network will easily overfit, regardless of the data augmentation.
# 
# ### Initialization


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(32, 64, (3, 3))
        self.pool3 = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 43)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # "flatten" 2D to 1D
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        return self.fc2(x)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.03)
#optimizer = optim.RMSprop(model.parameters())
criterion = nn.CrossEntropyLoss()

print(model)


# ### Learning

def train(epoch, scores=None):
    # Set model to training mode
    model.train()
    epoch_loss = 0

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        loss = criterion(output, target)
        epoch_loss += loss.data.item()

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

    epoch_loss /= len(train_loader.dataset)
    print('Train Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss))

    if scores is not None:
        if 'loss' not in scores:
            scores['loss'] = []
        scores['loss'].append(epoch_loss)

    if log is not None:
        log.add_scalar('loss', epoch_loss, epoch-1)


def evaluate(loader, scores=None, iteration=-1):
    model.eval()
    loss, correct = 0, 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        loss += criterion(output, target).data.item()

        _, pred = output.max(1)
        correct += pred.eq(target.data).cpu().sum()

    loss /= len(loader.dataset)

    accuracy = 100. * correct.to(torch.float32) / len(loader.dataset)

    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
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


epochs = 50

train_scores = {}
valid_scores = {}

start_time = datetime.now()
for epoch in range(1, epochs + 1):
    train(epoch, train_scores)

    with torch.no_grad():
        print('\nValidation:')
        evaluate(validation_loader, valid_scores, epoch-1)

end_time = datetime.now()
print('Total training time: {}.'.format(end_time - start_time))

# torch.save("gtsrb-small-cnn.pt")

# ### Inference

print('\nTesting:')
with torch.no_grad():
    evaluate(test_loader)

