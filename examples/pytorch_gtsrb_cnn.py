# coding: utf-8

# Traffic sign classification with CNNs
#
# In this notebook, we'll train a convolutional neural network (CNN,
# ConvNet) to classify images of traffic signs from [The German
# Traffic Sign Recognition
# Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)
# using PyTorch. This notebook is largely based on the blog post
# [Building powerful image classification models using very little
# data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) by François Chollet.
#
# **Note that using a GPU with this notebook is highly recommended.**

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from distutils.version import LooseVersion as LV
import os

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
assert(LV(torch.__version__) >= LV("1.0.0"))

subpath = 'gtsrb/train-5535'

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2003747/data/"

datapath = os.path.join(DATADIR, subpath)

print('Reading data from path:', datapath)

(nimages_train, nimages_validation, nimages_test) = (5535, 999, 12630)


def get_tensorboard(log_name):
    try:
        import tensorboardX
        import os
        import datetime
        logdir = os.path.join(os.getcwd(), "logs",
                              "gtsrb-" + log_name + "-" +
                              datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print('Logging TensorBoard to:', logdir)
        os.makedirs(logdir)
        return tensorboardX.SummaryWriter(logdir)
    except (ImportError, FileExistsError):
        return None


def train(model, loader, criterion, optimizer, epoch, log=None):
    # Set model to training mode
    model.train()
    epoch_loss = 0.

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)
        epoch_loss += loss.data.item()

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

    epoch_loss /= len(loader.dataset)
    print('Train Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss))

    if log is not None:
        log.add_scalar('loss', epoch_loss, epoch-1)


def evaluate(model, loader, criterion=None, epoch=None, log=None):
    model.eval()
    loss, correct = 0, 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        if criterion is not None:
            loss += criterion(output, target).data.item()

        _, pred = output.max(1)
        correct += pred.eq(target.data).cpu().sum()

    if criterion is not None:
        loss /= len(loader.dataset)

    accuracy = 100. * correct.to(torch.float32) / len(loader.dataset)

    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss, correct, len(loader.dataset), accuracy))

    if log is not None and epoch is not None:
        log.add_scalar('val_loss', loss, epoch-1)
        log.add_scalar('val_acc', accuracy, epoch-1)


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


def get_train_loader(batch_size=50):
    print('Train: ', end="")
    train_dataset = datasets.ImageFolder(root=datapath+'/train',
                                         transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    print('Found', len(train_dataset), 'images belonging to',
          len(train_dataset.classes), 'classes')
    return train_loader


def get_validation_loader(batch_size=50):
    print('Validation: ', end="")
    validation_dataset = datasets.ImageFolder(root=datapath+'/validation',
                                              transform=noop_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=4)
    print('Found', len(validation_dataset), 'images belonging to',
          len(validation_dataset.classes), 'classes')
    return validation_loader

def get_test_loader(batch_size=50):
    print('Test: ', end="")
    test_dataset = datasets.ImageFolder(root=datapath+'/test',
                                        transform=noop_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4)
    print('Found', len(test_dataset), 'images belonging to',
          len(test_dataset.classes), 'classes')
    return test_loader

if __name__ == '__main__':
    print('\nThis Python script is only for common functions. *DON\'T RUN IT DIRECTLY!* :-)')

