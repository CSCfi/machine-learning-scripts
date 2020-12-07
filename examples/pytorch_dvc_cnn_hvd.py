# coding: utf-8

# Dogs-vs-cats classification with CNNs
#
# In this script, we'll train a convolutional neural network (CNN,
# ConvNet) to classify images of dogs from images of cats using
# PyTorch. This script is largely based on the blog post [Building
# powerful image classification models using very little
# data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
# by François Chollet.
#
# **Note that using a GPU with this script is highly recommended.**

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.utils.data.distributed
from distutils.version import LooseVersion as LV
import os
import horovod.torch as hvd

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

hvd.init()
torch.cuda.set_device(hvd.local_rank())

if hvd.rank() == 0:
    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    assert(LV(torch.__version__) >= LV("1.0.0"))


subpath = 'dogs-vs-cats/train-2000'

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2003747/data/"

datapath = os.path.join(DATADIR, subpath)

if hvd.rank() == 0:
    print('Reading data from path:', datapath)

(nimages_train, nimages_validation, nimages_test) = (2000, 1000, 22000)


def get_tensorboard(log_name):
    if hvd.rank() != 0:
        return None
    try:
        import tensorboardX
        import os
        import datetime
        logdir = os.path.join(os.getcwd(), "logs",
                              "dvc-" + log_name + "-" +
                              datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print('Logging TensorBoard to:', logdir)
        os.makedirs(logdir)
        return tensorboardX.SummaryWriter(logdir)
    except ImportError:
        return None


def train(model, loader, sampler, criterion, optimizer, epoch, log=None):
    # Set model to training mode
    model.train()

    # Horovod: set epoch to sampler for shuffling.
    sampler.set_epoch(epoch)

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
        output = torch.squeeze(output)

        # Calculate loss
        loss = criterion(output, target.to(torch.float32))
        epoch_loss += loss.data.item()

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

    epoch_loss /= len(loader.dataset)
    print('Train Epoch: {}:{}, Loss: {:.4f}'.format(epoch, hvd.rank(), epoch_loss))

    if log is not None:
        log.add_scalar('loss', epoch_loss, epoch-1)


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def evaluate(model, loader, sampler, criterion=None, epoch=None, log=None):
    model.eval()
    loss, correct = 0, 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        output = torch.squeeze(model(data))

        if criterion is not None:
            loss += criterion(output, target.to(torch.float32)).data.item()

        pred = output > 0.5
        pred = pred.to(torch.int64)
        correct += pred.eq(target.data).cpu().sum()

    if criterion is not None:
        loss /= len(sampler)

    accuracy = 100. * correct.to(torch.float32) / len(sampler)

    # Horovod: average metric values across workers.
    loss = metric_average(loss, 'avg_loss')
    accuracy = metric_average(accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            loss, correct, len(loader.dataset), accuracy))

        if log is not None and epoch is not None:
            log.add_scalar('val_loss', loss, epoch-1)
            log.add_scalar('val_acc', accuracy, epoch-1)


input_image_size = (150, 150)

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


def get_train_loader(batch_size=25):
    if hvd.rank() == 0:
        print('Train: ', end="")
    train_dataset = datasets.ImageFolder(root=datapath+'/train',
                                         transform=data_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler, num_workers=4, pin_memory=True)

    if hvd.rank() == 0:
        print('Found', len(train_dataset), 'images belonging to',
              len(train_dataset.classes), 'classes')
    return train_loader, train_sampler


def get_validation_loader(batch_size=25):
    if hvd.rank() == 0:
        print('Validation: ', end="")
    val_dataset = datasets.ImageFolder(root=datapath+'/validation',
                                       transform=noop_transform)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            sampler=val_sampler, shuffle=False, num_workers=4)
    if hvd.rank() == 0:
        print('Found', len(val_dataset), 'images belonging to',
              len(val_dataset.classes), 'classes')
    return val_loader, val_sampler


def get_test_loader(batch_size=25):
    if hvd.rank() == 0:
        print('Test: ', end="")
    test_dataset = datasets.ImageFolder(root=datapath+'/test',
                                        transform=noop_transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             sampler=test_sampler, shuffle=False, num_workers=4)
    if hvd.rank() == 0:
        print('Found', len(test_dataset), 'images belonging to',
              len(test_dataset.classes), 'classes')
    return test_loader

if __name__ == '__main__':
    print('\nThis Python script is only for common functions. *DON\'T RUN IT DIRECTLY!* :-)')

