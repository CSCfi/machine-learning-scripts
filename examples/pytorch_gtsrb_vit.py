#!/usr/bin/env python
# coding: utf-8

# # Traffic sign classification with CNNs
# 
# In this script, we'll finetune a Vision Transformer
# (https://arxiv.org/abs/2010.11929) (ViT) to classify images of
# traffic signs using PyTorch and HuggingFace Transformers:
# https://github.com/huggingface/transformers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from packaging.version import Version as LV
from datetime import datetime
import os
import sys

from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import __version__ as transformers_version

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__,
      'Transformers version:', transformers_version,
      'Device:', device)
assert LV(torch.__version__) >= LV("1.0.0")

#
# There are some broken folders, but we need to keep the class indices
# the same. We created a custom Dataset class to handle this.
#
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
        output = model(data).logits #.squeeze()

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
            output = model(data).logits #.squeeze()

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


class ImageClassificationCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        data = self.processor([x[0] for x in batch], do_rescale=False,
                              return_tensors='pt').pixel_values
        targets = torch.tensor([x[1] for x in batch]) #, dtype=torch.float32)
        return data, targets


def main():
    # TensorBoard for logging
    try:
        import tensorboardX
        time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logdir = os.path.join(os.getcwd(), "logs", "gtsrb-vit-" + time_str)
        print('TensorBoard log directory:', logdir)
        os.makedirs(logdir)
        log = tensorboardX.SummaryWriter(logdir)
    except ImportError:
        log = None

    # The training dataset consists of 2000 images of dogs and cats, split
    # in half.  In addition, the validation set consists of 1000 images,
    # and the test set of 22000 images.
    #
    # First, we'll resize all training and validation images to a fixed
    # size.
    #
    # Then, to make the most of our limited number of training examples,
    # we'll apply random transformations to them each time we are looping
    # over them. This way, we "augment" our training dataset to contain
    # more data. There are various transformations available in
    # torchvision, see:
    # https://pytorch.org/docs/stable/torchvision/transforms.html

    datapath = os.getenv('DATADIR')
    if datapath is None:
        print("Please set DATADIR environment variable!")
        sys.exit(1)
    datapath = os.path.join(datapath, 'gtsrb/train-5535')

    # Data loaders
    batch_size = 32

    vitmodel = 'google/vit-base-patch16-224'
    processor = AutoImageProcessor.from_pretrained(vitmodel)
    collator = ImageClassificationCollator(processor)

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

    # Define the network and training parameters
    model = ViTForImageClassification.from_pretrained(
        vitmodel, num_labels=43, ignore_mismatched_sizes=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    print(model)

    num_epochs = 5

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
