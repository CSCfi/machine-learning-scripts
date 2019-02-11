# coding: utf-8

# Dogs-vs-cats classification with CNNs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import horovod.torch as hvd

from pytorch_dvc_cnn_hvd import get_train_loader, get_validation_loader, get_test_loader
from pytorch_dvc_cnn_hvd import device, train, evaluate, get_tensorboard

model_file = 'dvc_simple_cnn_hvd.pt'


# Option 1: Train a small CNN from scratch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(32, 64, (3, 3))
        self.pool3 = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(17*17*64, 64)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # "flatten" 2D to 1D
        x = x.view(-1, 17*17*64)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        return torch.sigmoid(self.fc2(x))


def train_main():
    model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.05)

    if hvd.rank() == 0:
        print(model)

    # Horovod: broadcast parameters.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # Horovod: scale learning rate by the number of GPUs.
    lr = 0.05
    optimizer = optim.SGD(model.parameters(), lr=lr * hvd.size())

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters())
    criterion = nn.BCELoss()

    batch_size = 25
    train_loader, train_sampler = get_train_loader(batch_size)
    validation_loader, validation_sampler = get_validation_loader(batch_size)

    log = get_tensorboard('simple_hvd')
    epochs = 50

    start_time = datetime.now()
    for epoch in range(1, epochs + 1):
        train(model, train_loader, train_sampler, criterion, optimizer, epoch, log)

        with torch.no_grad():
            if hvd.rank() == 0:
                print('\nValidation:')
            evaluate(model, validation_loader, validation_sampler, criterion, epoch, log)

    end_time = datetime.now()

    if hvd.rank() == 0:
        print('Total training time: {}.'.format(end_time - start_time))
        torch.save(model.state_dict(), model_file)
        print('Wrote model to', model_file)


def test_main():
    model = Net()
    model.load_state_dict(torch.load(model_file))
    model.to(device)

    test_loader = get_test_loader(25)

    print('=========')
    print('Test set:')
    with torch.no_grad():
        evaluate(model, test_loader)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        test_main()
    else:
        train_main()
