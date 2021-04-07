# coding: utf-8

# Dogs-vs-cats classification with CNNs

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

from datetime import datetime

from pytorch_dvc_cnn import get_train_loader, get_validation_loader, \
    get_test_loader


# Option 1: Train a small CNN from scratch

class Net(pl.LightningModule):
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
        x = torch.sigmoid(self.fc2(x))
        return x.type_as(x)

    def shared_step(self, batch):
        data, target = batch
        output = self(data)
        output = torch.squeeze(output)
        loss = F.binary_cross_entropy(output, target.to(torch.float32))
        self.log('train_loss', loss)
        return output, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.shared_step(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output, loss = self.shared_step(batch)
        acc = FM.accuracy(output, target)

        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)

        metrics = {'test_acc': metrics['val_acc'],
                   'test_loss': metrics['val_loss']}
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.05)


def main():
    model = Net()

    batch_size = 25
    train_loader = get_train_loader(batch_size)
    validation_loader = get_validation_loader(batch_size)

    trainer = pl.Trainer(gpus=-1, max_epochs=50, accelerator='ddp')
    # trainer = pl.Trainer(gpus=1, max_epochs=50, accelerator='horovod', checkpoint_callback=False)

    start_time = datetime.now()
    trainer.fit(model, train_loader, validation_loader)
    end_time = datetime.now()
    print('Total training time: {}.'.format(end_time - start_time))

    # torch.save(model.state_dict(), model_file)
    # print('Wrote model to', model_file)

    test_loader = get_test_loader(batch_size)
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    main()
