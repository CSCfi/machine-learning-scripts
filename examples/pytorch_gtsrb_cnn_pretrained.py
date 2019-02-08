# coding: utf-8

# Traffic sign classification with CNNs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from datetime import datetime

from pytorch_gtsrb_cnn import get_train_loader, get_validation_loader, get_test_loader
from pytorch_gtsrb_cnn import device, train, evaluate, get_tensorboard

model_file = 'gtsrb_pretrained_cnn.pt'
model_file_ft = 'gtsrb_pretrained_finetune.pt'


# Option 2: Reuse a pre-trained CNN

class PretrainedNet(nn.Module):
    def __init__(self):
        super(PretrainedNet, self).__init__()
        self.vgg_features = models.vgg16(pretrained=True).features

        # Freeze the VGG16 layers
        for param in self.vgg_features.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(512*2*2, 256)
        self.fc2 = nn.Linear(256, 43)

    def forward(self, x):
        x = self.vgg_features(x)

        # flattened 2D to 1D
        x = x.view(-1, 512*2*2)

        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_main():
    # Learning 1: New layers

    model = PretrainedNet().to(device)

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(params, lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print(model)

    batch_size = 50
    train_loader = get_train_loader(batch_size)
    validation_loader = get_validation_loader(batch_size)

    log = get_tensorboard('pretrained')
    epochs = 20

    start_time = datetime.now()
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch, log)

        with torch.no_grad():
            print('\nValidation:')
            evaluate(model, validation_loader, criterion, epoch, log)

    end_time = datetime.now()
    print('Total training time: {}.'.format(end_time - start_time))

    torch.save(model.state_dict(), model_file)
    print('Wrote model to', model_file)

    # Learning 2: Fine-tuning
    log = get_tensorboard('finetuned')

    for name, layer in model.vgg_features.named_children():
        note = ' '
        for param in layer.parameters():
            note = '-'
            if int(name) >= 24:
                param.requires_grad = True
                note = '+'
        print(name, note, layer, len(param))

    params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=1e-3)
    optimizer = optim.RMSprop(params, lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    print(model)

    prev_epochs = epoch
    epochs = 20

    start_time = datetime.now()
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, prev_epochs+epoch, log)

        with torch.no_grad():
            print('\nValidation:')
            evaluate(model, validation_loader, criterion, prev_epochs+epoch, log)

    end_time = datetime.now()
    print('Total training time: {}.'.format(end_time - start_time))

    torch.save(model.state_dict(), model_file_ft)
    print('Wrote finetuned model to', model_file_ft)


def test_main():
    model = PretrainedNet()
    model.load_state_dict(torch.load(model_file))
    model.to(device)

    test_loader = get_test_loader(50)

    print('=========')
    print('Pretrained:')
    with torch.no_grad():
        evaluate(model, test_loader)

    model = PretrainedNet()
    model.load_state_dict(torch.load(model_file_ft))
    model.to(device)

    print('=========')
    print('Finetuned:')
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
