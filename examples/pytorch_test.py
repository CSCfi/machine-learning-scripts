# Script for testing the PyTorch setup

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

from torchtext import datasets
import torchtext.transforms as T
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from packaging.version import Version as LV
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

print('Using PyTorch version:', torch.__version__)
assert(LV(torch.__version__) >= LV("2.0"))

if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.') 
    device = torch.device('cpu')

# Create some tensors
x = torch.ones(3, 4)
data = [[1, 2, 3],[4, 5, 6]]
y = torch.tensor(data, dtype=torch.float)

# Copy them to the GPU
x = x.to(device)
y = y.to(device)

# Perform matrix multiplication on GPU
z = y.matmul(x)

print("z =", z)
