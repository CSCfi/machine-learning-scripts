from __future__ import print_function
import torch

print('torch.__version__:', torch.__version__)
print('torch.cuda.is_available():', torch.cuda.is_available())

x = torch.rand(5, 3)
print(x.size())
y = torch.rand(5, 3)
print(x + y)

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x + y)
