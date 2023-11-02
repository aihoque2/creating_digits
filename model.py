import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import math
import matplotlib.pyplot as plt

torch.manual_seed(111)

class Discriminator(nn.Module):
    def __init__(self, input_size, out_size):
        pass

    def forward(self, x):
        pass

class Generator(nn.Module):
    def __init__(self, input_size, out_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, out_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return F.tanh(self.fc4(x))
    