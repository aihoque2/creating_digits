import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import math
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(111)
EPS = 0.003

def fanin_init(size, fanin=None):
	"""
	Set the weights in some random numbers 
	uniformly distributed
	"""
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Discriminator(nn.Module):
    """
    Discriminator Model. The neural net that outputs
    probability that the data came from the dataset 
    samples
    """

    def __init__(self, input_size):

        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())

        if torch.cuda.is_available():
            self.cuda() 

    def forward(self, x):
        """
        calculate probability that the data came from 
        the dataset, not the Generator
        """
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return F.sigmoid(self.fc4(x))


class Generator(nn.Module):
    """
    Generator Model. This is the model that generates
    the data we wish to make look like our dataset

    comes with its own probability distribution of generated images!
    """

    def __init__(self, input_size, out_size):
        
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(256, 512)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(512, 1024)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(1024, out_size)
        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, z):
        """
        generate the new image from the noise sample
        """
        z = F.leaky_relu(self.fc1(z), 0.2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        return F.tanh(self.fc4(z))
    