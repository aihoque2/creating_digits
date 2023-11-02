from model import *

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import math
import matplotlib.pyplot as plt
import numpy as np

def test_model(generator: nn.Module):
    pass

if __name__=="__main__":
    generator = Generator()
    discriminator = Discriminator()
    print("I did the  main method!")