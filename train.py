import torch
from torch import nn
import torch.optim as optim

from model import *

"""
train.py

this file is meant to be where the Generator 
and Discriminator have their zero-sum game 
"""

class Trainer:

    def __init__(self, generator: nn.Module, discriminator: nn.Module, n_epochs: int, lr: float, z_dim: int = 100):
    
        self.generator = generator
        self.discriminator = discriminator
        self.n_epochs = n_epochs
        self.z_dim = z_dim

        self.criterion = nn.BCELoss()

        self.lr = lr
        self.G_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr) 

    def train(self):
        """
        full training process       
        """
        pass

    def train_disc(self):
        """
        training step of Discriminator model
        """
        pass

    def train_gen(self):
        """
        training step of Generator model
        """
        pass
