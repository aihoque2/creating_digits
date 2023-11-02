import torch
from torch import nn

from model import *

"""
train.py

this file is meant to be where the Generator 
and Discriminator have their zero-sum game 
"""

class Trainer:
    def __init__(self, generator: nn.Module, discriminator: nn.Module, n_epochs: int):
        self.generator = generator
        self.discriminator = discriminator
        self.n_epochs = n_epochs

    def train(self):
        pass