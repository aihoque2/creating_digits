import torch
from torch import nn
import torch.optim as optim
import torch.utils
from torch.autograd import Variable

from model import *

"""
train.py

this file is meant to be where the Generator 
and Discriminator have their zero-sum game 
"""

class Trainer:

    def __init__(self, generator: nn.Module, discriminator: nn.Module, num_epochs: int, lr: float, train_loader: torch.utils.data.DataLoader, z_dim: int = 100):
    
        self.generator = generator
        self.discriminator = discriminator
        self.num_epochs = num_epochs
        self.z_dim = z_dim
        self.loss_fn = nn.BCELoss() # AKA self.criterion to many ppl
        self.lr = lr

        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr) 
        self.G_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)

        self.data_loader = train_loader

    def train(self):
        """
        full training process       
        """
        for epoch in range(self.num_epochs):
            for idx, (images, labels)  in enumerate(self.data_loader):
                real_images = Variable(images.cuda())
                real_labels = Variable(torch.ones(images.size(0)).cuda())

                # Sample from generator
                noise = Variable(torch.randn(images.size(0), self.z_dim).cuda())
                fake_images = self.generator(noise)
                fake_labels = Variable(torch.zeros(images.size(0)).cuda())

                # train discriminator
                discrim_loss, real_score, fake_score = self.disc_step(real_images, real_labels, fake_images, fake_labels)

                # Re-sample from generator and get the discriminator's thoughts
                new_noise = Variable(torch.randn(images.size(0), self.z_dim).cuda())
                fake_images = self.generator(new_noise).cuda()
                outputs = self.discriminator(fake_images)
                gen_loss = self.gen_step(outputs, real_labels)

            # print the loss statistics for this step
            print("Here's the Epoch: ", epoch)
            print("Discriminator Statistics:\n\t Discriminator Loss: {},\n Real Loss: {},\n Fake Loss: {}\n".format(discrim_loss, real_score, fake_score))
            
            print("Generator Statistics:\n\t Generator Loss: ", gen_loss)

    def disc_step(self, real_images, real_labels, fake_images, fake_labels):
        """
        training step of Discriminator model
        """
        self.D_optimizer.zero_grad()

        probs = self.discriminator(real_images) 
        real_L = self.loss_fn(probs, real_labels) # real loss
        real_score = probs

        probs = self.discriminator(fake_images)
        fake_L = self.loss_fn(probs, fake_labels)
        fake_score = probs

        total_L = real_L + fake_L
        total_L.backward() # back propogate the loss
        self.D_optimizer.step() # perform a step
        return total_L, real_score, fake_score

    def gen_step(self, discriminator_outputs, real_labels):
        """
        training step of Generator model
        """
        self.G_optimizer.zero_grad()
        
        generator_L = self.loss_fn(discriminator_outputs, real_labels)
        generator_L.backward()
        self.G_optimizer.step()
        return generator_L
