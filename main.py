from model import *
from train import *

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import math
import matplotlib.pyplot as plt
import itertools

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def test_model(generator: nn.Module):
    
    # generate some images
    num_test_samples = 16
    test_noise = Variable(torch.randn(num_test_samples, 100).cuda())
    test_images = generator(test_noise)


    # create figure for plotting
    size_figure_grid = int(math.sqrt(num_test_samples))
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)

        for k in range(num_test_samples):
            i = k//4
            j = k%4
            ax[i,j].cla()
        pass

if __name__=="__main__":
    generator = Generator()
    discriminator = Discriminator()

    # Create the dataloaders
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)    
    print("training loader is created!")