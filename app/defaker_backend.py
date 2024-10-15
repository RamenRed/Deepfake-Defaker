from torch import *
import torch.nn as nn
import torch.nn.parallel
import torch.optim as opt
from torchvision import *
import torch.utils.data as u_data
import torchvision.datasets as tv_dset
import torchvision.transforms as tv_transforms
import torchvision.utils as vutils
import numpy as np
import random

# =======================================================
#                 Initial Parameters
# =======================================================

# Initial value for Noise Dimension
d_noise = 100

# Number of epochs used for training
tot_epochs = 50

# Max Batch
max_batch_size = 512

# Rate of Learning
l_rate = 0.01

# Check if NVIDIA GPU is available for use
use_cuda = torch.cuda.is_available()

# GPU's in machine
num_gpu = 1

# Decide what the device will run
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

torch.use_deterministic_algorithms(False)

# =======================================================
#                  Helper Functions
# =======================================================

def model_probability_opinion(opinions: list): # Used to calculate how many times out of length(opinions) the discriminator detected an image as fake
    dfd_average = 0 # Initial value of zero
    cycle_count = 0
    while cycle_count < len(opinions):
        if opinions[cycle_count] == False:
            dfd_average += 1
        cycle_count += 1
    return dfd_average/len(opinions)

def gan_logic(dfg, dfd):
    pass

# =======================================================
#                      Models
# =======================================================

class Defaker_generator(nn.Module):
    def __init__(self, image_data: list[Tensor], num_gpu):
        super(Generator, self).__init__(self, image_data, num_gpu)
        self.num_gpu = num_gpu
        self.image_data = image_data
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        

class Defaker_discriminator(nn.Module):
    def __init__(self, d_noise, image_data: list[Tensor], num_gpu):
        super().__init__(self, d_noise, image_data, num_gpu)
        self.num_gpu = num_gpu
        self.image_data = image_data
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

# =======================================================

if __name__=="__main__":
    dfg = Defaker_generator()
    dfd = Defaker_discriminator()
    dfd_opinions: list = []
