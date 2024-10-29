from torch import *
import torch
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
import app.defaker_api as df_api

# With help from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

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
l_rate = 0.001

# Check if NVIDIA GPU is available for use
use_cuda = torch.cuda.is_available()

# GPU's in machine
num_gpu = 1

# Decide what the device will run
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

# Prevent deterministic outputs and algorithms
torch.use_deterministic_algorithms(False)

# Number of examples
num_examples = 32

# Random seed for use
r_seed = torch.normal(num_examples, d_noise)

# Load training images
image_arrays, image_names = df_api.get_van_gogh_paintings()

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

x_entropy = nn.CrossEntropyLoss()

def weights_init(weights_inst):
    c_name = weights_inst.__class__.__name__
    if c_name.find('Conv') != -1:
        nn.init.normal_(weights_inst.weight.data, 0.0, 0.02)
    elif c_name.find('BatchNorm') != -1:
        nn.init.normal_(weights_inst.weight.data, 1.0, 0.02)
        nn.init.constant_(weights_inst.bias.data, 0)
# =======================================================
#                      Models
# =======================================================

class Defaker_generator(nn.Module):
    def __init__(self, image_data: list[Tensor], num_gpu):
        super(Generator, self).__init__(self, image_data, num_gpu)
        optimizer = opt.Adam(self.parameters(), l_rate)
        self.num_gpu = num_gpu
        self.image_data = image_data
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def loss(fake):
        return x_entropy(torch.ones_like(fake), fake)
    
    def gen_forward(self, gen_in):
        return self.model(gen_in)

class Defaker_discriminator(nn.Module):
    def __init__(self, d_noise, image_data: list[Tensor], num_gpu):
        super().__init__(self, d_noise, image_data, num_gpu)
        self.optimizer = opt.Adam(self.parameters(), l_rate)
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
    
    def discrim_forward(self, discrim_in):
        return self.model(discrim_in)
    
    def loss(real, fake):
        r_loss = x_entropy(torch.ones_like(real), real)
        f_loss = x_entropy(torch.zeros_like(fake), fake)
        t_loss = r_loss + f_loss
        return f_loss
    
    def trainer_function(self, images, train_load):
        noise = torch.normal(num_examples, d_noise)
        for epoch in range(tot_epochs):
            run_loss = 0
            prev_loss = 0
            for i, data in enumerate(train_load):

                #Real image batch
                inputs, labels = data
                self.optimizer.zero_grad()
                out_images = Defaker_discriminator(image_data=images)
                loss = x_entropy

                
                

                

# =======================================================

if __name__=="__main__":
    dfg = Defaker_generator().to(device)
    dfd = Defaker_discriminator().to(device)

    # Setup use of multiple GPU's if present and capable for Discriminator and Generator 
    if (device.type == "cuda") and (num_examples >1):
        dfg = torch.nn.parallel.DistributedDataParallel
    if (device.type == "cuda") and (num_examples >1):
        dfd = torch.nn.parallel.DistributedDataParallel

    # Initialize Weights
    dfg.apply(weights_init)
    dfd.apply(weights_init)

    dfd_opinions: list = []
