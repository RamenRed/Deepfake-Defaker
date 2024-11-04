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

def img_to_tensor(image):
    t_form = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    t_img = t_form(image)
    return t_img

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
    



def trainer_function(dfd: Defaker_discriminator, dfg: Defaker_generator, images: list[Tensor], train_load, fake):
    noise = torch.normal(num_examples, d_noise)
    Gen_losses = []
    Disc_losses = []
    for epoch in range(tot_epochs):
        run_loss = 0
        prev_loss = 0
        for i, data in enumerate(train_load):

            #Real image batch
            inputs, labels = data
            dfd.zero_grad()
            r_cpu = data[0].to(device)
            b_size = r_cpu.size(0)
            label = torch.full((b_size,), dtype = torch.float, device=device)
            out_images = dfd(image_data=images)
            r_loss = x_entropy
            r_loss.backward()
            D_x = out_images.mean().item()

            #Fake image batch
            noise = torch.randn(b_size, 1, 1, device=device)
            fakers = dfg(noise)
            
            label.fill_(f_label)

            out_images = dfd(fakers.detach()).view(-1)

            f_loss = x_entropy
            f_loss.backward()

            D_G_z1 = out_images.mean().item()

            d_loss = r_loss + f_loss

            dfd.optimizer.step()

            dfg.zero_grad()

            out_images = dfd(fakers).view(-1)


            g_loss = x_entropy(out_images)
            g.backward()

            D_G_z2 = out_images.mean().item()

            
            dfg.optimizer.step()
            

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, tot_epochs, i, len(train_load),
                d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
            
            Gen_losses.append(g_loss.item())
            Disc_losses.append(d_loss.item())

            if (iters % 500 == 0) or ((epoch == tot_epochs-1) and (i == len(train_load)-1)):
                with torch.no_grad():
                    fake = dfg(noise).detach().cpu()
                images.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

                
                






                
                

                

# =======================================================

def run_model(image):
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
    
    tensors_from_imgs: list = [Tensor]
    for i in image_arrays:
        tensors_from_imgs.append(img_to_tensor(i))
    trainer_function(dfd, dfg, tensors_from_imgs, )
    dfd_opinions: list = []
