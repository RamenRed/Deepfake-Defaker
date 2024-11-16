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

import sys

MY_UTILS_PATH = 'F:/GitHub/Deepfake-Defaker/app'
if not MY_UTILS_PATH in sys.path:
    sys.path.append(MY_UTILS_PATH)
import defaker_api as df_api


#import app.defaker_api as df_api
#import torch.utils.data.dataloader as torchDL
from torch.utils.data import DataLoader
#import torch.utils.data.dataset as torchDS
from torch.utils.data import Dataset


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
r_seed = torch.normal(mean=0, std=d_noise, size = (num_examples,))

# Load training images
image_arrays = []


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
    return (dfd_average/len(opinions)) * 100

def gan_logic(dfg, dfd):
    pass

def img_to_tensor(image):
    t_form = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    t_img = t_form(image)
    return t_img

def tensor_array(images: list):
    tens_arr = []
    for each in images:
        t_add = img_to_tensor(each)
        tens_arr.append(t_add)
    return tens_arr

class images_data(Dataset):
    def __init__(self, images, transform=None, device=None):
        self.images = images
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        image = self.images[idx]

        i_to_t = img_to_tensor(image)
        if self.device:
            i_to_t=i_to_t.to(self.device)
        return i_to_t


dataset = images_data(image_arrays, device=device)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

x_entropy = nn.BCELoss()

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
    def __init__(self):
        super(Defaker_generator, self).__init__(self)
        optimizer = opt.Adam(self.parameters(), l_rate)
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
    
    def forward(self, gen_in):
        return self.model(gen_in)

class Defaker_discriminator(nn.Module):
    def __init__(self):
        super(Defaker_discriminator, self).__init__(self)
        self.optimizer = opt.Adam(self.parameters(), l_rate)
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
    
    def forward(self, discrim_in):
        return self.model(discrim_in)
    
    def loss(real, fake):
        r_loss = x_entropy(torch.ones_like(real), real)
        f_loss = x_entropy(torch.zeros_like(fake), fake)
        t_loss = r_loss + f_loss
        return f_loss
    



def trainer_function(dfd: Defaker_discriminator, dfg: Defaker_generator, dataloader):
    noise = torch.randn(max_batch_size, d_noise, 1, 1, device=device)
    Gen_losses = []
    Disc_losses = []
    for epoch in range(tot_epochs):
        for i, real_images in enumerate(dataloader):
            
            real_images = real_images.to(device)

            #Real image batch
            dfd.optimizer.zero_grad()
            label = torch.ones(real_images.size(0), device=device)
            output = dfd(real_images)
            d_loss_real = x_entropy(output, label)
            d_loss_real.backward()

            noise = torch.randn(real_images.size(0), d_noise, 1, 1, device=device)
            fakers = dfg(noise)
            label.fill_(0)
            output = dfd(fakers.detach())

            d_loss_fake = x_entropy(output, label)
            d_loss_fake.backward()
            
            d_loss_total = d_loss_real + d_loss_fake

            #Fake image batch
            dfg.optimizer.zero_grad()
            label.fill_(1)
            output = dfd(fakers)

            g_loss = x_entropy(output, label)
            g_loss.backward()
            dfg.optimizer.step()
            
            

            
            

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                % (epoch, tot_epochs, i, len(dataset.images),
                d_loss_total.item(), g_loss.item()))
            
            Gen_losses.append(g_loss.item())
            Disc_losses.append(d_loss_total.item())

            if (iters % 500 == 0) or ((epoch == tot_epochs-1) and (i == len(dataset.images)-1)):
                with torch.no_grad():
                    fake = dfg(noise).detach().cpu()
                dataset.images.append(vutils.make_grid(fake, padding=2, normalize=True))

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
    
    tensors_from_imgs: list = tensor_array(image_arrays)
    
    fake_images = dfg(d_noise)
    print(f"Fake Images Shape: {fake_images.shape}")

    d_output = dfd(fake_images)
    print(f"Discriminator Output Shape: {d_output.shape}")
    print(f"Discriminator Output: {d_output}")


    real_images = torch.randn(max_batch_size,3, 256, 256, device=device)

    d_real_output = dfd(real_images)
    
    print(f"Real Discriminator Output Shape: {d_real_output.shape}")
    print(f"Real Discriminator Output: {d_real_output}")

    trainer_function(dfd, dfg, dataloader)
    dfd_opinions: list = [True]
    return model_probability_opinion(dfd_opinions)

test_img_dir = 'F:/GitHub/Deepfake-Defaker/ui/background.jpg'
for image in os.listdir(test_img_dir):
    test_img = image        
test_num = run_model(test_img)
print(test_num)