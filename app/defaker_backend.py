from torch import *
from torchvision import *

# =======================================================
#                 Initial Parameters
# =======================================================

# Initial value for Noise Dimension
d_noise = 100

# Number of epochs used for training
tot_epochs = 50

# Max Batch
max_batch_size = 512

# Check if NVIDIA GPU is available for use
use_cuda = torch.cuda.is_available()

# =======================================================
#                  Helper Functions
# =======================================================

def model_probability_opinion(dfd): # Used to calculate how many times out of 20 the discriminator detected an image as fake
    dfd_average = 0 # Initial value of zero
    cycle_count = 0
    while cycle_count < 20:
        # If discriminator.decision == FALSE; dfd_average++
        cycle_count += 1
    return dfd_average/20

# =======================================================
#                      Models
# =======================================================

class Defaker_generator(nn.Module):
    def __init__(self):
        super().__init__(self)
        self.model = nn.Sequential(
        )

class Defaker_discriminator(nn.Module):
    def __init__(self, d_noise):
        super().__init__(self, d_noise)
        self.model = nn.Sequential(
        )

# =======================================================

if __name__=="__main__":
    dfg = Defaker_generator()
    dfd = Defaker_discriminator()
