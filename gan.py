from discriminator import Discriminator
from generator import Generator
import torch.nn as nn
import torch
from torch import optim
import matplotlib.pyplot as plt

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
# fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


class GAN:
    def __init__(self):
        self.device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.gen = Generator().to(self.device)
        self.dis = Discriminator().to(self.device)

        # self.gen.apply(self.weights_init)
        # self.dis.apply(self.weights_init)


    # def weights_init(self, m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         nn.init.normal_(m.conv.weight.data, 0.0, 0.02)
    #     elif classname.find('BatchNorm') != -1:
    #         nn.init.normal_(m.conv.weight.data, 1.0, 0.02)
    #         nn.init.constant_(m.bias.data, 0)


    def train(self):
        x = torch.randn((8, 3, 256, 256), dtype=self.dtype, device=self.device)
        mask = torch.zeros((8, 1, 256, 256), dtype=self.dtype, device=self.device)

        out = self.gen(x, mask)
        img = out[0].permute(1,2,0)
        img = img.cpu()
        img = img.detach().numpy()
        img = (img /2) + 0.5
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    gan = GAN()
    gan.train()