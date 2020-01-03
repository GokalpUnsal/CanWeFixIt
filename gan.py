from .discriminator import Discriminator
from .generator import Generator
import torch.nn as nn
import torch
from torch import optim


class GAN:
    def __init__(self, device):
        self.device = device
        self.dtype = torch.float32
        self.gen = Generator().to(self.device)
        self.dis = Discriminator().to(self.device)

        # Number of training epochs
        self.num_epochs = 5

        # Learning rate for optimizers
        self.lr = 0.0002

        # Beta1 hyperparam for Adam optimizers
        self.beta1 = 0.5

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        # fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = 0

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.dis.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

    # def weights_init(self, m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         nn.init.normal_(m.conv.weight.data, 0.0, 0.02)
    #     elif classname.find('BatchNorm') != -1:
    #         nn.init.normal_(m.conv.weight.data, 1.0, 0.02)
    #         nn.init.constant_(m.bias.data, 0)


    def train(self, data):
        x = torch.randn((8, 3, 256, 256), dtype=self.dtype, device=self.device)
        mask = torch.zeros((8, 1, 256, 256), dtype=self.dtype, device=self.device)
        for epoch in range(self.num_epochs):
            pass
        # out = self.gen(x, mask)
        # img = out[0].permute(1,2,0)
        # img = img.cpu()
        # img = img.detach().numpy()
        # img = (img /2) + 0.5
        # plt.imshow(img)
        # plt.show()


if __name__ == '__main__':
    gan = GAN()
    gan.train()