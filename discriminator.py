import torch
from torch import nn

from layers import SpectralConv2D


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        cnum = 64
        self.conv1 = SpectralConv2D(4, cnum)
        self.conv2 = SpectralConv2D(cnum, 2 * cnum)
        self.conv3 = SpectralConv2D(2 * cnum, 4 * cnum)
        self.conv4 = SpectralConv2D(4 * cnum, 4 * cnum)
        self.conv5 = SpectralConv2D(4 * cnum, 4 * cnum)
        self.conv6 = SpectralConv2D(4 * cnum, 4 * cnum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.flatten(x)
        return x
